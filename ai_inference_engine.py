import os
import sys
import time
import json
import pickle
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime
import pika
from fastapi import FastAPI

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import layers

app = FastAPI(title="AI Inference Engine")

# -- Model Configs --
MODEL_DIR = 'deploy_models'
scaler_path = os.path.join(MODEL_DIR, 'minmax_scaler.pkl')
if_path = os.path.join(MODEL_DIR, 'isolation_forest.pkl')
encoder_path = os.path.join(MODEL_DIR, 'vae_encoder.keras')
decoder_path = os.path.join(MODEL_DIR, 'vae_decoder.keras')
config_path = os.path.join(MODEL_DIR, 'soar_config.json')

# Global Variables
scaler = None
iso_forest = None
encoder = None
decoder = None
vae_threshold = 0.0
feature_cols = []
contamination_ratio = 0.0
stream_df = None
is_streaming = True

# Global for Adaptive Threshold
mae_history_buffer = []
BUFFER_SIZE = 100
ALPHA = 0.95

# Custom Layer
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# RabbitMQ Publisher
class RabbitMQPublisher:
    def __init__(self, queue_name='soar_events_queue'):
        self.queue_name = queue_name
        self.connection = None
        self.channel = None
        self._connect()

    def _connect(self):
        try:
            self.connection = pika.BlockingConnection(pika.ConnectionParameters('127.0.0.1'))
            self.channel = self.connection.channel()
            self.channel.queue_declare(queue=self.queue_name, durable=True)
            print(f"[RabbitMQ] Connected to queue: {self.queue_name}")
        except Exception as e:
            print(f"[RabbitMQ Error] Connection failed: {e}")

    def publish(self, message: dict):
        if not self.connection or self.connection.is_closed:
            self._connect()
        if self.channel:
            try:
                self.channel.basic_publish(
                    exchange='',
                    routing_key=self.queue_name,
                    body=json.dumps(message)
                )
            except Exception as e:
                print(f"[RabbitMQ Error] Failed to publish message: {e}")

rabbitmq_pub = None

def get_vae_mae(x_scaled):
    if encoder is None or decoder is None:
        return 0.0
    z_mean, _, _ = encoder(x_scaled, training=False)
    reconstruction = decoder(z_mean, training=False)
    mae = np.mean(np.abs(x_scaled - reconstruction.numpy()), axis=1)
    return float(mae[0])

@app.on_event("startup")
async def startup_event():
    global scaler, iso_forest, encoder, decoder, vae_threshold, feature_cols, contamination_ratio, stream_df, rabbitmq_pub
    
    print("Initializing Models and RabbitMQ...")
    rabbitmq_pub = RabbitMQPublisher()
    
    if not os.path.exists(scaler_path) or not os.path.exists(encoder_path):
        print("[⚠️ WARNING] Models not found. Ensure Notebook has been run.")
        return
        
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(if_path, 'rb') as f:
            iso_forest = pickle.load(f)
        with open(config_path, 'r', encoding='utf-8') as f:
            soar_config = json.load(f)
            
        encoder = tf.keras.models.load_model(encoder_path, custom_objects={'Sampling': Sampling}, compile=False)
        decoder = tf.keras.models.load_model(decoder_path, compile=False)
        
        vae_threshold = soar_config['vae_threshold']
        feature_cols = soar_config['feature_cols']
        contamination_ratio = soar_config['contamination_ratio']
        
        df = pd.read_csv('lms_training_dataset_final.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by=['ip', 'timestamp']).reset_index(drop=True)
        df['time_gap'] = df.groupby('ip')['timestamp'].diff().dt.total_seconds().fillna(300.0)

        df_idx = df.set_index('timestamp')
        mean_gap = df_idx.groupby('ip')['time_gap'].rolling('5min', min_periods=1).mean().reset_index()
        std_gap = df_idx.groupby('ip')['time_gap'].rolling('5min', min_periods=1).std().reset_index()

        df['time_gap_per_IP'] = mean_gap['time_gap'].values
        df['req_regularity_per_IP'] = std_gap['time_gap'].fillna(300.0).values
        df['hour_of_day'] = df['timestamp'].dt.hour
        
        normal_df = df[df['Label'] == 0]
        malicious_df = df[df['Label'] == 1]
        _, val_normal_df = np.split(normal_df, [int(0.8*len(normal_df))])
        
        stream_df = pd.concat([val_normal_df, malicious_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print("[✓] AI Models and Log Data loaded successfully!")
        
        # Start streaming task
        asyncio.create_task(log_stream_task())
        
    except Exception as e:
        print(f"[❌ ERROR] Model loading error: {e}")

async def log_stream_task():
    global vae_threshold, mae_history_buffer
    stream_index = 0
    
    while True:
        if is_streaming and stream_df is not None and stream_index < len(stream_df):
            row = stream_df.iloc[stream_index]
            stream_index += 1
            
            ip = row['ip']
            timestamp_str = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            actual_label = int(row['Label'])
            
            feature_vector = row[feature_cols].values.reshape(1, -1).astype(np.float32)
            feature_vector_scaled = scaler.transform(feature_vector)
            
            vae_error = get_vae_mae(feature_vector_scaled)
            
            # Adaptive threshold
            mae_history_buffer.append(vae_error)
            if len(mae_history_buffer) > BUFFER_SIZE:
                mae_history_buffer.pop(0)
            if len(mae_history_buffer) >= 20:
                p95_current = float(np.percentile(mae_history_buffer, 95))
                vae_threshold = float(ALPHA * vae_threshold + (1 - ALPHA) * p95_current)
                
            is_vae_anomaly = bool(vae_error > vae_threshold)
            
            if_pred = iso_forest.predict(feature_vector_scaled)[0]
            is_if_anomaly = bool(if_pred == -1)
            
            event_data = {
                "timestamp": timestamp_str,
                "ip": ip,
                "vae_error": float(vae_error),
                "is_vae_anomaly": is_vae_anomaly,
                "is_if_anomaly": is_if_anomaly,
                "label": actual_label,
                "vae_threshold": float(vae_threshold),
                "features": {col: float(row[col]) for col in feature_cols}
            }
            
            if rabbitmq_pub:
                rabbitmq_pub.publish(event_data)
                print(f"[AI] Sent log for IP {ip} to RabbitMQ (VAE Anomaly: {is_vae_anomaly})")
                
        await asyncio.sleep(1)

@app.post("/api/engine/toggle")
def toggle_stream():
    global is_streaming
    is_streaming = not is_streaming
    return {"is_streaming": is_streaming}

@app.get("/api/engine/status")
def status():
    return {"is_streaming": is_streaming, "model_loaded": stream_df is not None}

if __name__ == "__main__":
    import uvicorn
    print("\n=======================================================")
    print(" AI INFERENCE ENGINE ĐÃ SẴN SÀNG KHỞI CHẠY!")
    print(" API: http://127.0.0.1:8000")
    print("=======================================================\n")
    uvicorn.run(app, host="127.0.0.1", port=8000)
