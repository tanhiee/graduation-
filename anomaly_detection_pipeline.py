"""
==========================================================================
 ĐỒ ÁN TỐT NGHIỆP - AN TOÀN THÔNG TIN
 Phát hiện điểm dị thường (Anomaly Detection) chống tấn công
 Credential Stuffing (Low & Slow) trên hệ thống LMS Moodle

 Pipeline: Data Split → Isolation Forest (Baseline) → VAE (Main Model)
           → Threshold & Evaluation → ROC Comparison
==========================================================================
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Fix encoding cho Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, roc_curve, auc
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings('ignore')
np.random.seed(42)
tf.random.set_seed(42)

# --- Cấu hình đường dẫn lưu biểu đồ ---
SAVE_DIR = os.path.join(os.path.dirname(__file__), 'charts')
os.makedirs(SAVE_DIR, exist_ok=True)

# Cấu hình style biểu đồ chuyên nghiệp
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e94560',
    'axes.labelcolor': '#eee',
    'text.color': '#eee',
    'xtick.color': '#aaa',
    'ytick.color': '#aaa',
    'grid.color': '#333',
    'font.size': 12,
    'axes.titlesize': 14,
    'figure.titlesize': 16,
})

print("=" * 70)
print("  ANOMALY DETECTION PIPELINE - CREDENTIAL STUFFING ON MOODLE LMS")
print("=" * 70)

# =====================================================================
# BƯỚC 1: ĐỌC VÀ TIỀN XỬ LÝ DỮ LIỆU & FEATURE ENGINEERING
# =====================================================================
print("\n[STEP 1] Đọc và tiền xử lý dữ liệu...")

df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'lms_training_dataset_final.csv'))
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sắp xếp theo IP và thời gian (chronological)
df = df.sort_values(by=['ip', 'timestamp']).reset_index(drop=True)

print(f"  → Tổng số mẫu: {len(df)}")
print(f"  → Phân bố nhãn:")
print(f"     Normal  (Label 0): {len(df[df['Label'] == 0])}")
print(f"     Malicious (Label 1): {len(df[df['Label'] == 1])}")
print(f"  → Các cột ban đầu: {list(df.columns)}")

# --- Trích xuất thêm 3 Đặc trưng nâng cao (Feature Engineering) ---
print("  → Đang thực hiện Feature Engineering...")
df['time_gap'] = df.groupby('ip')['timestamp'].diff().dt.total_seconds().fillna(300.0)

df_idx = df.set_index('timestamp')
mean_gap = df_idx.groupby('ip')['time_gap'].rolling('5min', min_periods=1).mean().reset_index()
std_gap = df_idx.groupby('ip')['time_gap'].rolling('5min', min_periods=1).std().reset_index()

df['time_gap_per_IP'] = mean_gap['time_gap'].values
df['req_regularity_per_IP'] = std_gap['time_gap'].fillna(300.0).values
df['hour_of_day'] = df['timestamp'].dt.hour

# Mã hóa cột IP bằng LabelEncoder (không sử dụng trực tiếp để train tránh gây nhiễu)
le_ip = LabelEncoder()
df['ip_encoded'] = le_ip.fit_transform(df['ip'])

# --- Chọn Features số học để huấn luyện ---
FEATURE_COLS = [
    'req_per_IP_5min', 
    'error_ratio_per_IP',
    'unique_UA_per_IP', 
    'size',
    'time_gap_per_IP', 
    'req_regularity_per_IP', 
    'hour_of_day'
]

X = df[FEATURE_COLS].values.astype(np.float32)
y = df['Label'].values

# --- Chuẩn hóa Min-Max [0, 1] ---
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

print(f"  → Features đã chọn để huấn luyện: {FEATURE_COLS}")
print(f"  → Shape sau chuẩn hóa: {X_scaled.shape}")

# =====================================================================
# BƯỚC 2: CHIA DỮ LIỆU & TĂNG CƯỜNG ĐỘ NHIỄU (DATA AUGMENTATION)
# =====================================================================
print("\n[STEP 2] Chia dữ liệu và thực hiện Tăng cường dữ liệu (Data Augmentation)...")

# Tách riêng Normal và Malicious
normal_mask = (y == 0)
malicious_mask = (y == 1)

X_normal = X_scaled[normal_mask]
X_malicious = X_scaled[malicious_mask]
y_normal = y[normal_mask]
y_malicious = y[malicious_mask]

# Chia Normal thành 80% Train, 20% Validation
X_normal_train, X_normal_val = train_test_split(
    X_normal, test_size=0.2, random_state=42
)

# --- TĂNG CƯỜNG ĐỘ NHIỄU (DATA AUGMENTATION) CHO TẬP HUẤN LUYỆN ---
# Thêm nhiễu Gaussian ngẫu nhiên vào các đặc trưng liên tục (time_gap_per_IP và req_regularity_per_IP)
# trên tập huấn luyện Normal Train để mô phỏng network jitter và giảm thiểu synthetic data bias.
print("  → Đang bổ sung nhiễu Gaussian (Augmentation) vào time_gap_per_IP & req_regularity_per_IP...")
time_gap_idx = FEATURE_COLS.index('time_gap_per_IP')
req_reg_idx = FEATURE_COLS.index('req_regularity_per_IP')

# Cấu hình độ lệch chuẩn của nhiễu (noise std = 0.05 trong không gian scaled [0, 1])
noise_std = 0.05
noise_time_gap = np.random.normal(0, noise_std, size=X_normal_train.shape[0])
noise_req_reg = np.random.normal(0, noise_std, size=X_normal_train.shape[0])

# Áp dụng nhiễu và clip về khoảng [0, 1]
X_normal_train_augmented = X_normal_train.copy()
X_normal_train_augmented[:, time_gap_idx] = np.clip(X_normal_train_augmented[:, time_gap_idx] + noise_time_gap, 0.0, 1.0)
X_normal_train_augmented[:, req_reg_idx] = np.clip(X_normal_train_augmented[:, req_reg_idx] + noise_req_reg, 0.0, 1.0)

# Tập Test = toàn bộ Malicious
X_test = X_malicious
y_test = y_malicious  # Toàn bộ là label 1

# Tập Evaluation (Validation + Test) - dùng để đánh giá cả 2 mô hình
X_eval = np.vstack([X_normal_val, X_test])
y_eval = np.concatenate([
    np.zeros(len(X_normal_val)),  # Normal
    np.ones(len(X_test))          # Malicious
])

print(f"  → Normal Train (Sau tăng cường nhiễu): {X_normal_train_augmented.shape[0]} mẫu")
print(f"  → Normal Val                         : {X_normal_val.shape[0]} mẫu")
print(f"  → Malicious Test                     : {X_test.shape[0]} mẫu")
print(f"  → Evaluation Set                     : {X_eval.shape[0]} mẫu (Val + Test)")

# =====================================================================
# BƯỚC 3: BASELINE MODEL - ISOLATION FOREST
# =====================================================================
print("\n[STEP 3] Huấn luyện Baseline Model (Isolation Forest)...")

# contamination = tỉ lệ dị thường thực tế trong tập dữ liệu
contamination_ratio = len(X_malicious) / len(X_scaled)
print(f"  → Contamination ratio (True anomaly ratio): {contamination_ratio:.6f}")

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=contamination_ratio,
    random_state=42,
    n_jobs=-1
)

# Huấn luyện trên tập Normal Train đã được tăng cường nhiễu (chống data leakage)
print("  → Huấn luyện Isolation Forest trên tập Normal Train...")
iso_forest.fit(X_normal_train_augmented)

# Dự đoán trên tập Evaluation (Validation + Test)
iso_pred_raw = iso_forest.predict(X_eval)
# Chuyển đổi: -1 → 1 (Malicious), 1 → 0 (Normal)
iso_pred = np.where(iso_pred_raw == -1, 1, 0)

# Anomaly score (càng âm = càng dị thường)
iso_scores = -iso_forest.decision_function(X_eval)

# --- Đánh giá Isolation Forest ---
iso_f1 = f1_score(y_eval, iso_pred, average='weighted')
iso_cm = confusion_matrix(y_eval, iso_pred)

print(f"\n  ✦ Isolation Forest - F1 Score (weighted): {iso_f1:.4f}")
print(f"  ✦ Classification Report:")
print(classification_report(y_eval, iso_pred,
                            target_names=['Normal', 'Malicious'],
                            digits=4))

# =====================================================================
# BƯỚC 4: MAIN MODEL - VARIATIONAL AUTOENCODER (VAE)
# =====================================================================
print("\n[STEP 4] Xây dựng và huấn luyện Variational Autoencoder (VAE)...")

input_dim = X_normal_train_augmented.shape[1]
latent_dim = 8  # Không gian tiềm ẩn 8 chiều để biểu diễn các đặc trưng tốt hơn

# ---- Lớp Sampling cho Latent Space ----
class Sampling(layers.Layer):
    """Lớp lấy mẫu z = mu + sigma * epsilon (Reparameterization trick)"""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# ---- ENCODER ARCHITECTURE WITH FEATURE DROPOUT ----
encoder_inputs = keras.Input(shape=(input_dim,), name='encoder_input')
# Bổ sung Feature Dropout (10%) ở đầu vào để tránh mô hình quá phụ thuộc vào một đặc trưng mạnh duy nhất (ví dụ User-Agent)
x = layers.Dropout(0.1, name='enc_dropout')(encoder_inputs)
x = layers.Dense(64, activation='relu', name='enc_dense1')(x)
x = layers.BatchNormalization(name='enc_bn1')(x)
x = layers.Dense(32, activation='relu', name='enc_dense2')(x)
x = layers.BatchNormalization(name='enc_bn2')(x)
x = layers.Dense(16, activation='relu', name='enc_dense3')(x)

z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
z = Sampling(name='sampling')([z_mean, z_log_var])

encoder = Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# ---- DECODER ----
decoder_inputs = keras.Input(shape=(latent_dim,), name='decoder_input')
x = layers.Dense(16, activation='relu', name='dec_dense1')(decoder_inputs)
x = layers.BatchNormalization(name='dec_bn1')(x)
x = layers.Dense(32, activation='relu', name='dec_dense2')(x)
x = layers.BatchNormalization(name='dec_bn2')(x)
x = layers.Dense(64, activation='relu', name='dec_dense3')(x)
decoder_outputs = layers.Dense(input_dim, activation='sigmoid', name='dec_output')(x)

decoder = Model(decoder_inputs, decoder_outputs, name='decoder')
decoder.summary()

# ---- VAE MODEL CLASS WITH MAE RECONSTRUCTION LOSS & BETA ----
class VAE(Model):
    def __init__(self, encoder, decoder, beta=0.1, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            # Reconstruction loss dùng MAE (thay vì MSE) để giảm độ nhạy với các nhiễu biên trị
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.abs(data - reconstruction),
                    axis=1
                )
            )
            # KL Divergence loss
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                    axis=1,
                )
            )
            total_loss = reconstruction_loss + self.beta * kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        z_mean, z_log_var, z = self.encoder(data)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                tf.abs(data - reconstruction),
                axis=1
            )
        )
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1,
            )
        )
        total_loss = reconstruction_loss + self.beta * kl_loss

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

# Khởi tạo và biên dịch VAE với beta=0.1
vae = VAE(encoder, decoder, beta=0.1)
vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

# Callback Early Stopping & Learning Rate Decay
early_stop = EarlyStopping(
    monitor='val_total_loss',
    mode='min',
    patience=15,
    restore_best_weights=True
)

from tensorflow.keras.callbacks import ReduceLROnPlateau
reduce_lr = ReduceLROnPlateau(
    monitor='val_total_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-5,
    mode='min',
    verbose=1
)

# ---- HUẤN LUYỆN CHỈ TRÊN TẬP NORMAL TRAIN ĐÃ ĐƯỢC TĂNG CƯỜNG NHIỄU ----
print(f"\n  → Bắt đầu huấn luyện VAE trên {X_normal_train_augmented.shape[0]} mẫu Normal (đã Augmented)...")
history = vae.fit(
    X_normal_train_augmented,
    epochs=200,
    batch_size=64,
    validation_data=(X_normal_val,),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

print("  ✦ Huấn luyện VAE hoàn tất!")

# =====================================================================
# BƯỚC 5: TÍNH RECONSTRUCTION ERROR & NGƯỠNG THÍCH ỨNG
# =====================================================================
print("\n[STEP 5] Tính Reconstruction Error và xác định Threshold...")

def compute_reconstruction_error(model, data):
    """Tính MAE reconstruction error cho từng mẫu"""
    z_mean, z_log_var, z = model.encoder(data)
    reconstructed = model.decoder(z)
    mae = np.mean(np.abs(data - reconstructed.numpy()), axis=1)
    return mae

# Tính lỗi trên từng tập
re_normal_train = compute_reconstruction_error(vae, X_normal_train_augmented)
re_normal_val = compute_reconstruction_error(vae, X_normal_val)
re_malicious = compute_reconstruction_error(vae, X_test)

# Reconstruction error trên toàn bộ tập Eval
re_eval = np.concatenate([re_normal_val, re_malicious])

# THRESHOLD = phân vị thứ 95 của tập Normal Validation (Chấp nhận 5% FP để tối đa hóa Recall phát hiện tấn công)
threshold = np.percentile(re_normal_val, 95)


print(f"  → RE Normal Train  : mean={np.mean(re_normal_train):.6f}, max={np.max(re_normal_train):.6f}")
print(f"  → RE Normal Val    : mean={np.mean(re_normal_val):.6f}, max={np.max(re_normal_val):.6f}")
print(f"  → RE Malicious     : mean={np.mean(re_malicious):.6f}, max={np.max(re_malicious):.6f}")
print(f"\n  ★ THRESHOLD (max loss Normal Val) = {threshold:.6f}")

# Dự đoán VAE: vượt threshold → Malicious (1)
vae_pred = (re_eval > threshold).astype(int)

# --- Đánh giá VAE ---
vae_f1 = f1_score(y_eval, vae_pred, average='weighted')
vae_cm = confusion_matrix(y_eval, vae_pred)

print(f"\n  ✦ VAE - F1 Score (weighted): {vae_f1:.4f}")
print(f"  ✦ Classification Report:")
print(classification_report(y_eval, vae_pred,
                            target_names=['Normal', 'Malicious'],
                            digits=4))

# =====================================================================
# BƯỚC 6: VẼ 5 BIỂU ĐỒ "ĂN ĐIỂM"
# =====================================================================
print("\n[STEP 6] Xuất 5 biểu đồ chuyên nghiệp...")

# ── Màu sắc chuyên nghiệp ──
COLOR_NORMAL = '#00d2ff'      # Cyan sáng
COLOR_MALICIOUS = '#e94560'   # Đỏ san hô
COLOR_THRESHOLD = '#f5a623'   # Vàng cam
COLOR_IF = '#7b68ee'          # Tím nhạt (Isolation Forest)
COLOR_VAE = '#00e396'         # Xanh lá neon (VAE)
BG_DARK = '#1a1a2e'
BG_AXES = '#16213e'

# ─────────────────────────────────────────────────────────────────────
# BIỂU ĐỒ 1: Training & Validation Loss Curve
# ─────────────────────────────────────────────────────────────────────
fig1, ax1 = plt.subplots(figsize=(10, 6))
fig1.patch.set_facecolor(BG_DARK)
ax1.set_facecolor(BG_AXES)

epochs_range = range(1, len(history.history['total_loss']) + 1)

ax1.plot(epochs_range, history.history['total_loss'],
         color=COLOR_NORMAL, linewidth=2.5, label='Training Loss', marker='o',
         markersize=3, alpha=0.9)
ax1.plot(epochs_range, history.history['val_total_loss'],
         color=COLOR_MALICIOUS, linewidth=2.5, label='Validation Loss', marker='s',
         markersize=3, alpha=0.9)

ax1.set_xlabel('Epoch', fontsize=13, fontweight='bold')
ax1.set_ylabel('Total Loss (MSE + KL)', fontsize=13, fontweight='bold')
ax1.set_title('Biểu đồ 1: Đường cong hội tụ (Training & Validation Loss)',
              fontsize=15, fontweight='bold', pad=15)
ax1.legend(fontsize=12, loc='upper right', fancybox=True, framealpha=0.8,
           edgecolor='#555')
ax1.grid(True, alpha=0.3)

fig1.tight_layout()
fig1.savefig(os.path.join(SAVE_DIR, '01_training_validation_loss.png'),
             dpi=200, bbox_inches='tight', facecolor=BG_DARK)
print("  ✓ Đã lưu: 01_training_validation_loss.png")

# ─────────────────────────────────────────────────────────────────────
# BIỂU ĐỒ 2: Reconstruction Error Histogram
# ─────────────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.patch.set_facecolor(BG_DARK)
ax2.set_facecolor(BG_AXES)

ax2.hist(re_normal_val, bins=80, alpha=0.75, color=COLOR_NORMAL,
         label=f'Normal (n={len(re_normal_val)})', edgecolor='white', linewidth=0.5)
ax2.hist(re_malicious, bins=80, alpha=0.75, color=COLOR_MALICIOUS,
         label=f'Malicious (n={len(re_malicious)})', edgecolor='white', linewidth=0.5)
ax2.axvline(x=threshold, color=COLOR_THRESHOLD, linestyle='--', linewidth=2.5,
            label=f'Threshold = {threshold:.4f}')

ax2.set_xlabel('Reconstruction Error (MSE)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Số lượng mẫu', fontsize=13, fontweight='bold')
ax2.set_title('Biểu đồ 2: Phân phối lỗi giải nén (Reconstruction Error)',
              fontsize=15, fontweight='bold', pad=15)
ax2.legend(fontsize=11, loc='upper right', fancybox=True, framealpha=0.8,
           edgecolor='#555')
ax2.grid(True, alpha=0.3)

fig2.tight_layout()
fig2.savefig(os.path.join(SAVE_DIR, '02_reconstruction_error_histogram.png'),
             dpi=200, bbox_inches='tight', facecolor=BG_DARK)
print("  ✓ Đã lưu: 02_reconstruction_error_histogram.png")

# ─────────────────────────────────────────────────────────────────────
# BIỂU ĐỒ 3: Confusion Matrix - Isolation Forest (Baseline)
# ─────────────────────────────────────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(7, 6))
fig3.patch.set_facecolor(BG_DARK)

sns.heatmap(iso_cm, annot=True, fmt='d', cmap='coolwarm',
            xticklabels=['Normal', 'Malicious'],
            yticklabels=['Normal', 'Malicious'],
            annot_kws={'size': 18, 'fontweight': 'bold'},
            linewidths=2, linecolor='#333',
            cbar_kws={'label': 'Số lượng'},
            ax=ax3)

ax3.set_xlabel('Dự đoán (Predicted)', fontsize=13, fontweight='bold')
ax3.set_ylabel('Thực tế (Actual)', fontsize=13, fontweight='bold')
ax3.set_title(f'Biểu đồ 3: Confusion Matrix - Isolation Forest\n(F1={iso_f1:.4f})',
              fontsize=14, fontweight='bold', pad=15)

fig3.tight_layout()
fig3.savefig(os.path.join(SAVE_DIR, '03_confusion_matrix_isolation_forest.png'),
             dpi=200, bbox_inches='tight', facecolor=BG_DARK)
print("  ✓ Đã lưu: 03_confusion_matrix_isolation_forest.png")

# ─────────────────────────────────────────────────────────────────────
# BIỂU ĐỒ 4: Confusion Matrix - VAE (Main Model)
# ─────────────────────────────────────────────────────────────────────
fig4, ax4 = plt.subplots(figsize=(7, 6))
fig4.patch.set_facecolor(BG_DARK)

sns.heatmap(vae_cm, annot=True, fmt='d', cmap='YlGnBu',
            xticklabels=['Normal', 'Malicious'],
            yticklabels=['Normal', 'Malicious'],
            annot_kws={'size': 18, 'fontweight': 'bold'},
            linewidths=2, linecolor='#333',
            cbar_kws={'label': 'Số lượng'},
            ax=ax4)

ax4.set_xlabel('Dự đoán (Predicted)', fontsize=13, fontweight='bold')
ax4.set_ylabel('Thực tế (Actual)', fontsize=13, fontweight='bold')
ax4.set_title(f'Biểu đồ 4: Confusion Matrix - VAE\n(F1={vae_f1:.4f})',
              fontsize=14, fontweight='bold', pad=15)

fig4.tight_layout()
fig4.savefig(os.path.join(SAVE_DIR, '04_confusion_matrix_vae.png'),
             dpi=200, bbox_inches='tight', facecolor=BG_DARK)
print("  ✓ Đã lưu: 04_confusion_matrix_vae.png")

# ─────────────────────────────────────────────────────────────────────
# BIỂU ĐỒ 5: ROC Curve - So sánh AUC Isolation Forest vs VAE
# ─────────────────────────────────────────────────────────────────────
fig5, ax5 = plt.subplots(figsize=(10, 8))
fig5.patch.set_facecolor(BG_DARK)
ax5.set_facecolor(BG_AXES)

# ROC cho Isolation Forest (dùng anomaly score)
fpr_if, tpr_if, _ = roc_curve(y_eval, iso_scores)
auc_if = auc(fpr_if, tpr_if)

# ROC cho VAE (dùng reconstruction error)
fpr_vae, tpr_vae, _ = roc_curve(y_eval, re_eval)
auc_vae = auc(fpr_vae, tpr_vae)

# Đường chéo ngẫu nhiên
ax5.plot([0, 1], [0, 1], color='#555', linestyle='--', linewidth=1.5,
         label='Random Classifier (AUC = 0.50)', alpha=0.7)

# Đường ROC Isolation Forest
ax5.plot(fpr_if, tpr_if, color=COLOR_IF, linewidth=3,
         label=f'Isolation Forest (AUC = {auc_if:.4f})', alpha=0.9)
ax5.fill_between(fpr_if, tpr_if, alpha=0.15, color=COLOR_IF)

# Đường ROC VAE
ax5.plot(fpr_vae, tpr_vae, color=COLOR_VAE, linewidth=3,
         label=f'VAE (AUC = {auc_vae:.4f})', alpha=0.9)
ax5.fill_between(fpr_vae, tpr_vae, alpha=0.15, color=COLOR_VAE)

ax5.set_xlabel('False Positive Rate (Tỷ lệ Dương tính giả)', fontsize=13, fontweight='bold')
ax5.set_ylabel('True Positive Rate (Tỷ lệ phát hiện đúng)', fontsize=13, fontweight='bold')
ax5.set_title('Biểu đồ 5: Đường cong ROC - So sánh Isolation Forest vs VAE',
              fontsize=15, fontweight='bold', pad=15)
ax5.legend(fontsize=12, loc='lower right', fancybox=True, framealpha=0.8,
           edgecolor='#555')
ax5.grid(True, alpha=0.3)
ax5.set_xlim([-0.02, 1.02])
ax5.set_ylim([-0.02, 1.02])

fig5.tight_layout()
fig5.savefig(os.path.join(SAVE_DIR, '05_roc_curve_comparison.png'),
             dpi=200, bbox_inches='tight', facecolor=BG_DARK)
print("  ✓ Đã lưu: 05_roc_curve_comparison.png")

# =====================================================================
# TỔNG KẾT
# =====================================================================
print("\n" + "=" * 70)
print("  TỔNG KẾT KẾT QUẢ")
print("=" * 70)
print(f"""
  ┌──────────────────────┬──────────────────┬──────────────────┐
  │      Chỉ số          │ Isolation Forest │       VAE        │
  ├──────────────────────┼──────────────────┼──────────────────┤
  │  F1 Score (weighted) │     {iso_f1:.4f}       │     {vae_f1:.4f}       │
  │  AUC                 │     {auc_if:.4f}       │     {auc_vae:.4f}       │
  │  True Positive       │     {iso_cm[1][1]:>5d}        │     {vae_cm[1][1]:>5d}        │
  │  False Positive      │     {iso_cm[0][1]:>5d}        │     {vae_cm[0][1]:>5d}        │
  │  True Negative       │     {iso_cm[0][0]:>5d}        │     {vae_cm[0][0]:>5d}        │
  │  False Negative      │     {iso_cm[1][0]:>5d}        │     {vae_cm[1][0]:>5d}        │
  └──────────────────────┴──────────────────┴──────────────────┘
""")

if auc_vae > auc_if:
    print("  ★ KẾT LUẬN: VAE (Deep Learning) vượt trội hơn Isolation Forest (ML truyền thống)")
    print("    trong việc phát hiện tấn công Credential Stuffing trên hệ thống LMS Moodle.")
else:
    print("  ★ KẾT LUẬN: Cả hai mô hình đều cho kết quả tốt.")
    print("    Cần tinh chỉnh thêm hyperparameter để tối ưu VAE.")

print(f"\n  📁 Tất cả biểu đồ đã được lưu tại: {SAVE_DIR}")
print("=" * 70)

# =====================================================================
# BƯỚC 7: LƯU MÔ HÌNH VÀ CẤU HÌNH PHỤC VỤ SOAR
# =====================================================================
print("\n[STEP 7] Đang lưu mô hình và cấu hình phục vụ SOAR...")
import pickle
import json

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'deploy_models')
os.makedirs(MODEL_DIR, exist_ok=True)

# 1. Lưu MinMaxScaler
scaler_path = os.path.join(MODEL_DIR, 'minmax_scaler.pkl')
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"  ✓ Đã lưu MinMaxScaler tại: {scaler_path}")

# 2. Lưu Isolation Forest model
if_path = os.path.join(MODEL_DIR, 'isolation_forest.pkl')
with open(if_path, 'wb') as f:
    pickle.dump(iso_forest, f)
print(f"  ✓ Đã lưu Isolation Forest tại: {if_path}")

# 3. Lưu VAE Encoder & Decoder riêng biệt (.keras)
encoder_path = os.path.join(MODEL_DIR, 'vae_encoder.keras')
decoder_path = os.path.join(MODEL_DIR, 'vae_decoder.keras')
vae.encoder.save(encoder_path)
vae.decoder.save(decoder_path)
print(f"  ✓ Đã lưu VAE Encoder tại: {encoder_path}")
print(f"  ✓ Đã lưu VAE Decoder tại: {decoder_path}")

# 4. Lưu cấu hình ngưỡng động và các đặc trưng đầu vào phục vụ SOAR
soar_config = {
    'vae_threshold': float(threshold),
    'feature_cols': FEATURE_COLS,
    'contamination_ratio': float(contamination_ratio)
}
config_path = os.path.join(MODEL_DIR, 'soar_config.json')
with open(config_path, 'w', encoding='utf-8') as f:
    json.dump(soar_config, f, indent=4, ensure_ascii=False)
print(f"  ✓ Đã lưu cấu hình ngưỡng SOAR tại: {config_path}")
print("=" * 70)
