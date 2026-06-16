import os
import sys
import json
import time
import random
import requests
import threading
from datetime import datetime
from flask import Flask, jsonify, render_template, request
import pika

# Fix encoding cho Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)

# Biến toàn cục lưu trữ trạng thái hệ thống (Shared with Consumer Thread)
blocklist = {} # IP -> Timestamp blocked
playbook_logs = []
history_logs = []
unfetched_events = [] # Sự kiện chưa được UI fetch
total_scanned = 0
anomalies_detected = 0

is_running = True # Cờ điều khiển trạng thái xử lý
is_high_traffic_season = False

vae_threshold = 0.0 # Được cập nhật từ FastAPI qua message queue
contamination_ratio = 0.01

# --- HÀM GIẢ LẬP GỌI REST API VỚI RETRY & EXPONENTIAL BACKOFF ---
def simulate_rest_api_call(api_name, payload, timeout=2.0):
    time.sleep(random.uniform(0.05, 0.2))
    if random.random() < 0.05:
        raise requests.exceptions.Timeout("Connection timed out (Timeout 2.0s)")
    if random.random() < 0.10:
        raise requests.exceptions.ConnectionError("Failed to establish a new connection: Connection refused")
    if random.random() < 0.05:
        response = requests.Response()
        response.status_code = 500
        raise requests.exceptions.HTTPError("500 Internal Server Error", response=response)
    return {"status": "success", "message": f"Đã gửi lệnh chặn tới {api_name} cho payload {payload}"}

def call_rest_api_with_backoff(api_name, payload, max_retries=3, initial_delay=0.5):
    delay = initial_delay
    for attempt in range(1, max_retries + 1):
        try:
            add_playbook_log(f"   [API] Đang gọi {api_name} (Lần thử {attempt}/{max_retries})...")
            res = simulate_rest_api_call(api_name, payload, timeout=2.0)
            return True, f"API {api_name} thành công: {res['message']}"
        except Exception as e:
            add_playbook_log(f"   ⚠️ LỖI KẾT NỐI API {api_name}: Lần thử {attempt} thất bại. Chi tiết: {e}")
            if attempt == max_retries:
                return False, f"Thất bại hoàn toàn sau {max_retries} lần thử. Lỗi cuối: {e}"
            add_playbook_log(f"   ↳ Đang tự động thử lại sau {delay:.1f} giây...")
            time.sleep(delay)
            delay *= 2
    return False, "Unknown API Error"

def add_playbook_log(message):
    global playbook_logs
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_msg = f"[{timestamp}] {message}"
    playbook_logs.insert(0, log_msg)
    if len(playbook_logs) > 40:
        playbook_logs.pop()

# --- RABBITMQ CONSUMER THREAD ---
def rabbitmq_consumer():
    global total_scanned, anomalies_detected, blocklist, history_logs, unfetched_events, vae_threshold
    
    connection = None
    while True:
        try:
            connection = pika.BlockingConnection(pika.ConnectionParameters(host='127.0.0.1'))
            channel = connection.channel()
            channel.queue_declare(queue='soar_events_queue', durable=True)
            
            def callback(ch, method, properties, body):
                global total_scanned, anomalies_detected, blocklist, history_logs, unfetched_events, vae_threshold
                
                if not is_running:
                    # Nếu SOAR đang tạm dừng, bỏ qua event xử lý playbook (hoặc có thể ACK và bỏ qua)
                    return
                    
                event_data = json.loads(body)
                total_scanned += 1
                
                ip = event_data['ip']
                vae_error = event_data['vae_error']
                is_vae_anomaly = event_data['is_vae_anomaly']
                is_if_anomaly = event_data['is_if_anomaly']
                vae_threshold = event_data.get('vae_threshold', vae_threshold)
                
                is_blocked = ip in blocklist
                event_data['is_blocked'] = is_blocked
                
                # Lưu vào history và danh sách chờ fetch
                history_logs.insert(0, event_data)
                if len(history_logs) > 100:
                    history_logs.pop()
                    
                unfetched_events.append(event_data)
                
                if not is_blocked:
                    if is_vae_anomaly:
                        if is_high_traffic_season:
                            add_playbook_log(f"⚠️ HIGH TRAFFIC SEASON WARNING: Phát hiện dị thường từ IP {ip}!")
                            add_playbook_log(f"   [Chi tiết] VAE MAE Error: {vae_error:.5f} (Ngưỡng thích ứng: {vae_threshold:.5f})")
                            add_playbook_log(f"   🛡️ SOAR PLAYBOOK (Tier 2 - Downgraded): Hạ cấp quy trình cấm IP để tránh chặn nhầm sinh viên.")
                            with open('soar_security_events.log', 'a', encoding='utf-8') as lf:
                                lf.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SOAR WARNING (HIGH TRAFFIC SEASON): IP={ip}, MAE={vae_error:.6f}, Threshold={vae_threshold:.6f}\n")
                        else:
                            anomalies_detected += 1
                            blocklist[ip] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            event_data['is_blocked'] = True
                            
                            add_playbook_log(f"🚨 CRITICAL ALERT: Phát hiện Credential Stuffing từ IP {ip}!")
                            add_playbook_log(f"   [Chi tiết] VAE MAE Error: {vae_error:.5f} (Ngưỡng thích ứng: {vae_threshold:.5f})")
                            add_playbook_log(f"   🛡️ SOAR PLAYBOOK: Kích hoạt Kịch bản ngăn chặn khẩn cấp...")
                            
                            # Chạy Playbook APIs
                            firewall_success, firewall_msg = call_rest_api_with_backoff("Firewall API", {"ip": ip, "action": "block"})
                            moodle_success, moodle_msg = call_rest_api_with_backoff("Moodle API", {"ip": ip, "action": "lock_user"})
                            
                            if firewall_success: add_playbook_log(f"   ↳ [FIREWALL] {firewall_msg}")
                            else: add_playbook_log(f"   ❌ [FIREWALL LỖI] {firewall_msg}")
                                
                            if moodle_success: add_playbook_log(f"   ↳ [MOODLE API] {moodle_msg}")
                            else: add_playbook_log(f"   ❌ [MOODLE API LỖI] {moodle_msg}")
                            
                            with open('soar_security_events.log', 'a', encoding='utf-8') as lf:
                                lf.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SOAR BLOCK: IP={ip}, MAE={vae_error:.6f}, Threshold={vae_threshold:.6f}\n")
                                
                    elif is_if_anomaly:
                        add_playbook_log(f"⚠️ WARNING: Isolation Forest phát hiện hành vi bất thường từ IP {ip}.")

            channel.basic_consume(queue='soar_events_queue', on_message_callback=callback, auto_ack=True)
            print("[SOAR Worker] Đang lắng nghe sự kiện từ RabbitMQ (soar_events_queue)...")
            channel.start_consuming()
        except pika.exceptions.AMQPConnectionError as e:
            print(f"[SOAR Worker] Không thể kết nối tới RabbitMQ (AMQPConnectionError). Lỗi chi tiết: {e}. Thử lại sau 5 giây...")
            time.sleep(5)
        except Exception as e:
            print(f"[SOAR Worker] Lỗi bất ngờ: {e}")
            time.sleep(5)

# Khởi chạy luồng chạy ngầm RabbitMQ
threading.Thread(target=rabbitmq_consumer, daemon=True).start()

# --- ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status', methods=['GET'])
def get_status():
    global unfetched_events
    try:
        blocklist_list = [{"ip": k, "time": v} for k, v in blocklist.items()]
        
        # Pop các events mới để gửi cho Frontend hiển thị biểu đồ
        events_to_send = unfetched_events.copy()
        unfetched_events.clear()
        
        return jsonify({
            "is_running": is_running,
            "is_high_traffic_season": is_high_traffic_season,
            "metrics": {
                "total_scanned": int(total_scanned),
                "anomalies_detected": int(anomalies_detected),
                "active_blocked_ips": int(len(blocklist)),
                "vae_threshold": float(vae_threshold),
                "contamination_ratio": float(contamination_ratio)
            },
            "new_events": events_to_send,
            "blocklist": blocklist_list,
            "playbook_logs": playbook_logs
        })
    except Exception as e:
        import traceback
        return jsonify({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }), 500

@app.route('/api/toggle', methods=['POST'])
def toggle_stream():
    global is_running
    is_running = not is_running
    add_playbook_log(f"⚙️ SYSTEM: Đã {'BẮT ĐẦU' if is_running else 'TẠM DỪNG'} xử lý sự kiện SOAR.")
    
    # Optional: Gửi API toggle tới FastAPI để đồng bộ trạng thái stream
    try:
        requests.post("http://127.0.0.1:8000/api/engine/toggle", timeout=1.0)
    except Exception:
        pass # Nếu FastAPI chưa chạy thì bỏ qua
        
    return jsonify({"is_running": is_running})

@app.route('/api/toggle-high-traffic', methods=['POST'])
def toggle_high_traffic():
    global is_high_traffic_season
    is_high_traffic_season = not is_high_traffic_season
    add_playbook_log(f"⚙️ SYSTEM: Đã {'BẬT' if is_high_traffic_season else 'TẮT'} chế độ Mùa cao điểm (Concept Drift Mode).")
    return jsonify({"is_high_traffic_season": is_high_traffic_season})

@app.route('/api/reset-blocklist', methods=['POST'])
def reset_blocklist():
    global blocklist, anomalies_detected
    blocklist.clear()
    add_playbook_log("⚙️ SYSTEM: Đã xóa toàn bộ IP khỏi Blocklist và khôi phục cài đặt gốc Firewall.")
    return jsonify({"success": True})

@app.route('/api/manual-block', methods=['POST'])
def manual_block():
    global blocklist
    data = request.get_json()
    ip = data.get('ip')
    if ip:
        blocklist[ip] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        add_playbook_log(f"🛡️ MANUAL ACTION: Quản trị viên chủ động thêm IP {ip} vào Blocklist.")
        add_playbook_log(f"   ↳ [FIREWALL] Cập nhật Firewall chặn IP {ip} lập tức.")
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "IP is required"}), 400

@app.route('/api/manual-unblock', methods=['POST'])
def manual_unblock():
    global blocklist
    data = request.get_json()
    ip = data.get('ip')
    if ip in blocklist:
        del blocklist[ip]
        add_playbook_log(f"🛡️ MANUAL ACTION: Quản trị viên gỡ bỏ IP {ip} khỏi danh sách hạn chế.")
        add_playbook_log(f"   ↳ [FIREWALL] Gỡ bỏ luật cấm IP {ip} khỏi Firewall.")
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "IP not in blocklist"}), 400

if __name__ == '__main__':
    print("\n=======================================================")
    print(" SOAR WEB APP ĐÃ SẴN SÀNG KHỞI CHẠY (Microservices Mode)!")
    print(" 👉 Mở trình duyệt truy cập: http://127.0.0.1:5000")
    print("=======================================================\n")
    # use_reloader=False rất quan trọng để tránh RabbitMQ worker khởi chạy 2 lần trong chế độ debug
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False)
