# SOAR AI-Driven Security System for Moodle LMS

Hệ thống SOAR (Security Orchestration, Automation, and Response) tích hợp Trí tuệ Nhân tạo (AI) giúp tự động phát hiện và ngăn chặn các cuộc tấn công mạng (Credential Stuffing, DDoS, Low-and-Slow) nhắm vào hệ thống học trực tuyến Moodle.

##  Tính năng nổi bật

- **Phát hiện Dị thường bằng AI:** Sử dụng mô hình Deep Learning (Variational Autoencoder - VAE) kết hợp với Isolation Forest để phát hiện các mẫu hành vi độc hại theo thời gian thực.
- **Adaptive Threshold (Ngưỡng thích ứng):** Tự động điều chỉnh ngưỡng phát hiện tấn công theo thời gian thực dựa trên kỹ thuật Exponential Moving Average (EMA) để đối phó với hiện tượng Concept Drift.
- **Kiến trúc Microservices & Message Queue:** Tách biệt hoàn toàn khối xử lý AI nặng nề và khối Web App, kết nối bằng hàng đợi **RabbitMQ** giúp hệ thống chịu tải cực cao.
- **SOAR Playbooks Tự động:** Khi phát hiện tấn công, hệ thống lập tức gọi REST API để:
  - Khóa IP trên Firewall.
  - Khóa tài khoản sinh viên bị lộ mật khẩu trên Moodle.
  - Hỗ trợ cơ chế *Exponential Backoff Retry* đảm bảo API luôn được gửi thành công dù mạng chập chờn.
- **High Traffic Season Mode (Mùa thi):** Tính năng thông minh giúp hạ cấp độ (Downgrade) cảnh báo từ Block xuống Warning để tránh việc vô tình chặn nhầm một lượng lớn sinh viên truy cập chung IP (từ Ký túc xá/Thư viện) trong mùa thi cử.

## Kiến trúc hệ thống

Dự án được chia làm 3 thành phần chính:
1. **AI Inference Engine (FastAPI):** `ai_inference_engine.py` - Nạp mô hình AI, đọc log liên tục, phân tích và đẩy cảnh báo (Alert) vào Queue.
2. **Message Broker (RabbitMQ):** Tiếp nhận dữ liệu từ AI Engine và phân phối cho các Worker, tránh nghẽn luồng.
3. **SOAR Web App (Flask):** `soar_web_app.py` - Chạy Worker ngầm (Consumer) lắng nghe RabbitMQ để kích hoạt Playbook, đồng thời cung cấp giao diện Dashboard giám sát.

##  Yêu cầu hệ thống (Prerequisites)

- **Python 3.10+**
- **RabbitMQ Server** (được cài đặt và chạy trên cổng mặc định `5672`).
  - *Lưu ý cho Windows:* Bạn phải tải cài đặt `Erlang` (bản 26+) trước khi cài `RabbitMQ` (bản 4.x+).

##  Cài đặt

1. Clone project về máy.
2. Cài đặt các thư viện Python cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
   *(Chú ý: Đảm bảo cài đúng bản `tensorflow>=2.16.1` để tương thích với cấu trúc Keras 3 của model).*

##  Hướng dẫn Khởi chạy

Để hệ thống hoạt động, bạn cần mở **2 cửa sổ Terminal** riêng biệt.

**Terminal 1: Khởi động AI Inference Engine**
```bash
python ai_inference_engine.py
```
*(Mô hình AI sẽ nạp và bắt đầu bắn dữ liệu vào RabbitMQ. API của AI chạy tại: `http://127.0.0.1:8000`)*

**Terminal 2: Khởi động SOAR Web App & Worker**
```bash
python soar_web_app.py
```
*(Giao diện giám sát và SOAR Playbook sẽ chạy tại: `http://127.0.0.1:5000`)*

**Sử dụng:**
- Mở trình duyệt và truy cập `http://127.0.0.1:5000`.
- Nhấn nút "Bắt đầu Stream" trên giao diện để hệ thống bắt đầu quá trình quét log tự động và thực thi kịch bản (Playbook) khi có tấn công.

##  Cấu trúc thư mục

```text
 graduation-project
 ┣  deploy_models/             # Thư mục chứa các model đã huấn luyện (VAE, Isolation Forest, Scaler)
 ┣  templates/                 # Thư mục giao diện HTML của SOAR Web App
 ┣  ai_inference_engine.py     # Source code khối AI Model Serving
 ┣  soar_web_app.py            # Source code khối Web App & Playbook Worker
 ┣  requirements.txt           # Danh sách các thư viện cần cài đặt
 ┣  lms_training_dataset_final.csv # File log mô phỏng dữ liệu đầu vào
 ┗  soar_security_events.log   # File lưu trữ nhật ký phòng thủ của SOAR
```
