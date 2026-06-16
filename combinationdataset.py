import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
import random
# ==========================================
# 1. ĐỌC VÀ CHUẨN HÓA KAGGLE (Traffic Nền LMS - Nhãn 0)
# ==========================================
print("1. Đang xử lý Kaggle dataset và chuyển đổi sang giao thức Moodle...")
log_pattern = re.compile(
    r'(?P<ip>\S+) \S+ \S+ \[(?P<time>.*?)\] "(?P<method>\S+) (?P<uri>\S+) \S+" (?P<status>\d{3}) (?P<size>\d+) "(?P<referer>.*?)" "(?P<user_agent>.*?)"'
)

kaggle_data = []
with open('access.log', 'r') as f:
    for i, line in enumerate(f):
        if len(kaggle_data) >= 50000: break # Lấy 50k dòng làm nền
        match = log_pattern.match(line)
        if match: kaggle_data.append(match.groupdict())

df_kaggle = pd.DataFrame(kaggle_data)
df_kaggle['status'] = df_kaggle['status'].astype(int)
df_kaggle['size'] = df_kaggle['size'].astype(int)
df_kaggle['Label'] = 0
df_kaggle['timestamp'] = pd.to_datetime(df_kaggle['time'], format='%d/%b/%Y:%H:%M:%S %z', exact=False).dt.tz_localize(None)
df_kaggle = df_kaggle.drop(columns=['time', 'referer'])

# ---------------------------------------------------------
# ĐIỂM MỚI: CHUYỂN ĐỔI TOÀN BỘ URI SANG MÔI TRƯỜNG MOODLE
# ---------------------------------------------------------
# Tập hợp các đường dẫn (URI) phổ biến khi sinh viên sử dụng Moodle hợp lệ
moodle_normal_uris = [
    "/moodle/my/",                                                # Xem trang Dashboard
    "/moodle/course/view.php?id=2",                               # Truy cập khóa học A
    "/moodle/course/view.php?id=5",                               # Truy cập khóa học B
    "/moodle/mod/resource/view.php?id=12",                        # Xem tài liệu tham khảo
    "/moodle/pluginfile.php/54/mod_resource/content/1/slide.pdf", # Tải file PDF bài giảng
    "/moodle/mod/assign/view.php?id=34",                          # Xem yêu cầu nộp bài tập
    "/moodle/mod/forum/view.php?id=8",                            # Đọc thảo luận trên Forum
    "/moodle/theme/image.php/boost/theme/168/logo"                # Trình duyệt tự tải logo hệ thống
]

# 1. Ghi đè toàn bộ URI của trang bán hàng thành URI của Moodle
df_kaggle['uri'] = np.random.choice(moodle_normal_uris, size=len(df_kaggle))

# 2. Sinh viên lướt Moodle tải tài liệu chủ yếu dùng lệnh GET
df_kaggle['method'] = 'GET'

# 3. Để tăng độ chân thực, cho khoảng 5% request là hành động "Mở trang đăng nhập" (GET request)
# (Lưu ý: Mở trang đăng nhập là GET, còn khi gõ pass bấm Submit thì mới là POST)
mask_login_page = np.random.rand(len(df_kaggle)) < 0.05
df_kaggle.loc[mask_login_page, 'uri'] = '/moodle/login/index.php'
df_kaggle.loc[mask_login_page, 'method'] = 'GET' 
df_kaggle.loc[mask_login_page, 'status'] = 200 # Mở trang thành công, không phải lỗi 303

# ==========================================
# 2. XỬ LÝ DATASET LAB (Tách Nhãn 0 và Nhãn 1)
# ==========================================
print("2. Đang phân tách và nâng cấp Lab dataset...")
df_lab = pd.read_csv('lms_credential_stuffing_dataset.csv')

start_time = df_kaggle['timestamp'].min()
end_time = df_kaggle['timestamp'].max()
total_duration_seconds = int((end_time - start_time).total_seconds())

# --- XỬ LÝ LAB NHÃN 0 (Sinh viên đăng nhập hợp lệ) ---
df_lab_0 = df_lab[df_lab['Label'] == 0].copy()
# Giả lập 50 sinh viên dùng các mạng khác nhau (VD: dải IP 113.190.x.x)
student_ips = [f"113.190.10.{i}" for i in range(1, 51)]
df_lab_0['ip'] = [random.choice(student_ips) for _ in range(len(df_lab_0))]
# Rải rác thời gian đăng nhập hợp lệ ra toàn bộ khoảng thời gian
time_increments_0 = [start_time + timedelta(seconds=random.randint(0, total_duration_seconds)) for _ in range(len(df_lab_0))]
df_lab_0['timestamp'] = sorted(time_increments_0)


# --- XỬ LÝ LAB NHÃN 1 (Botnet Credential Stuffing) ---
df_lab_1 = df_lab[df_lab['Label'] == 1].copy()
# Giả lập mạng Botnet 200 IP (VD: dải IP 103.22.x.x)
botnet_ips = [f"103.22.4.{i}" for i in range(1, 201)]
df_lab_1['ip'] = [random.choice(botnet_ips) for _ in range(len(df_lab_1))]
# Tấn công thường xảy ra dồn dập, ép 533 requests này vào một khoảng 10 phút (600s) giữa chừng
attack_start_time = start_time + timedelta(hours=1) 
time_increments_1 = [attack_start_time + timedelta(seconds=random.randint(0, 600)) for _ in range(len(df_lab_1))]
df_lab_1['timestamp'] = sorted(time_increments_1)


# ==========================================
# 3. GỘP VÀ TRÍCH XUẤT ĐẶC TRƯNG HÀNH VI
# ==========================================
print("3. Đang gộp dữ liệu và tính toán Features...")
# Trộn cả 3 tệp lại: Kaggle nền + Lab Sinh viên + Lab Hacker
df_mixed = pd.concat([df_kaggle, df_lab_0, df_lab_1], ignore_index=True)
# Phải sắp xếp lại theo thời gian để cửa sổ trượt (rolling) hoạt động đúng
df_mixed = df_mixed.sort_values(by='timestamp').reset_index(drop=True)
df_mixed = df_mixed.set_index('timestamp')

# Đặc trưng 1: Số Request / 1 IP / 5 phút
df_mixed['req_per_IP_5min'] = df_mixed.groupby('ip')['method'].transform(lambda x: x.rolling('5T').count())

# Đặc trưng 2: Tỷ lệ báo lỗi sai pass / 1 IP / 5 phút (Moodle trả về 303, Kaggle có thể có 401, 403)
df_mixed['is_error'] = df_mixed['status'].isin([303, 401, 403]).astype(int)
df_mixed['error_ratio_per_IP'] = df_mixed.groupby('ip')['is_error'].transform(lambda x: x.rolling('5T').mean())

# Đặc trưng 3: Số lượng trình duyệt (User-Agent) dùng bởi 1 IP trong 5 phút
df_mixed['ua_id'] = df_mixed['user_agent'].astype('category').cat.codes
df_mixed['unique_UA_per_IP'] = df_mixed.groupby('ip')['ua_id'].transform(lambda x: x.rolling('5T').apply(lambda y: len(set(y)), raw=False))

# Chuyển index timestamp về lại thành cột
df_mixed = df_mixed.reset_index()

# Chọn các cột có giá trị cho AI học
features_to_train = ['timestamp', 'ip', 'req_per_IP_5min', 'error_ratio_per_IP', 'unique_UA_per_IP', 'size', 'Label']
df_ai_ready = df_mixed[features_to_train]

# Fill các giá trị NaN (nếu có lúc rolling) bằng 0
df_ai_ready = df_ai_ready.fillna(0)

print("\n--- HOÀN THÀNH ---")
print(df_ai_ready['Label'].value_counts())

# Xuất file CSV để đưa vào thuật toán Machine Learning
df_ai_ready.to_csv('lms_training_dataset_final.csv', index=False)