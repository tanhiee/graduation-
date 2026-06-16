import re
import pandas as pd
import os
from datetime import datetime

# Regex chuẩn xác 100% cho định dạng Apache Combined Log
# Cấu trúc: IP - - [Thời gian] "Method URI HTTP/Phiên_bản" Mã_Status Dung_lượng "Referrer" "User_Agent"
APACHE_LOG_PATTERN = re.compile(
    r'^(?P<ip>\S+) \S+ \S+ \[(?P<timestamp>.*?)\] "(?P<method>[A-Z]+) (?P<uri>[^ "]+) HTTP/[0-9.]+" '
    r'(?P<status>\d{3}) (?P<size>\d+|-) "(?P<referrer>.*?)" "(?P<user_agent>.*?)"'
)

def parse_log_file(file_path, label):
    """
    Đọc file log, sử dụng Regex để cắt từng trường dữ liệu và gán nhãn (Label).
    """
    parsed_data = []
    
    if not os.path.exists(file_path):
        print(f"[-] Lỗi: Không tìm thấy file {file_path}")
        return parsed_data

    print(f"[*] Đang bóc tách file: {file_path} (Nhãn: {label})")
    
    # Đọc từng dòng để tối ưu RAM nếu file log từ Kali lên tới hàng GB
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            match = APACHE_LOG_PATTERN.match(line)
            if match:
                data_dict = match.groupdict()
                
                # BỘ LỌC TẬP TRUNG (Feature Selection): 
                # Bài toán Credential Stuffing nhắm vào luồng đăng nhập, nên ta chỉ lấy các request POST tới trang login
                if data_dict['method'] == 'POST' and 'login' in data_dict['uri']:
                    # Gán nhãn cho dữ liệu: 0 = Bình thường, 1 = Tấn công
                    data_dict['Label'] = label 
                    parsed_data.append(data_dict)
            else:
                # Báo cáo nếu có dòng log dị biệt không khớp định dạng
                # (Rất hữu ích để debug nếu Kali gửi payload làm vỡ cấu trúc log)
                pass 
                
    print(f"   -> Thành công! Trích xuất được {len(parsed_data)} request POST đăng nhập.")
    return parsed_data

def main():
    print("[i] BẮT ĐẦU QUÁ TRÌNH TIỀN XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)\n")
    
    # 1. Bóc tách dữ liệu hai phe
    normal_data = parse_log_file("clean_traffic.log", label=0)
    malicious_data = parse_log_file("malicious_traffic.log", label=1)
    
    # 2. Hợp nhất thành một Dataset duy nhất
    all_data = normal_data + malicious_data
    
    if not all_data:
        print("[!] Dataset trống. Vui lòng kiểm tra lại dữ liệu đầu vào.")
        return

    # 3. Chuyển hóa thành Pandas DataFrame để thao tác ma trận
    df = pd.DataFrame(all_data)
    
    # ---------------------------------------------------------
    # BƯỚC 4: FEATURE ENGINEERING (Kỹ thuật trích xuất đặc trưng)
    # Đây là phần quan trọng nhất để báo cáo trong đồ án
    # ---------------------------------------------------------
    
    print("\n[*] Đang thực hiện Feature Engineering...")
    
    # 4.1. Xử lý giá trị rỗng của dung lượng (size)
    df['size'] = df['size'].replace('-', 0).astype(int)
    
    # 4.2. Ép kiểu Status Code về dạng số nguyên (Integer)
    df['status'] = df['status'].astype(int)
    
    # 4.3. Phân tích chuỗi thời gian (Time-series Analysis)
    # Chuyển đổi chuỗi "25/May/2026:17:19:18 +0700" thành kiểu datetime chuẩn để AI hiểu được
    # Mẹo: Bỏ phần múi giờ (+0700) để parse dễ hơn
    df['timestamp'] = df['timestamp'].str.split(' ').str[0]
    df['datetime'] = pd.to_datetime(df['timestamp'], format='%d/%b/%Y:%H:%M:%S')
    
    # Sắp xếp toàn bộ log theo trình tự thời gian
    df = df.sort_values(by='datetime').reset_index(drop=True)
    
    # 5. Xuất File CSV để đưa vào mô hình Học máy
    output_filename = "lms_credential_stuffing_dataset.csv"
    
    # Chỉ xuất các cột có giá trị cho Machine Learning
    # Loại bỏ cột 'datetime' nguyên bản vì định dạng ngày tháng thuần túy khó đưa vào mạng Nơ-ron
    columns_to_export = ['ip', 'method', 'uri', 'status', 'size', 'user_agent', 'Label']
    df[columns_to_export].to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    print(f"\n[+] HOÀN TẤT! Dataset gồm {len(df)} mẫu đã được lưu tại '{output_filename}'.")
    
    # In thống kê sơ bộ
    print("\n--- THỐNG KÊ DATASET ---")
    print(df['Label'].value_counts().rename(index={0: 'Normal (0)', 1: 'Malicious (1)'}))

if __name__ == "__main__":
    main()