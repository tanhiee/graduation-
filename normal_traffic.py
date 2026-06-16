import csv
import random
import time
import requests
from bs4 import BeautifulSoup

# --- CẤU HÌNH ---
# THAY ĐỔI ĐỊA CHỈ IP NÀY THÀNH IP MÁY ẢO UBUNTU CỦA BẠN
MOODLE_URL = "http://192.168.16.128/moodle" 
LOGIN_URL = f"{MOODLE_URL}/login/index.php"

# Mật khẩu chung đã tạo ở bước trước
DEFAULT_PASSWORD = "StudentPassword123@"

# Danh sách User-Agent để giả lập sinh viên dùng nhiều loại trình duyệt/điện thoại khác nhau
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/114.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Mobile/15E148 Safari/604.1"
]

def load_students(csv_file="moodle_students.csv"):
    students = []
    try:
        # Nhớ dùng utf-8-sig để xử lý BOM nếu có
        with open(csv_file, mode='r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                students.append(row['username'])
    except FileNotFoundError:
        print(f"[-] Không tìm thấy file {csv_file}. Hãy chắc chắn file nằm cùng thư mục.")
    return students

def simulate_login(username):
    # Khởi tạo một Session để giữ Cookie giống như trình duyệt thật
    session = requests.Session()
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    
    try:
        # Bước 1: Truy cập trang đăng nhập để lấy Cookie và logintoken
        print(f"[*] {username} đang mở trang đăng nhập...")
        response = session.get(LOGIN_URL, headers=headers, timeout=10)
        
        # Parse HTML để tìm logintoken
        soup = BeautifulSoup(response.text, 'html.parser')
        token_input = soup.find('input', {'name': 'logintoken'})
        
        if not token_input:
            print(f"[-] Không tìm thấy logintoken cho {username}. Bỏ qua.")
            return
            
        logintoken = token_input.get('value')
        
        # Bước 2: Gửi request POST chứa username, password và logintoken
        login_data = {
            'username': username,
            'password': DEFAULT_PASSWORD,
            'logintoken': logintoken
        }
        
        print(f"[*] {username} đang gửi thông tin đăng nhập...")
        login_post = session.post(LOGIN_URL, data=login_data, headers=headers, timeout=10)
        
        # Kiểm tra đăng nhập thành công (Moodle thường chuyển hướng hoặc mất nút login)
        if "loginerrormessage" not in login_post.text:
            print(f"[+] {username} đã đăng nhập THÀNH CÔNG!")
            # Tùy chọn: Có thể viết thêm code GET trang dashboard để sinh thêm log truy cập tài liệu
        else:
            print(f"[-] {username} đăng nhập THẤT BẠI (Sai mật khẩu).")
            
    except Exception as e:
        print(f"[!] Lỗi kết nối đối với {username}: {e}")

def main():
    students = load_students()
    if not students:
        return
        
    print(f"[i] Đã tải {len(students)} tài khoản sinh viên. Bắt đầu mô phỏng...")
    
    # Vòng lặp vô hạn để tạo log liên tục cho đến khi bạn bấm Ctrl+C
    try:
        while True:
            # Chọn ngẫu nhiên 1 sinh viên trong danh sách
            random_student = random.choice(students)
            simulate_login(random_student)
            
            # Tạm nghỉ ngẫu nhiên từ 15 giây đến 60 giây trước khi có sinh viên khác đăng nhập
            # Điều này giúp log giãn cách tự nhiên, giống hệ thống thật
            sleep_time = random.randint(3,5)
            print(f"[i] Chờ {sleep_time} giây trước lượt truy cập tiếp theo...\n")
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\n[i] Đã dừng mô phỏng bằng phím tắt.")

if __name__ == "__main__":
    main()