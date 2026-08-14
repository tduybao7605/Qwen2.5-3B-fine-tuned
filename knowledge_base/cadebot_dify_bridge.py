import requests
import json
import os

# ==========================================
# CẤU HÌNH DIFY API CHO ROBOT CADEBOT
# ==========================================
DIFY_API_BASE = os.getenv("DIFY_API_BASE", "http://localhost/v1")  # Hoặc IP server Dify của bạn
DIFY_API_KEY = os.getenv("DIFY_API_KEY", "YOUR_DIFY_APP_API_KEY")

class CadebotDifyBridge:
    def __init__(self, api_key: str, api_base: str):
        self.api_key = api_key
        self.api_base = api_base.rstrip('/')
        self.conversation_id = ""  # Lưu vết cuộc trò chuyện với từng khách hàng

    def reset_conversation(self):
        """Reset cuộc trò chuyện khi đổi khách hàng mới"""
        self.conversation_id = ""
        print("🤖 Cadebot: Đã làm mới phiên trò chuyện cho khách hàng mới.")

    def chat_with_robot(self, user_text: str, user_id: str = "cadebot_guest") -> str:
        """
        Gửi câu hỏi từ giọng nói/văn bản của khách trên Cadebot tới Dify API
        và nhận câu trả lời để Cadebot phát loa (TTS) hoặc hiển thị trên màn hình.
        """
        url = f"{self.api_base}/chat-messages"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": {},
            "query": user_text,
            "response_mode": "blocking",  # Hoặc "streaming" nếu muốn robot nói ngay khi có từng từ
            "user": user_id,
            "conversation_id": self.conversation_id
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=15)
            if response.status_code == 200:
                data = response.json()
                # Cập nhật conversation_id để giữ ngữ cảnh hội thoại
                self.conversation_id = data.get("conversation_id", "")
                answer = data.get("answer", "")
                return answer
            else:
                print(f"Lỗi Dify API ({response.status_code}): {response.text}")
                return "Dạ xin lỗi quý khách, em đang gặp chút gián đoạn kết nối. Quý khách vui lòng hỏi lại giúp em nhé!"
        except Exception as e:
            print(f"Lỗi kết nối Robot -> Dify: {e}")
            return "Dạ em chưa nghe rõ, quý khách có thể nói lại được không ạ?"

# ==========================================
# VÍ DỤ CHẠY THỬ KIỂM TRA
# ==========================================
if __name__ == "__main__":
    print("=== DEMO TÍCH HỢP CADEBOT x DIFY KNOWLEDGE BASE (VIVA RESERVE) ===")
    
    # Nhập API Key từ Dify ứng dụng của bạn để test
    api_key = DIFY_API_KEY
    if api_key == "YOUR_DIFY_APP_API_KEY":
        print("⚠️ Lưu ý: Bạn cần thay YOUR_DIFY_APP_API_KEY bằng API Key thực tế lấy từ Dify App!")
    
    bot = CadebotDifyBridge(api_key=api_key, api_base=DIFY_API_BASE)

    # Giả lập khách hàng nói chuyện với robot Cadebot tại quán Viva Reserve
    sample_questions = [
        "Quán mình có những món cà phê pha thủ công nào vậy em?",
        "Không gian quán có chỗ ngồi làm việc cắm máy tính không?",
        "Giờ mở cửa của quán Viva Reserve là mấy giờ?",
        "Địa chỉ chi nhánh này ở đâu vậy?"
    ]

    for q in sample_questions:
        print(f"\n👤 Khách nói với Cadebot: {q}")
        reply = bot.chat_with_robot(q)
        print(f"🤖 Cadebot phản hồi (Phát loa TTS): {reply}")
