"""Gọi Cadebot API (có RAG) thay vì gọi thẳng Ollama.

Trước đây file này POST tới Ollama /api/chat với model cadebot-viva, không gửi
system prompt, không gửi history, không có context — tức là bỏ qua hoàn toàn
knowledge base. Nay đi qua Cadebot API để đường giọng nói dùng chung một
lớp RAG với client Android (cùng ngưỡng, cùng logic chặn out-of-scope).
"""
import json

import requests

API_URL = "http://127.0.0.1:8000/chat"
TIMEOUT = 300  # CPU chậm: ~78s/lượt cho câu in-scope; câu out-of-scope <2s

FALLBACK = {
    "intent": "FALLBACK",
    "confidence": 1.0,
    "answerText": "Xin lỗi bạn, mình chưa kết nối được hệ thống. Bạn gọi nhân viên giúp mình nhé.",
    "spokenText": "Xin lỗi bạn, mình chưa kết nối được hệ thống. Bạn gọi nhân viên giúp mình nhé.",
    "recommendedItems": [],
    "draftCartItems": [],
    "requiresHumanSupport": True,
    "sourceIds": [],
}


def chat(user_text: str, history: list | None = None) -> dict:
    try:
        resp = requests.post(
            API_URL,
            json={"message": user_text, "history": history or []},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
    except requests.RequestException as exc:
        print(f"[llm] lỗi gọi API: {exc}")
        return dict(FALLBACK)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Model đôi khi xuất text thường thay vì JSON — vẫn nói ra được.
        return {**FALLBACK, "answerText": raw, "spokenText": raw}
