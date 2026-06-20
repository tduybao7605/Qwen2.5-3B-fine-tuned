"""Gọi Ollama API (cadebot-viva) → parse JSON response."""
import json
import requests

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "cadebot-viva"
TIMEOUT = 60


def chat(user_text: str, model: str = DEFAULT_MODEL) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_text}],
        "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
        resp.raise_for_status()
        raw = resp.json()["message"]["content"]
        print(f"[LLM] Raw: {raw}")
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"intent": "FALLBACK", "spokenText": raw, "raw": raw}
    except Exception as e:
        print(f"[LLM] Lỗi: {e}")
        return {"intent": "FALLBACK", "spokenText": "Xin lỗi, tôi đang gặp sự cố."}
