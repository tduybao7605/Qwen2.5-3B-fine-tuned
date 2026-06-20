"""Test từng module riêng lẻ."""
import argparse
import sys


def test_ollama(model="cadebot-viva"):
    import requests
    url = "http://127.0.0.1:11434/api/tags"
    try:
        resp = requests.get(url, timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        print(f"[OK] Ollama đang chạy. Models: {models}")
        if model not in models and f"{model}:latest" not in models:
            print(f"[WARN] Model '{model}' chưa có trong Ollama!")
        else:
            print(f"[OK] Model '{model}' sẵn sàng.")
    except Exception as e:
        print(f"[FAIL] Không kết nối được Ollama: {e}")


def test_llm(model="cadebot-viva"):
    from llm import chat
    cases = [
        ("Latte giá bao nhiêu?", "MENU_QA"),
        ("Cho tôi 1 Americano", "ADD_TO_CART_DRAFT"),
        ("Bạn có thể gợi ý gì cho tôi không?", "RECOMMENDATION"),
        ("Thời tiết hôm nay thế nào?", "FALLBACK"),
        ("Gọi nhân viên giúp tôi", "CALL_STAFF"),
        ("Hôm nay có khuyến mãi gì không?", "PROMOTION_QA"),
    ]
    print(f"\n[LLM] Test {len(cases)} kịch bản với model '{model}':\n")
    ok = 0
    for text, expected in cases:
        result = chat(text, model=model)
        intent = result.get("intent", "?")
        spoken = result.get("spokenText", "")[:60]
        status = "✅" if intent == expected else "❓"
        print(f"  {status} '{text}'\n     intent={intent} (expected={expected})\n     spoken='{spoken}'\n")
        if intent == expected:
            ok += 1
    print(f"[LLM] Kết quả: {ok}/{len(cases)}")


def test_tts():
    from tts import speak
    print("[TTS] Đang phát giọng nói thử...")
    speak("Xin chào! Tôi là Cadebot, trợ lý robot tại Viva Reserve Coffee.")
    print("[TTS] Hoàn thành.")


def test_stt():
    from stt import load_stt_model
    model = load_stt_model()
    print("[STT] Whisper model đã load thành công.")


def test_vad():
    from vad import load_vad_model, record_until_silence
    from stt import transcribe
    vad_model, _ = load_vad_model()
    print("[VAD] Nói một câu gì đó...")
    wav = record_until_silence(vad_model)
    text = transcribe(wav)
    print(f"[VAD+STT] Kết quả: '{text}'")


MODULES = {
    "ollama": test_ollama,
    "llm": test_llm,
    "tts": test_tts,
    "stt": test_stt,
    "vad": test_vad,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=list(MODULES.keys()), required=True)
    parser.add_argument("--model", default="cadebot-viva")
    args = parser.parse_args()

    fn = MODULES[args.module]
    import inspect
    sig = inspect.signature(fn)
    if "model" in sig.parameters:
        fn(model=args.model)
    else:
        fn()
