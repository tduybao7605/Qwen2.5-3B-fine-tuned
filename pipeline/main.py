"""Vòng lặp chính: VAD → STT → LLM → TTS."""
from vad import load_vad_model, record_until_silence
from stt import transcribe
from llm import chat
from tts import speak


def run():
    print("=== Cadebot Voice Pipeline ===")
    print("Nhấn Ctrl+C để thoát.\n")
    vad_model, _ = load_vad_model()

    while True:
        try:
            wav_bytes = record_until_silence(vad_model)
            text = transcribe(wav_bytes)

            if not text:
                print("[Main] Không nhận dạng được, thử lại...")
                continue

            result = chat(text)
            spoken = result.get("spokenText", result.get("answer", str(result)))
            speak(spoken)

        except KeyboardInterrupt:
            print("\n[Main] Thoát.")
            break


if __name__ == "__main__":
    run()
