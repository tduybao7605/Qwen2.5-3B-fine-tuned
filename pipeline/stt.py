"""faster-whisper small — audio bytes (WAV) → text tiếng Việt."""
import io
import numpy as np
from faster_whisper import WhisperModel

_model: WhisperModel | None = None


def load_stt_model(model_size="small", device="cpu", compute_type="int8"):
    global _model
    if _model is None:
        print(f"[STT] Đang load Whisper {model_size}...")
        _model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("[STT] Whisper sẵn sàng.")
    return _model


def transcribe(wav_bytes: bytes, language="vi") -> str:
    model = load_stt_model()
    audio = np.frombuffer(wav_bytes[44:], dtype=np.int16).astype(np.float32) / 32767.0
    segments, info = model.transcribe(audio, language=language, beam_size=5)
    text = " ".join(s.text.strip() for s in segments).strip()
    print(f"[STT] '{text}'")
    return text
