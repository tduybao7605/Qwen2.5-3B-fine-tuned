"""WebRTC VAD — nghe micro, phát hiện giọng nói, cắt khi im lặng 600ms."""
import io
import collections
import numpy as np
import sounddevice as sd
import webrtcvad
from scipy.io import wavfile

SAMPLE_RATE = 16000
FRAME_MS = 30          # webrtcvad hỗ trợ 10, 20, 30ms
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
SILENCE_TIMEOUT_MS = 600
SILENCE_FRAMES = SILENCE_TIMEOUT_MS // FRAME_MS
MIN_SPEECH_FRAMES = 5   # ít nhất ~150ms giọng nói mới ghi


def load_vad_model(aggressiveness=2):
    """aggressiveness: 0 (ít lọc) → 3 (lọc nhiều)."""
    vad = webrtcvad.Vad(aggressiveness)
    return vad, None   # trả tuple để tương thích giao diện cũ


def record_until_silence(vad_model, threshold=None, min_speech_ms=None) -> bytes:
    """Ghi âm từ mic cho đến khi im lặng, trả về bytes WAV 16kHz mono."""
    vad = vad_model
    print("[VAD] Đang lắng nghe... (nói để bắt đầu)")

    ring = collections.deque(maxlen=SILENCE_FRAMES)
    buffer = []
    speech_started = False
    speech_frame_count = 0

    with sd.RawInputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16",
                           blocksize=FRAME_SAMPLES) as stream:
        while True:
            data, _ = stream.read(FRAME_SAMPLES)
            frame = bytes(data)

            is_speech = vad.is_speech(frame, SAMPLE_RATE)
            ring.append(is_speech)

            if is_speech:
                if not speech_started:
                    print("[VAD] Phát hiện giọng nói...")
                    speech_started = True
                buffer.append(frame)
                speech_frame_count += 1
            elif speech_started:
                buffer.append(frame)
                if len(ring) == SILENCE_FRAMES and not any(ring):
                    if speech_frame_count >= MIN_SPEECH_FRAMES:
                        print("[VAD] Im lặng — dừng ghi.")
                        break
                    else:
                        # Quá ngắn, reset và chờ tiếp
                        buffer.clear()
                        ring.clear()
                        speech_started = False
                        speech_frame_count = 0

    audio = np.frombuffer(b"".join(buffer), dtype=np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, SAMPLE_RATE, audio)
    return buf.getvalue()
