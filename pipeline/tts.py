"""edge-tts vi-VN-HoaiMyNeural → phát audio qua pygame."""
import asyncio
import io
import tempfile
import os
import edge_tts
import pygame

VOICE = "vi-VN-HoaiMyNeural"

_pygame_init = False


def _ensure_pygame():
    global _pygame_init
    if not _pygame_init:
        pygame.mixer.init()
        _pygame_init = True


async def _synthesize(text: str) -> bytes:
    communicate = edge_tts.Communicate(text, VOICE)
    buf = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            buf.write(chunk["data"])
    return buf.getvalue()


def speak(text: str):
    print(f"[TTS] '{text}'")
    audio_bytes = asyncio.run(_synthesize(text))

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        _ensure_pygame()
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
    finally:
        os.unlink(tmp_path)
