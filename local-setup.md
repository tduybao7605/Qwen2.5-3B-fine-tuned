# Local Setup — Môi trường test Cadebot Voice Pipeline

> Ghi lại toàn bộ những gì đã cài đặt trên máy local để phục vụ việc test voice pipeline Cadebot.
> Máy: Intel i5-1235U · 24 GB RAM · Ubuntu 22.04 (Linux 6.8.0) · Không có GPU NVIDIA

---

## 1. Hệ thống

### Python
| Thành phần | Phiên bản | Đường dẫn |
|---|---|---|
| Python | **3.11.8** | `/home/ncd/.pyenv/versions/3.11.8/bin/python3` |
| PyTorch | 2.12.0+cu130 | CPU only (CUDA không khả dụng) |

> ⚠️ Luôn chạy pipeline bằng `/home/ncd/.pyenv/versions/3.11.8/bin/python3 main.py` nếu lệnh `python3` trỏ sai phiên bản.

### System library
| Thư viện | Phiên bản | Lý do cài |
|---|---|---|
| `libportaudio2` | 19.6.0 | `sounddevice` cần để thu âm từ micro |

Lệnh cài:
```bash
sudo apt-get install -y libportaudio2
```

---

## 2. Ollama

| Thành phần | Chi tiết |
|---|---|
| Phiên bản | 0.30.10 |
| Service | Đang chạy tại `http://127.0.0.1:11434` |
| Cài đặt | Script chính thức: `curl -fsSL https://ollama.com/install.sh \| sh` |

### Models đã có trong Ollama

| Model | ID | Kích thước | Mô tả |
|---|---|---|---|
| `cadebot-viva:latest` | 1d704a6609bc | **3.3 GB** | Qwen2.5-3B fine-tuned cho Viva Reserve Coffee (Q8_0) |
| `qwen2.5:3b` | 357c53fb659c | 1.9 GB | Base model gốc (Q4) |

### Modelfile của cadebot-viva

```
FROM ./cadebot-viva.gguf
SYSTEM "Bạn là Cadebot, trợ lý robot phục vụ tại Viva Reserve Coffee..."
PARAMETER temperature 0.1
PARAMETER num_ctx 2048
PARAMETER repeat_penalty 1.1
```

File GGUF gốc: `Qwen2.5-3B-fine-tuned/cadebot-viva.gguf` (3.1 GB trên disk)

---

## 3. Python packages cho pipeline

Cài bằng:
```bash
pip install faster-whisper sounddevice silero-vad edge-tts scipy requests
```

| Package | Phiên bản | Dùng cho |
|---|---|---|
| `faster-whisper` | 1.2.1 | STT — nhận dạng giọng nói tiếng Việt (Whisper small, CPU) |
| `sounddevice` | 0.5.5 | Thu âm từ micro laptop |
| `silero-vad` | 6.2.1 | VAD — phát hiện khi người dùng nói/dừng nói |
| `edge-tts` | 7.2.8 | TTS — chuyển text thành giọng nói (vi-VN-HoaiMyNeural) |
| `scipy` | 1.17.1 | Đọc/ghi file WAV |
| `requests` | 2.34.2 | Gọi Ollama API |
| `numpy` | 2.3.0 | Xử lý buffer audio |
| `pygame` | 2.6.1 | Phát audio qua loa/tai nghe |
| `torch` | 2.12.0 | Silero VAD cần |
| `torchaudio` | 2.11.0 | Silero VAD cần |

---

## 4. Code pipeline

Tất cả nằm trong thư mục `pipeline/`:

| File | Mô tả |
|---|---|
| `pipeline/vad.py` | Silero VAD — nghe micro, phát hiện giọng nói, cắt khi im lặng 500ms |
| `pipeline/stt.py` | faster-whisper small — audio → text tiếng Việt |
| `pipeline/llm.py` | Gọi Ollama API (`cadebot-viva`) → parse JSON response |
| `pipeline/tts.py` | edge-tts `vi-VN-HoaiMyNeural` → phát audio qua pygame |
| `pipeline/main.py` | Vòng lặp chính: VAD → STT → LLM → TTS |
| `pipeline/test_pipeline.py` | Test từng module riêng lẻ |

---

## 5. Lệnh chạy test

### Test từng module (không cần micro)

```bash
cd /home/ncd/learnspaces/Cadebot/pipeline

# Kiểm tra Ollama + model cadebot-viva có sẵn
python3 test_pipeline.py --module ollama --model cadebot-viva

# Test LLM — 6 kịch bản intent (MENU_QA, ADD_TO_CART, REC, FALLBACK, CALL_STAFF, PROMO)
python3 test_pipeline.py --module llm --model cadebot-viva

# Test TTS — nghe thử giọng đọc tiếng Việt
python3 test_pipeline.py --module tts

# Test STT — load Whisper model (không cần micro)
python3 test_pipeline.py --module stt
```

### Test có micro

```bash
# Test VAD phát hiện giọng nói
python3 test_pipeline.py --module vad

# Chạy pipeline đầy đủ
python3 main.py
```

### Gọi API thủ công (không qua pipeline)

```bash
curl -s http://localhost:11434/api/chat \
  -d '{"model":"cadebot-viva","messages":[{"role":"user","content":"Latte giá bao nhiêu?"}],"stream":false}' \
  | python3 -c "import sys,json; r=json.load(sys.stdin); j=json.loads(r['message']['content']); print('intent:',j['intent']); print('answer:',j['spokenText'])"
```

---

## 6. Kết quả test đã chạy

| Test | Kết quả |
|---|---|
| LLM — MENU_QA | ✅ |
| LLM — ADD_TO_CART_DRAFT | ✅ |
| LLM — RECOMMENDATION | ✅ |
| LLM — FALLBACK | ✅ |
| LLM — CALL_STAFF | ✅ |
| LLM — PROMOTION_QA | ✅ |
| JSON parse 6/6 hợp lệ | ✅ |
| TTS giọng đọc tiếng Việt | ✅ |
| Pipeline đầy đủ với micro | ⬜ Chưa test |

---

## 7. Lưu ý khi chạy

- Nếu `python3 main.py` báo `ModuleNotFoundError` → dùng đường dẫn đầy đủ: `/home/ncd/.pyenv/versions/3.11.8/bin/python3 main.py`
- Nếu Ollama chưa chạy → `ollama serve` (hoặc service đã tự start khi boot)
- Timeout LLM mặc định 60s trong `pipeline/llm.py` — đủ cho CPU inference
- STT dùng Whisper `small` (CPU) — nhận dạng tiếng Việt mất ~1.5–3s/câu
- TTS cần kết nối internet (edge-tts gọi Microsoft API)
