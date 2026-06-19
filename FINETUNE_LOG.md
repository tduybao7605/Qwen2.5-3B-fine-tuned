# Cadebot Fine-tune — Nhật ký thực hiện

> Người thực hiện: Claude (AI assistant)
> Ngày: 2026-06-19
> Mục tiêu: Fine-tune Qwen2.5-3B-Instruct thành Cadebot — trợ lý robot phục vụ tại Viva Reserve Coffee

---

## Tóm tắt nhanh

| Hạng mục | Kết quả |
|----------|---------|
| Base model | Qwen/Qwen2.5-3B-Instruct (5.8 GB) |
| Phương pháp | LoRA (fp16, không cần quantization) |
| Dataset | 144 train + 26 val (JSONL chat format) |
| Thời gian train | ~98 giây |
| Loss cuối | 0.16 (từ 2.53) |
| Token accuracy cuối | 95.7% (từ 56.6%) |
| Output | `./cadebot-lora/` |

---

## 1. Khám phá môi trường

**Vấn đề phát hiện:**
- Máy không có `pip` / `pip3` / `pip3 install` trực tiếp
- Python hệ thống (`/usr/bin/python3`) không có module nào cả
- Không có `conda`, không có venv riêng cho project

**Giải pháp:**
Phát hiện venv sẵn có của Isaac-GR00T tại `/home/team3/Isaac-GR00T/.venv/` đã có sẵn:
- `torch 2.7.1+cu128`
- `transformers 4.57.6`
- `peft 0.17.1`
- `datasets 4.8.5`
- GPU: **NVIDIA RTX 5880 Ada Generation — 50.9 GB VRAM** (guide gốc giả định 7.5 GB)

---

## 2. Tải model Qwen2.5-3B-Instruct

Dùng Python của GR00T venv để tải model từ HuggingFace:

```bash
/home/team3/Isaac-GR00T/.venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen2.5-3B-Instruct',
    local_dir='./qwen2.5-3b-base',
    local_dir_use_symlinks=False,
    ignore_patterns=['*.pt', '*.bin', 'original/*'],
)
"
```

**Kết quả:** Model lưu tại `./qwen2.5-3b-base/` — 5.8 GB, gồm:
- `config.json`, `tokenizer.json`, `tokenizer_config.json`
- `model-00001-of-00002.safetensors`, `model-00002-of-00002.safetensors`

---

## 3. Cài thư viện còn thiếu

Dùng `uv` (có sẵn tại `~/.local/bin/uv`) để cài vào GR00T venv:

```bash
uv pip install trl bitsandbytes --python /home/team3/Isaac-GR00T/.venv/bin/python
```

**Lý do cần 2 thư viện này:**
- **`trl`**: cung cấp `SFTTrainer` và `SFTConfig` — training loop chuyên dụng cho LLM, tự xử lý chat template và loss masking chỉ trên phần assistant response
- **`bitsandbytes`**: ban đầu cần cho QLoRA 4-bit, cuối cùng không dùng quantization (VRAM đủ) nhưng vẫn cần để import không lỗi

**Sự cố với bitsandbytes:**
bitsandbytes 0.49.2 phụ thuộc vào `triton`, và `triton` cần compile CUDA helper khi import lần đầu. Compilation thất bại vì thiếu `Python.h` tại `/usr/include/python3.12`.

**Giải pháp:** Triton đã có fallback built-in — set biến môi trường:
```bash
export PYTHON_INCLUDE_DIR=/home/team3/python_headers/python3.12
```
(headers có sẵn tại `/home/team3/python_headers/python3.12/Python.h`)

---

## 4. Viết script fine-tune (`finetune_cadebot.py`)

Script tại `./finetune_cadebot.py`. Các điểm kỹ thuật chính:

### Dataset
- Format: JSONL, mỗi dòng là `{"messages": [system, user, assistant]}`
- Assistant output là JSON có cấu trúc (intent, answerText, spokenText, draftCartItems, ...)
- Dùng `tokenizer.apply_chat_template()` để convert sang text

### Lý do chọn LoRA thay vì QLoRA
Guide gốc thiết kế cho 7.5 GB VRAM → cần QLoRA 4-bit. Thực tế GPU có 50.9 GB → dùng fp16 trực tiếp:
- Nhanh hơn (không overhead dequantization)
- Stable hơn (fp16 thuần)
- Dùng `r=64, alpha=128` (cao hơn guide gốc r=16) vì VRAM dư dả

### Cấu hình LoRA
```python
LoraConfig(
    r=64, lora_alpha=128, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
)
```
Trainable params: **119,734,272 / 3,205,672,960 = 3.7%**

### Cấu hình training
```python
SFTConfig(
    num_train_epochs=5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,   # effective batch = 16
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    fp16=True,
    optim="adamw_torch",
    dataset_text_field="text",
    max_length=1024,
)
```

### Sự cố API TRL 1.6
TRL 1.6 thay đổi API so với các version cũ:
| Cũ (TRL 0.x) | Mới (TRL 1.6) |
|---|---|
| `TrainingArguments` | `SFTConfig` |
| `evaluation_strategy` | `eval_strategy` |
| `max_seq_length` | `max_length` |
| `tokenizer=...` | `processing_class=...` |

---

## 5. Chạy fine-tune

```bash
PYTHON_INCLUDE_DIR=/home/team3/python_headers/python3.12 \
/home/team3/Isaac-GR00T/.venv/bin/python finetune_cadebot.py
```

---

## 6. Kết quả training

| Step | Epoch | Loss | Token Accuracy |
|------|-------|------|---------------|
| 5 | 0.56 | 2.533 | 56.6% |
| 10 | 1.11 | 0.787 | 82.6% |
| 15 | 1.67 | 0.426 | 89.7% |
| 20 | 2.22 | 0.365 | 90.7% |
| 25 | 2.78 | 0.267 | 93.0% |
| 30 | 3.33 | 0.229 | 93.6% |
| 35 | 3.89 | 0.191 | 94.8% |
| 40 | 4.44 | 0.158 | 95.7% |
| 45 | 5.00 | 0.164 | 95.6% |

- **Train runtime:** 98.8 giây
- **Train loss cuối:** 0.569 (trung bình)
- **Loss giảm:** 2.53 → 0.16 (~94%)

---

## 7. Output

LoRA adapter lưu tại `./cadebot-lora/`:
```
cadebot-lora/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── ...
```

### Cách load model sau khi fine-tune

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    "./qwen2.5-3b-base",
    dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "./cadebot-lora")
tokenizer = AutoTokenizer.from_pretrained("./cadebot-lora")
```

### Cách merge adapter vào base model (để deploy)

```python
merged = model.merge_and_unload()
merged.save_pretrained("./cadebot-merged")
tokenizer.save_pretrained("./cadebot-merged")
```

---

## Lệnh chạy lại (tóm tắt)

```bash
# Bước 1: Đảm bảo biến môi trường
export PYTHON_INCLUDE_DIR=/home/team3/python_headers/python3.12
export PYTHON=/home/team3/Isaac-GR00T/.venv/bin/python

# Bước 2: Chạy fine-tune
$PYTHON finetune_cadebot.py

# Bước 3: Kết quả ở ./cadebot-lora/
```
