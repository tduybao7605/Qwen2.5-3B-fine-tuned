# Báo cáo đánh giá Fine-tune Cadebot

> **Model:** Qwen2.5-3B-Instruct → Cadebot (LoRA)
> **Ngày thực hiện:** 2026-06-19
> **Script:** `finetune_cadebot.py`

---

## 1. Dataset

### Thống kê tổng quan

| Tập | Số mẫu | Tỷ lệ |
|-----|--------|-------|
| Train | 144 | 84.7% |
| Validation | 26 | 15.3% |
| **Tổng** | **170** | 100% |

### Thống kê độ dài mẫu (tính theo ký tự)

| | Giá trị |
|--|---------|
| Trung bình | 908 ký tự/mẫu |
| Ngắn nhất | 668 ký tự |
| Dài nhất | 2 175 ký tự |

### Phân phối intent trong tập train

| Intent | Số mẫu | Tỷ lệ |
|--------|--------|-------|
| MENU_QA | 51 | 35.4% |
| ADD_TO_CART_DRAFT | 33 | 22.9% |
| RECOMMENDATION | 22 | 15.3% |
| FALLBACK | 18 | 12.5% |
| CALL_STAFF | 12 | 8.3% |
| PROMOTION_QA | 8 | 5.6% |

### Định dạng dữ liệu

Mỗi mẫu là một object JSON theo chuẩn **chat format**:

```json
{
  "messages": [
    {"role": "system",    "content": "Bạn là Cadebot..."},
    {"role": "user",      "content": "<câu hỏi của khách>"},
    {"role": "assistant", "content": "{\"intent\": \"...\", \"answerText\": \"...\", ...}"}
  ]
}
```

Assistant luôn trả về JSON có cấu trúc gồm các trường: `intent`, `confidence`, `answerText`, `spokenText`, `recommendedItems`, `draftCartItems`, `requiresHumanSupport`, `sourceIds`.

---

## 2. Cấu hình fine-tune

### Base model

| Thuộc tính | Giá trị |
|-----------|---------|
| Model ID | `Qwen/Qwen2.5-3B-Instruct` |
| Đường dẫn local | `./qwen2.5-3b-base/` |
| Kích thước | 5.8 GB (2 file safetensors) |
| Tổng tham số | 3,205,672,960 (~3.2B) |
| Dtype khi load | float16 (fp16) |
| Quantization | Không dùng (VRAM đủ) |

### Phương pháp: LoRA (Low-Rank Adaptation)

Lý do chọn LoRA thay vì full fine-tune: chỉ cần update một phần nhỏ tham số (~3.7%), giảm thời gian train và tránh catastrophic forgetting trên kiến thức gốc của model.

| Tham số LoRA | Giá trị |
|-------------|---------|
| Rank (`r`) | 64 |
| Alpha (`lora_alpha`) | 128 |
| Dropout | 0.05 |
| Bias | none |
| Scaling (alpha/r) | 2.0 |

**Target modules** (7 layer loại):
```
q_proj, k_proj, v_proj, o_proj,
gate_proj, up_proj, down_proj
```

| Loại tham số | Số lượng |
|---|---|
| Trainable (LoRA) | 119,734,272 |
| Frozen (base) | 3,085,938,688 |
| **Tổng** | **3,205,672,960** |
| % trainable | **3.74%** |

### Hyperparameter training

| Hyperparameter | Giá trị | Ghi chú |
|----------------|---------|---------|
| Epochs | 5 | |
| Batch size / GPU | 8 | |
| Gradient accumulation | 2 | |
| **Effective batch size** | **16** | 8 × 2 |
| Learning rate | 2e-4 | |
| LR scheduler | cosine | |
| Warmup ratio | 0.05 | ~2 steps |
| Optimizer | adamw_torch | |
| Precision | fp16 | |
| Max sequence length | 1,024 tokens | |
| Packing | Không | |
| Total steps | 45 | 144/8/2 × 5 epochs |
| Logging mỗi | 5 steps | |
| Save/eval mỗi | 50 steps | |

### Phần cứng

| | Thông tin |
|--|-----------|
| GPU | NVIDIA RTX 5880 Ada Generation |
| VRAM tổng | 50.9 GB |
| VRAM free khi bắt đầu | 44.5 GB |
| Framework | PyTorch 2.7.1+cu128 |
| Python | 3.12 (venv: `/home/team3/Isaac-GR00T/.venv`) |
| transformers | 4.57.6 |
| peft | 0.17.1 |
| trl | 1.6.0 |
| bitsandbytes | 0.49.2 |

---

## 3. Kết quả training

### Lịch sử loss & accuracy

| Step | Epoch | Train Loss | Token Accuracy | Learning Rate | Grad Norm |
|------|-------|-----------|----------------|---------------|-----------|
| 5 | 0.56 | 2.5334 | 56.63% | 1.997e-4 | 1.6745 |
| 10 | 1.11 | 0.7868 | 82.56% | 1.901e-4 | 0.8797 |
| 15 | 1.67 | 0.4260 | 89.70% | 1.680e-4 | 0.5665 |
| 20 | 2.22 | 0.3648 | 90.67% | 1.365e-4 | 0.4278 |
| 25 | 2.78 | 0.2666 | 93.02% | 1.000e-4 | 0.4046 |
| 30 | 3.33 | 0.2292 | 93.61% | 6.347e-5 | 0.3451 |
| 35 | 3.89 | 0.1913 | 94.78% | 3.198e-5 | 0.4265 |
| 40 | 4.44 | 0.1575 | **95.66%** | 9.903e-6 | 0.3793 |
| **45** | **5.00** | **0.1642** | **95.63%** | 2.796e-7 | 0.3471 |

### Biểu đồ loss (text)

```
Loss
2.5 │▓
    │
1.5 │
    │
0.8 │  ▓
    │
0.4 │     ▓
0.3 │        ▓
0.2 │           ▓  ▓  ▓
0.1 │                    ▓  ▓
    └──────────────────────────── Step
       5  10  15  20  25  30  35  40  45
```

### Biểu đồ token accuracy (text)

```
Accuracy
95% │                             ██
90% │        ██  ██
85% │  ██
56% │██
    └─────────────────────────────── Step
       5   10  15  20  25  30  35  40  45
```

### Tóm tắt kết quả

| Chỉ số | Đầu training | Cuối training | Thay đổi |
|--------|-------------|--------------|---------|
| Train Loss | 2.5334 | 0.1642 | ↓ 93.5% |
| Token Accuracy | 56.63% | 95.63% | ↑ 39.0 pp |
| Grad Norm | 1.6745 | 0.3471 | ↓ 79.3% |
| Train Loss TB | — | 0.5689 | — |

**Tổng thời gian train:** 98.78 giây (~1 phút 39 giây)
**Throughput:** 7.29 samples/giây | 0.456 steps/giây
**Total FLOPs:** 4.48 × 10¹⁵

> **Nhận xét:** Loss giảm mạnh và nhanh từ epoch 1→2 (2.53 → 0.43), sau đó hội tụ dần. Gradient norm giảm liên tục — model học ổn định, không có dấu hiệu diverge hay gradient explosion. Accuracy 95.6% ở epoch 5 là tốt với 144 mẫu.

---

## 4. Checkpoint & Output

### Vị trí lưu trữ

```
hrc2026-team3/
├── qwen2.5-3b-base/               ← Base model (5.8 GB, không thay đổi)
│   ├── config.json
│   ├── model-00001-of-00002.safetensors
│   ├── model-00002-of-00002.safetensors
│   └── tokenizer.json
│
└── cadebot-lora/                  ← Output fine-tune (1.9 GB tổng)
    ├── adapter_config.json        ← Cấu hình LoRA
    ├── adapter_model.safetensors  ← Trọng số LoRA (457 MB)
    ├── tokenizer.json             ← Tokenizer đã lưu
    ├── training_args.bin          ← Training arguments
    └── checkpoint-45/             ← Checkpoint cuối (step 45 = epoch 5)
        ├── adapter_model.safetensors
        ├── optimizer.pt           (914 MB — optimizer states)
        ├── trainer_state.json
        └── ...
```

### Số checkpoint

| Checkpoint | Step | Epoch | Ghi chú |
|-----------|------|-------|---------|
| `checkpoint-45` | 45 | 5.0 | Checkpoint duy nhất (cuối training) |

> Chỉ có 1 checkpoint vì `save_steps=50` > `max_steps=45` — training kết thúc trước khi trigger save theo interval. Checkpoint-45 được lưu tự động ở cuối training.

### Kích thước output

| File | Kích thước | Mô tả |
|------|-----------|-------|
| `adapter_model.safetensors` | 457 MB | LoRA weights (final) |
| `checkpoint-45/optimizer.pt` | 914 MB | Optimizer states (để resume) |
| Tokenizer files | ~15 MB | vocab, merges, template |
| **Tổng cộng** | **~1.9 GB** | |

> LoRA adapter chỉ **457 MB** so với base model **5.8 GB** — nhỏ hơn 12.7 lần.

---

## 5. Cách sử dụng model đã fine-tune

### Load để inference

```python
import os
os.environ["PYTHON_INCLUDE_DIR"] = "/home/team3/python_headers/python3.12"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# Load base + adapter
base = AutoModelForCausalLM.from_pretrained(
    "./qwen2.5-3b-base",
    dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(base, "./cadebot-lora")
tokenizer = AutoTokenizer.from_pretrained("./cadebot-lora")
model.eval()
```

### Chạy thử inference

```python
SYSTEM = (
    "Bạn là Cadebot, trợ lý robot phục vụ tại Viva Reserve Coffee. "
    "Chỉ sử dụng Knowledge Hub được cung cấp để trả lời. "
    "Không bịa giá, thành phần, khuyến mãi. "
    "Nếu không tìm thấy thông tin, hãy nói chưa có thông tin chính xác và đề nghị hỏi nhân viên. "
    "Trả lời ngắn gọn, thân thiện, phù hợp môi trường quán cà phê. "
    "Xưng là Cadebot hoặc mình, gọi khách là bạn."
)

messages = [
    {"role": "system",  "content": SYSTEM},
    {"role": "user",    "content": "Cho tôi 1 Latte size M ít đường"},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.1, do_sample=True)

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```

### Merge adapter vào base (để deploy không cần peft)

```python
merged = model.merge_and_unload()
merged.save_pretrained("./cadebot-merged")
tokenizer.save_pretrained("./cadebot-merged")
# ./cadebot-merged/ dùng như model HuggingFace thông thường, ~5.8 GB
```

### Lệnh chạy lại fine-tune

```bash
PYTHON_INCLUDE_DIR=/home/team3/python_headers/python3.12 \
/home/team3/Isaac-GR00T/.venv/bin/python finetune_cadebot.py
```

---

## 6. Đánh giá & hướng cải thiện

### Điểm mạnh

- Loss hội tụ tốt (2.53 → 0.16) trong thời gian rất ngắn (~99 giây)
- Token accuracy 95.6% — model học được cấu trúc JSON output
- Gradient norm ổn định, không có spike bất thường

### Hạn chế hiện tại

- **Không có eval loss:** `eval_steps=50` > `max_steps=45` nên validation set chưa được đánh giá trong quá trình train. Cần chạy thêm evaluation riêng trên val set sau khi train.
- **Dataset nhỏ (144 mẫu):** Có nguy cơ overfit — model thuộc lòng training set thay vì tổng quát hóa. Cần đánh giá trên các câu hỏi ngoài dataset.
- **Chỉ có 1 checkpoint:** Không thể quay lại best checkpoint giữa chừng vì training xong trước khi save_steps kích hoạt.

### Hướng cải thiện

| Vấn đề | Giải pháp |
|--------|----------|
| Không có eval trong training | Giảm `eval_steps=10` để eval mỗi 10 steps |
| Chỉ 1 checkpoint | Giảm `save_steps=10` hoặc thêm `save_total_limit=3` |
| Dataset nhỏ | Sinh thêm dữ liệu bằng `generate_dataset.py` hoặc augmentation |
| Chưa đo val loss | Chạy eval riêng sau training (xem script bên dưới) |

### Script đánh giá nhanh trên val set

```bash
PYTHON_INCLUDE_DIR=/home/team3/python_headers/python3.12 \
/home/team3/Isaac-GR00T/.venv/bin/python - << 'EOF'
import json, torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained("./cadebot-lora")
base  = AutoModelForCausalLM.from_pretrained("./qwen2.5-3b-base", dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base, "./cadebot-lora")
model.eval()

correct = 0
with open("./dataset/val.jsonl") as f:
    samples = [json.loads(l) for l in f if l.strip()]

for s in samples:
    msgs = s["messages"][:-1]  # bỏ assistant turn
    expected_intent = json.loads(s["messages"][-1]["content"])["intent"]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    try:
        predicted_intent = json.loads(response.strip())["intent"]
        if predicted_intent == expected_intent:
            correct += 1
        else:
            print(f"WRONG: expected={expected_intent}, got={predicted_intent}")
    except:
        print(f"PARSE ERROR: {response[:80]}")

print(f"\nIntent Accuracy: {correct}/{len(samples)} = {correct/len(samples)*100:.1f}%")
EOF
```
