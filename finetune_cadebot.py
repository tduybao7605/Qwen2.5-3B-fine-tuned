"""
Fine-tune Qwen2.5-3B-Instruct → Cadebot (LoRA fp16)
Dataset: dataset/train.jsonl + val.jsonl
Output : ./cadebot-lora/

Run:
    PYTHON_INCLUDE_DIR=/home/team3/python_headers/python3.12 \
    /home/team3/Isaac-GR00T/.venv/bin/python finetune_cadebot.py
"""

import json
import os
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODEL_PATH   = "./qwen2.5-3b-base"
TRAIN_FILE   = "./dataset/train.jsonl"
VAL_FILE     = "./dataset/val.jsonl"
OUTPUT_DIR   = "./cadebot-lora"
MAX_SEQ_LEN  = 1024

# LoRA — r=64 vì có nhiều VRAM (50 GB)
LORA_R       = 64
LORA_ALPHA   = 128
LORA_DROPOUT = 0.05

# Training — batch lớn hơn vì VRAM dư dả
EPOCHS       = 5
BATCH_SIZE   = 8
GRAD_ACCUM   = 2    # effective batch = 16
LR           = 2e-4
WARMUP_RATIO = 0.05
SAVE_STEPS   = 50
EVAL_STEPS   = 50

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
def load_jsonl(path):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def apply_chat_template(sample, tokenizer):
    text = tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Cadebot LoRA Fine-tune — Qwen2.5-3B-Instruct (fp16)")
    print("=" * 60)

    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        free, _ = torch.cuda.mem_get_info()
        print(f"  GPU  : {p.name}")
        print(f"  VRAM : {round(p.total_memory/1e9,1)} GB total | {round(free/1e9,1)} GB free")
    else:
        raise RuntimeError("CUDA not available!")

    # 1. Tokenizer
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2. Dataset
    print("[2/4] Loading dataset...")
    train_raw = load_jsonl(TRAIN_FILE)
    val_raw   = load_jsonl(VAL_FILE)
    print(f"      Train: {len(train_raw)} | Val: {len(val_raw)}")

    train_ds = Dataset.from_list(train_raw).map(lambda x: apply_chat_template(x, tokenizer))
    val_ds   = Dataset.from_list(val_raw).map(lambda x: apply_chat_template(x, tokenizer))

    # 3. Model (fp16, no quantization — VRAM đủ)
    print("[3/4] Loading model in fp16...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 4. Train
    print("[4/4] Starting training...")
    sft_cfg = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=WARMUP_RATIO,
        fp16=True,
        bf16=False,
        logging_steps=5,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        optim="adamw_torch",
        group_by_length=True,
        dataloader_num_workers=0,
        dataset_text_field="text",
        max_length=MAX_SEQ_LEN,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    trainer.train()

    print("\nSaving final model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n Done! LoRA adapter saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
