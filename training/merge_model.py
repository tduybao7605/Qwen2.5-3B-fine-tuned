from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path
import os
import torch

ROOT = Path(__file__).resolve().parent.parent
BASE = os.getenv("CADEBOT_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
LORA = os.getenv("CADEBOT_MODEL_DIR", str(ROOT / "cadebot-lora"))
OUT  = str(ROOT / "cadebot-merged")

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    BASE, dtype=torch.float16, device_map="auto"
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base, LORA)

print("Merging weights...")
merged = model.merge_and_unload()

print(f"Saving to {OUT} ...")
merged.save_pretrained(OUT)
AutoTokenizer.from_pretrained(LORA).save_pretrained(OUT)

print("Xong! Merge hoàn tất.")
