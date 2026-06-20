from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

BASE = "/home/team3/hrc2026-team3/qwen2.5-3b-base"
LORA = "/home/team3/Qwen2.5-3B-fine-tuned/cadebot-lora"
OUT  = "/home/team3/Qwen2.5-3B-fine-tuned/cadebot-merged"

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
