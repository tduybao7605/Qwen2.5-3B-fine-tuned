"""Nạp model. Không import FastAPI ở đây — để tầng API import được mà chưa cần trọng số."""
import torch

from cadebot.rag import config


def load_stt():
    """PhoWhisper-large cho tiếng Việt. CPU, fp32."""
    from transformers import pipeline

    print(f"Loading {config.STT_MODEL} (STT)...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=config.STT_MODEL,
        device="cpu",
        torch_dtype=torch.float32,
    )
    print("✅ STT ready!")
    return pipe


def load_chat():
    """Qwen2.5-3B-Instruct base + LoRA adapter đã fine-tune trên dataset Cadebot."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    lora_path = str(config.MODEL_DIR)

    print("Loading tokenizer (chat)...")
    tokenizer = AutoTokenizer.from_pretrained(lora_path)

    print(f"Loading {config.BASE_MODEL} base model...")
    base = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
    )
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    print("✅ Chat model ready!")
    return model, tokenizer
