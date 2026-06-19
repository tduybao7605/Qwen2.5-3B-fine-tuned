"""
Cadebot API Server
Chạy: python3 serve_model.py
Mặc định: http://0.0.0.0:8000
"""

import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

model = None
tokenizer = None

SYSTEM_PROMPT = (
    "Bạn là Cadebot, trợ lý robot phục vụ tại Viva Reserve Coffee. "
    "Chỉ sử dụng Knowledge Hub được cung cấp để trả lời. "
    "Không bịa giá, thành phần, khuyến mãi. "
    "Nếu không tìm thấy thông tin, hãy nói chưa có thông tin chính xác và đề nghị hỏi nhân viên. "
    "Trả lời ngắn gọn, thân thiện, phù hợp môi trường quán cà phê. "
    "Xưng là Cadebot hoặc mình, gọi khách là bạn.\n\n"
    "Luôn trả lời theo định dạng JSON:\n"
    '{"intent":"MENU_QA|RECOMMENDATION|ADD_TO_CART_DRAFT|PROMOTION_QA|CALL_STAFF|FALLBACK",'
    '"confidence":0.9,"answerText":"...","spokenText":"...",'
    '"recommendedItems":[],"draftCartItems":[],"requiresHumanSupport":false,"sourceIds":[]}'
)


def load_model():
    global model, tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model_name = "Qwen/Qwen2.5-3B-Instruct"
    lora_path = "./cadebot-lora"

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(lora_path)

    print("Loading base model (fp16)...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    print("✅ Model ready! Server listening on http://0.0.0.0:8000")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Cadebot API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class HistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[HistoryItem] = []


@app.post("/chat")
async def chat(req: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for h in req.history[-8:]:
        messages.append({"role": h.role, "content": h.content})
    messages.append({"role": "user", "content": req.message})

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(next(model.parameters()).device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return {"response": response}


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
