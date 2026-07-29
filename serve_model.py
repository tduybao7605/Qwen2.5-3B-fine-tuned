"""
Cadebot API Server
Chạy: python3 serve_model.py
Endpoints:
  POST /stt    — Speech-to-Text bằng PhoWhisper-large
  POST /chat   — Trả lời bằng Qwen2.5-3B + LoRA (có RAG qua Dify + BGE-M3)
  POST /retrieve — Chỉ chạy retrieval để debug, không gọi LLM
  GET  /health — Kiểm tra trạng thái server
"""

import json
import os
import tempfile
import torch
import uvicorn
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

from rag import config as rag_config
from rag.prompt import build_context_block, fallback_response
from rag.retriever import Retriever

# ── Models ─────────────────────────────────────────────────────────────
chat_model = None
chat_tokenizer = None
stt_pipeline = None
retriever = None

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


def load_stt():
    global stt_pipeline
    from transformers import pipeline
    print("Loading PhoWhisper-large (STT)...")
    stt_pipeline = pipeline(
        "automatic-speech-recognition",
        model="vinai/PhoWhisper-large",
        device="cpu",
        torch_dtype=torch.float32,
    )
    print("✅ PhoWhisper-large ready!")


def load_chat():
    global chat_model, chat_tokenizer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    base_model_name = "Qwen/Qwen2.5-3B-Instruct"
    lora_path = "./cadebot-lora"

    print("Loading tokenizer (chat)...")
    chat_tokenizer = AutoTokenizer.from_pretrained(lora_path)

    print("Loading Qwen2.5-3B base model...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=torch.float16,
        device_map="auto",
    )
    print("Loading LoRA adapter...")
    chat_model = PeftModel.from_pretrained(base, lora_path)
    chat_model.eval()
    print("✅ Qwen2.5-3B + LoRA ready!")


def load_retriever():
    global retriever
    if not rag_config.DIFY_DATASET_API_KEY or not rag_config.DIFY_DATASET_ID:
        print("⚠️  Chưa cấu hình Dify — chạy KHÔNG có RAG. Xem docs/RAG_SETUP.md")
        return
    retriever = Retriever()
    probe = retriever.retrieve("Viva Latte giá bao nhiêu")
    if probe.in_scope:
        print(f"✅ RAG ready (bge-m3, threshold={rag_config.SCORE_THRESHOLD}, "
              f"probe top_score={probe.top_score:.3f})")
    else:
        print(f"⚠️  RAG probe không tìm thấy gì (top_score={probe.top_score:.3f}) — "
              "KB đã sync và index xong chưa?")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_stt()
    load_chat()
    load_retriever()
    yield


# ── App ────────────────────────────────────────────────────────────────
app = FastAPI(title="Cadebot API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────
class HistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[HistoryItem] = []
    use_rag: bool = True          # optional — Android không gửi, mặc định bật
    top_k: int | None = None      # optional — để debug


class RetrieveRequest(BaseModel):
    query: str
    top_k: int | None = None


# ── Routes ─────────────────────────────────────────────────────────────
@app.post("/stt")
async def transcribe(file: UploadFile = File(...)):
    import subprocess, soundfile as sf

    audio_bytes = await file.read()

    suffix = "." + (file.filename.split(".")[-1] if file.filename else "m4a")
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    wav_path = tmp_path + ".wav"
    try:
        # ffmpeg chuyển bất kỳ định dạng nào (m4a, mp4...) sang WAV 16kHz mono
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-ar", "16000", "-ac", "1", wav_path],
            capture_output=True, check=True
        )
        audio, _ = sf.read(wav_path, dtype="float32")
        result = stt_pipeline({"array": audio, "sampling_rate": 16000})
        return {"text": result["text"].strip()}
    finally:
        os.unlink(tmp_path)
        if os.path.exists(wav_path):
            os.unlink(wav_path)


@app.post("/chat")
async def chat(req: ChatRequest):
    retrieval = None
    context_block = ""

    if req.use_rag and retriever is not None:
        retrieval = retriever.retrieve(req.message)
        if not retrieval.in_scope:
            # CHẶN CỨNG: không gọi LLM. Tiết kiệm ~78s và loại bỏ nguy cơ bịa.
            return {
                "response": json.dumps(fallback_response(), ensure_ascii=False),
                "retrieval": {
                    "in_scope": False,
                    "top_score": retrieval.top_score,
                    "threshold": retriever.threshold,
                    "sourceIds": [],
                },
            }
        context_block = build_context_block(retrieval)

    system_content = SYSTEM_PROMPT
    if context_block:
        system_content = f"{SYSTEM_PROMPT}\n\n{context_block}"

    messages = [{"role": "system", "content": system_content}]
    for h in req.history[-8:]:
        messages.append({"role": h.role, "content": h.content})
    messages.append({"role": "user", "content": req.message})

    text = chat_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = chat_tokenizer([text], return_tensors="pt").to(
        next(chat_model.parameters()).device
    )

    with torch.no_grad():
        output_ids = chat_model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.7,
            do_sample=True,
            pad_token_id=chat_tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = chat_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    payload = {"response": response}
    if retrieval is not None:
        payload["retrieval"] = {
            "in_scope": True,
            "top_score": retrieval.top_score,
            "threshold": retriever.threshold,
            "sourceIds": retrieval.source_ids,
        }
    return payload


@app.post("/retrieve")
async def retrieve_only(req: RetrieveRequest):
    """Xem retrieval trả về gì mà không tốn 78s chạy LLM."""
    if retriever is None:
        return {"error": "RAG chưa được cấu hình"}
    result = retriever.retrieve(req.query)
    return {
        "in_scope": result.in_scope,
        "top_score": result.top_score,
        "threshold": retriever.threshold,
        "chunks": [
            {"chunk_id": c.chunk_id, "score": c.score, "text": c.text[:300]}
            for c in result.chunks
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "stt_ready": stt_pipeline is not None,
        "chat_ready": chat_model is not None,
        "rag_ready": retriever is not None,
        "embedding_model": rag_config.EMBEDDING_MODEL,
        "score_threshold": rag_config.SCORE_THRESHOLD,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
