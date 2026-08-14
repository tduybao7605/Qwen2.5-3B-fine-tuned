"""
Cadebot API Server
Chạy: python3 -m cadebot
Endpoints:
  POST /stt    — Speech-to-Text bằng PhoWhisper-large
  POST /chat   — Trả lời bằng Qwen2.5-3B + LoRA (có RAG qua Dify + BGE-M3)
  POST /retrieve — Chỉ chạy retrieval để debug, không gọi LLM
  GET  /health — Kiểm tra trạng thái server
"""

import json
import os
import tempfile
from contextlib import asynccontextmanager
from typing import List

import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cadebot import models
from cadebot.rag import config as rag_config
from cadebot.rag.prompt import build_context_block, fallback_response, sanitize_response
from cadebot.rag.retriever import Retriever

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
    # Ràng buộc chống \"nói thêm\": model từng gán nhầm \"best seller\" của Viva Latte
    # sang Trà Đào Cam Sả — thông tin lấy từ trí nhớ lúc fine-tune, không có trong context.
    "QUY TẮC BẮT BUỘC:\n"
    "- Chỉ trả lời ĐÚNG điều khách hỏi. Không thêm nhận xét, không quảng cáo thêm.\n"
    "- Không gán cho một món bất kỳ tính chất nào (best seller, ngon nhất, được ưa chuộng, "
    "signature...) trừ khi Knowledge Hub nói RÕ về CHÍNH món đó.\n"
    "- Không suy diễn từ món này sang món khác.\n"
    "- Thà trả lời ngắn còn hơn thêm chi tiết không có trong Knowledge Hub.\n\n"
    "Luôn trả lời theo định dạng JSON:\n"
    '{"intent":"MENU_QA|RECOMMENDATION|ADD_TO_CART_DRAFT|PROMOTION_QA|CALL_STAFF|FALLBACK",'
    '"confidence":0.9,"answerText":"...","spokenText":"...",'
    '"recommendedItems":[],"draftCartItems":[],"requiresHumanSupport":false,"sourceIds":[]}'
)


def load_retriever():
    global retriever
    if not rag_config.DIFY_DATASET_API_KEY or not rag_config.DIFY_DATASET_ID:
        print("⚠️  Chưa cấu hình Dify — chạy KHÔNG có RAG. Xem docs/rag-setup.md")
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
    global stt_pipeline, chat_model, chat_tokenizer
    stt_pipeline = models.load_stt()
    chat_model, chat_tokenizer = models.load_chat()
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
    # Tắt mặc định 2026-08-04: có RAG mất ~190s, vượt giới hạn 100s của
    # Cloudflare edge proxy (lỗi 524). Không RAG còn ~95s, qua được ngưỡng.
    use_rag: bool = False          # optional — Android không gửi field này
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
        # Whisper chỉ nuốt được 30s mỗi lần. Dài hơn thì phải bật long-form,
        # mà long-form bắt buộc model dự đoán timestamp token — không truyền
        # return_timestamps là nó ném ValueError. Robot lắng nghe liên tục nên
        # đây là đường thường gặp, không phải ngoại lệ hiếm.
        duration_s = len(audio) / 16000
        extra = {"return_timestamps": True} if duration_s > 30 else {}
        result = stt_pipeline({"array": audio, "sampling_rate": 16000}, **extra)
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
            temperature=rag_config.GEN_TEMPERATURE,
            do_sample=True,
            pad_token_id=chat_tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = chat_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Bỏ các sourceIds model tự bịa; vá luôn ID thiếu tiền tố.
    if retrieval is not None:
        response = sanitize_response(response, retrieval.source_ids)

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
