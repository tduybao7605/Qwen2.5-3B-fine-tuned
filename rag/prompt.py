"""Dựng context block cho LLM và câu trả lời FALLBACK cố định."""
from rag.retriever import RetrievalResult

FALLBACK_TEXT = (
    "Xin lỗi bạn, mình chưa có thông tin chính xác về điều này. "
    "Bạn vui lòng hỏi nhân viên Viva để được hỗ trợ nhé!"
)


def fallback_response() -> dict:
    """Trả về ĐÚNG schema JSON mà SYSTEM_PROMPT quy định (serve_model.py)."""
    return {
        "intent": "FALLBACK",
        "confidence": 1.0,
        "answerText": FALLBACK_TEXT,
        "spokenText": FALLBACK_TEXT,
        "recommendedItems": [],
        "draftCartItems": [],
        "requiresHumanSupport": True,
        "sourceIds": [],
    }


def build_context_block(result: RetrievalResult) -> str:
    if not result.in_scope or not result.chunks:
        return ""
    return (
        "### KNOWLEDGE HUB (chỉ được dùng thông tin dưới đây):\n"
        f"{result.context_text}\n\n"
        "### YÊU CẦU:\n"
        "- Chỉ trả lời dựa trên KNOWLEDGE HUB ở trên. Tuyệt đối không bịa "
        "giá, thành phần hay khuyến mãi không có ở trên.\n"
        "- Điền vào sourceIds đúng các mã trong ngoặc vuông [] mà bạn đã dùng.\n"
        "- Nếu KNOWLEDGE HUB không đủ để trả lời, dùng intent FALLBACK."
    )
