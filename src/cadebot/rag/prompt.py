"""Dựng context block cho LLM và câu trả lời FALLBACK cố định."""
import json
from cadebot.rag.retriever import RetrievalResult

FALLBACK_TEXT = (
    "Xin lỗi bạn, mình chưa có thông tin chính xác về điều này. "
    "Bạn vui lòng hỏi nhân viên Viva để được hỗ trợ nhé!"
)


def fallback_response() -> dict:
    """Trả về ĐÚNG schema JSON mà SYSTEM_PROMPT quy định (cadebot/api.py)."""
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


def sanitize_response(raw: str, allowed_ids: list[str]) -> str:
    """Lọc sourceIds: chỉ giữ ID thật sự có trong context vừa truy xuất.

    Model hay bịa ID không tồn tại, hoặc trả thiếu tiền tố ('VR_TIRAMISU' thay vì
    'menu:VR_TIRAMISU'). Cái sau sửa được nên sửa, cái trước bỏ.

    Trả về chuỗi JSON đã chỉnh. Nếu raw không phải JSON thì trả nguyên xi —
    không được làm hỏng thêm output vốn đã sai định dạng.
    """
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw
    if not isinstance(payload, dict):
        return raw

    # Bản đồ hậu tố -> id đầy đủ, để vá được ID thiếu tiền tố.
    by_suffix = {cid.split(":", 1)[-1]: cid for cid in allowed_ids}
    allowed = set(allowed_ids)

    cleaned: list[str] = []
    for cid in payload.get("sourceIds") or []:
        if not isinstance(cid, str):
            continue
        resolved = cid if cid in allowed else by_suffix.get(cid.split(":", 1)[-1])
        if resolved and resolved not in cleaned:
            cleaned.append(resolved)

    payload["sourceIds"] = cleaned
    return json.dumps(payload, ensure_ascii=False)
