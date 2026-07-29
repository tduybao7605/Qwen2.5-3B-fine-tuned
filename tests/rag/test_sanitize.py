"""Mức 2: lọc sourceIds model tự bịa ra."""
import json

from rag.prompt import sanitize_response


def test_drops_source_ids_not_in_context():
    raw = json.dumps({"intent": "MENU_QA", "answerText": "x",
                      "sourceIds": ["menu:VR_LATTE_M", "menu:KHONG_CO_THAT"]})
    out = json.loads(sanitize_response(raw, ["menu:VR_LATTE_M", "faq:md_001"]))
    assert out["sourceIds"] == ["menu:VR_LATTE_M"]


def test_repairs_id_missing_its_prefix():
    """Base model hay trả 'VR_TIRAMISU' thay vì 'menu:VR_TIRAMISU'."""
    raw = json.dumps({"sourceIds": ["VR_TIRAMISU", "md_001"]})
    out = json.loads(sanitize_response(raw, ["menu:VR_TIRAMISU", "faq:md_001"]))
    assert out["sourceIds"] == ["menu:VR_TIRAMISU", "faq:md_001"]


def test_keeps_order_and_drops_duplicates():
    raw = json.dumps({"sourceIds": ["faq:md_001", "menu:VR_LATTE_M", "faq:md_001"]})
    out = json.loads(sanitize_response(raw, ["menu:VR_LATTE_M", "faq:md_001"]))
    assert out["sourceIds"] == ["faq:md_001", "menu:VR_LATTE_M"]


def test_missing_or_null_source_ids_becomes_empty_list():
    assert json.loads(sanitize_response(json.dumps({"intent": "MENU_QA"}), ["a"]))["sourceIds"] == []
    assert json.loads(sanitize_response(json.dumps({"sourceIds": None}), ["a"]))["sourceIds"] == []


def test_non_json_output_is_returned_untouched():
    """Model đôi khi xuất text thường — không được làm hỏng thêm."""
    assert sanitize_response("xin chào bạn", ["a"]) == "xin chào bạn"


def test_other_fields_are_preserved_exactly():
    raw = json.dumps({"intent": "MENU_QA", "confidence": 0.9, "answerText": "Giá 45.000đ",
                      "spokenText": "Giá 45 nghìn", "recommendedItems": [],
                      "draftCartItems": [], "requiresHumanSupport": False,
                      "sourceIds": ["bịa"]}, ensure_ascii=False)
    out = json.loads(sanitize_response(raw, ["menu:VR_PEACH_TEA"]))
    assert out["answerText"] == "Giá 45.000đ"
    assert out["confidence"] == 0.9
    assert out["requiresHumanSupport"] is False
    assert out["sourceIds"] == []
