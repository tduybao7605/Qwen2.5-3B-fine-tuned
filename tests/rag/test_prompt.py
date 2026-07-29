import json

from rag.prompt import build_context_block, fallback_response
from rag.retriever import RetrievalResult, RetrievedChunk


def test_fallback_matches_android_json_schema():
    resp = fallback_response()
    for key in ["intent", "confidence", "answerText", "spokenText",
                "recommendedItems", "draftCartItems", "requiresHumanSupport", "sourceIds"]:
        assert key in resp, f"thiếu field {key}"
    assert resp["intent"] == "FALLBACK"
    assert resp["requiresHumanSupport"] is True
    assert resp["sourceIds"] == []
    json.dumps(resp)  # phải serialize được


def test_fallback_text_is_vietnamese_and_non_empty():
    resp = fallback_response()
    assert len(resp["answerText"]) > 10
    assert len(resp["spokenText"]) > 10


def test_context_block_lists_chunk_ids_for_citation():
    result = RetrievalResult(
        chunks=[RetrievedChunk("menu:VR_LATTE", "Giá 55.000đ", 0.9)],
        in_scope=True, top_score=0.9,
    )
    block = build_context_block(result)
    assert "menu:VR_LATTE" in block
    assert "55.000đ" in block
    assert "sourceIds" in block  # phải chỉ thị model trích dẫn


def test_context_block_empty_when_out_of_scope():
    assert build_context_block(RetrievalResult()) == ""
