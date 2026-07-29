from unittest.mock import patch

from rag.retriever import RetrievalResult, RetrievedChunk, Retriever, parse_chunk_id


def _fake_response(records):
    class R:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"query": {"content": "q"}, "records": records}

    return R()


def _record(content, score):
    return {"segment": {"id": "seg", "content": content, "document": {"name": "d"}},
            "score": score}


def test_parse_chunk_id_strips_the_id_line():
    cid, body = parse_chunk_id("[menu:VR_LATTE]\nGiá 55.000đ")
    assert cid == "menu:VR_LATTE"
    assert body == "Giá 55.000đ"


def test_parse_chunk_id_returns_unknown_when_missing():
    cid, body = parse_chunk_id("không có id")
    assert cid == "unknown"
    assert body == "không có id"


def test_scores_below_threshold_are_dropped_and_marked_out_of_scope():
    with patch("rag.retriever.requests.post",
               return_value=_fake_response([_record("[menu:VR_LATTE]\nx", 0.21)])):
        result = Retriever(threshold=0.55).retrieve("hôm nay trời mưa không")
    assert result.in_scope is False
    assert result.chunks == []
    assert result.source_ids == []


def test_scores_above_threshold_are_kept_and_marked_in_scope():
    with patch("rag.retriever.requests.post", return_value=_fake_response([
        _record("[menu:VR_LATTE]\nGiá 55.000đ", 0.82),
        _record("[faq:md_001]\nQ: vị thế nào\nA: béo nhẹ", 0.61),
        _record("[doc:03_Khong_Gian#1]\nkhông liên quan", 0.30),
    ])):
        result = Retriever(threshold=0.55).retrieve("Viva Latte giá bao nhiêu")
    assert result.in_scope is True
    assert result.source_ids == ["menu:VR_LATTE", "faq:md_001"]
    assert result.top_score == 0.82
    assert "55.000đ" in result.context_text
    assert "không liên quan" not in result.context_text


def test_empty_records_is_out_of_scope_not_a_crash():
    with patch("rag.retriever.requests.post", return_value=_fake_response([])):
        result = Retriever(threshold=0.55).retrieve("xyz")
    assert result.in_scope is False
    assert result.top_score == 0.0


def test_network_failure_degrades_to_out_of_scope():
    import requests as _rq
    with patch("rag.retriever.requests.post", side_effect=_rq.Timeout("boom")):
        result = Retriever(threshold=0.55).retrieve("Viva Latte giá bao nhiêu")
    # Dify chết thì thà từ chối còn hơn để LLM bịa
    assert result.in_scope is False


def test_context_text_is_truncated_to_budget():
    long_chunk = _record("[menu:VR_X]\n" + "dài " * 5000, 0.9)
    with patch("rag.retriever.requests.post", return_value=_fake_response([long_chunk])):
        result = Retriever(threshold=0.55).retrieve("q")
    assert len(result.context_text) <= 2100  # MAX_CONTEXT_CHARS + nhãn
