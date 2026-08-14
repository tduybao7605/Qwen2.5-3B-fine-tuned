"""Kiểm tra hành vi chặn cứng out-of-scope ở /chat.

Không dùng `with TestClient(...)` — context manager sẽ chạy lifespan và load
Qwen2.5-3B + PhoWhisper (nhiều phút trên CPU). Ở đây chat_model cố tình để None:
nếu route lỡ gọi tới LLM, test sẽ nổ AttributeError thay vì âm thầm pass.
"""
import json

import pytest
from fastapi.testclient import TestClient

from cadebot import api
from cadebot.rag.retriever import RetrievalResult, RetrievedChunk


class FakeRetriever:
    threshold = 0.55

    def __init__(self, result):
        self.result = result
        self.calls = []

    def retrieve(self, query, top_k=None):
        self.calls.append(query)
        self.top_k_seen = top_k
        return self.result


@pytest.fixture
def client():
    return TestClient(api.app)


@pytest.fixture(autouse=True)
def _no_models():
    """Đảm bảo LLM chưa được load — chạm vào nó là lỗi ngay."""
    api.chat_model = None
    api.chat_tokenizer = None
    yield
    api.retriever = None


def test_out_of_scope_returns_fallback_without_touching_llm(client):
    fake = FakeRetriever(RetrievalResult(chunks=[], in_scope=False, top_score=0.21))
    api.retriever = fake

    resp = client.post("/chat", json={"message": "hôm nay trời có mưa không"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["retrieval"]["in_scope"] is False
    assert body["retrieval"]["sourceIds"] == []

    payload = json.loads(body["response"])
    assert payload["intent"] == "FALLBACK"
    assert payload["requiresHumanSupport"] is True
    assert payload["sourceIds"] == []
    assert fake.calls == ["hôm nay trời có mưa không"]


def test_android_payload_shape_still_works(client):
    """Android chỉ gửi {message, history} — không có use_rag."""
    api.retriever = FakeRetriever(
        RetrievalResult(chunks=[], in_scope=False, top_score=0.1)
    )
    resp = client.post("/chat", json={"message": "giá vàng hôm nay", "history": []})
    assert resp.status_code == 200
    # `response` phải luôn là chuỗi JSON đúng schema cũ
    assert isinstance(resp.json()["response"], str)
    json.loads(resp.json()["response"])


def test_use_rag_defaults_on_for_android_payload():
    """Android không gửi use_rag, nên mặc định quyết định luôn hành vi thật.

    Phải là BẬT: tắt RAG nghĩa là model trả lời bằng trí nhớ fine-tune — bịa
    giá, bịa khuyến mãi, và không còn chặn cứng câu ngoài phạm vi. Deployment
    nào bị chặn bởi timeout của proxy thì tự gửi use_rag=false, đừng hạ mặc
    định của cả hệ thống xuống. Xem docs/deployment.md.
    """
    req = api.ChatRequest(message="giá vàng hôm nay", history=[])
    assert req.use_rag is True


def test_retrieve_endpoint_reports_scores(client):
    api.retriever = FakeRetriever(
        RetrievalResult(
            chunks=[RetrievedChunk("menu:VR_LATTE_M", "Giá 55,000 VNĐ", 0.82)],
            in_scope=True,
            top_score=0.82,
        )
    )
    resp = client.post("/retrieve", json={"query": "Viva Latte giá bao nhiêu"})
    body = resp.json()
    assert body["in_scope"] is True
    assert body["chunks"][0]["chunk_id"] == "menu:VR_LATTE_M"


def test_health_exposes_rag_state(client):
    api.retriever = None
    body = client.get("/health").json()
    assert body["rag_ready"] is False
    assert body["embedding_model"] == "bge-m3"


def test_top_k_from_the_request_reaches_the_retriever(client):
    """Field top_k từng được nhận rồi bỏ qua — mọi request đều dùng config.TOP_K."""
    fake = FakeRetriever(RetrievalResult(chunks=[], in_scope=False, top_score=0.1))
    api.retriever = fake

    client.post("/retrieve", json={"query": "Viva Latte giá bao nhiêu", "top_k": 7})
    assert fake.top_k_seen == 7

    client.post("/chat", json={"message": "Viva Latte giá bao nhiêu", "top_k": 5})
    assert fake.top_k_seen == 5
