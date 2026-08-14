"""Truy vấn Dify Retrieval API và quyết định in-scope / out-of-scope."""
from dataclasses import dataclass, field

import requests

from cadebot.rag import config


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk] = field(default_factory=list)
    in_scope: bool = False
    top_score: float = 0.0

    @property
    def source_ids(self) -> list[str]:
        return [c.chunk_id for c in self.chunks]

    @property
    def context_text(self) -> str:
        parts, budget = [], config.MAX_CONTEXT_CHARS
        for c in self.chunks:
            piece = f"[{c.chunk_id}]\n{c.text}"
            if len(piece) > budget:
                piece = piece[:budget]
            parts.append(piece)
            budget -= len(piece)
            if budget <= 0:
                break
        return "\n\n".join(parts)


def parse_chunk_id(content: str) -> tuple[str, str]:
    """Dòng đầu dạng `[menu:VR_LATTE]` -> ('menu:VR_LATTE', phần còn lại)."""
    stripped = content.lstrip()
    if stripped.startswith("["):
        end = stripped.find("]")
        newline = stripped.find("\n")
        if 0 < end < (newline if newline != -1 else len(stripped)):
            return stripped[1:end], stripped[end + 1:].lstrip("\n")
    return "unknown", content


class Retriever:
    def __init__(self, api_key: str | None = None, dataset_id: str | None = None,
                 base_url: str | None = None, threshold: float | None = None,
                 top_k: int | None = None):
        self.api_key = api_key or config.DIFY_DATASET_API_KEY
        self.dataset_id = dataset_id or config.DIFY_DATASET_ID
        self.base_url = (base_url or config.DIFY_BASE_URL).rstrip("/")
        self.threshold = config.SCORE_THRESHOLD if threshold is None else threshold
        self.top_k = top_k or config.TOP_K

    def retrieve(self, query: str) -> RetrievalResult:
        payload = {
            "query": query,
            "retrieval_model": {
                "search_method": config.SEARCH_METHOD,
                "reranking_enable": False,
                "reranking_model": None,
                "weights": None,
                "top_k": self.top_k,
                # Lọc ngưỡng ở phía ta, không nhờ Dify — để log được điểm thật
                "score_threshold_enabled": False,
                "score_threshold": None,
            },
        }
        try:
            resp = requests.post(
                f"{self.base_url}/datasets/{self.dataset_id}/retrieve",
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"},
                json=payload,
                timeout=config.RETRIEVAL_TIMEOUT,
            )
            resp.raise_for_status()
            records = resp.json().get("records", [])
        except (requests.RequestException, ValueError) as exc:
            # Dify chết -> coi như out-of-scope. Thà từ chối còn hơn để LLM bịa.
            print(f"[retriever] lỗi truy vấn: {exc}")
            return RetrievalResult()

        if not records:
            return RetrievalResult()

        scored = []
        for rec in records:
            score = float(rec.get("score") or 0.0)
            content = rec.get("segment", {}).get("content", "")
            chunk_id, body = parse_chunk_id(content)
            scored.append(RetrievedChunk(chunk_id=chunk_id, text=body, score=score))
        scored.sort(key=lambda c: c.score, reverse=True)

        kept = [c for c in scored if c.score >= self.threshold]
        return RetrievalResult(
            chunks=kept,
            in_scope=bool(kept),
            top_score=scored[0].score,
        )
