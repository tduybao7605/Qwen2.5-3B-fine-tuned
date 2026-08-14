"""Client Dify Dataset API — ghi KB. Dùng Dataset API key (KHÔNG phải App API key)."""
import requests

from cadebot.rag import config


class DifyKnowledgeClient:
    def __init__(self, api_key: str | None = None, dataset_id: str | None = None,
                 base_url: str | None = None):
        self.api_key = api_key or config.DIFY_DATASET_API_KEY
        self.dataset_id = dataset_id or config.DIFY_DATASET_ID
        self.base_url = (base_url or config.DIFY_BASE_URL).rstrip("/")
        if not self.api_key or not self.dataset_id:
            raise RuntimeError(
                "Thiếu DIFY_DATASET_API_KEY / DIFY_DATASET_ID. "
                "Xem docs/rag-setup.md Step 6."
            )
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def find_document_id(self, name: str) -> str | None:
        resp = requests.get(
            f"{self.base_url}/datasets/{self.dataset_id}/documents",
            headers=self._headers,
            params={"limit": 100},
            timeout=config.SYNC_TIMEOUT,
        )
        resp.raise_for_status()
        for doc in resp.json().get("data", []):
            if doc.get("name") == name:
                return doc.get("id")
        return None

    def upsert_document(self, name: str, text: str) -> dict:
        """Có rồi thì update, chưa có thì create — chạy nhiều lần không sinh bản trùng."""
        payload = {
            "name": name,
            "text": text,
            "indexing_technique": "high_quality",
            "process_rule": {
                "mode": "custom",
                "rules": {
                    "pre_processing_rules": [
                        {"id": "remove_extra_spaces", "enabled": True},
                        {"id": "remove_urls_emails", "enabled": False},
                    ],
                    "segmentation": {
                        "separator": config.CHUNK_SEPARATOR,
                        "max_tokens": config.CHUNK_MAX_TOKENS,
                        "chunk_overlap": 0,
                    },
                },
            },
        }
        doc_id = self.find_document_id(name)
        if doc_id:
            url = f"{self.base_url}/datasets/{self.dataset_id}/documents/{doc_id}/update_by_text"
        else:
            url = f"{self.base_url}/datasets/{self.dataset_id}/document/create_by_text"

        resp = requests.post(url, headers=self._headers, json=payload,
                             timeout=config.SYNC_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
