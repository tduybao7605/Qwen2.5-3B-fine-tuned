"""Cấu hình RAG — nguồn sự thật duy nhất. Không hardcode các giá trị này ở nơi khác."""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Embedding (ĐÃ CHỐT — đổi = phải re-embed toàn bộ KB) ────────────────
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_DIM = 1024
EMBEDDING_MAX_TOKENS = 8192

# ── Dify ───────────────────────────────────────────────────────────────
DIFY_BASE_URL = os.getenv("DIFY_BASE_URL", "http://localhost/v1")
# Dataset API key (Knowledge → API Access) — KHÁC với App API key
DIFY_DATASET_API_KEY = os.getenv("DIFY_DATASET_API_KEY", "")
DIFY_DATASET_ID = os.getenv("DIFY_DATASET_ID", "")
RETRIEVAL_TIMEOUT = int(os.getenv("RETRIEVAL_TIMEOUT", "15"))
SYNC_TIMEOUT = int(os.getenv("SYNC_TIMEOUT", "60"))

# ── Retrieval ──────────────────────────────────────────────────────────
# SCORE_THRESHOLD được hiệu chỉnh bằng scripts/calibrate_threshold.py (Task 7).
# 0.55 là giá trị khởi đầu, PHẢI chạy calibrate rồi cập nhật lại.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.55"))
TOP_K = int(os.getenv("TOP_K", "3"))
SEARCH_METHOD = "semantic_search"
MAX_CONTEXT_CHARS = 2000

# ── KB sources ─────────────────────────────────────────────────────────
KB_DIR = REPO_ROOT / "knowledge_Base_cadebot"
DB_FILE = KB_DIR / "demo_cafe.db"
MARKDOWN_FILES = [
    "01_Tong_Quan_Thuong_Hieu.md",
    "02_Menu_Va_Phuong_Phap_Pha_Che.md",
    "03_Khong_Gian_Va_Dich_Vu.md",
    "04_Dia_Chi_Va_Lien_He.md",
    "05_Bo_Cau_Hoi_Thuong_Gap_FAQ.md",
]

# ── Document naming trong Dify (dùng để update idempotent) ──────────────
KB_DOC_NAME_MARKDOWN = "cadebot_kb_markdown.md"
KB_DOC_NAME_DB = "cadebot_kb_database.md"
CHUNK_SEPARATOR = "\n---\n"
CHUNK_MAX_TOKENS = 500
