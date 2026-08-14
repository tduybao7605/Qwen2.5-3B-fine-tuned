"""Cấu hình RAG — nguồn sự thật duy nhất. Không hardcode các giá trị này ở nơi khác."""
import os
from pathlib import Path

# config.py nằm ở src/cadebot/rag/config.py -> parents[3] là gốc repo.
# Cho override được để container (WORKDIR /app) và host cùng hiểu một gốc.
REPO_ROOT = Path(os.getenv("CADEBOT_ROOT", Path(__file__).resolve().parents[3]))

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
# Hiệu chỉnh 2026-07-29 bằng scripts/calibrate_threshold.py trên KB thật
# (bge-m3, 69 segments, semantic_search, top_k=3):
#   in-scope : min=0.526 mean=0.659  (15/15 câu)
#   out-scope: max=0.540 mean=0.442  (15/15 câu)
#   F1=0.968 precision=0.938 recall=1.000
# Hai phân phối chồng nhau 0.014, do DUY NHẤT câu "cho tôi số điện thoại của
# bạn" (0.540) — mà KB có hotline thật nên bot trả lời được, tức là nhãn
# out-of-scope của câu đó đáng ngờ chứ không phải retrieval sai.
# Nâng ngưỡng lên >0.540 sẽ chặn nhầm "có chỗ đậu xe" (0.526) và "quán mở cửa
# mấy giờ" (0.530) — đánh đổi không đáng.
# hybrid_search đã thử: kết quả y hệt. full_text_search: trả 0 (KB high_quality).
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.51"))
TOP_K = int(os.getenv("TOP_K", "3"))
SEARCH_METHOD = "semantic_search"
MAX_CONTEXT_CHARS = 2000

# ── Sinh văn ───────────────────────────────────────────────────────────
# Hạ từ 0.7 xuống để model bớt "sáng tạo" — nó hay tự gắn từ khen ngợi
# ("best seller", "được yêu thích nhất") lấy từ trọng số fine-tune chứ không
# có trong context. Modelfile của bản Ollama vốn đã để 0.1.
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.2"))

# ── Model artifacts ────────────────────────────────────────────────────
# Trọng số KHÔNG bake vào image Docker — mount lúc chạy, xem docker-compose.yml.
# Đường dẫn lấy từ env để container và host cùng trỏ về một chỗ.
MODEL_DIR = Path(os.getenv("CADEBOT_MODEL_DIR", REPO_ROOT / "cadebot-lora"))
BASE_MODEL = os.getenv("CADEBOT_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
STT_MODEL = os.getenv("CADEBOT_STT_MODEL", "vinai/PhoWhisper-large")

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
# Ta đã tự chẻ chunk ở rag/chunker.py, nên đây chỉ là lưới an toàn — phải đặt
# CAO HƠN HẲN chunk lớn nhất (694 ký tự). Để 500 thì Dify cắt đôi các chunk
# markdown tiếng Việt (dấu tokenize rất tốn) và đoạn sau mất dòng [chunk_id]
# -> parse_chunk_id trả 'unknown' -> hỏng sourceIds. Đã đo: 500 làm 34 chunk
# markdown nở thành 46 segment, 13 segment mất id.
CHUNK_MAX_TOKENS = 2000
