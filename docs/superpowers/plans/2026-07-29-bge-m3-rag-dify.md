# BGE-M3 Embedding + Dynamic RAG cho Cadebot — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây lớp RAG dùng embedding model **BGE-M3** trên Dify self-hosted + Qdrant, để `serve_model.py` (Qwen2.5-3B + LoRA) trả lời bám sát knowledge base Viva Reserve và **chặn cứng** câu hỏi ngoài phạm vi thay vì bịa.

**Architecture:** Dify self-hosted quản lý knowledge base (chunk + embed + lưu vào Qdrant), BGE-M3 chạy qua Ollama và được đăng ký làm Text Embedding provider trong Dify. Một module `rag/` tự viết lo phần *dynamic*: sinh chunk có ID ổn định từ 5 file markdown + SQLite, đẩy/cập nhật idempotent qua Dify Dataset API, truy vấn qua Dify Retrieval API, và áp ngưỡng điểm để quyết định trả lời hay từ chối. `serve_model.py` chỉ gọi LLM khi retrieval có chunk vượt ngưỡng.

**Tech Stack:** Dify (self-hosted, Docker Compose) · Qdrant (port 6333) · Ollama + `bge-m3` (1024 chiều) · Python 3.11 · FastAPI · transformers + peft (Qwen2.5-3B + LoRA) · SQLite (`demo_cafe.db`) · pytest

---

## Context

`serve_model.py:26-37` đã có `SYSTEM_PROMPT` hứa hẹn *"Chỉ sử dụng Knowledge Hub được cung cấp để trả lời"* và một trường `sourceIds` trong JSON schema đầu ra — **nhưng không hề có Knowledge Hub nào tồn tại**. Toàn bộ kiến thức hiện chỉ nằm trong trọng số LoRA, huấn luyện từ vỏn vẹn 144 mẫu train + 26 val. Hệ quả:

1. **Bịa thông tin**: giá, thành phần, khuyến mãi không có gì ràng buộc model.
2. **Không cập nhật được**: đổi giá 1 món = phải fine-tune lại.
3. **`sourceIds` luôn rỗng**: không truy vết được câu trả lời đến từ đâu.
4. **Chống out-of-scope yếu**: chỉ dựa vào 18 mẫu FALLBACK trong training data.

Kế hoạch này hiện thực hoá Giai đoạn 1-5 của `knowledge_base/cadebot-plan.md`, với hai điều chỉnh so với bản gốc: embedding model chốt là **BGE-M3** (không phải Qwen3-Embedding), và nguồn KB là **5 file markdown 01-05 + sync động từ `demo_cafe.db`** (không dùng file tĩnh `db_exported_kb.md`).

Kết quả mong đợi: hỏi "Trà Đào Cam Sả bao nhiêu tiền?" → retrieval trả chunk `menu:VR_PEACH_TEA` → Qwen trả lời đúng 45.000đ kèm `sourceIds:["menu:VR_PEACH_TEA"]`. Hỏi "Hôm nay trời có mưa không?" → không chunk nào vượt ngưỡng → trả JSON FALLBACK cố định, **không tốn 78 giây gọi LLM**.

---

## Global Constraints

Mọi task đều ngầm chịu các ràng buộc sau. Giá trị chép nguyên văn, không được đổi giữa chừng:

- **Embedding model: `bge-m3`, dimension = 1024, max input 8192 tokens.** Đổi model giữa chừng = phải re-embed toàn bộ KB. Chốt từ đây.
- **Vector store: Qdrant**, `VECTOR_STORE=qdrant`, cổng `6333`.
- **Dify base URL: `http://localhost/v1`** (cổng 80, nginx của Dify Docker Compose) — trùng với `knowledge_base/cadebot_dify_bridge.py` và `demo_db_to_dify.py`.
- **Ollama phải nghe được từ trong container Dify**: dùng `knowledge_base/ollama_docker_bridge.py` (listen `172.17.0.1:11434` → forward `127.0.0.1:11434`). Trong Dify, base URL của Ollama provider là `http://172.17.0.1:11434`.
- **Indexing mode: `high_quality`** (có embedding). Không hạ được về `economical` sau khi đã tạo KB.
- **Chunk ID convention (bắt buộc, đã tồn tại trong `dataset/train.jsonl`)**: `menu:<item_code>`, `faq:<faq_id>`, `promo:<promo_code>`, `doc:<tên_file>#<số_section>`. Ví dụ: `menu:VR_LATTE`, `faq:faq_001`, `promo:VR_COMBO_A`, `doc:01_Tong_Quan#2`.
- **Mọi chunk mở đầu bằng đúng một dòng `[<chunk_id>]`** rồi mới tới nội dung. Đây là cách lấy provenance mà không cần Dify metadata API.
- **Separator giữa các chunk trong document đẩy lên Dify: `\n---\n`**, khai báo trong `process_rule` mode `custom`.
- **Backward compatibility bắt buộc**: Android (`Cadebot_UI/.../CadebotApiService.kt`) POST đúng `{message, history}` tới `/chat`. Mọi field mới trong `ChatRequest` phải **optional có default**.
- **Ngôn ngữ KB và câu hỏi: tiếng Việt.** Không dịch, không normalize bỏ dấu.
- **Không đưa bảng đơn hàng/giao dịch POS vào KB** — chỉ `menu_items`, `promotions`, `faqs`.
- **Máy deploy hiện tại không có GPU** (torch `+cpu`, i5-1235U, 24 GB RAM). BGE-M3 chạy trong Ollama (CPU), không load vào process Python.

---

## File Structure

| File | Trách nhiệm |
|---|---|
| `rag/config.py` | Toàn bộ hằng số cấu hình + đọc env. Một nguồn sự thật duy nhất. |
| `rag/chunker.py` | 5 file markdown 01-05 → `list[Chunk]`. Thuần hàm, không I/O mạng. |
| `rag/db_source.py` | `demo_cafe.db` → `list[Chunk]`. Chỗ duy nhất biết về SQL (Giai đoạn 7 chỉ sửa file này). |
| `rag/kb_builder.py` | Gộp chunk từ 2 nguồn → 1 chuỗi markdown có `---` phân cách. |
| `rag/dify_kb.py` | Client Dify Dataset API: list/create/update document. Idempotent. |
| `rag/retriever.py` | Client Dify Retrieval API + áp ngưỡng + trích `sourceIds`. |
| `rag/prompt.py` | Dựng context block và JSON FALLBACK cố định. |
| `scripts/sync_kb.py` | CLI: build + đẩy KB lên Dify. |
| `scripts/calibrate_threshold.py` | CLI: đo điểm in-scope vs out-of-scope → chọn ngưỡng. |
| `tests/rag/` | pytest cho chunker, db_source, retriever, prompt. |
| `serve_model.py` | *Sửa*: thêm retrieval vào `/chat`, thêm `/health` field, endpoint `/retrieve` để debug. |
| `requirements.txt` | *Tạo mới* — repo hiện chưa có file dependency nào. |
| `docs/RAG_SETUP.md` | Hướng dẫn dựng Dify + Ollama + sync KB. |

**Nguyên tắc phân tách:** `chunker.py` và `db_source.py` là hai *nguồn* độc lập, cùng trả về cùng một dataclass `Chunk`. `dify_kb.py` (ghi) và `retriever.py` (đọc) tách riêng vì dùng **hai API key khác nhau** của Dify và có vòng đời khác nhau (sync chạy theo cron, retrieve chạy mỗi request).

---

## Task 1: Config module + xác minh môi trường máy deploy

**Files:**
- Create: `rag/__init__.py`, `rag/config.py`
- Create: `requirements.txt`
- Test: `tests/rag/__init__.py`, `tests/rag/test_config.py`

**Interfaces:**
- Consumes: (không có)
- Produces: `rag.config` với các hằng `EMBEDDING_MODEL: str`, `EMBEDDING_DIM: int`, `DIFY_BASE_URL: str`, `DIFY_DATASET_API_KEY: str`, `DIFY_DATASET_ID: str`, `KB_DOC_NAME_MARKDOWN: str`, `KB_DOC_NAME_DB: str`, `CHUNK_SEPARATOR: str`, `SCORE_THRESHOLD: float`, `TOP_K: int`, `KB_DIR: Path`, `DB_FILE: Path`, `RETRIEVAL_TIMEOUT: int`

- [ ] **Step 1: Xác minh phiên bản torch/python trên đúng máy deploy**

Bạn đã lưu ý phải check lại cho khớp máy deploy. Chạy trên máy sẽ deploy (không phải máy dev):

```bash
which python3 && python3 --version
python3 -c "import torch, transformers, peft, numpy; \
print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); \
print('transformers', transformers.__version__); \
print('peft', peft.__version__); \
print('numpy', numpy.__version__)"
free -g | head -2
```

Ghi lại output. Trên máy dev hiện tại kết quả là: torch `2.10.0+cpu`, cuda `False`, transformers `5.12.1`, peft `0.19.1`, numpy `1.26.4`, RAM 23 GB.

**Lưu ý có 2 interpreter khác nhau trên máy này** — `python3` mặc định (torch 2.10.0+cpu, numpy 1.26.4) và `/home/ncd/.pyenv/versions/3.11.8/bin/python3` (torch 2.12.0, numpy 2.3.0). Chốt **một** interpreter cho toàn bộ dự án và ghi vào `requirements.txt`. Nếu output khác các số trên, cập nhật pin ở Step 2 cho khớp — **không** ép cài đè phiên bản torch đang chạy được.

- [ ] **Step 2: Tạo `requirements.txt`**

Repo hiện **không có** file dependency nào (không `requirements.txt`, không `pyproject.toml`) — deps chỉ được ghi bằng văn xuôi trong `local-setup.md`. Tạo file, điền phiên bản đúng theo output Step 1:

```
# Cadebot — pin theo máy deploy, xem docs/RAG_SETUP.md
# Core serving (đã cài sẵn, pin lại cho reproducible)
torch==2.10.0
transformers==5.12.1
peft==0.19.1
accelerate==1.14.0
fastapi==0.128.8
uvicorn==0.49.0
numpy==1.26.4
soundfile==0.13.1

# RAG layer (mới)
requests==2.32.3

# Dev
pytest==8.3.4
```

Lưu ý: **không** thêm `sentence-transformers`, `FlagEmbedding`, `faiss`, `qdrant-client`. BGE-M3 chạy trong Ollama và Qdrant do Dify quản lý — phía Python chỉ cần `requests` gọi HTTP.

- [ ] **Step 3: Viết test cho config**

```python
# tests/rag/test_config.py
import pytest
from rag import config


def test_embedding_model_locked():
    assert config.EMBEDDING_MODEL == "bge-m3"
    assert config.EMBEDDING_DIM == 1024


def test_chunk_separator_is_markdown_hr():
    assert config.CHUNK_SEPARATOR == "\n---\n"


def test_kb_paths_exist():
    assert config.KB_DIR.is_dir()
    assert config.DB_FILE.is_file()


def test_threshold_in_valid_range():
    assert 0.0 < config.SCORE_THRESHOLD < 1.0


def test_dify_base_url_has_v1_suffix():
    assert config.DIFY_BASE_URL.endswith("/v1")
```

- [ ] **Step 4: Chạy test, xác nhận FAIL**

Run: `python3 -m pytest tests/rag/test_config.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'rag'`

- [ ] **Step 5: Viết `rag/config.py`**

```python
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
# SCORE_THRESHOLD được hiệu chỉnh bằng scripts/calibrate_threshold.py (Task 8).
# 0.55 là giá trị khởi đầu, PHẢI chạy calibrate rồi cập nhật lại.
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.55"))
TOP_K = int(os.getenv("TOP_K", "3"))
SEARCH_METHOD = "semantic_search"
MAX_CONTEXT_CHARS = 2000

# ── KB sources ─────────────────────────────────────────────────────────
KB_DIR = REPO_ROOT / "knowledge_base"
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
```

- [ ] **Step 6: Chạy test, xác nhận PASS**

Run: `python3 -m pytest tests/rag/test_config.py -v`
Expected: 5 passed

- [ ] **Step 7: Commit**

```bash
git add rag/ tests/ requirements.txt
git commit -m "feat(rag): add config module and pin dependencies"
```

---

## Task 2: Chunker cho 5 file markdown

**Files:**
- Create: `rag/chunker.py`
- Test: `tests/rag/test_chunker.py`

**Interfaces:**
- Consumes: `rag.config.KB_DIR`, `rag.config.MARKDOWN_FILES`
- Produces:
  - `@dataclass(frozen=True) class Chunk: id: str; text: str; source: str`
  - `chunk_markdown_file(path: Path) -> list[Chunk]`
  - `chunk_faq_file(path: Path) -> list[Chunk]`
  - `chunk_all_markdown() -> list[Chunk]`
  - `Chunk.render() -> str` trả `"[<id>]\n<text>"`

**Bối cảnh cần biết:** File 05 (FAQ) có cấu trúc **khác hẳn** 4 file còn lại — nó là 21 cặp `Q:` / `A:` phẳng, không có heading `##` nào. Bốn file kia dùng `#` (title) → `##` (section đánh số) → đôi khi `###` (nhóm con A-E). **Không file nào có bảng markdown.** Vì vậy cần hai chiến lược chunk riêng.

- [ ] **Step 1: Viết test thất bại**

```python
# tests/rag/test_chunker.py
from rag import config
from rag.chunker import Chunk, chunk_all_markdown, chunk_faq_file, chunk_markdown_file


def test_chunk_render_prefixes_id():
    c = Chunk(id="menu:VR_LATTE", text="Giá 55.000đ", source="test")
    assert c.render() == "[menu:VR_LATTE]\nGiá 55.000đ"


def test_faq_file_yields_one_chunk_per_qa_pair():
    chunks = chunk_faq_file(config.KB_DIR / "05_Bo_Cau_Hoi_Thuong_Gap_FAQ.md")
    assert len(chunks) == 21
    assert all(c.id.startswith("faq:md_") for c in chunks)
    # mỗi chunk phải chứa CẢ câu hỏi lẫn câu trả lời
    assert all("Q:" in c.text and "A:" in c.text for c in chunks)


def test_faq_chunk_keeps_question_and_answer_together():
    chunks = chunk_faq_file(config.KB_DIR / "05_Bo_Cau_Hoi_Thuong_Gap_FAQ.md")
    latte = next(c for c in chunks if "Viva Latte có vị như thế nào" in c.text)
    assert "sữa béo nhẹ" in latte.text


def test_section_file_splits_on_h2():
    chunks = chunk_markdown_file(config.KB_DIR / "01_Tong_Quan_Thuong_Hieu.md")
    # file 01 có 3 section `##`
    assert len(chunks) == 3
    assert chunks[0].id == "doc:01_Tong_Quan_Thuong_Hieu#1"


def test_every_chunk_carries_document_title_for_context():
    chunks = chunk_markdown_file(config.KB_DIR / "01_Tong_Quan_Thuong_Hieu.md")
    # chunk lẻ mất ngữ cảnh nếu không mang tiêu đề file
    assert all("VIVA RESERVE" in c.text.upper() for c in chunks)


def test_menu_file_splits_on_h3_subgroups():
    chunks = chunk_markdown_file(config.KB_DIR / "02_Menu_Va_Phuong_Phap_Pha_Che.md")
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk id phải duy nhất"
    joined = " ".join(c.text for c in chunks)
    assert "Trà Đào Cam Sả" in joined
    assert "Extra Shot" in joined  # section toppings không được rơi mất


def test_chunk_all_markdown_ids_are_globally_unique():
    chunks = chunk_all_markdown()
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert len(chunks) > 25


def test_no_chunk_is_empty_or_whitespace_only():
    for c in chunk_all_markdown():
        assert c.text.strip(), f"chunk rỗng: {c.id}"
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `python3 -m pytest tests/rag/test_chunker.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'rag.chunker'`

- [ ] **Step 3: Viết `rag/chunker.py`**

```python
"""Chuyển 5 file markdown KB thành các chunk có ID ổn định.

Hai chiến lược:
  - File 05 (FAQ): 1 chunk / 1 cặp Q&A — giữ câu hỏi và câu trả lời cùng nhau.
  - File 01-04: 1 chunk / 1 section `##`, có kèm tiêu đề file để chunk không mất ngữ cảnh.
"""
import re
from dataclasses import dataclass
from pathlib import Path

from rag import config


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str

    def render(self) -> str:
        """Dòng đầu là [id] để retriever trích lại được sourceIds."""
        return f"[{self.id}]\n{self.text}"


def _doc_slug(path: Path) -> str:
    """01_Tong_Quan_Thuong_Hieu.md -> 01_Tong_Quan_Thuong_Hieu"""
    return path.stem


def chunk_faq_file(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    slug = _doc_slug(path)
    # Bắt từng cặp Q:/A: — A: chạy tới khi gặp Q: kế tiếp hoặc hết file.
    pattern = re.compile(
        r"^Q:\s*(?P<q>.+?)\s*\nA:\s*(?P<a>.+?)(?=\n\s*\nQ:|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    chunks: list[Chunk] = []
    for i, m in enumerate(pattern.finditer(raw), start=1):
        q = m.group("q").strip()
        a = m.group("a").strip()
        chunks.append(
            Chunk(
                id=f"faq:md_{i:03d}",
                text=f"Q: {q}\nA: {a}",
                source=path.name,
            )
        )
    return chunks


def chunk_markdown_file(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    slug = _doc_slug(path)

    title_match = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else slug

    # Cắt theo heading `##`. Bỏ phần trước `##` đầu tiên (chỉ là tiêu đề file).
    parts = re.split(r"^##\s+", raw, flags=re.MULTILINE)[1:]

    chunks: list[Chunk] = []
    for i, part in enumerate(parts, start=1):
        body = part.strip()
        if not body:
            continue
        # Gắn tiêu đề file vào đầu chunk — nếu không, chunk "## 2. Tiện Ích"
        # bị lấy ra một mình sẽ không biết là tiện ích của quán nào.
        text = f"{title}\n\n## {body}"
        chunks.append(Chunk(id=f"doc:{slug}#{i}", text=text, source=path.name))
    return chunks


def chunk_all_markdown() -> list[Chunk]:
    chunks: list[Chunk] = []
    for name in config.MARKDOWN_FILES:
        path = config.KB_DIR / name
        if name.startswith("05_"):
            chunks.extend(chunk_faq_file(path))
        else:
            chunks.extend(chunk_markdown_file(path))
    return chunks
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `python3 -m pytest tests/rag/test_chunker.py -v`
Expected: 8 passed

Nếu `test_faq_file_yields_one_chunk_per_qa_pair` báo 20 thay vì 21 — đếm lại bằng `grep -c '^Q:' knowledge_base/05_Bo_Cau_Hoi_Thuong_Gap_FAQ.md` và sửa con số kỳ vọng trong test cho khớp thực tế, đừng sửa regex cho vừa số.

- [ ] **Step 5: Kiểm tra mắt thường output chunk**

```bash
python3 -c "
from rag.chunker import chunk_all_markdown
cs = chunk_all_markdown()
print(f'Tổng: {len(cs)} chunks')
for c in cs[:3] + cs[-2:]:
    print('=' * 60); print(c.render()[:300])
"
```

Đọc kỹ: mỗi chunk phải **tự đứng vững** — đọc riêng nó vẫn hiểu đang nói về Viva Reserve.

- [ ] **Step 6: Commit**

```bash
git add rag/chunker.py tests/rag/test_chunker.py
git commit -m "feat(rag): chunk 5 markdown KB files with stable chunk IDs"
```

---

## Task 3: Chunker cho SQLite database

**Files:**
- Create: `rag/db_source.py`
- Test: `tests/rag/test_db_source.py`

**Interfaces:**
- Consumes: `rag.chunker.Chunk`, `rag.config.DB_FILE`
- Produces:
  - `get_menu_data() -> dict` với keys `"menu"`, `"promotions"`, `"faqs"` (mỗi value là `list[sqlite3.Row]`)
  - `chunk_database() -> list[Chunk]`

**Bối cảnh cần biết (quan trọng):**

- `knowledge_base/demo_db_to_dify.py` đã có `get_menu_data_from_db()` nhưng nó **string-build ra một khối markdown lớn** — không dùng lại trực tiếp được vì ta cần chunk rời có ID. Ta viết lại, giữ nguyên **tên hàm `get_menu_data()`** theo đúng Giai đoạn 4/7 của `cadebot-plan.md`: sau này đổi sang DB thật của quán thì **chỉ sửa bên trong hàm này**.
- **Có drift tên cột giữa hai schema**: `knowledge_base/schema.sql` (PostgreSQL) dùng `is_available`, còn `demo_cafe.db` (SQLite) dùng `available`. Code phải dò tên cột thay vì hardcode, nếu không sẽ vỡ khi chuyển sang Postgres.
- Số dòng thực tế trong `demo_cafe.db`: `menu_items` = 12, `promotions` = 3, `faqs` = 20.
- Cột `attributes` là JSON lưu dạng TEXT, ví dụ `{"caffeine": true, "sizeOptions": ["S","M","L"], "toppings": [...]}`. Cần trải phẳng thành tiếng Việt đọc được thì embedding mới bắt được câu hỏi kiểu "Latte có size L không?".

- [ ] **Step 1: Viết test thất bại**

```python
# tests/rag/test_db_source.py
import json

from rag.db_source import chunk_database, get_menu_data


def test_get_menu_data_returns_three_sections():
    data = get_menu_data()
    assert set(data.keys()) == {"menu", "promotions", "faqs"}
    assert len(data["menu"]) == 12
    assert len(data["promotions"]) == 3
    assert len(data["faqs"]) == 20


def test_menu_chunk_ids_use_item_code():
    chunks = chunk_database()
    latte = next(c for c in chunks if c.id == "menu:VR_LATTE_M")
    assert "55,000" in latte.text or "55.000" in latte.text
    assert "Viva Latte" in latte.text


def test_promo_and_faq_chunks_have_correct_id_prefix():
    chunks = chunk_database()
    assert any(c.id.startswith("promo:") for c in chunks)
    assert any(c.id.startswith("faq:db_") for c in chunks)


def test_attributes_json_is_flattened_into_readable_text():
    chunks = chunk_database()
    latte = next(c for c in chunks if c.id == "menu:VR_LATTE_M")
    # câu hỏi "có size L không" chỉ match được nếu size đã được trải phẳng
    assert "Size" in latte.text
    assert "L" in latte.text


def test_db_chunk_ids_do_not_collide_with_markdown_chunk_ids():
    from rag.chunker import chunk_all_markdown

    all_ids = [c.id for c in chunk_database()] + [c.id for c in chunk_all_markdown()]
    assert len(all_ids) == len(set(all_ids))


def test_unavailable_items_are_excluded():
    # KB không được quảng cáo món đã ngừng bán
    chunks = chunk_database()
    data = get_menu_data()
    assert len([c for c in chunks if c.id.startswith("menu:")]) == len(data["menu"])
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `python3 -m pytest tests/rag/test_db_source.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'rag.db_source'`

- [ ] **Step 3: Viết `rag/db_source.py`**

```python
"""Đọc menu/khuyến mãi/FAQ từ database và sinh chunk.

GIAI ĐOẠN 7 (DB thật của quán): CHỈ sửa bên trong get_menu_data().
Toàn bộ logic chunk + sync phía dưới giữ nguyên.
"""
import json
import sqlite3

from rag import config
from rag.chunker import Chunk


def _available_column(conn: sqlite3.Connection) -> str:
    """schema.sql (Postgres) dùng `is_available`, demo_cafe.db (SQLite) dùng `available`."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(menu_items)")}
    for candidate in ("available", "is_available"):
        if candidate in cols:
            return candidate
    raise RuntimeError(f"menu_items không có cột available/is_available. Có: {sorted(cols)}")


def get_menu_data() -> dict:
    conn = sqlite3.connect(config.DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        avail = _available_column(conn)
        return {
            "menu": conn.execute(
                f"SELECT * FROM menu_items WHERE {avail} = 1 ORDER BY id"
            ).fetchall(),
            "promotions": conn.execute("SELECT * FROM promotions ORDER BY id").fetchall(),
            "faqs": conn.execute("SELECT * FROM faqs ORDER BY id").fetchall(),
        }
    finally:
        conn.close()


_ATTR_LABELS = {
    "sizeOptions": "Size",
    "sweetnessOptions": "Độ ngọt",
    "iceOptions": "Tùy chọn đá",
    "temperatureOptions": "Nóng/Lạnh",
    "toppings": "Topping",
    "toppingOptions": "Topping",
}


def _flatten_attributes(raw: str | None) -> str:
    """JSON attributes -> dòng tiếng Việt để embedding bắt được."""
    if not raw:
        return ""
    try:
        attrs = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return ""

    lines = []
    if attrs.get("caffeine") is True:
        lines.append("  + Có caffeine")
    elif attrs.get("caffeine") is False:
        lines.append("  + Không có caffeine")

    for key, label in _ATTR_LABELS.items():
        values = attrs.get(key)
        if isinstance(values, list) and values:
            lines.append(f"  + {label}: {', '.join(str(v) for v in values)}")
    return "\n".join(lines)


def chunk_database() -> list[Chunk]:
    data = get_menu_data()
    chunks: list[Chunk] = []

    for row in data["menu"]:
        body = (
            f"Món: {row['name']} (Mã: {row['item_code']} | Nhóm: {row['category']})\n"
            f"  + Giá: {row['price']:,} VNĐ\n"
            f"  + Mô tả: {row['description']}\n"
            f"  + Phương pháp: {row['brewing_method']}"
        )
        attrs = _flatten_attributes(row["attributes"] if "attributes" in row.keys() else None)
        if attrs:
            body += "\n" + attrs
        chunks.append(Chunk(id=f"menu:{row['item_code']}", text=body, source="demo_cafe.db"))

    for row in data["promotions"]:
        body = (
            f"Khuyến mãi: {row['title']} (Mã: {row['promo_code']})\n"
            f"  + Chi tiết: {row['discount_detail']}\n"
            f"  + Thời gian: {row['start_date']} đến {row['end_date']}\n"
            f"  + Điều kiện: {row['conditions']}"
        )
        chunks.append(Chunk(id=f"promo:{row['promo_code']}", text=body, source="demo_cafe.db"))

    for row in data["faqs"]:
        body = f"Q: {row['question']}\nA: {row['answer']}"
        # tiền tố db_ để không đụng id faq:md_* của chunker markdown
        chunks.append(Chunk(id=f"faq:db_{row['faq_id']}", text=body, source="demo_cafe.db"))

    return chunks
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `python3 -m pytest tests/rag/test_db_source.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add rag/db_source.py tests/rag/test_db_source.py
git commit -m "feat(rag): chunk menu/promo/faq from SQLite with schema-drift guard"
```

---

## Task 4: Dựng hạ tầng Dify + Qdrant + BGE-M3 qua Ollama

Đây là task **vận hành**, không có unit test — nghiệm thu bằng lệnh kiểm tra thực tế. Làm đúng thứ tự, mỗi bước phải xanh mới sang bước sau.

**Files:**
- Create: `docs/RAG_SETUP.md` (ghi lại toàn bộ những gì làm ở task này, kèm dataset ID/API key thu được)

- [ ] **Step 1: Cài Ollama và pull BGE-M3**

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull bge-m3
ollama list | grep bge-m3
```

Expected: dòng `bge-m3:latest` (~1.2 GB). Kiểm tra số chiều đúng 1024:

```bash
curl -s http://127.0.0.1:11434/api/embed \
  -d '{"model":"bge-m3","input":"Viva Latte giá bao nhiêu"}' \
  | python3 -c "import json,sys; print('dim =', len(json.load(sys.stdin)['embeddings'][0]))"
```

Expected: `dim = 1024`. **Nếu khác 1024, dừng lại** — cập nhật `EMBEDDING_DIM` trong `rag/config.py` trước khi tạo KB, vì sau khi tạo KB không đổi được.

- [ ] **Step 2: Cài Docker và dựng Dify với Qdrant**

```bash
docker --version || (curl -fsSL https://get.docker.com | sh)
git clone https://github.com/langgenius/dify.git ~/dify
cd ~/dify/docker && cp .env.example .env
```

Sửa `~/dify/docker/.env`:

```
VECTOR_STORE=qdrant
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=difyai123456
QDRANT_CLIENT_TIMEOUT=20
```

```bash
cd ~/dify/docker && docker compose --profile qdrant up -d
docker compose ps
```

Expected: các container `api`, `worker`, `web`, `db`, `redis`, `qdrant`, `nginx` đều `running`.

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost/
```

Expected: `200` hoặc `307`. Mở `http://localhost` trên trình duyệt, tạo tài khoản admin.

- [ ] **Step 3: Mở đường cho Dify container gọi được Ollama**

Ollama chỉ nghe `127.0.0.1:11434`, container Dify không với tới được. Repo **đã có sẵn** cầu nối — dùng lại, đừng viết mới:

```bash
python3 knowledge_base/ollama_docker_bridge.py &
```

Xác minh từ **bên trong** container Dify:

```bash
docker compose -f ~/dify/docker/docker-compose.yaml exec api \
  curl -s http://172.17.0.1:11434/api/tags | head -c 200
```

Expected: JSON có `bge-m3`. Nếu timeout, kiểm tra firewall trên interface `docker0`.

Để bridge sống sau reboot, tạo systemd unit (ghi lại vào `docs/RAG_SETUP.md`):

```bash
sudo tee /etc/systemd/system/ollama-docker-bridge.service >/dev/null <<'EOF'
[Unit]
Description=Ollama docker0 bridge for Dify
After=network.target ollama.service

[Service]
ExecStart=/usr/bin/python3 /home/ncd/learnspaces/Qwen2.5-3B-fine-tuned/knowledge_base/ollama_docker_bridge.py
Restart=always
User=ncd

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now ollama-docker-bridge
```

- [ ] **Step 4: Đăng ký BGE-M3 làm Text Embedding provider trong Dify**

Trên UI Dify: **Settings → Model Providers → Ollama → Add Model**

| Trường | Giá trị |
|---|---|
| Model Type | `Text Embedding` |
| Model Name | `bge-m3` |
| Base URL | `http://172.17.0.1:11434` |
| Model context size | `8192` |
| Max token limit | `8192` |

Bấm Save. Nếu báo "Connection refused" → quay lại Step 3.

- [ ] **Step 5: Tạo Knowledge base rỗng**

**Knowledge → Create Knowledge → Import from text → tạo document tạm bất kỳ**, chọn:
- Chunk setting: **Custom**, Delimiter `---`, Max chunk length `500`, Overlap `0`
- Index Method: **High Quality**
- Embedding Model: **bge-m3**
- Retrieval Setting: **Vector Search**

Sau khi tạo, **xoá document tạm** (script ở Task 5 sẽ tự tạo document thật).

- [ ] **Step 6: Lấy Dataset ID và Dataset API key**

- **Dataset ID**: nằm trong URL khi mở knowledge base — `http://localhost/datasets/<DATASET_ID>/documents`
- **Dataset API key**: **Knowledge → API Access** (icon góc trên) → **API Key → Create**. ⚠️ Đây là key **khác** với App API key mà `cadebot_dify_bridge.py` dùng.

Lưu vào `.env` ở gốc repo (nhớ thêm `.env` vào `.gitignore` — **không commit key**):

```bash
cat >> .env <<'EOF'
DIFY_BASE_URL=http://localhost/v1
DIFY_DATASET_API_KEY=dataset-xxxxxxxxxxxxxxxx
DIFY_DATASET_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
EOF
grep -qxF '.env' .gitignore || echo '.env' >> .gitignore
```

Xác minh key hoạt động:

```bash
set -a && source .env && set +a
curl -s "$DIFY_BASE_URL/datasets/$DIFY_DATASET_ID/documents" \
  -H "Authorization: Bearer $DIFY_DATASET_API_KEY" | head -c 300
```

Expected: JSON có `"data": []` (hoặc danh sách document). Nếu `401` → sai loại key.

- [ ] **Step 7: Viết `docs/RAG_SETUP.md`**

Chép lại toàn bộ Step 1-6 thành hướng dẫn tái lập được, kèm bảng cấu hình đã chốt (model `bge-m3`, dim 1024, Qdrant 6333, delimiter `---`). **Không ghi API key thật vào file này** — chỉ ghi tên biến env.

- [ ] **Step 8: Commit**

```bash
git add docs/RAG_SETUP.md .gitignore
git commit -m "docs: Dify + Qdrant + BGE-M3 (Ollama) setup guide"
```

---

## Task 5: Đẩy KB lên Dify (idempotent)

**Files:**
- Create: `rag/kb_builder.py`, `rag/dify_kb.py`, `scripts/sync_kb.py`
- Test: `tests/rag/test_kb_builder.py`

**Interfaces:**
- Consumes: `rag.chunker.chunk_all_markdown`, `rag.db_source.chunk_database`, `rag.config.*`
- Produces:
  - `rag.kb_builder.build_document(chunks: list[Chunk]) -> str`
  - `rag.dify_kb.DifyKnowledgeClient` với `.find_document_id(name) -> str | None`, `.upsert_document(name, text) -> dict`
  - `scripts/sync_kb.py` CLI

**Bối cảnh:** `knowledge_base/demo_db_to_dify.py` đã có `sync_to_dify_dataset()` gọi `create_by_text`, nhưng nó **luôn tạo document mới** → chạy 2 lần là KB có 2 bản trùng, retrieval trả kết quả nhân đôi. Ta phải list trước, có rồi thì `update_by_text`.

- [ ] **Step 1: Viết test cho kb_builder**

```python
# tests/rag/test_kb_builder.py
from rag import config
from rag.chunker import Chunk
from rag.kb_builder import build_document


def test_chunks_joined_by_separator():
    doc = build_document([
        Chunk(id="a:1", text="alpha", source="t"),
        Chunk(id="b:2", text="beta", source="t"),
    ])
    assert doc == "[a:1]\nalpha\n---\n[b:2]\nbeta"


def test_splitting_document_recovers_original_chunk_count():
    chunks = [Chunk(id=f"x:{i}", text=f"text {i}", source="t") for i in range(10)]
    doc = build_document(chunks)
    assert len(doc.split(config.CHUNK_SEPARATOR)) == 10


def test_build_document_rejects_text_containing_separator():
    # nội dung chứa `\n---\n` sẽ làm Dify cắt nhầm chỗ
    bad = Chunk(id="a:1", text="trước\n---\nsau", source="t")
    try:
        build_document([bad])
        assert False, "phải raise ValueError"
    except ValueError as e:
        assert "a:1" in str(e)


def test_real_kb_builds_without_separator_collision():
    from rag.chunker import chunk_all_markdown
    from rag.db_source import chunk_database

    doc = build_document(chunk_all_markdown() + chunk_database())
    assert len(doc) > 5000
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `python3 -m pytest tests/rag/test_kb_builder.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'rag.kb_builder'`

- [ ] **Step 3: Viết `rag/kb_builder.py`**

```python
"""Gộp chunk thành một document markdown để Dify cắt lại theo delimiter `---`."""
from rag import config
from rag.chunker import Chunk


def build_document(chunks: list[Chunk]) -> str:
    for c in chunks:
        if config.CHUNK_SEPARATOR in c.render():
            raise ValueError(
                f"Chunk {c.id} chứa separator {config.CHUNK_SEPARATOR!r} — "
                "Dify sẽ cắt nhầm. Sửa nội dung nguồn hoặc đổi CHUNK_SEPARATOR."
            )
    return config.CHUNK_SEPARATOR.join(c.render() for c in chunks)
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `python3 -m pytest tests/rag/test_kb_builder.py -v`
Expected: 4 passed

- [ ] **Step 5: Viết `rag/dify_kb.py`**

```python
"""Client Dify Dataset API — ghi KB. Dùng Dataset API key (KHÔNG phải App API key)."""
import requests

from rag import config


class DifyKnowledgeClient:
    def __init__(self, api_key: str | None = None, dataset_id: str | None = None,
                 base_url: str | None = None):
        self.api_key = api_key or config.DIFY_DATASET_API_KEY
        self.dataset_id = dataset_id or config.DIFY_DATASET_ID
        self.base_url = (base_url or config.DIFY_BASE_URL).rstrip("/")
        if not self.api_key or not self.dataset_id:
            raise RuntimeError(
                "Thiếu DIFY_DATASET_API_KEY / DIFY_DATASET_ID. "
                "Xem docs/RAG_SETUP.md Step 6."
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
```

- [ ] **Step 6: Viết `scripts/sync_kb.py`**

```python
"""Build KB từ markdown + SQLite rồi đẩy lên Dify.

    python3 scripts/sync_kb.py            # đẩy lên Dify
    python3 scripts/sync_kb.py --dry-run  # chỉ in ra, không gọi mạng
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import config
from rag.chunker import chunk_all_markdown
from rag.db_source import chunk_database
from rag.dify_kb import DifyKnowledgeClient
from rag.kb_builder import build_document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    md_chunks = chunk_all_markdown()
    db_chunks = chunk_database()
    print(f"Markdown: {len(md_chunks)} chunks | Database: {len(db_chunks)} chunks")

    all_ids = [c.id for c in md_chunks + db_chunks]
    duplicates = {i for i in all_ids if all_ids.count(i) > 1}
    if duplicates:
        print(f"❌ Chunk ID trùng: {duplicates}", file=sys.stderr)
        return 1

    md_doc = build_document(md_chunks)
    db_doc = build_document(db_chunks)

    if args.dry_run:
        print("=" * 70)
        print(md_doc[:1500])
        print("=" * 70)
        print(db_doc[:1500])
        return 0

    client = DifyKnowledgeClient()
    for name, doc in ((config.KB_DOC_NAME_MARKDOWN, md_doc),
                      (config.KB_DOC_NAME_DB, db_doc)):
        result = client.upsert_document(name, doc)
        doc_id = result.get("document", {}).get("id", "?")
        print(f"✅ {name} → document id {doc_id}")

    print("\n⚠️  Đợi Dify index xong (Knowledge → Documents, trạng thái 'Available') "
          "rồi mới chạy retrieval.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 7: Chạy dry-run rồi sync thật**

```bash
python3 scripts/sync_kb.py --dry-run
set -a && source .env && set +a
python3 scripts/sync_kb.py
```

Expected: hai dòng `✅`. Vào Dify UI → Knowledge → Documents, chờ trạng thái chuyển **Available**, rồi mở document xem số segment có khớp số chunk không.

- [ ] **Step 8: Xác nhận tính idempotent — chạy lại lần hai**

```bash
python3 scripts/sync_kb.py
```

Vào UI kiểm tra: vẫn đúng **2 document**, không phải 4. Đây chính là bug của `demo_db_to_dify.py` mà ta đang tránh.

- [ ] **Step 9: Test end-to-end sửa DB → KB đổi theo (Giai đoạn 4 của cadebot-plan.md)**

```bash
sqlite3 knowledge_base/demo_cafe.db \
  "UPDATE menu_items SET price = 59000 WHERE item_code = 'VR_LATTE_M';"
python3 scripts/sync_kb.py
# Dify UI → Retrieval Testing → "Viva Latte giá bao nhiêu" → phải thấy 59,000
sqlite3 knowledge_base/demo_cafe.db \
  "UPDATE menu_items SET price = 55000 WHERE item_code = 'VR_LATTE_M';"
python3 scripts/sync_kb.py
```

- [ ] **Step 10: Commit**

```bash
git add rag/kb_builder.py rag/dify_kb.py scripts/sync_kb.py tests/rag/test_kb_builder.py
git commit -m "feat(rag): idempotent KB sync to Dify dataset"
```

---

## Task 6: Retrieval client + logic out-of-scope

**Files:**
- Create: `rag/retriever.py`
- Test: `tests/rag/test_retriever.py`

**Interfaces:**
- Consumes: `rag.config.*`
- Produces:
  - `@dataclass class RetrievedChunk: chunk_id: str; text: str; score: float`
  - `@dataclass class RetrievalResult: chunks: list[RetrievedChunk]; in_scope: bool; top_score: float`
  - `RetrievalResult.source_ids -> list[str]`
  - `RetrievalResult.context_text -> str`
  - `parse_chunk_id(content: str) -> tuple[str, str]` → `(chunk_id, body)`
  - `Retriever.retrieve(query: str) -> RetrievalResult`

**Bối cảnh:** Dify `POST /v1/datasets/{id}/retrieve` trả `{"query": {...}, "records": [{"segment": {"id":..., "content":..., "document": {...}}, "score": 0.87}]}`. Nội dung `segment.content` chính là chunk ta đẩy lên, **dòng đầu là `[chunk_id]`** — đó là chỗ lấy `sourceIds`.

- [ ] **Step 1: Viết test thất bại (mock HTTP, không cần Dify chạy)**

```python
# tests/rag/test_retriever.py
from unittest.mock import patch

from rag.retriever import RetrievalResult, RetrievedChunk, Retriever, parse_chunk_id


def _fake_response(records):
    class R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return {"query": {"content": "q"}, "records": records}
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
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `python3 -m pytest tests/rag/test_retriever.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'rag.retriever'`

- [ ] **Step 3: Viết `rag/retriever.py`**

```python
"""Truy vấn Dify Retrieval API và quyết định in-scope / out-of-scope."""
from dataclasses import dataclass, field

import requests

from rag import config


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
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `python3 -m pytest tests/rag/test_retriever.py -v`
Expected: 7 passed

- [ ] **Step 5: Thử với Dify thật**

```bash
set -a && source .env && set +a
python3 -c "
from rag.retriever import Retriever
r = Retriever()
for q in ['Viva Latte giá bao nhiêu', 'quán mở cửa mấy giờ', 'hôm nay trời có mưa không']:
    res = r.retrieve(q)
    print(f'{q!r:45} in_scope={res.in_scope} top={res.top_score:.3f} ids={res.source_ids}')
"
```

Expected: hai câu đầu `in_scope=True` với ID hợp lý, câu cuối điểm thấp. Nếu chưa tách bạch, đó là việc của Task 7.

- [ ] **Step 6: Commit**

```bash
git add rag/retriever.py tests/rag/test_retriever.py
git commit -m "feat(rag): Dify retrieval client with score threshold and sourceIds"
```

---

## Task 7: Hiệu chỉnh score threshold bằng dữ liệu thật

**Files:**
- Create: `scripts/calibrate_threshold.py`, `eval/rag_queries.json`
- Modify: `rag/config.py` (cập nhật `SCORE_THRESHOLD` bằng con số đo được)

**Bối cảnh:** Đây là bước quyết định chất lượng chống-bịa. `PIPELINE_ANALYSIS.md` §5 đã liệt kê sẵn 144 câu training theo intent, trong đó **18 câu FALLBACK** — dùng luôn làm tập out-of-scope. Không đoán ngưỡng, phải đo.

- [ ] **Step 1: Tạo tập câu hỏi đánh giá**

```json
{
  "in_scope": [
    {"q": "Viva Latte giá bao nhiêu", "expect": "menu:VR_LATTE_M"},
    {"q": "Trà Đào Cam Sả có vị chua không", "expect": null},
    {"q": "món nào không có cà phê", "expect": null},
    {"q": "quán mở cửa mấy giờ", "expect": null},
    {"q": "địa chỉ chi nhánh Bình Thới ở đâu", "expect": null},
    {"q": "có combo nào tiết kiệm không", "expect": null},
    {"q": "Matcha Latte có sữa không", "expect": null},
    {"q": "thanh toán bằng MoMo được không", "expect": null},
    {"q": "có wifi không", "expect": null},
    {"q": "Latte size L có không", "expect": "menu:VR_LATTE_M"},
    {"q": "Tiramisu bao nhiêu tiền", "expect": null},
    {"q": "robot có giao món đến bàn không", "expect": null},
    {"q": "có chỗ đậu xe ô tô không", "expect": null},
    {"q": "topping oat milk có không", "expect": null},
    {"q": "hotline đặt bàn số mấy", "expect": null}
  ],
  "out_of_scope": [
    {"q": "hôm nay trời có mưa không"},
    {"q": "giá vàng hôm nay bao nhiêu"},
    {"q": "ai là tổng thống Mỹ"},
    {"q": "kể tôi nghe một câu chuyện cười"},
    {"q": "đội tuyển Việt Nam đá mấy giờ"},
    {"q": "cách nấu phở bò"},
    {"q": "dịch câu này sang tiếng Anh"},
    {"q": "2 cộng 2 bằng mấy"},
    {"q": "bitcoin đang bao nhiêu"},
    {"q": "gợi ý cho tôi một bộ phim hay"},
    {"q": "làm sao để giảm cân"},
    {"q": "cho tôi số điện thoại của bạn"},
    {"q": "quán có bán xe máy không"},
    {"q": "tôi muốn đặt vé máy bay"},
    {"q": "Starbucks có chi nhánh nào gần đây"}
  ]
}
```

Lưu ý câu cuối tập out-of-scope (`Starbucks`) là **ca khó có chủ đích**: nó nói về cà phê nên điểm sẽ cao, đúng loại nhiễu ta cần ngưỡng phân biệt được.

- [ ] **Step 2: Viết `scripts/calibrate_threshold.py`**

```python
"""Đo điểm retrieval trên tập in-scope vs out-of-scope, đề xuất ngưỡng tối ưu.

    python3 scripts/calibrate_threshold.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag import config
from rag.retriever import Retriever

EVAL_FILE = Path(__file__).resolve().parent.parent / "eval" / "rag_queries.json"


def main() -> int:
    data = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    # threshold=0 để lấy điểm thô, tự lọc sau
    retriever = Retriever(threshold=0.0)

    in_scores, out_scores, recall_hits, recall_total = [], [], 0, 0

    print("── IN-SCOPE ──")
    for item in data["in_scope"]:
        res = retriever.retrieve(item["q"])
        in_scores.append(res.top_score)
        ids = res.source_ids[:3]
        print(f"  {res.top_score:.3f}  {item['q'][:42]:44} {ids}")
        if item.get("expect"):
            recall_total += 1
            recall_hits += int(item["expect"] in res.source_ids)

    print("\n── OUT-OF-SCOPE ──")
    for item in data["out_of_scope"]:
        res = retriever.retrieve(item["q"])
        out_scores.append(res.top_score)
        print(f"  {res.top_score:.3f}  {item['q'][:42]}")

    print(f"\nin-scope : min={min(in_scores):.3f} mean={sum(in_scores)/len(in_scores):.3f}")
    print(f"out-scope: max={max(out_scores):.3f} mean={sum(out_scores)/len(out_scores):.3f}")
    if recall_total:
        print(f"recall@{config.TOP_K} (câu có expect): {recall_hits}/{recall_total}")

    # Quét ngưỡng, tối ưu F1 với in-scope là lớp positive
    best = None
    for i in range(20, 90):
        t = i / 100
        tp = sum(s >= t for s in in_scores)
        fn = len(in_scores) - tp
        fp = sum(s >= t for s in out_scores)
        if tp == 0:
            continue
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
        if best is None or f1 > best[1]:
            best = (t, f1, prec, rec)

    if best:
        t, f1, prec, rec = best
        print(f"\n✅ Ngưỡng đề xuất: {t:.2f}  (F1={f1:.3f} precision={prec:.3f} recall={rec:.3f})")
        print(f"   Cập nhật SCORE_THRESHOLD trong rag/config.py thành {t:.2f}")

    gap = min(in_scores) - max(out_scores)
    if gap <= 0:
        print("\n⚠️  Hai phân phối CHỒNG NHAU — không có ngưỡng nào tách sạch được.")
        print("   Cân nhắc: bật hybrid search, thêm reranker, hoặc chẻ nhỏ chunk hơn.")
    else:
        print(f"\n   Khoảng cách an toàn: {gap:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Chạy calibration**

```bash
set -a && source .env && set +a
python3 scripts/calibrate_threshold.py
```

Đọc kỹ output. Với BGE-M3 + cosine trên tiếng Việt, kỳ vọng in-scope rơi vào ~0.55-0.80, out-of-scope ~0.25-0.45.

- [ ] **Step 4: Cập nhật ngưỡng trong config**

Sửa `rag/config.py`, thay giá trị mặc định bằng con số script đề xuất, và **ghi kèm comment ngày đo**:

```python
# Hiệu chỉnh 2026-07-29 bằng scripts/calibrate_threshold.py
# in-scope min=0.xx, out-of-scope max=0.yy, F1=0.zz
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.XX"))
```

Nếu script báo hai phân phối chồng nhau: **đừng đi tiếp**. Xử lý theo thứ tự — (a) đổi `SEARCH_METHOD` thành `"hybrid_search"` và đo lại; (b) chẻ nhỏ chunk file 02 theo từng món thay vì theo nhóm A-E; (c) cân nhắc thêm reranker `bge-reranker-v2-m3` trong Dify.

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_threshold.py eval/rag_queries.json rag/config.py
git commit -m "feat(rag): calibrate retrieval score threshold on in/out-of-scope query sets"
```

---

## Task 8: Tích hợp RAG vào `serve_model.py`

**Files:**
- Create: `rag/prompt.py`
- Test: `tests/rag/test_prompt.py`
- Modify: `serve_model.py` — `SYSTEM_PROMPT` (dòng 26-37), `lifespan` (76-80), `ChatRequest` (99-101), `chat()` (132-157), `health()` (160-166)

**Interfaces:**
- Consumes: `rag.retriever.RetrievalResult`, `rag.retriever.Retriever`
- Produces:
  - `rag.prompt.build_context_block(result: RetrievalResult) -> str`
  - `rag.prompt.fallback_response() -> dict` — JSON envelope y hệt schema hiện có
  - `serve_model.py` `/chat` trả thêm `retrieval` metadata; endpoint mới `POST /retrieve`

**Bối cảnh quan trọng:**
- `SYSTEM_PROMPT` hiện tại **đã** yêu cầu model xuất JSON với các field `intent | confidence | answerText | spokenText | recommendedItems | draftCartItems | requiresHumanSupport | sourceIds`. JSON FALLBACK ta trả về khi out-of-scope phải **khớp chính xác schema đó**, nếu không Android sẽ crash khi parse.
- Android POST đúng `{message, history}` → field mới phải optional.
- `/chat` hiện mất ~78 giây/lượt trên CPU. Chặn cứng out-of-scope tiết kiệm trọn 78 giây đó — đây là lợi ích lớn nhất về mặt UX của task này.

- [ ] **Step 1: Viết test cho prompt**

```python
# tests/rag/test_prompt.py
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
```

- [ ] **Step 2: Chạy test, xác nhận FAIL**

Run: `python3 -m pytest tests/rag/test_prompt.py -v`
Expected: FAIL với `ModuleNotFoundError: No module named 'rag.prompt'`

- [ ] **Step 3: Viết `rag/prompt.py`**

```python
"""Dựng context block cho LLM và câu trả lời FALLBACK cố định."""
from rag.retriever import RetrievalResult

FALLBACK_TEXT = (
    "Xin lỗi bạn, mình chưa có thông tin chính xác về điều này. "
    "Bạn vui lòng hỏi nhân viên Viva để được hỗ trợ nhé!"
)


def fallback_response() -> dict:
    """Trả về ĐÚNG schema JSON mà SYSTEM_PROMPT quy định (serve_model.py:33-36)."""
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
```

- [ ] **Step 4: Chạy test, xác nhận PASS**

Run: `python3 -m pytest tests/rag/test_prompt.py -v`
Expected: 4 passed

- [ ] **Step 5: Sửa `serve_model.py` — thêm import và loader**

Thêm vào phần import đầu file:

```python
from rag import config as rag_config
from rag.prompt import build_context_block, fallback_response
from rag.retriever import Retriever
```

Thêm global và loader cạnh `load_stt` / `load_chat`:

```python
retriever = None


def load_retriever():
    global retriever
    if not rag_config.DIFY_DATASET_API_KEY or not rag_config.DIFY_DATASET_ID:
        print("⚠️  Chưa cấu hình Dify — chạy KHÔNG có RAG. Xem docs/RAG_SETUP.md")
        return
    retriever = Retriever()
    probe = retriever.retrieve("Viva Latte giá bao nhiêu")
    if probe.in_scope:
        print(f"✅ RAG ready (bge-m3, threshold={rag_config.SCORE_THRESHOLD}, "
              f"probe top_score={probe.top_score:.3f})")
    else:
        print(f"⚠️  RAG probe không tìm thấy gì (top_score={probe.top_score:.3f}) — "
              "KB đã sync và index xong chưa?")
```

Gọi trong `lifespan` (hiện ở dòng 76-80):

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_stt()
    load_chat()
    load_retriever()
    yield
```

- [ ] **Step 6: Sửa `ChatRequest` — thêm field optional (giữ tương thích Android)**

```python
class ChatRequest(BaseModel):
    message: str
    history: List[HistoryItem] = []
    use_rag: bool = True          # optional — Android không gửi, mặc định bật
    top_k: int | None = None      # optional — để debug
```

- [ ] **Step 7: Viết lại `chat()` — chặn cứng out-of-scope**

Thay toàn bộ thân hàm `chat()` (hiện dòng 132-157):

```python
@app.post("/chat")
async def chat(req: ChatRequest):
    retrieval = None
    context_block = ""

    if req.use_rag and retriever is not None:
        retrieval = retriever.retrieve(req.message)
        if not retrieval.in_scope:
            # CHẶN CỨNG: không gọi LLM. Tiết kiệm ~78s và loại bỏ nguy cơ bịa.
            return {
                "response": json.dumps(fallback_response(), ensure_ascii=False),
                "retrieval": {
                    "in_scope": False,
                    "top_score": retrieval.top_score,
                    "threshold": retriever.threshold,
                    "sourceIds": [],
                },
            }
        context_block = build_context_block(retrieval)

    system_content = SYSTEM_PROMPT
    if context_block:
        system_content = f"{SYSTEM_PROMPT}\n\n{context_block}"

    messages = [{"role": "system", "content": system_content}]
    for h in req.history[-8:]:
        messages.append({"role": h.role, "content": h.content})
    messages.append({"role": "user", "content": req.message})

    text = chat_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = chat_tokenizer([text], return_tensors="pt").to(
        next(chat_model.parameters()).device
    )

    with torch.no_grad():
        output_ids = chat_model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.7,
            do_sample=True,
            pad_token_id=chat_tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = chat_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    payload = {"response": response}
    if retrieval is not None:
        payload["retrieval"] = {
            "in_scope": True,
            "top_score": retrieval.top_score,
            "threshold": retriever.threshold,
            "sourceIds": retrieval.source_ids,
        }
    return payload
```

Thêm `import json` vào đầu file (hiện chưa có).

- [ ] **Step 8: Thêm endpoint `/retrieve` để debug và cập nhật `/health`**

```python
class RetrieveRequest(BaseModel):
    query: str
    top_k: int | None = None


@app.post("/retrieve")
async def retrieve_only(req: RetrieveRequest):
    """Xem retrieval trả về gì mà không tốn 78s chạy LLM."""
    if retriever is None:
        return {"error": "RAG chưa được cấu hình"}
    result = retriever.retrieve(req.query)
    return {
        "in_scope": result.in_scope,
        "top_score": result.top_score,
        "threshold": retriever.threshold,
        "chunks": [
            {"chunk_id": c.chunk_id, "score": c.score, "text": c.text[:300]}
            for c in result.chunks
        ],
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "stt_ready": stt_pipeline is not None,
        "chat_ready": chat_model is not None,
        "rag_ready": retriever is not None,
        "embedding_model": rag_config.EMBEDDING_MODEL,
        "score_threshold": rag_config.SCORE_THRESHOLD,
    }
```

- [ ] **Step 9: Chạy server và test end-to-end**

```bash
set -a && source .env && set +a
python3 serve_model.py
```

Ở terminal khác:

```bash
curl -s localhost:8000/health | python3 -m json.tool
# kỳ vọng rag_ready: true

# Retrieval nhanh, không tốn LLM
curl -s localhost:8000/retrieve -H 'Content-Type: application/json' \
  -d '{"query":"Viva Latte giá bao nhiêu"}' | python3 -m json.tool

# Out-of-scope: PHẢI trả về gần như tức thì (không chạy LLM)
time curl -s localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"message":"hôm nay trời có mưa không"}' | python3 -m json.tool

# In-scope: chậm (~78s trên CPU) nhưng phải có sourceIds
time curl -s localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"message":"Viva Latte giá bao nhiêu"}' | python3 -m json.tool
```

Nghiệm thu: câu out-of-scope trả về **dưới 2 giây** với `intent: FALLBACK`; câu in-scope có `retrieval.sourceIds` không rỗng và giá **đúng 55.000đ**.

- [ ] **Step 10: Xác nhận Android không vỡ**

```bash
# đúng payload mà CadebotApiService.kt gửi — không có use_rag
curl -s localhost:8000/chat -H 'Content-Type: application/json' \
  -d '{"message":"quán mở cửa mấy giờ","history":[]}' | python3 -m json.tool
```

Field `response` phải vẫn là một chuỗi JSON đúng schema cũ. Field `retrieval` là bổ sung — client cũ bỏ qua được.

- [ ] **Step 11: Commit**

```bash
git add rag/prompt.py tests/rag/test_prompt.py serve_model.py
git commit -m "feat(rag): wire retrieval into /chat with hard out-of-scope block"
```

---

## Task 9: Đánh giá end-to-end và trỏ pipeline giọng nói về server RAG

**Files:**
- Create: `scripts/eval_rag.py`
- Modify: `pipeline/llm.py` (dòng 11 — hiện gọi thẳng Ollama)
- Modify: `docs/RAG_SETUP.md` (thêm mục kết quả đánh giá)

**Bối cảnh:** `pipeline/llm.py` là **đường vào thứ hai, hoàn toàn độc lập** — nó POST tới Ollama `http://127.0.0.1:11434/api/chat` với model `cadebot-viva`, **không gửi system prompt, không gửi history, không có context**. Nếu để nguyên, đường giọng nói sẽ hoàn toàn bỏ qua lớp RAG vừa xây. Thêm nữa, `Modelfile` đặt `num_ctx 2048` — quá chật để nhét context. Cách sạch nhất là trỏ nó về `serve_model.py:8000`.

- [ ] **Step 1: Viết `scripts/eval_rag.py`**

```python
"""Đánh giá end-to-end: out-of-scope có bị chặn không, in-scope có trích nguồn không.

    python3 scripts/eval_rag.py            # cần serve_model.py đang chạy
    python3 scripts/eval_rag.py --fast     # chỉ /retrieve, bỏ qua LLM
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

API = "http://localhost:8000"
EVAL_FILE = ROOT / "eval" / "rag_queries.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="chỉ test retrieval")
    args = parser.parse_args()

    data = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
    endpoint = "/retrieve" if args.fast else "/chat"
    key = "query" if args.fast else "message"

    results = {"in_scope_ok": 0, "out_scope_blocked": 0}

    for item in data["in_scope"]:
        r = requests.post(f"{API}{endpoint}", json={key: item["q"]}, timeout=300).json()
        ok = r.get("in_scope") if args.fast else r.get("retrieval", {}).get("in_scope")
        results["in_scope_ok"] += int(bool(ok))
        print(f"{'✅' if ok else '❌'} IN  {item['q'][:45]}")

    for item in data["out_of_scope"]:
        t0 = time.time()
        r = requests.post(f"{API}{endpoint}", json={key: item["q"]}, timeout=300).json()
        elapsed = time.time() - t0
        blocked = not (r.get("in_scope") if args.fast
                       else r.get("retrieval", {}).get("in_scope"))
        results["out_scope_blocked"] += int(blocked)
        print(f"{'✅' if blocked else '❌'} OUT {item['q'][:45]:47} {elapsed:.1f}s")

    n_in, n_out = len(data["in_scope"]), len(data["out_of_scope"])
    print(f"\nIn-scope tìm được context : {results['in_scope_ok']}/{n_in}")
    print(f"Out-of-scope bị chặn      : {results['out_scope_blocked']}/{n_out}")

    # Chặn out-of-scope là tiêu chí quan trọng nhất — nó là lý do tồn tại của RAG.
    return 0 if results["out_scope_blocked"] == n_out else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Chạy đánh giá**

```bash
python3 scripts/eval_rag.py --fast    # nhanh, chỉ retrieval
python3 scripts/eval_rag.py           # đầy đủ, sẽ mất ~20 phút trên CPU
```

Mục tiêu nghiệm thu: **out-of-scope bị chặn 15/15**, in-scope tìm được context **≥ 13/15**. Nếu chưa đạt, quay lại Task 7 hiệu chỉnh ngưỡng — đừng nới lỏng tiêu chí.

- [ ] **Step 3: Trỏ `pipeline/llm.py` về serve_model.py**

Thay toàn bộ `pipeline/llm.py`:

```python
"""Gọi Cadebot API (có RAG) thay vì gọi thẳng Ollama.

Trước đây file này POST tới Ollama /api/chat, không có system prompt, không có
context — tức là bỏ qua hoàn toàn knowledge base. Nay đi qua serve_model.py để
dùng chung một đường RAG với client Android.
"""
import json

import requests

API_URL = "http://127.0.0.1:8000/chat"
TIMEOUT = 300  # CPU chậm: ~78s/lượt cho câu in-scope

FALLBACK = {
    "intent": "FALLBACK",
    "confidence": 1.0,
    "answerText": "Xin lỗi bạn, mình chưa kết nối được hệ thống. Bạn gọi nhân viên giúp mình nhé.",
    "spokenText": "Xin lỗi bạn, mình chưa kết nối được hệ thống. Bạn gọi nhân viên giúp mình nhé.",
    "recommendedItems": [],
    "draftCartItems": [],
    "requiresHumanSupport": True,
    "sourceIds": [],
}


def chat(user_text: str, history: list | None = None) -> dict:
    try:
        resp = requests.post(
            API_URL,
            json={"message": user_text, "history": history or []},
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
    except requests.RequestException as exc:
        print(f"[llm] lỗi gọi API: {exc}")
        return dict(FALLBACK)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Model đôi khi xuất text thường thay vì JSON — vẫn nói ra được.
        return {**FALLBACK, "answerText": raw, "spokenText": raw}
```

`pipeline/main.py` dùng `result["spokenText"]` — schema trên vẫn khớp, không cần sửa.

- [ ] **Step 4: Test lại pipeline giọng nói**

```bash
python3 pipeline/test_pipeline.py --module llm
```

Expected: các case in-scope có nội dung đúng KB, case out-of-scope trả `intent: FALLBACK`.

- [ ] **Step 5: Ghi kết quả vào `docs/RAG_SETUP.md`**

Thêm mục "Kết quả đánh giá" gồm: ngưỡng đã chốt, tỉ lệ chặn out-of-scope, recall in-scope, thời gian phản hồi trung bình cho hai loại câu, và ngày đo.

- [ ] **Step 6: Commit**

```bash
git add scripts/eval_rag.py pipeline/llm.py docs/RAG_SETUP.md
git commit -m "feat(rag): end-to-end eval and route voice pipeline through RAG server"
```

---

## Verification

Chạy toàn bộ chuỗi sau trên máy deploy, từ trạng thái sạch:

```bash
# 1. Unit test — không cần Dify chạy
python3 -m pytest tests/ -v
# Kỳ vọng: toàn bộ pass (config 5, chunker 8, db_source 6, kb_builder 4, retriever 7, prompt 4)

# 2. Hạ tầng
ollama list | grep bge-m3
docker compose -f ~/dify/docker/docker-compose.yaml ps    # tất cả running
curl -s http://127.0.0.1:11434/api/embed -d '{"model":"bge-m3","input":"test"}' \
  | python3 -c "import json,sys; assert len(json.load(sys.stdin)['embeddings'][0])==1024; print('dim OK')"

# 3. Sync KB (idempotent — chạy 2 lần vẫn đúng 2 document)
set -a && source .env && set +a
python3 scripts/sync_kb.py && python3 scripts/sync_kb.py

# 4. Ngưỡng vẫn hợp lệ
python3 scripts/calibrate_threshold.py

# 5. Server
python3 serve_model.py &
sleep 180   # chờ load PhoWhisper + Qwen + probe RAG
curl -s localhost:8000/health | python3 -m json.tool   # rag_ready: true

# 6. Nghiệm thu hành vi
python3 scripts/eval_rag.py --fast    # out-of-scope chặn 15/15
python3 scripts/eval_rag.py           # đầy đủ (~20 phút CPU)
```

**Tiêu chí đạt:**

| Hạng mục | Ngưỡng chấp nhận |
|---|---|
| Unit test | 100% pass |
| Chặn out-of-scope | 15/15 |
| In-scope tìm được context | ≥ 13/15 |
| Độ trễ câu out-of-scope | < 2 giây (không gọi LLM) |
| `sourceIds` câu in-scope | không rỗng, ID có thật trong KB |
| Sửa giá trong SQLite → sync → hỏi lại | trả giá mới |
| Android gửi `{message, history}` | vẫn hoạt động, `response` đúng schema JSON cũ |

**Kiểm thử thủ công bắt buộc** (không script nào thay được): mở Dify → Knowledge → **Retrieval Testing**, gõ 5-10 câu hỏi tiếng Việt tự nhiên và **đọc mắt** các chunk trả về. Chunk lấy ra có thực sự trả lời được câu hỏi không? Nếu retrieval đúng mà câu trả lời sai, vấn đề nằm ở prompt/LLM. Nếu retrieval sai, vấn đề nằm ở chunking hoặc ngưỡng.

---

## Ghi chú cho người thực thi

- **Không đổi embedding model giữa chừng.** Đổi `bge-m3` sang model khác = phải xoá knowledge base và tạo lại từ đầu (Dify không re-embed tại chỗ). Đây là cảnh báo lặp lại từ `cadebot-plan.md`.
- **Hai loại API key của Dify rất dễ nhầm**: Dataset API key (Knowledge → API Access) dùng cho sync + retrieve; App API key (dùng bởi `cadebot_dify_bridge.py`) là thứ khác. Dùng nhầm sẽ nhận 401.
- **`db_exported_kb.md` không được dùng làm nguồn** (theo quyết định của bạn) — nó là file tĩnh do `demo_db_to_dify.py` sinh ra và đã lệch so với DB (thiếu 1 FAQ, và cả hai combo bị gán nhầm `brewing_method = 'Đá Xay'`). Ta đọc thẳng SQLite. Nếu file này còn nằm trong thư mục KB, đừng để nó lọt vào index.
- **Giai đoạn 7 (DB thật của quán)**: chỉ sửa bên trong `get_menu_data()` ở `rag/db_source.py`. Toàn bộ chunker, builder, sync, retriever giữ nguyên. Đặt cron chạy `scripts/sync_kb.py` hằng ngày là đủ — giá/khuyến mãi đổi rất ít, không cần real-time.
- **Về hiệu năng**: ~78 giây/lượt cho câu in-scope trên CPU là con số đã đo trong `PIPELINE_ANALYSIS.md`, không phải do RAG gây ra (retrieval chỉ thêm ~0.3-1 giây). Nếu cần nhanh hơn, đó là bài toán riêng — lượng hoá INT4 hoặc chuyển sang GPU, xem §4 và §7 của `PIPELINE_ANALYSIS.md`.
- **File plan này** hiện nằm ở `~/.claude/plans/`. Khi bắt đầu thực thi, copy vào repo: `docs/superpowers/plans/2026-07-29-bge-m3-rag-dify.md`.
