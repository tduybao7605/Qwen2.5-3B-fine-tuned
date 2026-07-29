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
