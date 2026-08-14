"""Chuyển 5 file markdown KB thành các chunk có ID ổn định.

Hai chiến lược:
  - File 05 (FAQ): 1 chunk / 1 cặp Q&A — giữ câu hỏi và câu trả lời cùng nhau.
  - File 01-04: 1 chunk / 1 section `##`, có kèm tiêu đề file để chunk không mất ngữ cảnh.
"""
import re
from dataclasses import dataclass
from pathlib import Path

from cadebot.rag import config


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


_HR_LINE = {"---", "***", "___"}


def _strip_rules(text: str) -> str:
    """Bỏ đường kẻ ngang markdown khỏi nội dung chunk.

    Chúng không mang thông tin gì cho retrieval, và nếu lọt vào chunk thì khi
    nối bằng CHUNK_SEPARATOR (`\\n---\\n`) sẽ tạo ra `---\\n---\\n`: Dify cắt
    lệch một nhịp và segment kế tiếp mở đầu bằng `---` thay vì `[chunk_id]`,
    làm parse_chunk_id trả 'unknown' và mất sourceIds.
    """
    kept = [ln for ln in text.splitlines() if ln.strip() not in _HR_LINE]
    return "\n".join(kept).strip()


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

        # Section có `###` (vd. file 02: A. Cà Phê, B. Trà, ...) phải chẻ tiếp.
        # Để nguyên thì chunk vượt max_tokens=500, Dify tự cắt và đoạn sau mất
        # dòng [chunk_id] -> hỏng provenance đúng ở phần menu hay được hỏi nhất.
        subs = re.split(r"^###\s+", body, flags=re.MULTILINE)
        section_head = subs[0].strip()
        section_title = section_head.splitlines()[0] if section_head else ""

        if len(subs) == 1:
            # Gắn tiêu đề file vào đầu chunk — nếu không, chunk "## 2. Tiện Ích"
            # bị lấy ra một mình sẽ không biết là tiện ích của quán nào.
            text = _strip_rules(f"{title}\n\n## {body}")
            chunks.append(Chunk(id=f"doc:{slug}#{i}", text=text, source=path.name))
            continue

        for j, sub in enumerate(subs[1:], start=1):
            sub_body = sub.strip()
            if not sub_body:
                continue
            # Chunk con mang cả tên file lẫn tên section cha để tự đứng vững.
            text = _strip_rules(f"{title}\n\n## {section_title}\n\n### {sub_body}")
            chunks.append(
                Chunk(id=f"doc:{slug}#{i}.{j}", text=text, source=path.name)
            )
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
