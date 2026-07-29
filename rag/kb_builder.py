"""Gộp chunk thành một document markdown để Dify cắt lại theo delimiter `---`."""
from rag import config
from rag.chunker import Chunk


def build_document(chunks: list[Chunk]) -> str:
    sep_line = config.CHUNK_SEPARATOR.strip()
    for c in chunks:
        rendered = c.render()
        # Kiểm tra theo DÒNG, không theo chuỗi con: chunk kết thúc bằng `...\n---`
        # (thiếu newline cuối) vẫn lọt qua phép `in` nhưng khi nối vẫn tạo ra
        # `---\n---\n` làm Dify cắt lệch. Đã dính lỗi này một lần với file 02.
        if any(ln.strip() == sep_line for ln in rendered.splitlines()):
            raise ValueError(
                f"Chunk {c.id} chứa dòng {sep_line!r} — Dify sẽ cắt nhầm và "
                "segment kế tiếp mất dòng [chunk_id]. Lọc nó ở rag/chunker.py."
            )
    return config.CHUNK_SEPARATOR.join(c.render() for c in chunks)
