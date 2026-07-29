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
