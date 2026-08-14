from cadebot.rag import config
from cadebot.rag.chunker import Chunk
from cadebot.rag.kb_builder import build_document


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
    from cadebot.rag.chunker import chunk_all_markdown
    from cadebot.rag.db_source import chunk_database

    doc = build_document(chunk_all_markdown() + chunk_database())
    assert len(doc) > 5000
