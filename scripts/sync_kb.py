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
