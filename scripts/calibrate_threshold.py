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
