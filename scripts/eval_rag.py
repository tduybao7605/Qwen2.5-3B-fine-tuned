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
