# Cadebot RAG — Hướng dẫn dựng hạ tầng

Dify self-hosted + Qdrant + BGE-M3 (qua Ollama) cho `serve_model.py`.
Plan gốc: `docs/superpowers/plans/2026-07-29-bge-m3-rag-dify.md`.

## Cấu hình đã chốt

| Hạng mục | Giá trị | Ghi chú |
|---|---|---|
| Embedding model | `bge-m3` | **Đã xác minh dim = 1024** |
| Vector dimension | 1024 | Đổi = phải re-embed toàn bộ KB |
| Vector store | Qdrant | `http://qdrant:6333` (nội bộ, không publish ra host) |
| Dify | 1.16.1 | `http://localhost` (nginx cổng 80) |
| Ollama | `http://172.17.0.1:11434` | Từ trong container Dify |
| Chunk separator | `\n---\n` | `process_rule` mode `custom` |
| Max chunk tokens | 500 | Chunk lớn nhất hiện tại 694 ký tự |
| Indexing | High Quality | Không hạ về Economical được |

## Trạng thái đã dựng xong

- [x] Ollama + `bge-m3` (dim 1024 đã verify)
- [x] Dify 1.16.1 + Qdrant chạy, `http://localhost` trả 307 → `/install` trả 200
- [x] Ollama reachable từ container `docker-api-1`
- [ ] **Tài khoản admin Dify** — cần làm thủ công trên trình duyệt
- [ ] **Đăng ký `bge-m3` làm Text Embedding provider**
- [ ] **Tạo Knowledge base + lấy Dataset ID / API key**

---

## 1. Ollama + BGE-M3

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull bge-m3

# Bắt buộc verify — sai dim thì phải sửa rag/config.py TRƯỚC khi tạo KB
curl -s http://127.0.0.1:11434/api/embed \
  -d '{"model":"bge-m3","input":"Viva Latte giá bao nhiêu"}' \
  | python3 -c "import json,sys; print('dim =', len(json.load(sys.stdin)['embeddings'][0]))"
# → dim = 1024
```

## 2. Dify + Qdrant

```bash
git clone --depth 1 https://github.com/langgenius/dify.git ~/dify
cp ~/dify/docker/.env.example ~/dify/docker/.env
```

Sửa `~/dify/docker/.env`:

```
VECTOR_STORE=qdrant

QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=difyai123456
QDRANT_CLIENT_TIMEOUT=20
QDRANT_GRPC_ENABLED=false
QDRANT_GRPC_PORT=6334
```

> ⚠️ **Khác với plan gốc:** Dify 1.16.1 **không có sẵn biến `QDRANT_*`** trong `.env.example` — phải tự thêm.

Khởi động — **phải bật cả hai profile**:

```bash
cd ~/dify/docker && docker compose --profile postgresql --profile qdrant up -d
```

> ⚠️ **Khác với plan gốc:** plan chỉ ghi `--profile qdrant`. Ở 1.16.1 service Postgres tên là **`db_postgres`** và nằm sau profile **`postgresql`**. Thiếu profile này thì `plugin_daemon` crashloop (`lookup db_postgres ... server misbehaving`) và `nginx` cũng crashloop theo (`host not found in upstream "plugin_daemon"`).

Nếu `plugin_daemon`/`nginx` vẫn restarting sau lần `up` đầu (chúng khởi động trước khi Postgres kịp healthy):

```bash
docker restart docker-plugin_daemon-1
docker restart docker-nginx-1
```

Kiểm tra:

```bash
docker compose -f ~/dify/docker/docker-compose.yaml ps   # tất cả running
curl -s -o /dev/null -w "%{http_code}\n" http://localhost/   # 307
```

### Yêu cầu dung lượng

Bộ image Dify chiếm khoảng **12-15 GB**. Kiểm tra trước khi pull:

```bash
df -h /
docker system df          # xem phần RECLAIMABLE
docker builder prune -af  # thu hồi build cache nếu thiếu chỗ (an toàn, cache tự sinh lại)
```

## 3. Cầu nối Ollama → container Dify

Ollama chỉ nghe `127.0.0.1:11434` nên container không với tới được. Dùng script có sẵn:

```bash
python3 knowledge_Base_cadebot/ollama_docker_bridge.py &
```

Verify **từ trong container**:

```bash
docker exec docker-api-1 sh -c 'curl -s http://172.17.0.1:11434/api/tags'
# phải thấy bge-m3
```

Cho chạy vĩnh viễn (cần sudo — chạy thủ công):

```bash
sudo tee /etc/systemd/system/ollama-docker-bridge.service >/dev/null <<'EOF'
[Unit]
Description=Ollama docker0 bridge for Dify
After=network.target ollama.service

[Service]
ExecStart=/usr/bin/python3 /home/ncd/learnspaces/Qwen2.5-3B-fine-tuned/knowledge_Base_cadebot/ollama_docker_bridge.py
Restart=always
User=ncd

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now ollama-docker-bridge
```

---

## 4. Các bước thủ công còn lại (trình duyệt)

### 4.1 Tạo tài khoản admin
Mở `http://localhost/install`, tạo email + mật khẩu.

### 4.2 Đăng ký BGE-M3
**Settings → Model Providers → Ollama → Add Model**

| Trường | Giá trị |
|---|---|
| Model Type | `Text Embedding` |
| Model Name | `bge-m3` |
| Base URL | `http://172.17.0.1:11434` |
| Model context size | `8192` |
| Max token limit | `8192` |

Báo "Connection refused" → cầu nối ở mục 3 chưa chạy.

### 4.3 Tạo Knowledge base
**Knowledge → Create Knowledge → Import from text** (tạo document tạm bất kỳ):

- Chunk setting: **Custom**, Delimiter `---`, Max chunk length `500`, Overlap `0`
- Index Method: **High Quality**
- Embedding Model: **bge-m3**
- Retrieval Setting: **Vector Search**

Tạo xong thì **xóa document tạm** — `scripts/sync_kb.py` sẽ tạo document thật.

### 4.4 Lấy Dataset ID + API key

- **Dataset ID**: trong URL `http://localhost/datasets/<DATASET_ID>/documents`
- **Dataset API key**: **Knowledge → API Access → API Key → Create**
  ⚠️ Đây là key **khác** với App API key mà `cadebot_dify_bridge.py` dùng. Dùng nhầm sẽ nhận 401.

Lưu vào `.env` ở gốc repo (`.env` đã nằm trong `.gitignore`):

```bash
cat >> .env <<'EOF'
DIFY_BASE_URL=http://localhost/v1
DIFY_DATASET_API_KEY=dataset-xxxxxxxxxxxxxxxx
DIFY_DATASET_ID=xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
EOF
```

Verify:

```bash
set -a && source .env && set +a
curl -s "$DIFY_BASE_URL/datasets/$DIFY_DATASET_ID/documents" \
  -H "Authorization: Bearer $DIFY_DATASET_API_KEY" | head -c 300
```

---

## 5. Sync KB và hiệu chỉnh ngưỡng

```bash
set -a && source .env && set +a

python3 scripts/sync_kb.py --dry-run   # 34 markdown + 35 database = 69 chunks
python3 scripts/sync_kb.py             # đẩy lên Dify
python3 scripts/sync_kb.py             # chạy lại: vẫn ĐÚNG 2 document, không phải 4
```

Đợi Dify index xong (Knowledge → Documents, trạng thái **Available**) rồi:

```bash
python3 scripts/calibrate_threshold.py
```

Cập nhật `SCORE_THRESHOLD` trong `rag/config.py` bằng con số script đề xuất.
**Giá trị hiện tại `0.55` chỉ là tạm — chưa đo.**

Nếu hai phân phối in-scope/out-of-scope chồng nhau, xử lý theo thứ tự:
1. Đổi `SEARCH_METHOD` thành `"hybrid_search"` trong `rag/config.py`
2. Chẻ nhỏ chunk hơn nữa
3. Thêm reranker `bge-reranker-v2-m3`

## 6. Chạy server

```bash
set -a && source .env && set +a
python3 serve_model.py
curl -s localhost:8000/health | python3 -m json.tool   # rag_ready: true
python3 scripts/eval_rag.py --fast                      # out-of-scope phải chặn 15/15
```

---

## Kết quả đánh giá

Đo ngày **2026-07-29** trên KB thật (69 segments, bge-m3, `semantic_search`, `top_k=3`).

| Chỉ số | Giá trị |
|---|---|
| **SCORE_THRESHOLD đã chốt** | **0.51** |
| In-scope: min / mean | 0.526 / 0.659 |
| Out-of-scope: max / mean | 0.540 / 0.442 |
| F1 / precision / recall | 0.968 / 0.938 / 1.000 |
| In-scope tìm được context | **15/15** |
| Out-of-scope bị chặn | **14/15** |
| Độ trễ câu out-of-scope | **0.096 s** (không gọi LLM) |
| Độ trễ câu in-scope | **~2 phút 22 s** (CPU) |
| Segment giữ được `[chunk_id]` | **69/69** |
| Idempotency | 5 lần sync → vẫn đúng 2 document |

### Những điểm cần biết

**1. Một câu out-of-scope lọt lưới.** `"cho tôi số điện thoại của bạn"` đạt 0.540 > ngưỡng.
Nhưng KB *có* hotline thật (`096 607 70 88`), nên bot trả lời được — nhãn out-of-scope
của câu này trong `eval/rag_queries.json` mới là thứ đáng ngờ, không phải retrieval sai.
Nâng ngưỡng lên >0.540 sẽ chặn nhầm `"có chỗ đậu xe"` (0.526) và `"quán mở cửa mấy giờ"`
(0.530) — đánh đổi không đáng.

**2. `hybrid_search` không giúp gì.** Đã đo: kết quả *y hệt* `semantic_search`
(Dify bỏ qua `weights` khi `reranking_enable=false`). `full_text_search` trả 0 điểm
cho mọi câu vì KB dùng High Quality, không có keyword index. Muốn tách bạch hơn thì
phải thêm reranker `bge-reranker-v2-m3`.

**3. RAG chặn bịa *số liệu*, không chặn bịa *thuộc tính*.** Ví dụ thật: hỏi
"Trà Đào Cam Sả giá bao nhiêu" → giá 45.000đ **đúng**, `sourceIds` **đúng**
(`menu:VR_PEACH_TEA`), nhưng model thêm câu *"Đây là món best seller của Viva"* —
trong khi KB nói **Viva Latte** mới là best seller. Model trộn thuộc tính giữa các
chunk được truy xuất. Muốn siết thì phải sửa prompt hoặc fine-tune lại, RAG không lo được.

**4. Độ trễ in-scope tăng so với `PIPELINE_ANALYSIS.md`** (~78s → ~142s) vì context
injection làm prompt dài ra. Đây là bài toán riêng — lượng hoá INT4 hoặc chuyển GPU.
