# API reference

Four endpoints, served by `src/cadebot/api.py`. Base URL is
`http://localhost:8000` locally, or the Cloudflare Tunnel hostname in production.

| Method | Path | Purpose |
|---|---|---|
| `GET` | [`/health`](#get-health) | Readiness of each subsystem |
| `POST` | [`/stt`](#post-stt) | Vietnamese speech-to-text |
| `POST` | [`/chat`](#post-chat) | Answer a question, optionally grounded by RAG |
| `POST` | [`/retrieve`](#post-retrieve) | Retrieval only — debugging, no LLM |

There is **no authentication** and CORS is `allow_origins=["*"]`. See
[deployment.md](deployment.md#security-notes).

---

## `GET /health`

No parameters. Reports whether each subsystem finished loading.

```bash
curl -s localhost:8000/health | python3 -m json.tool
```

```json
{
    "status": "ok",
    "stt_ready": true,
    "chat_ready": true,
    "rag_ready": true,
    "embedding_model": "bge-m3",
    "score_threshold": 0.51
}
```

| Field | Type | Meaning |
|---|---|---|
| `status` | string | Always `"ok"` when the process is answering at all |
| `stt_ready` | bool | PhoWhisper-large pipeline loaded |
| `chat_ready` | bool | Qwen2.5-3B + LoRA loaded |
| `rag_ready` | bool | A `Retriever` was constructed — i.e. `DIFY_DATASET_API_KEY` and `DIFY_DATASET_ID` were both set at startup |
| `embedding_model` | string | `config.EMBEDDING_MODEL` — `"bge-m3"` |
| `score_threshold` | float | `config.SCORE_THRESHOLD` — `0.51` by default |

The three `*_ready` flags are all `false` until the lifespan hook finishes, which
on a cold container means after the ~6 GB weight download. `rag_ready` reflects
only that the retriever object exists; whether Dify actually answers is logged at
startup by a probe query.

---

## `POST /stt`

`multipart/form-data` with a single field `file`. Any format `ffmpeg` can decode
is accepted — the Android client sends MPEG-4/AAC `.m4a` — and is converted to
16 kHz mono WAV before transcription.

```bash
curl -s -X POST localhost:8000/stt -F "file=@question.m4a"
```

```json
{"text": "Viva Latte giá bao nhiêu"}
```

| Field | Type | Meaning |
|---|---|---|
| `text` | string | Transcript, whitespace-stripped |

**Long audio.** Whisper processes 30 s at a time. When the decoded audio exceeds
30 s the server switches to the long-form path and passes `return_timestamps=True`
— required there, or `transformers` raises `ValueError`. Clients see no
difference; the response shape is identical.

**Latency.** PhoWhisper-large on CPU runs at roughly 30× slower than real time,
so 5 s of audio takes about 150 s. See [performance.md](performance.md) §3.

---

## `POST /chat`

The main endpoint. `application/json`.

### Request

```json
{
  "message": "Viva Latte giá bao nhiêu",
  "history": [
    {"role": "user", "content": "chào bạn"},
    {"role": "assistant", "content": "Chào bạn! Mình là Cadebot."}
  ],
  "use_rag": true,
  "top_k": null
}
```

| Field | Type | Default | Meaning |
|---|---|---|---|
| `message` | string | *required* | The user's turn |
| `history` | array of `{role, content}` | `[]` | Prior turns. Only the **last 8** are sent to the model (`req.history[-8:]`) |
| `use_rag` | bool | **`true`** | Ground the answer in the knowledge base and hard-block out-of-scope questions. Send `false` only if your deployment cannot tolerate a 142-184 s grounded turn — the answer then comes from fine-tuning memory, unverified. See [deployment.md](deployment.md#the-100-s-limit-vs-grounded-answers) |
| `top_k` | int or null | `null` | Accepted for debugging but **not currently used** by this handler; retrieval uses `config.TOP_K` (3) |

If `use_rag` is `false`, or the server started without Dify credentials
(`retriever is None`), retrieval is skipped entirely and no `retrieval` key
appears in the response.

### Response

```json
{
  "response": "{\"intent\":\"MENU_QA\",\"confidence\":0.95,\"answerText\":\"Viva Latte có giá 55.000 VNĐ bạn nhé!\",\"spokenText\":\"Viva Latte giá 55 nghìn đồng.\",\"recommendedItems\":[],\"draftCartItems\":[],\"requiresHumanSupport\":false,\"sourceIds\":[\"menu:VR_LATTE_M\"]}",
  "retrieval": {
    "in_scope": true,
    "top_score": 0.82,
    "threshold": 0.51,
    "sourceIds": ["menu:VR_LATTE_M"]
  }
}
```

> **`response` is a *string containing JSON*, not a nested object.** It is the
> model's raw output, so clients must call their JSON parser a second time on it.
> The Android client does exactly that in
> `Cadebot_UI/.../data/remote/CadebotApiService.kt`.

| Field | Type | Present when |
|---|---|---|
| `response` | string (JSON) | Always |
| `retrieval` | object | Only when `use_rag: true` **and** a retriever is configured |
| `retrieval.in_scope` | bool | — |
| `retrieval.top_score` | float | Highest raw similarity score returned by Dify, before thresholding |
| `retrieval.threshold` | float | The threshold actually applied by this `Retriever` instance |
| `retrieval.sourceIds` | array of string | Chunk IDs that passed the threshold; `[]` when out of scope |

### The JSON schema inside `response`

Mandated by `SYSTEM_PROMPT` in `src/cadebot/api.py` and produced verbatim by
`fallback_response()` in `src/cadebot/rag/prompt.py`:

| Field | Type | Notes |
|---|---|---|
| `intent` | string | One of `MENU_QA`, `RECOMMENDATION`, `ADD_TO_CART_DRAFT`, `PROMOTION_QA`, `CALL_STAFF`, `FALLBACK` |
| `confidence` | number | Model's own estimate |
| `answerText` | string | Full answer for the chat bubble |
| `spokenText` | string | Shorter phrasing for text-to-speech |
| `recommendedItems` | array | Suggested menu items |
| `draftCartItems` | array | Draft order lines |
| `requiresHumanSupport` | bool | `true` in the fallback path |
| `sourceIds` | array of string | Chunk IDs backing the answer |

`sourceIds` is post-processed by `sanitize_response()` whenever RAG ran: IDs the
model invented are dropped, and IDs that lost their prefix (`VR_TIRAMISU`) are
repaired to the full form (`menu:VR_TIRAMISU`). Chunk ID namespaces are
`menu:*`, `promo:*`, `faq:db_*`, `faq:md_*` and `doc:*`.

Since it is the model's own text, malformed JSON is possible. If it cannot be
parsed, `sanitize_response()` returns it unchanged rather than corrupting it
further — clients should handle a parse failure.

### Out-of-scope requests

With grounding on (the default), a query whose top retrieval score falls below
the threshold never reaches the LLM. The server returns the fixed fallback
immediately:

```bash
curl -s -X POST localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"hôm nay trời có mưa không"}'
```

```json
{
  "response": "{\"intent\":\"FALLBACK\",\"confidence\":1.0,\"answerText\":\"Xin lỗi bạn, mình chưa có thông tin chính xác về điều này. Bạn vui lòng hỏi nhân viên Viva để được hỗ trợ nhé!\",\"spokenText\":\"Xin lỗi bạn, mình chưa có thông tin chính xác về điều này. Bạn vui lòng hỏi nhân viên Viva để được hỗ trợ nhé!\",\"recommendedItems\":[],\"draftCartItems\":[],\"requiresHumanSupport\":true,\"sourceIds\":[]}",
  "retrieval": {"in_scope": false, "top_score": 0.21, "threshold": 0.51, "sourceIds": []}
}
```

HTTP status is still `200` — the request succeeded; the assistant simply declined.

---

## `POST /retrieve`

Debugging endpoint. Runs retrieval and returns the scores without invoking the
LLM, which is what makes threshold calibration cheap.

```bash
curl -s -X POST localhost:8000/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"Viva Latte giá bao nhiêu"}' | python3 -m json.tool
```

```json
{
    "in_scope": true,
    "top_score": 0.82,
    "threshold": 0.51,
    "chunks": [
        {
            "chunk_id": "menu:VR_LATTE_M",
            "score": 0.82,
            "text": "Món: Viva Latte (Mã: VR_LATTE_M | Nhóm: Cà Phê)\n  + Giá: 55,000 VNĐ..."
        }
    ]
}
```

| Field | Type | Meaning |
|---|---|---|
| `query` | string | *required* — the text to retrieve for |
| `top_k` | int or null | Accepted but **not currently used**; retrieval uses `config.TOP_K` (3) |

| Response field | Type | Meaning |
|---|---|---|
| `in_scope` | bool | At least one chunk scored at or above the threshold |
| `top_score` | float | Highest raw score, before thresholding |
| `threshold` | float | Threshold applied |
| `chunks` | array | Only chunks that passed the threshold — empty when `in_scope` is `false` |
| `chunks[].chunk_id` | string | `"unknown"` if the segment lost its leading `[id]` line |
| `chunks[].score` | float | Similarity score from Dify |
| `chunks[].text` | string | Chunk body, **truncated to 300 characters** |

If the server started without Dify credentials, this endpoint returns
`{"error": "RAG chưa được cấu hình"}` with HTTP `200`.

`scripts/eval_rag.py --fast` drives this endpoint over `eval/rag_queries.json`
(15 in-scope + 15 out-of-scope queries) without touching the LLM.

---

## Generation parameters

Fixed in `src/cadebot/api.py` and `src/cadebot/rag/config.py`; not settable per
request.

| Parameter | Value | Source |
|---|---|---|
| `max_new_tokens` | 400 | `api.py` |
| `temperature` | `GEN_TEMPERATURE`, default `0.2` | `config.py` |
| `do_sample` | `true` | `api.py` |
| History window | last 8 turns | `api.py` |
| Retrieval `top_k` | `TOP_K`, default 3 | `config.py` |
| Retrieval search method | `semantic_search`, reranking off | `config.py`, `retriever.py` |
| Context budget | `MAX_CONTEXT_CHARS`, 2000 characters | `config.py` |
| Retrieval timeout | `RETRIEVAL_TIMEOUT`, 15 s | `config.py` |

---

## Latency

Measured on the deployment machine — CPU only, no GPU.

| Path | Time | Source |
|---|---|---|
| `/chat` out of scope, grounded (default) | **0.096 s** — the LLM is never invoked | [rag-setup.md](rag-setup.md) |
| `/chat` in scope, grounded (default) | **142-184 s** (measured 2026-07-29 / 2026-08-14) | [deployment.md](deployment.md#the-100-s-limit-vs-grounded-answers) |
| `/chat` with `use_rag: false` | **~95 s** | Measured 2026-08-04, when the default was briefly flipped |
| `/chat` LLM generation alone, no context injection | ~78 s | [performance.md](performance.md) §3 |
| `/stt`, 5 s of audio | ~150 s (~30× slower than real time) | [performance.md](performance.md) §3 |

The gap between 78 s and 142+ s is context injection: grounding makes the prompt
longer, and prefill on CPU is not free. A grounded turn therefore does not fit
inside the 100 s response limit of a Cloudflare-Tunnel-fronted deployment, which
returns HTTP 524. That is handled per deployment — see
[deployment.md](deployment.md#the-100-s-limit-vs-grounded-answers) — not by
turning grounding off system-wide.

---

**Related:** [architecture.md](architecture.md) · [deployment.md](deployment.md) ·
[android-client.md](android-client.md) (Vietnamese — the client side of these calls)
