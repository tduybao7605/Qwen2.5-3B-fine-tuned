# Cadebot — GitHub Repo Professional Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `Qwen2.5-3B-fine-tuned` from a working-but-messy research dump into a repo that reads as a professionally engineered product on GitHub — English README, a real `src/` package layout, complete architecture + Docker hosting documentation, CI, and zero personal information.

**Architecture:** Python code moves to a `src/cadebot/` package (`cadebot.api`, `cadebot.models`, `cadebot.rag`) so imports stop depending on CWD; `serve_model.py` splits into a FastAPI layer and a model-loading layer; every loose `.md` at the repo root moves under `docs/` with a written-from-scratch English README as the front page. Heavy build artifacts are untracked going forward (no history rewrite). All absolute paths and the personal tunnel domain are replaced with environment variables and placeholders.

**Tech Stack:** Python 3.12, FastAPI + Uvicorn, Transformers + PEFT (Qwen2.5-3B-Instruct + LoRA), PhoWhisper-large (STT), Dify 1.16.1 + Qdrant + BGE-M3 via Ollama (RAG), Docker Compose, Cloudflare Tunnel, pytest, GitHub Actions.

**Spec:** This document (derived from a live audit of the repo on 2026-08-14; no separate spec exists).

## Context

The project works end-to-end — a Vietnamese voice assistant for a coffee shop: Android app → FastAPI server on a home machine → PhoWhisper STT + fine-tuned Qwen2.5-3B → RAG over a Dify/Qdrant knowledge base, exposed publicly through a Cloudflare Tunnel. But the GitHub repo does not show any of that:

- `README.md` is **one line**: `# Qwen2.5-3B-fine-tuned`.
- Eight `.md` files sit loose at the root with no index and no entry point.
- There is no architecture diagram, and the Docker hosting setup — arguably the most interesting engineering in the project (external Dify network, CPU-only torch wheel, weights mounted not baked, Cloudflare's 100 s edge timeout dictating a default) — is documented only in scattered Vietnamese comments inside `docker-compose.yml`.
- Python modules live at the repo root and only import correctly because `conftest.py` and three `scripts/*.py` each patch `sys.path` by hand.
- Personal information leaks in several tracked files: absolute home paths (`/home/team3/...`, `/home/ncd/learnspaces/...`), an internal team slug (`hrc2026-team3`), and the owner's personal tunnel domain (`duybao.tdbao-brian.work`) hardcoded as the Android app's default API URL.
- 1.4 GB of Git LFS objects, most of it a 914 MB `optimizer.pt` training checkpoint that nobody needs to reproduce anything.

Intended outcome: a visitor lands on the repo, understands the system in 60 seconds from the README, can find how it is deployed in one click, and can run the test suite locally without reading anyone's home directory path.

## Global Constraints

- **Language:** `README.md` and all newly authored docs are in **English**. Existing Vietnamese technical docs stay Vietnamese — they are only moved, renamed, and sanitized, never translated.
- **No history rewrite.** Never run `git filter-repo`, `git filter-branch`, or force-push. Heavy artifacts are removed with `git rm --cached` only, so existing clones keep working.
- **No personal information in tracked files.** Forbidden strings anywhere under version control: `/home/team3`, `/home/ncd`, `hrc2026-team3`, `duybao`, `tdbao-brian.work`, `Isaac-GR00T`. `.env` stays gitignored; only `.env.example` with placeholder values is committed.
- **Python 3.12**, dependency versions stay exactly as pinned in `requirements.txt` — this plan does not upgrade any dependency.
- **Behavior is frozen.** No endpoint contract changes, no prompt changes, no threshold changes. The only runtime change permitted is replacing hardcoded paths/model names with environment-variable lookups that default to the current values.
- **Package name:** `cadebot`. Import root is `src/`.
- Every task ends with `python3 -m pytest tests/ -q` passing (46 tests currently collect across 8 files) and a commit.

---

## Target Repository Layout

```
.
├── README.md                     # NEW — English front page, the deliverable
├── LICENSE                       # NEW — MIT
├── CONTRIBUTING.md               # NEW
├── Makefile                      # sanitized (personal domain removed)
├── pyproject.toml                # NEW — packaging + pytest config
├── requirements.txt
├── .env.example                  # NEW
├── Dockerfile                    # updated for src/ layout
├── docker-compose.yml            # commented in English
├── .dockerignore / .gitignore / .gitattributes
├── .github/
│   ├── workflows/ci.yml          # NEW
│   ├── ISSUE_TEMPLATE/{bug_report.yml,feature_request.yml}
│   └── pull_request_template.md
├── src/cadebot/
│   ├── __init__.py
│   ├── __main__.py               # `python -m cadebot`
│   ├── api.py                    # ← serve_model.py (FastAPI app, routes, schemas)
│   ├── models.py                 # ← serve_model.py (STT + LLM loaders)
│   └── rag/                      # ← rag/  (config, chunker, db_source,
│                                 #          dify_kb, kb_builder, prompt, retriever)
├── training/                     # ← finetune_cadebot.py, merge_model.py, eval_cadebot.py
├── scripts/                      # sync_kb.py, eval_rag.py, calibrate_threshold.py
├── pipeline/                     # standalone local voice loop (VAD→STT→LLM→TTS)
├── tests/rag/
├── eval/rag_queries.json
├── dataset/
├── knowledge_base/               # ← knowledge_Base_cadebot/
├── cadebot-lora/                 # LFS adapter only; checkpoint-45/ untracked
├── docs/
│   ├── architecture.md           # NEW — system design + mermaid diagrams
│   ├── deployment.md             # NEW — Docker hosting (the highlight)
│   ├── api-reference.md          # NEW — endpoint contracts
│   ├── rag-setup.md              # ← docs/RAG_SETUP.md (sanitized)
│   ├── model-training.md         # ← FINETUNE_EVAL.md (sanitized)
│   ├── model-training-log.md     # ← FINETUNE_LOG.md (sanitized)
│   ├── performance.md            # ← PIPELINE_ANALYSIS.md
│   ├── android-client.md         # ← INTEGRATION_DOCS.md
│   ├── voice-pipeline.md         # ← local-setup.md (sanitized)
│   └── superpowers/plans/        # unchanged
└── Cadebot_UI/                   # Android client (build.gradle.kts sanitized)
```

---

### Task 1: Package scaffolding and `rag/` → `src/cadebot/rag/`

Moves the RAG package under `src/` and deletes the three hand-rolled `sys.path` patches. `rag/config.py` computes `REPO_ROOT` by walking up two directories — after the move that resolves to `src/cadebot`, which silently breaks `KB_DIR`. `tests/rag/test_config.py::test_kb_paths_exist` is the guard that catches it.

**Files:**
- Create: `pyproject.toml`, `src/cadebot/__init__.py`
- Move: `rag/*.py` → `src/cadebot/rag/*.py` (git mv)
- Modify: `src/cadebot/rag/config.py` (REPO_ROOT + new model settings), all `from rag ...` imports in `src/cadebot/rag/*.py`, `scripts/*.py`, `tests/rag/*.py`
- Delete: `conftest.py`
- Test: `tests/rag/test_config.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: importable `cadebot.rag.config`, `cadebot.rag.chunker`, `cadebot.rag.db_source`, `cadebot.rag.dify_kb`, `cadebot.rag.kb_builder`, `cadebot.rag.prompt`, `cadebot.rag.retriever` — all public names unchanged (`Chunk`, `chunk_all_markdown`, `chunk_faq_file`, `chunk_markdown_file`, `chunk_database`, `get_menu_data`, `build_document`, `DifyKnowledgeClient`, `Retriever`, `RetrievalResult`, `RetrievedChunk`, `parse_chunk_id`, `build_context_block`, `fallback_response`, `sanitize_response`). New in `config`: `MODEL_DIR: Path`, `BASE_MODEL: str`, `STT_MODEL: str`.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "cadebot"
version = "1.0.0"
description = "Vietnamese voice assistant for coffee shops: PhoWhisper STT + fine-tuned Qwen2.5-3B + RAG over Dify/Qdrant"
requires-python = ">=3.12"
license = { text = "MIT" }

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Move the package**

```bash
mkdir -p src/cadebot
git mv rag src/cadebot/rag
touch src/cadebot/__init__.py
git add src/cadebot/__init__.py
git rm conftest.py
```

- [ ] **Step 3: Rewrite the intra-package imports**

Every `from rag import config` becomes `from cadebot.rag import config`; every `from rag.X import Y` becomes `from cadebot.rag.X import Y`. Affected: `src/cadebot/rag/{chunker,db_source,dify_kb,kb_builder,prompt,retriever}.py`, `scripts/{sync_kb,calibrate_threshold}.py`, `tests/rag/{test_chunker,test_config,test_db_source,test_kb_builder,test_prompt,test_retriever,test_sanitize}.py`.

```bash
grep -rl '^from rag\|^ *from rag\.\|import rag' --include=*.py src scripts tests \
  | xargs sed -i 's/\bfrom rag\./from cadebot.rag./g; s/\bfrom rag import/from cadebot.rag import/g'
grep -rn '\brag\.' --include=*.py src scripts tests | grep -v cadebot.rag   # must print nothing
```

- [ ] **Step 4: Fix `REPO_ROOT` and add model settings in `src/cadebot/rag/config.py`**

Replace the `REPO_ROOT` line:

```python
# config.py lives at src/cadebot/rag/config.py -> parents[3] is the repo root.
# Overridable so the container (WORKDIR /app) and the host agree.
REPO_ROOT = Path(os.getenv("CADEBOT_ROOT", Path(__file__).resolve().parents[3]))
```

Add a new section directly above `# ── KB sources ──`:

```python
# ── Model artifacts ────────────────────────────────────────────────────
# Weights are never baked into the Docker image — mounted at runtime, see
# docker-compose.yml. Paths are env-driven so the container and host agree.
MODEL_DIR = Path(os.getenv("CADEBOT_MODEL_DIR", REPO_ROOT / "cadebot-lora"))
BASE_MODEL = os.getenv("CADEBOT_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
STT_MODEL = os.getenv("CADEBOT_STT_MODEL", "vinai/PhoWhisper-large")
```

- [ ] **Step 5: Fix the two scripts that patch `sys.path`**

In `scripts/sync_kb.py` and `scripts/calibrate_threshold.py`, replace the line
`sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` with:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
```

In `scripts/eval_rag.py`, replace `sys.path.insert(0, str(ROOT))` with `sys.path.insert(0, str(ROOT / "src"))`.

- [ ] **Step 6: Run the tests — `test_chat_endpoint.py` is expected to fail here**

Run: `python3 -m pytest tests/ -q --ignore=tests/rag/test_chat_endpoint.py`
Expected: all PASS. In particular `test_kb_paths_exist` proves `REPO_ROOT` still resolves to the repo root.

Then: `python3 -m pytest tests/rag/test_chat_endpoint.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'serve_model'`. That module moves in Task 2; do not fix it here.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src scripts tests
git commit -m "refactor: move rag package under src/cadebot, drop sys.path hacks"
```

---

### Task 2: `serve_model.py` → `src/cadebot/api.py` + `src/cadebot/models.py`

Splits the 272-line server into a model-loading layer with no FastAPI dependency and an API layer that owns the runtime state. `test_chat_endpoint.py` assigns to module globals (`serve_model.retriever = fake`), so the globals must stay module-level in `api.py` and the lifespan must assign to them via `global`.

**Files:**
- Create: `src/cadebot/models.py`, `src/cadebot/api.py`, `src/cadebot/__main__.py`
- Delete: `serve_model.py`
- Modify: `tests/rag/test_chat_endpoint.py`, `Dockerfile`
- Test: `tests/rag/test_chat_endpoint.py`

**Interfaces:**
- Consumes: `cadebot.rag.config` (`MODEL_DIR`, `BASE_MODEL`, `STT_MODEL`, `GEN_TEMPERATURE`, `SCORE_THRESHOLD`, `EMBEDDING_MODEL`, `DIFY_DATASET_API_KEY`, `DIFY_DATASET_ID`), `cadebot.rag.prompt`, `cadebot.rag.retriever.Retriever`.
- Produces:
  - `cadebot.models.load_stt() -> transformers.Pipeline`
  - `cadebot.models.load_chat() -> tuple[PeftModel, AutoTokenizer]`
  - `cadebot.api.app: FastAPI`, module globals `chat_model`, `chat_tokenizer`, `stt_pipeline`, `retriever`, and constant `SYSTEM_PROMPT`.

- [ ] **Step 1: Write `src/cadebot/models.py`**

Loaders return their objects instead of mutating globals, so `api.py` owns all state.

```python
"""Model loading. No FastAPI here — keeps the API layer importable without weights."""
import torch

from cadebot.rag import config


def load_stt():
    """PhoWhisper-large for Vietnamese speech-to-text. CPU fp32."""
    from transformers import pipeline

    print(f"Loading {config.STT_MODEL} (STT)...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=config.STT_MODEL,
        device="cpu",
        torch_dtype=torch.float32,
    )
    print("✅ STT ready!")
    return pipe


def load_chat():
    """Qwen2.5-3B-Instruct base + the LoRA adapter trained on the Cadebot dataset."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    lora_path = str(config.MODEL_DIR)

    print("Loading tokenizer (chat)...")
    tokenizer = AutoTokenizer.from_pretrained(lora_path)

    print(f"Loading {config.BASE_MODEL} base model...")
    base = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        dtype=torch.float16,
        device_map="auto",
    )
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base, lora_path)
    model.eval()
    print("✅ Chat model ready!")
    return model, tokenizer
```

- [ ] **Step 2: Write `src/cadebot/api.py`**

Copy `serve_model.py` verbatim, then apply exactly these changes: drop the two loader function bodies (now imported), change the three `rag.*` imports, and make `lifespan` assign the globals.

Header and imports:

```python
"""
Cadebot API server.

Endpoints:
  POST /stt      — Speech-to-Text (PhoWhisper-large)
  POST /chat     — Answer with Qwen2.5-3B + LoRA, optionally grounded by RAG
  POST /retrieve — Retrieval only, for debugging — does not invoke the LLM
  GET  /health   — Readiness of each subsystem
"""

import json
import os
import tempfile
from contextlib import asynccontextmanager
from typing import List

import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from cadebot import models
from cadebot.rag import config as rag_config
from cadebot.rag.prompt import build_context_block, fallback_response, sanitize_response
from cadebot.rag.retriever import Retriever
```

Keep `SYSTEM_PROMPT`, `load_retriever()`, `HistoryItem`, `ChatRequest`, `RetrieveRequest` and all four routes byte-identical to `serve_model.py`. Replace only the lifespan:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    global stt_pipeline, chat_model, chat_tokenizer
    stt_pipeline = models.load_stt()
    chat_model, chat_tokenizer = models.load_chat()
    load_retriever()
    yield
```

Delete the trailing `if __name__ == "__main__": uvicorn.run(...)` block and the `import uvicorn` / `import numpy as np` lines (`numpy` was never used; `uvicorn` moves to `__main__.py`).

- [ ] **Step 3: Write `src/cadebot/__main__.py`**

```python
"""Entrypoint: `python -m cadebot`. Port is env-driven for container flexibility."""
import os

import uvicorn

from cadebot.api import app


def main() -> None:
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Delete the old module**

```bash
git rm serve_model.py
```

- [ ] **Step 5: Update `tests/rag/test_chat_endpoint.py`**

Change the two imports and every `serve_model.` reference:

```python
from cadebot import api
from cadebot.rag.retriever import RetrievalResult, RetrievedChunk
```

```bash
sed -i 's/^import serve_model$/from cadebot import api/; s/\bserve_model\./api./g; \
        s/^from rag\.retriever import/from cadebot.rag.retriever import/' \
  tests/rag/test_chat_endpoint.py
grep -n 'serve_model' tests/rag/test_chat_endpoint.py   # must print nothing
```

- [ ] **Step 6: Run the full suite**

Run: `python3 -m pytest tests/ -q`
Expected: PASS, all files including `test_chat_endpoint.py`. This proves the out-of-scope hard-block still short-circuits before the LLM.

- [ ] **Step 7: Update the `Dockerfile` COPY and CMD**

Replace the two `COPY` lines after the pip install and the `CMD`:

```dockerfile
COPY src/ ./src/

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["python3", "-m", "cadebot"]
```

- [ ] **Step 8: Verify the image still builds and the app imports**

Run: `docker compose build cadebot-api`
Expected: build succeeds.

Run: `docker compose run --rm --no-deps cadebot-api python3 -c "from cadebot.api import app; print(len(app.routes))"`
Expected: prints a number ≥ 4 (four routes plus FastAPI defaults). No `ModuleNotFoundError`.

- [ ] **Step 9: Commit**

```bash
git add src tests Dockerfile
git commit -m "refactor: split serve_model into cadebot.api and cadebot.models"
```

---

### Task 3: Directory hygiene — rename KB, group training scripts, untrack artifacts

Renames the mixed-case `knowledge_Base_cadebot/`, moves the three training scripts out of the root, and stops tracking the 914 MB optimizer state and the four binary archives. No history rewrite: `git rm --cached` leaves the blobs in history but removes them from `HEAD`, so fresh clones stop paying for them and LFS quota stops growing.

**Files:**
- Move: `knowledge_Base_cadebot/` → `knowledge_base/`; `finetune_cadebot.py`, `merge_model.py`, `eval_cadebot.py` → `training/`
- Modify: `src/cadebot/rag/config.py` (`KB_DIR`), `.gitignore`, `.dockerignore`, `.gitattributes`
- Untrack: `cadebot-lora/checkpoint-45/`, `cadebot-debug.apk`, `Cadebot_UI_Source.zip`, `Cadebot_UI_v2_STT.zip`, `Cadebot_UI/*.docx`
- Test: `tests/rag/test_config.py`, `tests/rag/test_db_source.py`

**Interfaces:**
- Consumes: `cadebot.rag.config` from Task 1.
- Produces: `config.KB_DIR == REPO_ROOT / "knowledge_base"`. All chunk IDs are unchanged (`menu:*`, `promo:*`, `faq:db_*`, `faq:md_*`, `doc:*`) — the KB in Dify does **not** need re-syncing.

- [ ] **Step 1: Rename the knowledge base directory**

```bash
git mv knowledge_Base_cadebot knowledge_base
```

- [ ] **Step 2: Point `config.KB_DIR` at the new name**

In `src/cadebot/rag/config.py`: `KB_DIR = REPO_ROOT / "knowledge_base"`.

Then update the remaining references:

```bash
grep -rn 'knowledge_Base_cadebot' --include=*.py --include=*.md --include=Makefile \
  --include=.dockerignore --include=.gitignore . | grep -v '^./.git/'
```

Fix each hit (expected: `.gitignore`, `.dockerignore`, `docs/RAG_SETUP.md`, `docs/superpowers/plans/2026-07-29-bge-m3-rag-dify.md`).

- [ ] **Step 3: Group the training scripts**

```bash
mkdir -p training
git mv finetune_cadebot.py merge_model.py eval_cadebot.py training/
```

- [ ] **Step 4: Untrack heavy artifacts (files stay on disk)**

```bash
git rm -r --cached cadebot-lora/checkpoint-45
git rm --cached cadebot-debug.apk Cadebot_UI_Source.zip Cadebot_UI_v2_STT.zip
git rm --cached Cadebot_UI/*.docx
git status --short | head -40   # expect D entries, files still present via `ls`
```

- [ ] **Step 5: Rewrite `.gitignore`**

```gitignore
# Secrets
.env

# Python
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
build/
dist/
.venv/

# Build artifacts — distributed via GitHub Releases, not Git
*.apk
*.zip
*.docx

# Training intermediates — reproducible from training/finetune_cadebot.py
cadebot-lora/checkpoint-*/
cadebot-merged/
*.gguf

# Generated reports
PIPELINE_ANALYSIS.html
PIPELINE_ANALYSIS_files/
```

- [ ] **Step 6: Rewrite `.dockerignore` for the `src/` layout**

The image needs `requirements.txt` and `src/` and nothing else. Deny-by-default is safer than the current allow-everything-then-subtract list:

```dockerignore
# Deny everything, then re-admit only what the image needs.
*
!requirements.txt
!src/
**/__pycache__
**/*.pyc
```

- [ ] **Step 7: Trim `.gitattributes` to the adapter that is still tracked**

```gitattributes
cadebot-lora/*.safetensors filter=lfs diff=lfs merge=lfs -text
```

- [ ] **Step 8: Verify tests and the Docker build context**

Run: `python3 -m pytest tests/ -q`
Expected: PASS — `test_kb_paths_exist` confirms `knowledge_base/demo_cafe.db` resolves, `test_db_source.py` confirms the DB still chunks.

Run: `docker compose build cadebot-api`
Expected: build succeeds and the reported build context is a few hundred KB, not gigabytes.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "chore: rename knowledge base dir, group training scripts, untrack build artifacts"
```

---

### Task 4: Remove personal information and add `.env.example`

Every tracked file that names someone's home directory, internal team, or personal domain gets a placeholder or an environment variable. The Android default API URL is the important one — it currently ships the owner's personal tunnel hostname inside every built APK.

**Files:**
- Modify: `Cadebot_UI/app/build.gradle.kts:27`, `Makefile`, `Modelfile`, `training/{finetune_cadebot,merge_model,eval_cadebot}.py`, `FINETUNE_EVAL.md`, `FINETUNE_LOG.md`, `local-setup.md`, `docs/RAG_SETUP.md`, `docs/superpowers/plans/2026-07-29-bge-m3-rag-dify.md`, `docker-compose.yml`
- Create: `.env.example`
- Test: repo-wide grep for the forbidden strings

**Interfaces:**
- Consumes: nothing.
- Produces: `.env.example` documenting `DIFY_BASE_URL`, `DIFY_DATASET_API_KEY`, `DIFY_DATASET_ID`, `SCORE_THRESHOLD`, `TOP_K`, `GEN_TEMPERATURE`, `RETRIEVAL_TIMEOUT`, `SYNC_TIMEOUT`, `CADEBOT_MODEL_DIR`, `CADEBOT_BASE_MODEL`, `CADEBOT_STT_MODEL`.

- [ ] **Step 1: Create `.env.example`**

```bash
# Copy to .env and fill in. .env is gitignored — never commit real keys.

# ── Dify (RAG) ────────────────────────────────────────────────────────
# Host:      http://localhost/v1
# Container: http://nginx/v1   (service name inside the Dify compose network)
DIFY_BASE_URL=http://localhost/v1
# Knowledge → API Access → API Key. NOT the App API key — that returns 401.
DIFY_DATASET_API_KEY=dataset-xxxxxxxxxxxxxxxxxxxxxxxx
DIFY_DATASET_ID=00000000-0000-0000-0000-000000000000

# ── Retrieval tuning ──────────────────────────────────────────────────
# 0.51 was calibrated against the real KB; see docs/rag-setup.md.
SCORE_THRESHOLD=0.51
TOP_K=3
RETRIEVAL_TIMEOUT=15
SYNC_TIMEOUT=60

# ── Generation ────────────────────────────────────────────────────────
GEN_TEMPERATURE=0.2

# ── Model artifacts (optional — sensible defaults in cadebot/rag/config.py) ──
# CADEBOT_MODEL_DIR=/app/cadebot-lora
# CADEBOT_BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
# CADEBOT_STT_MODEL=vinai/PhoWhisper-large
```

- [ ] **Step 2: Remove the personal domain from the Android client**

In `Cadebot_UI/app/build.gradle.kts:27`, change the fallback from the personal hostname to the Android emulator loopback:

```kotlin
buildConfigField("String", "CADEBOT_API_URL", "\"${localProps.getProperty("cadebot.api.url", "http://10.0.2.2:8000")}\"")
```

`10.0.2.2` is the emulator's alias for the host machine, so the default now points at a locally running server. Real deployments set `cadebot.api.url` in `local.properties`, which is already gitignored by the Android project.

- [ ] **Step 3: Remove the personal domain from the `Makefile` health target**

```makefile
health: ## Probe /health locally, and at PUBLIC_URL if it is set
	@echo "── localhost:8000 ──"
	@curl -s http://localhost:8000/health || echo "(no response)"
	@echo ""
	@if [ -n "$$PUBLIC_URL" ]; then \
		echo "── $$PUBLIC_URL ──"; \
		curl -s "$$PUBLIC_URL/health" || echo "(no response)"; echo ""; \
	else \
		echo "(set PUBLIC_URL=https://your-tunnel.example.com to also probe the public endpoint)"; \
	fi
```

While in this file, translate the remaining Vietnamese `##` help comments to English — they are user-facing via `make help`.

- [ ] **Step 4: Replace absolute paths in the training scripts**

`training/eval_cadebot.py` and `training/merge_model.py` hardcode `/home/team3/...`. Make them repo-relative with env overrides. In `training/eval_cadebot.py`:

```python
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = os.getenv("CADEBOT_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
LORA_PATH = os.getenv("CADEBOT_MODEL_DIR", str(ROOT / "cadebot-lora"))
VAL_DATA = str(ROOT / "dataset" / "val.jsonl")
```

In `training/merge_model.py`:

```python
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = os.getenv("CADEBOT_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
LORA = os.getenv("CADEBOT_MODEL_DIR", str(ROOT / "cadebot-lora"))
OUT = str(ROOT / "cadebot-merged")
```

In `training/finetune_cadebot.py`, the paths appear only in the module docstring's usage example — rewrite it to `python3 training/finetune_cadebot.py` with a note that a CUDA-capable environment is required.

- [ ] **Step 5: Make `Modelfile` relative**

```
FROM ./cadebot-viva.gguf
```

Add a leading comment line: `# Build with: ollama create cadebot-viva -f Modelfile  (run from the repo root)`.

- [ ] **Step 6: Sanitize the Markdown docs**

```bash
grep -rlE '/home/team3|/home/ncd|hrc2026-team3|Isaac-GR00T|duybao|tdbao-brian' \
  --include=*.md . | grep -v '^./.git/'
```

For each file, replace:
- `/home/team3/Isaac-GR00T/.venv/bin/python` → `$VENV/bin/python` (add one line noting `$VENV` is the training virtualenv)
- `/home/team3/python_headers/python3.12` → `$PYTHON_INCLUDE_DIR`
- `/home/team3/hrc2026-team3/qwen2.5-3b-base` and `hrc2026-team3/` → `$MODEL_CACHE/qwen2.5-3b-base`
- `/home/team3/Qwen2.5-3B-fine-tuned` and `/home/ncd/learnspaces/Qwen2.5-3B-fine-tuned` → `$REPO_ROOT`
- `/home/ncd/.pyenv/versions/3.11.8/bin/python3` → `$PYENV_ROOT/versions/3.11.8/bin/python3`
- `/home/ncd/learnspaces/Cadebot/pipeline` → `$REPO_ROOT/pipeline`
- `User=ncd` in the systemd unit → `User=<your-user>`

- [ ] **Step 7: Translate the `docker-compose.yml` comments to English**

The comments explain real constraints and belong in the reader's language. Keep every key and value identical; translate only the `#` lines, and generalize the Dify path comment from a home directory to `<path-to>/dify/docker`.

- [ ] **Step 8: Verify nothing leaks**

Run:

```bash
git grep -nIE '/home/team3|/home/ncd|hrc2026-team3|Isaac-GR00T|duybao|tdbao-brian\.work' \
  -- . ':!cadebot-lora' ':!dataset'
```

Expected: **no output**. (`cadebot-lora/adapter_config.json` records the base-model path baked in at training time and `dataset/*.jsonl` are tokenizer/vocab artifacts — excluded because they are data, not documentation; note this exclusion in the commit message.)

Run: `python3 -m pytest tests/ -q`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "chore: remove personal paths and domains from tracked files, add .env.example"
```

---

### Task 5: `docs/architecture.md`

The system-design document a reviewer opens second. Explains the request paths and — importantly — the decisions that look strange without context.

**Files:**
- Create: `docs/architecture.md`

**Interfaces:**
- Consumes: the final layout from Tasks 1–3; all paths referenced must be the post-move ones.
- Produces: anchors `#components`, `#request-flows`, `#key-design-decisions` for the README to link to.

- [ ] **Step 1: Write the component overview**

Sections, in order:

1. **System context** — a mermaid `flowchart LR` showing: Android tablet client → Cloudflare Tunnel → `cadebot-api` (FastAPI, port 8000) → three dependencies: PhoWhisper-large (in-process), Qwen2.5-3B + LoRA (in-process), and the Dify stack (`nginx` → `api` → Qdrant, plus Ollama serving `bge-m3` through the `docker0` bridge).
2. **Components** — a table: component, where it lives (`src/cadebot/api.py`, `src/cadebot/models.py`, `src/cadebot/rag/`, `scripts/`, `Cadebot_UI/`, `pipeline/`), and one-line responsibility.
3. **The RAG package** — walk `chunker.py` (markdown → stable chunk IDs), `db_source.py` (SQLite → `menu:*` / `promo:*` / `faq:db_*`), `kb_builder.py` (join with `\n---\n`, refuse chunks containing a bare `---`), `dify_kb.py` (idempotent upsert by document name), `retriever.py` (query, parse the leading `[chunk_id]`, threshold locally), `prompt.py` (context block, fixed FALLBACK, `sourceIds` sanitizing).

- [ ] **Step 2: Write the request flows**

Two mermaid `sequenceDiagram` blocks:

- **Voice turn:** Client records → `POST /stt` (ffmpeg → 16 kHz mono WAV → PhoWhisper; `return_timestamps` when audio exceeds 30 s) → text → `POST /chat`.
- **Grounded answer:** `POST /chat` with `use_rag: true` → `Retriever.retrieve()` → Dify `/datasets/{id}/retrieve` → if `top_score < 0.51`, return the canned FALLBACK **without invoking the LLM**; otherwise build the context block, generate, then strip any `sourceIds` the model invented.

- [ ] **Step 3: Write "Key design decisions"**

One subsection each, stating the decision, the reason, and the evidence — all of it already established in the codebase, so cite the source:

| Decision | Why | Source |
|---|---|---|
| Score threshold 0.51 | F1 0.968 / recall 1.000 over 30 calibration queries; distributions overlap by only 0.014 | `src/cadebot/rag/config.py`, `docs/rag-setup.md` |
| Hard-block out-of-scope before the LLM | 0.096 s vs ~142 s, and removes the fabrication path entirely | `src/cadebot/api.py` |
| Chunk IDs embedded as a `[id]` first line | Dify's API returns segment text, not our metadata — the ID has to survive inside the content | `src/cadebot/rag/chunker.py` |
| `max_tokens` 2000, not 500 | At 500, Dify re-split Vietnamese chunks and 13 of 46 segments lost their `[chunk_id]` line | `src/cadebot/rag/config.py` |
| Markdown horizontal rules stripped from chunks | A `---` inside a chunk collides with the `\n---\n` separator and shifts every later segment | `src/cadebot/rag/kb_builder.py` |
| Generation temperature 0.2 | At 0.7 the model attached "best seller" to items the KB never described that way | `src/cadebot/rag/config.py` |
| `use_rag` defaults to `false` | RAG turns take ~190 s; Cloudflare's edge proxy times out at 100 s (error 524) | `src/cadebot/api.py` |
| `semantic_search`, not `hybrid_search` | Measured identical (Dify ignores `weights` when reranking is off); `full_text_search` scores 0 on a High Quality index | `docs/rag-setup.md` |

- [ ] **Step 4: Write the "Known limitations" section**

State them plainly — this is what makes the doc read as engineering rather than marketing:
- ~142 s per grounded answer on CPU; the fix is INT4 quantization or a GPU, not more RAG.
- RAG blocks fabricated *numbers*, not fabricated *attributes* — the model still mixes properties across retrieved chunks.
- One of 15 out-of-scope calibration queries scores above threshold, and the label is arguably wrong (the KB does contain a hotline).
- Single-process server, no request queueing; concurrent `/chat` calls serialize behind the GIL and the model.

- [ ] **Step 5: Verify the mermaid renders and links resolve**

Run: `grep -c '```mermaid' docs/architecture.md`
Expected: `3`.

Check every relative link target exists:

```bash
grep -oE '\]\(([^)]+\.md[^)]*)\)' docs/architecture.md | sed 's/](//; s/)//; s/#.*//' \
  | while read -r p; do [ -e "docs/$p" ] || [ -e "$p" ] || echo "BROKEN: $p"; done
```
Expected: no `BROKEN` lines.

- [ ] **Step 6: Commit**

```bash
git add docs/architecture.md
git commit -m "docs: add architecture overview with request flows and design decisions"
```

---

### Task 6: `docs/deployment.md` and `docs/api-reference.md`

The deployment doc is the centerpiece the user specifically asked for: how this is actually hosted with Docker. Everything in it must be true of the committed `Dockerfile`, `docker-compose.yml`, and `Makefile` — no aspirational infrastructure.

**Files:**
- Create: `docs/deployment.md`, `docs/api-reference.md`

**Interfaces:**
- Consumes: `Dockerfile`, `docker-compose.yml`, `Makefile`, `.env.example` as they stand after Task 4.
- Produces: anchors `#topology`, `#quick-start`, `#operations`, `#troubleshooting` for the README.

- [ ] **Step 1: Write the deployment topology**

A mermaid `flowchart TB` with three subgraphs, matching the real setup:

- **Public** — Android client → `https://<your-tunnel-domain>`
- **Host machine** — `cloudflared` (systemd) → `localhost:8000`; Ollama bound to `127.0.0.1:11434` plus the `ollama_docker_bridge.py` relay on `172.17.0.1:11434`
- **Docker** — the `cadebot-api` container on two networks (`default` and the external `docker_default`), reaching Dify's `nginx` by service name; Dify's own stack (`nginx`, `api`, `plugin_daemon`, `db_postgres`, `qdrant`)

Follow with a table of every published port and why it is or is not exposed: only `8000` is published to the host; Qdrant, Postgres and the Dify API stay internal.

- [ ] **Step 2: Write "Why the image is built this way"**

Four annotated points, each quoting the relevant `Dockerfile` / `docker-compose.yml` line:

1. **CPU-only torch wheel** (`--index-url https://download.pytorch.org/whl/cpu`) — the default PyPI wheel pulls ~3 GB of CUDA libraries that are dead weight on a GPU-less host.
2. **Weights are not baked in** — `cadebot-lora/` is mounted read-only and the HuggingFace cache is a named volume (`hf_cache`), so the ~6 GB of base weights download once and survive rebuilds. Baking them would make the image unpushable and every rebuild a re-download.
3. **Two networks** — `dify_network` is declared `external: true` pointing at `docker_default`, letting the API address Dify as `http://nginx/v1`. Inside a container, `localhost` is the container itself, which is why `DIFY_BASE_URL` is overridden in `docker-compose.yml` rather than taken from `.env`.
4. **`ffmpeg` + `libsndfile1`** — `/stt` shells out to `ffmpeg` to normalize any client format to 16 kHz mono WAV; `soundfile` needs `libsndfile1`.

- [ ] **Step 3: Write the quick start**

Numbered, copy-pasteable, in dependency order — Dify stack must be up first because the compose file joins its network:

```bash
# 1. Prerequisites: Docker + Compose, ~20 GB free disk, and Ollama with bge-m3
ollama pull bge-m3

# 2. Bring up the Dify stack (see docs/rag-setup.md for first-time setup)
cd <path-to>/dify/docker && docker compose --profile postgresql --profile qdrant up -d

# 3. Configure Cadebot
cd <path-to>/Qwen2.5-3B-fine-tuned
cp .env.example .env && $EDITOR .env      # fill in the Dify dataset key + id

# 4. Sync the knowledge base and start the API
set -a && source .env && set +a
python3 scripts/sync_kb.py                # 69 chunks across 2 Dify documents
make up                                   # build + run in the background
make logs                                 # wait for "✅ Chat model ready!"
make health
```

Call out that first boot downloads ~6 GB of weights and takes several minutes; subsequent boots reuse `hf_cache`.

- [ ] **Step 4: Write the operations and exposure sections**

- **`make` target reference** — a table of `up`, `down`, `restart`, `build`, `logs`, `ps`, `health`, `clean`, `tunnel-status`, `tunnel-restart`, `tunnel-logs`, each with what it does and when to reach for it. Warn that `clean` runs `docker compose down -v` and destroys `hf_cache`, forcing a full re-download.
- **Public exposure** — describe the Cloudflare Tunnel approach generically (`cloudflared` as a systemd service, ingress mapping a hostname to `http://localhost:8000`), with placeholder hostnames only. State the trade-off explicitly: no inbound ports opened, TLS terminated at the edge, **but a hard 100 s response limit**, which is exactly why `use_rag` defaults to `false`.
- **Security notes** — CORS is currently `allow_origins=["*"]` and there is no authentication on `/chat`; acceptable for a demo behind a private hostname, and the first thing to change for production.

- [ ] **Step 5: Write the troubleshooting table**

Symptom → cause → fix, drawn from problems already recorded in the repo:

| Symptom | Cause | Fix |
|---|---|---|
| `docker compose up` fails: network `docker_default` not found | The Dify stack was never started | Start Dify first (quick start step 2) |
| `plugin_daemon` / `nginx` crash-looping | Started before Postgres was healthy | `docker restart docker-plugin_daemon-1 docker-nginx-1` |
| `/health` shows `rag_ready: false` | `DIFY_DATASET_API_KEY` / `DIFY_DATASET_ID` unset or wrong | Check `.env`; confirm it is the **Dataset** key, not the App key |
| Dify returns 401 | App API key used instead of the Dataset API key | Knowledge → API Access → create a key |
| Client gets HTTP 524 | Response exceeded Cloudflare's 100 s limit | Keep `use_rag: false`, or move generation to a GPU |
| First boot appears hung | Downloading ~6 GB of weights | `make logs`; wait for `✅ Chat model ready!` |
| Ollama unreachable from Dify | Ollama listens on `127.0.0.1` only | Run `knowledge_base/ollama_docker_bridge.py` (see `docs/rag-setup.md` §3) |

- [ ] **Step 6: Write `docs/api-reference.md`**

One section per endpoint, each with method + path, request schema, a `curl` example, and a real response body. Source the field names from `src/cadebot/api.py` — do not invent any.

- `GET /health` → `{status, stt_ready, chat_ready, rag_ready, embedding_model, score_threshold}`
- `POST /stt` → multipart `file`; returns `{text}`. Note the >30 s long-form path.
- `POST /chat` → `{message, history[], use_rag=false, top_k=null}`; returns `{response: <JSON string>, retrieval?: {in_scope, top_score, threshold, sourceIds}}`. Document the inner JSON schema the model emits: `intent` (one of `MENU_QA|RECOMMENDATION|ADD_TO_CART_DRAFT|PROMOTION_QA|CALL_STAFF|FALLBACK`), `confidence`, `answerText`, `spokenText`, `recommendedItems`, `draftCartItems`, `requiresHumanSupport`, `sourceIds`. State clearly that `response` is a **string containing JSON**, not a nested object — clients must parse it.
- `POST /retrieve` → `{query, top_k}`; returns `{in_scope, top_score, threshold, chunks[{chunk_id, score, text}]}`. Flag it as a debugging endpoint that skips the LLM.

Add a latency table (out-of-scope 0.096 s; in-scope with RAG ~142 s; without RAG ~95 s on CPU) and note that `history` is truncated to the last 8 turns.

- [ ] **Step 7: Verify every documented value against the code**

```bash
grep -n 'allow_origins\|max_new_tokens\|history\[-8:\]\|port=' src/cadebot/api.py src/cadebot/__main__.py
grep -n 'SCORE_THRESHOLD\|TOP_K\|GEN_TEMPERATURE\|EMBEDDING_MODEL' src/cadebot/rag/config.py
grep -n 'make\|docker compose' Makefile | head -20
```

Reconcile any mismatch by fixing the **doc**, never the code.

- [ ] **Step 8: Commit**

```bash
git add docs/deployment.md docs/api-reference.md
git commit -m "docs: add deployment topology, operations runbook and API reference"
```

---

### Task 7: New `README.md`, `LICENSE`, `CONTRIBUTING.md`, and doc relocation

The front page. It must communicate what the system is, show it working, and route to the deep docs — without becoming a deep doc itself. Target 150–200 lines.

**Files:**
- Create: `README.md` (replacing the one-line file), `LICENSE`, `CONTRIBUTING.md`
- Move: `FINETUNE_EVAL.md` → `docs/model-training.md`; `FINETUNE_LOG.md` → `docs/model-training-log.md`; `PIPELINE_ANALYSIS.md` → `docs/performance.md`; `INTEGRATION_DOCS.md` → `docs/android-client.md`; `local-setup.md` → `docs/voice-pipeline.md`; `docs/RAG_SETUP.md` → `docs/rag-setup.md`

**Interfaces:**
- Consumes: `docs/architecture.md`, `docs/deployment.md`, `docs/api-reference.md` from Tasks 5–6.
- Produces: the repo front page. No code depends on it.

- [ ] **Step 1: Move and rename the remaining docs**

```bash
git mv FINETUNE_EVAL.md docs/model-training.md
git mv FINETUNE_LOG.md docs/model-training-log.md
git mv PIPELINE_ANALYSIS.md docs/performance.md
git mv INTEGRATION_DOCS.md docs/android-client.md
git mv local-setup.md docs/voice-pipeline.md
git mv docs/RAG_SETUP.md docs/rag-setup.md
```

Then repair inbound references:

```bash
git grep -nE 'RAG_SETUP\.md|FINETUNE_(EVAL|LOG)\.md|PIPELINE_ANALYSIS\.md|INTEGRATION_DOCS\.md|local-setup\.md'
```

Expected hits in `src/cadebot/api.py`, `src/cadebot/rag/{config,dify_kb}.py` and several docs — update each to the new path.

- [ ] **Step 2: Write the README opening**

```markdown
# Cadebot

A Vietnamese-language voice assistant for coffee shops. A tablet at the table
listens, transcribes, answers questions about the menu, and drafts an order —
running on a fine-tuned 3B model that is grounded in the shop's own knowledge
base, so it declines to answer rather than inventing a price.

**Stack:** FastAPI · PhoWhisper-large (STT) · Qwen2.5-3B-Instruct + LoRA ·
BGE-M3 retrieval over Dify + Qdrant · Jetpack Compose (Android) · Docker Compose

| | |
|---|---|
| Out-of-scope questions blocked | 14/15, in **0.096 s** — without invoking the LLM |
| Retrieval F1 / recall | 0.968 / 1.000 (30 calibration queries, threshold 0.51) |
| Answers carrying source IDs | 69/69 knowledge segments keep their provenance tag |
| Fine-tuning | LoRA over 144 examples, 6 intents |
```

- [ ] **Step 3: Write the architecture-at-a-glance section**

A single mermaid `flowchart LR` — deliberately simpler than the one in `docs/architecture.md`: Android → Tunnel → FastAPI → {STT, LLM, Retriever → Dify/Qdrant}. One paragraph beneath it, then "→ Full detail in [docs/architecture.md](docs/architecture.md)".

- [ ] **Step 4: Write the quick start and repository map**

- **Run it** — two paths: Docker (`cp .env.example .env` → `make up` → `make health`, linking to `docs/deployment.md`) and local development (`pip install -r requirements.txt` → `python3 -m pytest tests/ -q` → `python3 -m cadebot`).
- **Repository map** — a table of the top-level directories from the target layout, one line each. This is what tells a visitor the repo is organized on purpose.
- **Documentation index** — a table linking all nine `docs/*.md` files with a one-line description each, marking which are English and which are Vietnamese.

- [ ] **Step 5: Write the "How grounding works" section**

Four sentences plus one short example showing an out-of-scope question returning `intent: FALLBACK` with `sourceIds: []`, and an in-scope one returning a price with `sourceIds: ["menu:VR_PEACH_TEA"]`. This is the project's most distinctive engineering — show it, don't just assert it.

Close the README with **Limitations** (three bullets lifted from `docs/architecture.md`: CPU latency, attribute-level fabrication, single-process serving), **License**, and an acknowledgements line for Qwen, VinAI PhoWhisper, BAAI BGE-M3, and Dify.

- [ ] **Step 6: Add `LICENSE` (MIT) and `CONTRIBUTING.md`**

`LICENSE`: standard MIT text, `Copyright (c) 2026 Cadebot contributors` — no personal name or email.

`CONTRIBUTING.md`: development setup (Python 3.12, `pip install -r requirements.txt`, `python3 -m pytest tests/ -q`), project conventions (source under `src/cadebot/`, tests mirror the package under `tests/rag/`, Conventional Commits as already used in this repo's history), and a short PR checklist (tests pass, no secrets, no absolute paths, docs updated alongside behavior).

- [ ] **Step 7: Verify no broken links and the docs actually moved**

```bash
ls docs/                                        # 9 .md files + superpowers/
ls *.md                                         # README.md and CONTRIBUTING.md only
grep -oE '\]\(([^)h][^)]*)\)' README.md | sed 's/](//; s/)//; s/#.*//' \
  | while read -r p; do [ -e "$p" ] || echo "BROKEN: $p"; done
```
Expected: no `BROKEN` lines.

Run: `python3 -m pytest tests/ -q`
Expected: PASS (the doc-path strings in `config.py` and `dify_kb.py` are only messages, but the suite confirms nothing was mangled).

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "docs: professional README, license, contributing guide; consolidate docs under docs/"
```

---

### Task 8: GitHub metadata — CI, issue templates, PR template

Makes the repo look and behave like a maintained project: a green check on every push, and structured intake for issues and PRs.

**Files:**
- Create: `.github/workflows/ci.yml`, `.github/ISSUE_TEMPLATE/bug_report.yml`, `.github/ISSUE_TEMPLATE/feature_request.yml`, `.github/ISSUE_TEMPLATE/config.yml`, `.github/pull_request_template.md`

**Interfaces:**
- Consumes: `pyproject.toml` (`pythonpath = ["src"]`), `requirements.txt`.
- Produces: a `test` job named `Tests (Python 3.12)`, referenced by the README badge.

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

The CPU torch wheel is installed from PyTorch's index — the same way the Dockerfile does — otherwise the runner pulls ~3 GB of CUDA libraries.

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    name: Tests (Python 3.12)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: false   # the LoRA adapter is not needed to run the test suite

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
          pip install -r requirements.txt

      - name: Run tests
        run: python -m pytest tests/ -q

  secrets-scan:
    name: No secrets or absolute paths
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          lfs: false

      - name: Grep for forbidden strings
        run: |
          if git grep -nIE '/home/[a-z0-9_-]+/|dataset-[A-Za-z0-9]{20,}|tdbao-brian\.work' \
               -- . ':!cadebot-lora' ':!dataset' ':!docs/superpowers'; then
            echo "::error::Found an absolute home path or a leaked key above."
            exit 1
          fi
```

- [ ] **Step 2: Verify the workflow locally before pushing**

The `secrets-scan` job is just a grep — run its exact command:

```bash
git grep -nIE '/home/[a-z0-9_-]+/|dataset-[A-Za-z0-9]{20,}|tdbao-brian\.work' \
  -- . ':!cadebot-lora' ':!dataset' ':!docs/superpowers'
```
Expected: no output (exit code 1 from `git grep`, which the workflow reads as "clean").

Validate the YAML parses:

```bash
python3 -c "import yaml,sys; [yaml.safe_load(open(f)) for f in sys.argv[1:]]; print('ok')" \
  .github/workflows/ci.yml
```
Expected: `ok`.

- [ ] **Step 3: Write the issue templates**

`.github/ISSUE_TEMPLATE/bug_report.yml` — a GitHub form with required fields: what happened, expected behavior, reproduction steps, and a dropdown for **Component** (`API server`, `RAG / retrieval`, `STT`, `Model / fine-tuning`, `Android client`, `Docker / deployment`, `Docs`). Include a textarea prompting for the output of `curl -s localhost:8000/health`, since that answers most deployment questions immediately.

`.github/ISSUE_TEMPLATE/feature_request.yml` — problem, proposed solution, alternatives considered.

`.github/ISSUE_TEMPLATE/config.yml`:

```yaml
blank_issues_enabled: false
contact_links:
  - name: Deployment questions
    url: https://github.com/tduybao7605/Qwen2.5-3B-fine-tuned/blob/main/docs/deployment.md
    about: Check the deployment guide and its troubleshooting table first.
```

- [ ] **Step 4: Write `.github/pull_request_template.md`**

Summary, type of change (checkboxes), how it was tested, and a checklist mirroring `CONTRIBUTING.md`: `python3 -m pytest tests/ -q` passes; no secrets or absolute paths; docs updated if behavior changed; `SCORE_THRESHOLD` / prompt changes are accompanied by re-run calibration numbers.

- [ ] **Step 5: Add the CI badge to the README**

Insert directly beneath the `# Cadebot` heading:

```markdown
[![CI](https://github.com/tduybao7605/Qwen2.5-3B-fine-tuned/actions/workflows/ci.yml/badge.svg)](https://github.com/tduybao7605/Qwen2.5-3B-fine-tuned/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
```

- [ ] **Step 6: Commit**

```bash
git add .github README.md
git commit -m "ci: add test and secret-scan workflows, issue and PR templates"
```

---

## Verification

Run all of the following from the repo root after Task 8. Every one must pass before the branch is considered done.

**1. Test suite**

```bash
python3 -m pytest tests/ -q
```
Expected: all tests pass, zero errors, zero warnings about import mode.

**2. Package imports cleanly from anywhere** — proves the `sys.path` hacks are genuinely gone:

```bash
cd /tmp && PYTHONPATH=<repo>/src python3 -c "
from cadebot.api import app
from cadebot.rag import config
print('routes:', len(app.routes))
print('kb dir exists:', config.KB_DIR.is_dir())
print('threshold:', config.SCORE_THRESHOLD)
"
```
Expected: routes ≥ 4, `kb dir exists: True`, `threshold: 0.51`.

**3. Scripts still run**

```bash
python3 scripts/sync_kb.py --dry-run
```
Expected: reports 34 markdown + 35 database = **69 chunks**, no network calls, no import errors.

**4. Docker image builds and serves**

```bash
docker compose build cadebot-api
docker compose up -d
make logs        # wait for "✅ Chat model ready!"  (several minutes on first boot)
curl -s localhost:8000/health | python3 -m json.tool
```
Expected: `status: ok`, `stt_ready: true`, `chat_ready: true`, `embedding_model: bge-m3`, `score_threshold: 0.51`.

**5. End-to-end behavior is unchanged**

```bash
set -a && source .env && set +a
python3 scripts/eval_rag.py --fast
```
Expected: the same out-of-scope block rate as before the refactor (14/15 at threshold 0.51). Any change here means the refactor altered behavior and must be fixed.

**6. No personal information anywhere in the tree**

```bash
git grep -nIE '/home/[a-z0-9_-]+/|hrc2026-team3|Isaac-GR00T|duybao|tdbao-brian\.work|ncongduy' \
  -- . ':!cadebot-lora' ':!dataset'
```
Expected: no output.

**7. Documentation links all resolve**

```bash
for f in README.md docs/*.md; do
  grep -oE '\]\(([^)h][^)]*)\)' "$f" | sed 's/](//; s/)//; s/#.*//' | while read -r p; do
    case "$f" in docs/*) base=docs;; *) base=.;; esac
    [ -e "$base/$p" ] || [ -e "$p" ] || echo "BROKEN in $f: $p"
  done
done
```
Expected: no `BROKEN` lines.

**8. Visual check** — render `README.md`, `docs/architecture.md`, and `docs/deployment.md` on GitHub (or with `gh markdown preview` / any Markdown viewer) and confirm the four mermaid diagrams render, tables are aligned, and no Vietnamese text leaked into the English documents.

## Out of Scope

Deliberately not done, and why:

- **Rewriting Git history.** The 914 MB `optimizer.pt` stays in past commits. Removing it requires `git filter-repo` and a force-push, which breaks every existing clone and changes every SHA. Task 3 stops the bleeding instead.
- **Publishing the APK to GitHub Releases and the adapter to HuggingFace.** Both need account credentials and are the owner's call; Task 3 untracks the files, and the README should link to a Release once one exists.
- **Dependency upgrades.** Pins stay exactly as measured on the deployment machine.
- **Latency work.** INT4 quantization or GPU serving would fix the ~142 s grounded-answer time, but that is a separate engineering project, documented as a known limitation rather than fixed here.
