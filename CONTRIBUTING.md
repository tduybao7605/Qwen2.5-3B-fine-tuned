# Contributing

Thanks for working on Cadebot. This file covers the development setup, the
conventions the codebase already follows, and what a reviewable pull request
looks like.

## Development setup

```bash
git clone https://github.com/tduybao7605/Qwen2.5-3B-fine-tuned.git
cd Qwen2.5-3B-fine-tuned
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest tests/ -q
```

Python **3.12** (`pyproject.toml` sets `requires-python = ">=3.12"`). The test
suite needs no model weights, no GPU and no network — it runs in seconds, so
there is no excuse for a PR that has not run it.

For anything touching retrieval you also need a running Dify stack and a synced
knowledge base; see the [README setup guide](README.md#setup-step-by-step) and
[docs/deployment.md](docs/deployment.md).

Dependency versions are pinned to what was measured on the deployment machine.
Do not bump them incidentally — an upgrade is its own PR, with a reason.

## Project conventions

**Layout.** Runtime code lives under `src/cadebot/`; the import root is `src/`
(`pyproject.toml` sets `pythonpath = ["src"]`). Nothing patches `sys.path` at
import time — the two helper scripts that need it do so explicitly, and new code
should not add more.

- `src/cadebot/api.py` — FastAPI routes and request/response schemas
- `src/cadebot/models.py` — model loading only, no FastAPI import, so the API
  module stays importable without weights
- `src/cadebot/rag/` — everything retrieval-related
- `scripts/` — operator tools run by hand
- `training/` — fine-tuning, requires CUDA, not part of serving

**Configuration.** Every tunable belongs in `src/cadebot/rag/config.py`, read
from an environment variable with a sensible default, and documented in
`.env.example`. Do not hardcode a threshold, path, model name or URL anywhere
else. Absolute paths — especially home directories — never go in tracked files.

**Comments.** Existing comments explain *why* a value is what it is, often
citing a measurement. Keep that. A comment that restates the code is noise; a
comment recording "500 made 13 of 46 segments lose their ID" is the reason the
next person does not undo your work.

**Tests.** `tests/rag/` mirrors the package. Behavior that took an experiment to
discover should be pinned by a test — `test_use_rag_defaults_on_for_android_payload`
exists precisely so a latency-motivated flag flip cannot silently remove
grounding.

**Language.** New documentation is written in **English**. The existing
Vietnamese documents (`docs/rag-setup.md`, `docs/model-training*.md`,
`docs/performance.md`, `docs/android-client.md`, `docs/voice-pipeline.md`) stay
Vietnamese — extend them in Vietnamese rather than half-translating them.
Code comments and user-facing strings in the Vietnamese assistant stay
Vietnamese.

**Commits.** [Conventional Commits](https://www.conventionalcommits.org/), as
the history already uses:

```
feat(rag): filter fabricated sourceIds and tighten anti-embellishment prompt
fix(stt): enable long-form transcription past 30s
docs: add deployment topology and operations runbook
chore(rag): make generation temperature configurable, default 0.2
test(rag): prove /chat hard-blocks out-of-scope without invoking the LLM
```

## Changes that need extra evidence

Some values in this repo were chosen by measurement, and changing them without
re-measuring quietly degrades the system. If your PR touches any of these,
include the numbers:

| Change | Evidence to include |
|---|---|
| `SCORE_THRESHOLD` | Fresh `scripts/calibrate_threshold.py` output: F1, precision, recall, and the in/out distribution overlap |
| `SYSTEM_PROMPT` or the context block | Before/after answers on at least a few in-scope and out-of-scope queries |
| `GEN_TEMPERATURE` | Evidence the model is not re-attaching attributes the KB never stated |
| `CHUNK_MAX_TOKENS`, `CHUNK_SEPARATOR`, chunking rules | `scripts/sync_kb.py --dry-run` counts, plus confirmation that every segment still keeps its `[chunk_id]` line |
| `use_rag` default | Why ungrounded answers are acceptable — a proxy timeout is a per-deployment problem, solved per request |
| Endpoint contracts | The Android client in `Cadebot_UI/` consumes them; update it or explain why it still works |

## Pull request checklist

- [ ] `python3 -m pytest tests/ -q` passes
- [ ] No secrets: no API keys, tokens or real `.env` contents — placeholders only
- [ ] No absolute paths, personal domains or home directories in tracked files
- [ ] New tunables are env-driven, defaulted in `config.py` and listed in `.env.example`
- [ ] Docs updated alongside behavior — if an endpoint, flag or default changed,
      `docs/api-reference.md`, `docs/architecture.md` and `docs/deployment.md`
      must still be true
- [ ] Calibration or measurement numbers included when the table above applies
- [ ] Commit messages follow Conventional Commits

## Reporting problems

Open an issue using one of the templates. For anything deployment-related,
paste the output of `curl -s localhost:8000/health` — it answers most questions
immediately — and check the troubleshooting table in
[docs/deployment.md](docs/deployment.md#troubleshooting) first.
