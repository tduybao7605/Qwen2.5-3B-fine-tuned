## Summary

<!-- What changed and why. One paragraph is usually enough. -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Refactor (no behavior change)
- [ ] Documentation
- [ ] Build / CI / deployment
- [ ] Model, prompt, or retrieval tuning

## How it was tested

<!-- Commands you ran and what came back. If an endpoint changed, paste the
     request and response. -->

```
python3 -m pytest tests/ -q
```

## Checklist

- [ ] `python3 -m pytest tests/ -q` passes
- [ ] No secrets: no API keys, tokens, or real `.env` contents
- [ ] No absolute paths, home directories, or personal domains in tracked files
- [ ] New tunables are env-driven, defaulted in `src/cadebot/rag/config.py`, and listed in `.env.example`
- [ ] Docs updated alongside behavior (`docs/api-reference.md`, `docs/architecture.md`, `docs/deployment.md`)
- [ ] Changes to `SCORE_THRESHOLD`, `GEN_TEMPERATURE`, the prompt, chunking, or the `use_rag` default include fresh calibration or comparison numbers — see [CONTRIBUTING.md](../CONTRIBUTING.md#changes-that-need-extra-evidence)
- [ ] Commit messages follow Conventional Commits

## Related issues

<!-- Closes #123 -->
