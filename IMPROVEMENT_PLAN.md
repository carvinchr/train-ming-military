# Improvement Plan

Design-review fixes applied on branch `fix/design-review-fixes`. Status legend: [x] done, [ ] todo.

## Completed

### Code & config
- [x] `config.py` — shared settings dataclass, env-var driven (`.env` via `python-dotenv`), used by pipeline/RAG scripts.
- [x] `01_data_pipeline/04_chunk_text.py` — token-aware chunking (bge-m3 tokenizer), Chinese sentence boundaries, whole-sentence overlap, SHA-256 chunk IDs, chapter tracking.
- [x] `01_data_pipeline/05_generate_qa.py` — structured JSON output (`response_format`), regex fallback, `normalize_question` dedup, checkpoint resume, quality filter, `--overwrite`.
- [x] `02_rag/build_index.py` — atomic collection swap (build into `ming_military_new`, then rename), chunk metadata persisted.
- [x] `02_rag/rag_server.py` — FastAPI lifespan, `CrossEncoder` reranker, configurable CORS, per-request `top_k`/`rerank_top_k`, `/health`.
- [x] `05_deployment/Modelfile` — `num_ctx` raised 4096 → 8191.
- [x] `00_setup/docker-compose.yml` — named volumes, healthchecks, env-based secret, `depends_on` conditions, removed deprecated `version`.

### Hygiene
- [x] `.gitignore` — pycache, `.venv`, data/, outputs/, results/, benchmark/, model artifacts, `chroma_db/`, `.env`.
- [x] `00_setup/.env.example` — documents every variable read by `config.py`.
- [x] `README.md` — quick-start includes `cp 00_setup/.env.example .env`; directory listing updated.
- [x] `00_setup/requirements.txt` — added `requests` (used by `benchmark/run_benchmark.py`); removed `pynvml` (not imported anywhere).

Commits: `3d2b276`, `1960a36`.

## Remaining work

### Tests
- [ ] Create `tests/` with pytest: chunking (boundary/overlap/ID), `normalize_question` dedup, config defaults + env overrides.
- [ ] Promote `02_rag/test_rag.py` from interactive script to an integration test (skip when server is down).

### CI
- [ ] Add `.github/workflows/ci.yml`: Python syntax check (`py_compile`), lint (ruff), `pytest` on push/PR.

### Docs sync
- [ ] Verify `RUNBOOK.md` matches current script flags (`--overwrite`, env vars) and reranker step.
- [ ] Confirm dependency list vs imports: `langchain*`, `marker-pdf`, `typer`, `datasets/peft/trl/bitsandbytes` are declared but not yet imported by any script (reserved for the training pipeline); decide keep vs trim.

## Verification
All rewritten scripts pass `py_compile`; overlap logic verified sentence-level; every third-party import found in `.py` files is covered by `requirements.txt`.
