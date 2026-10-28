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

## Completed (cont.)

### Tests
- [x] `tests/` pytest suite: chunking (boundary/overlap/ID/chapter tracking), `normalize_question` dedup + JSON fallback parsing + checkpoint, config defaults + env overrides. `pytest.ini` registers the `integration` marker.
- [x] `tests/test_rag_integration.py` reuses `ask()`/`search_only()` from `02_rag/test_rag.py`; health-check gate, auto-skip when the server is down or is not this project's RAG service. Server override via `RAG_TEST_SERVER`.

### CI
- [x] `.github/workflows/ci.yml`: `py_compile` all scripts, `ruff check` (`ruff.toml`, line-length 120, E/F/I rules), `pytest -m "not integration"` on push/PR.
- [x] `requirements-dev.txt` (pytest, ruff + minimal runtime deps for test imports).

### Docs sync
- [x] `RUNBOOK.md` fixed: chunk flags `--chunk-tokens`/`--overlap-tokens`, QA defaults + `--overwrite`/checkpoint note, rag_server flags (env-driven), reranker step, `--question` typo; added 2026-10-28 adjustment record. README quick-start + tree updated.
- [x] Dependency audit: added `pytesseract` + `pydantic` (imported but undeclared). Kept `langchain*`, `marker-pdf`, `typer`, `datasets/peft/trl/bitsandbytes`, `pandas`, `scikit-learn`, `pyyaml`, `sentencepiece` — marked as reserved for the training pipeline in `requirements.txt`.

### Bug fixes found by tests
- [x] `load_checkpoint` crashed on non-dict JSON lines (`AttributeError`) — now guards with `isinstance(qa, dict)`.
- [x] `run_pipeline.sh` used stale flags `--chunk-size`/`--overlap`; `build_and_serve.sh` passed `--db-path/--llm-base-url/--model` which `rag_server.py` no longer accepts (env-driven now). Import blocks ruff-normalized across all scripts.

## Verification
All `.py` files pass `py_compile`; `ruff check .` clean; `pytest`: 22 passed, 3 integration skipped (no live RAG service). Commits so far: `3d2b276`, `1960a36`, `c6d7d3f`.
