# FrontierOps AI

Enterprise multi-agent RAG operations platform: six LangGraph-style agents
(Orchestrator, Retrieval, Research, Analysis, Validation, Response),
grounding/citation/injection/PII/schema guardrails, human-in-the-loop
escalation, an evaluation framework, and observability.

## Design notes

Every module that needs an external package (FAISS, sentence-transformers,
LangGraph, the Anthropic API) is written with a **try/import fallback** to a
lightweight, dependency-free equivalent (a hashing-based embedder, a numpy
brute-force vector search, an extractive local reasoner, JSONL-based
tracing). This means the pipeline runs end-to-end even in a minimal
environment with only `numpy` installed, and automatically upgrades to the
production backend the moment the real dependency is available — no code
changes required. `data/evaluation/*.json` is real script output, not
hand-written sample data.

## Architecture

```
User → FastAPI → Orchestrator → Retrieval (FAISS/numpy) → Research
     → Analysis (LLM) → Validation (guardrails) → Response → Answer
                              ↓ confidence < 0.80
                         Human Review Queue
```

Each stage is implemented as an independent module under `backend/app/`, so
the flow above maps directly onto the code layout below.

## Repository layout

```
backend/app/
  agents/        orchestrator, retrieval, research, analysis, validation, response, graph
  rag/           chunking, embeddings (ST or hashing fallback), vector_store (FAISS or numpy fallback)
  guardrails/    grounding, citations, injection, pii, schema
  database/      SQLAlchemy models + session (SQLite default, Postgres via DATABASE_URL)
  observability/ tracing (OpenTelemetry or JSONL fallback)
  llm_client.py  Anthropic API client, or ExtractiveMockLLM fallback
  main.py        FastAPI gateway
backend/tests/   unit + integration tests
frontend/        React + TypeScript dashboard (source; needs `npm install`)
data/documents/  sample knowledge base (5 policy docs + 1 adversarial doc)
data/evaluation/ benchmark questions + real run output
scripts/         ingest.py, evaluate.py, guardrail_demo.py, build_dashboard.py
```

## Run it

### 1. Zero-dependency demo (works anywhere with Python 3 + numpy)

```bash
python scripts/ingest.py            # chunk + embed + index the sample docs
python scripts/evaluate.py          # run all 6 agents over 24 benchmark questions
python scripts/guardrail_demo.py    # adversarial test: hallucination/injection/PII catch rate
python scripts/build_dashboard.py   # generates data/evaluation/dashboard.html — open it in a browser
```

### 2. Full stack (needs network for `pip`/`npm`, and optionally an Anthropic API key)

```bash
cp .env.example .env                # optionally set ANTHROPIC_API_KEY
pip install -r backend/requirements.txt
cd backend && uvicorn app.main:app --reload    # http://localhost:8000/health

cd frontend && npm install && npm run dev       # http://localhost:5173
```

### 3. Docker

```bash
docker compose up --build
```

Runs Postgres + backend + frontend together. Without `ANTHROPIC_API_KEY`
set, the backend still runs fully — it uses the offline `ExtractiveMockLLM`.

### 4. Tests

```bash
pytest backend/tests/ -v
```

(13 tests, all passing — see `.github/workflows/ci.yml` for the full CI
gate, which also fails the build if faithfulness regresses below 0.85 or
any adversarial answer reaches the user unflagged.)

## Evaluation methodology

`scripts/evaluate.py` runs all 24 benchmark questions in
`data/evaluation/questions.json` through the complete pipeline and computes:

- **Context precision/recall**: retrieved chunks' `document_id`s vs. each
  question's `expected_documents`.
- **Faithfulness**: per-sentence claim/evidence token-overlap (see
  `guardrails/grounding.py`) — not an LLM judge, so it's fast and
  deterministic, at the cost of missing subtler unsupported claims (see
  limitations below).
- **Citation accuracy**: fraction of citations whose cited chunk actually
  overlaps with the claim it's attached to.
- **Latency, success rate, human-review rate**: measured directly from
  pipeline execution.

Because the default `ExtractiveMockLLM` only ever copies sentences that
already exist in the evidence, it never hallucinates by construction — so
the "unsupported answers before/after guardrails" comparison isn't
meaningfully demonstrated by the main benchmark run.
`scripts/guardrail_demo.py` covers that instead, with a fixed set of 7
adversarial answers (grounded claim + fabricated embellishment, PII, or a
prompt-injection payload) fed straight into the Validation Agent's
guardrails. Real, current output: **6/7 unsupported/unsafe answers would
reach the user unflagged before guardrails; 0/7 after.**

## Known limitations of the dependency-free fallback path

- **Retrieval quality**: the `HashingEmbedder` fallback (used when
  `sentence-transformers` isn't installed) is a TF-IDF-weighted hashing
  vectorizer, not a real sentence embedding model.
  It gets the right document most of the time on this small corpus but its
  cosine-similarity scores are compressed into a narrow range, which (a)
  lowers context precision with `top_k=3` on a 6-document corpus and (b)
  pushes the confidence gate toward human review more often than a real
  embedding model would. `rag/embeddings.py` auto-upgrades to
  `sentence-transformers` the moment it's importable — no code changes
  needed elsewhere.
- **Grounding check granularity**: `guardrails/grounding.py` scores claims
  at the sentence level. An embellishment merged into an otherwise-grounded
  sentence can be diluted enough to slip past the threshold (documented and
  tested in `test_grounding_flags_embellishment_within_same_sentence_only_if_dominant`).
  This is why validation also runs citation verification, not grounding
  alone — defense in depth, not a single silver-bullet check.
- **LangGraph**: `agents/graph.py` uses a hand-rolled `SequentialWorkflow`
  with the same state-machine shape (including the retry-on-failed-validation
  loop) as a dependency-free default. The `StateGraph` wiring for LangGraph
  itself is written out in a comment in the same file — installing
  `langgraph` and uncommenting it is a ~15-line change, not a rewrite.
- **Benchmark size**: 24 sample questions ship in `questions.json`
  (schema: `id`, `question`, `expected_documents`); add more entries and
  `evaluate.py` handles any size.

## Build order (if extending this)

1. Swap `HashingEmbedder` → `sentence-transformers` (network required) and
   re-run `evaluate.py` — expect context precision/recall to rise
   meaningfully; recalibrate or remove `RETRIEVAL_SCORE_CALIBRATION` in
   `agents/validation.py` once you do.
2. Set `ANTHROPIC_API_KEY` and re-run — the Analysis Agent will call a real
   LLM instead of `ExtractiveMockLLM`; re-run `guardrail_demo.py`-style
   adversarial cases against real model output to see actual hallucination
   rates rather than the synthetic ones used here.
3. Wire the commented `StateGraph` in `agents/graph.py` once `langgraph` is
   installed.
4. Point `DATABASE_URL` at Postgres and run via `docker compose up`.
5. Grow `data/evaluation/questions.json` to 100 questions and wire the CI
   gate thresholds in `.github/workflows/ci.yml` to your real numbers.
