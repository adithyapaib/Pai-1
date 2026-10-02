# Pai-1 API

> Local decision/classification API — ask "which option is best?" and get a ranked answer with probabilities. No cloud calls, no text generation, just decisions.

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-69%20passing-brightgreen)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Pai-1 pairs a **frozen `Qwen2.5-0.5B-Instruct` backbone** (understanding) with a **tiny trained decision head** (scoring, 1.4M params) to pick the best label from 2–20 options. It also speaks the **TypeSafe wire protocol**, so `langchain-typesafe` (`TypeSafeClassifier`, `ModelRouterMiddleware`, `AutoModeMiddleware`) can run against your laptop instead of the cloud.

```powershell
make start
curl.exe -X POST http://localhost:8000/v1/decide `
  -H "Content-Type: application/json" `
  -d '{"state":"Production API is returning HTTP 500 errors.","question":"What should the team do?","options":{"investigate":"Investigate logs.","rollback":"Rollback the deployment."}}'
# {"model":"pai-1-0.5b","choice":"rollback","confidence":0.96,...}
```

---

## Table of contents

- [How it works](#how-it-works)
  - [The two halves](#the-two-halves)
  - [Startup](#startup-what-happens-on-boot)
  - [A request, step by step](#a-request-step-by-step-pai1predict-appmodelpy116)
  - [How each endpoint reuses `predict`](#how-each-endpoint-reuses-predict)
  - [Failure modes](#failure-modes-by-design)
- [Quickstart](#quickstart)
- [Makefile](#makefile)
- [API reference](#api-reference)
- [LangChain / TypeSafe compatibility](#langchain--typesafe-compatibility)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Testing](#testing)
- [Docker](#docker)
- [Model files](#model-files)
- [Limitations](#limitations)

---

## How it works

Pai-1 is a **classifier, not a chatbot**. There is no text generation, no chat loop, no streaming — one request in, one ranked decision out:

```text
                 Qwen2.5-0.5B-Instruct  (frozen, understands text)
                          │
                   semantic embeddings
                          │
                          ▼
                  Pai-1 Decision Head  (trained, scores options)
                          │
               ┌──────────┴──────────┐
               │                     │
        option vectors          context vectors
               │                     │
               └──────────┬──────────┘
                          ▼
                     option logits → softmax
                          ▼
                 decision + confidence
```

### The two halves

Understanding and judging are deliberately split between two components (`app/model.py`):

| | Backbone | Decision head |
|---|---|---|
| Class | `Qwen2.5-0.5B-Instruct` (Hugging Face) | `MiniPaiHead` (`app/model.py:20`) |
| Parameters | ~0.5B, **frozen, never fine-tuned** | **1,414,657, trained** — the only learned part |
| Role | turns text into rich hidden states | turns hidden states into one score per option |
| Precision | FP16 on CUDA, FP32 on CPU | always FP32 inputs |
| Source | downloaded from HF on first boot, cached | `decision_head.pt` (Git LFS) |

The head architecture, in order: three `896→256` linear projections (state, question, options) → `LayerNorm` → 4-head self-attention over the sequence `[state, question, option₁…optionₙ]` with a residual add → residual feed-forward block (`LayerNorm → 256→512 → GELU → 512→256`) → per-option scorer that concatenates each option vector with the attended state and question context (`256×3=768 → 256 → GELU → 1`). Output: exactly one logit per option.

Because only the head is trained, swapping in a better checkpoint is a single-file change — the backbone, tokenizer, and API stay untouched.

### Startup: what happens on boot

The model is built **once** in the FastAPI `lifespan` handler (`app/main.py:23`) and shared by every request via `app.state.model`:

1. **Settings** — `Settings.from_environment()` reads `PAI1_*` env vars (legacy `PAICLEF_*` accepted when the new name is unset).
2. **Device** — `auto` picks CUDA when available, else CPU; requesting `cuda` on a CUDA-less box raises immediately (`app/model.py:99`).
3. **Config** — `config.json` supplies `hidden_size` (896), `decision_size` (256), `num_heads` (4); the model name is the export folder name (`pai-1-0.5b`).
4. **Tokenizer** — loaded from the local `tokenizer/` dir, including a compatibility shim that repairs older exports whose `extra_special_tokens` is a list instead of a mapping (`app/model.py:82`).
5. **Backbone** — `AutoModel.from_pretrained(...)` (first boot downloads it once, then it is cached), moved to the device, set to `.eval()`.
6. **Head** — `MiniPaiHead` is constructed, then `decision_head.pt` is loaded with `torch.load(..., weights_only=True)`. The loader accepts a bare state dict or a `{"state_dict": …}` / `{"model_state_dict": …}` wrapper, and applies it with **`strict=True` — a mismatched checkpoint crashes at startup instead of mis-scoring silently** (`app/model.py:72`).
7. The head moves to the device in FP32, eval mode, and its parameter count is logged.

If the model object is missing, `/health` reports `degraded` instead of `ok` — the server never pretends to work.

### A request, step by step (`Pai1.predict`, `app/model.py:116`)

Every endpoint funnels into the same `predict(state, question, options)` routine:

1. **Prompt assembly** — one text block is built, options numbered in insertion order:
   `STATE: …` ⏎ `QUESTION: …` ⏎ `OPTION_1: …` ⏎ `OPTION_2: …`.
   Note the model only ever sees the option **descriptions**; the **labels** (keys) are kept aside and re-attached to the scores afterwards.
2. **Tokenization** — truncated at `PAI1_MAX_LENGTH` (default 2048), requesting `return_offsets_mapping` so every token knows the exact character span it came from.
3. **Encoding** — a single backbone forward pass under `torch.inference_mode()` (no gradients, no dropout); only `last_hidden_state[0]` is kept, cast to float.
4. **Span pooling** — `_pool_span` (`app/model.py:107`) locates each section (`STATE: `, `QUESTION: `, `OPTION_i: `) by its character offsets, selects every token overlapping that span, and **mean-pools** them. Result: one 896-dim vector for the state, one for the question, and an `N×896` matrix for the options. A section that maps to zero tokens raises instead of returning a garbage zero-vector.
5. **Scoring** — the head runs its projections → attention (with a padding mask slot, all-true for real requests) → scorer, producing one logit per option.
6. **Decision** — softmax over the logits → probabilities, sorted highest-first. `choice` is the winning **label**, `confidence` is its probability, `probabilities` maps every label to its score (sums to 1).

Typical latency is tens of milliseconds on CPU — the backbone forward pass dominates; the 1.4M-param head is negligible.

### How each endpoint reuses `predict`

- **`/v1/decide`** (`app/main.py:57`) — thin wrapper: validates (2–20 options, non-blank everything, labels unique after stripping), calls `predict` once, stamps `model` + `inference_ms`.
- **`/v1/evaluate`** (`app/main.py:96`) — loops over typed questions sharing one state, translating each type into an options dict:
  - `noul` / `bool` / `boolean` → fixed `{"yes": "The answer to this question is yes.", "no": …}` options; `choice` is `yes` or `no`.
  - `choice` → the `criteria` object *is* the options dict; `choice` is the winning key.
  - `score` → the `criteria` list becomes `{"0": level₀, "1": level₁, …}`; after scoring, the winning index is mapped back to its **label** and probabilities are re-keyed by label.
- **`/v1/systemone`** (`app/main.py:67` + `app/systemone.py:137`) — same engine, JSON-tolerant protocol layer for LangChain:
  - `to_text` / `state_to_text` (`app/systemone.py:27`) accept **any JSON** as state, instructions, or criteria — objects and arrays are deterministically stringified (`sort_keys=True`) for the backbone.
  - `noul` honors custom `criteria.true` / `criteria.false` texts (falling back to *"The answer to this question is …"*), returning `noul = P(yes)`.
  - `choice` passes criteria values through as option descriptions.
  - `score` returns the **expected value** `Σ level × P(level)` plus a `legend` mapping levels back to rubric labels; confidence is the peak rubric probability.
  - `usage.input_tokens` is a best-effort count from the local tokenizer (`output_tokens` is 0 — classifiers generate nothing); a fresh `request_id` is returned in both the body and the `x-typesafe-request-id` header; the response is validated against `ClassifierResponse` **before** sending, so a malformed answer becomes a 500 instead of junk on the wire. Any bearer token is accepted — auth is not enforced locally.

### Failure modes (by design)

- **Bad input → `422`** with a message naming the offending field/question (Pydantic + custom validators), never a traceback.
- **Anything unexpected → generic `500 {"error": "Model inference failed"}`**; the real traceback goes to the server log only (`app/main.py:34`).
- **Bad checkpoint / missing CUDA / missing span → fail fast** at startup or request time, never a silent wrong answer.
- **What it can't do:** verify facts, enforce safety, or explain itself — confidence is a probability over *your* options, not a guarantee. Gate high-stakes actions on thresholds.

---

## Quickstart

```powershell
cd pai-1-api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Or simply `make start` (uses `.venv` automatically — see [Makefile](#makefile)).

Then open:

| URL | What |
|---|---|
| <http://localhost:8000/docs> | Swagger / OpenAPI playground |
| <http://localhost:8000/ui/> | Decision workbench — compose state + typed questions in the browser |
| <http://localhost:8000/health> | Model load status and device |

First boot downloads the Qwen backbone from Hugging Face once, then caches it.

---

## Makefile

Requires GNU Make 4.x (`winget install ezwinports.make` on Windows).

| Target | Command |
|---|---|
| `make help` | List targets |
| `make install` | Install `requirements.txt` into `.venv` if present |
| `make venv` | Create `.venv` |
| `make test` | Run the mocked suite (69 tests, ~1s, no downloads) |
| `make start` / `make run` | Serve with uvicorn (`HOST`/`PORT` overridable: `make start PORT=9000`) |
| `make docker-build` / `make docker-up` / `make docker-down` | Container lifecycle |
| `make clean` | Remove `__pycache__` and `.pytest_cache` |

---

## API reference

### `POST /v1/decide` — one question, N options

```powershell
curl.exe -X POST http://localhost:8000/v1/decide `
  -H "Content-Type: application/json" `
  -d '{"state":"Production API is returning HTTP 500 errors.","question":"What should the engineering team do?","options":{"investigate":"Investigate logs and identify the root cause.","rollback":"Rollback the latest deployment.","escalate":"Escalate the incident."}}'
```

```json
{
  "model": "pai-1-0.5b",
  "choice": "rollback",
  "confidence": 0.9635,
  "probabilities": {
    "rollback": 0.9635,
    "investigate": 0.0291,
    "escalate": 0.0074
  },
  "inference_ms": 42.7
}
```

Python:

```python
import requests

response = requests.post("http://localhost:8000/v1/decide", json={
    "state": "The deployment is returning HTTP 500 errors.",
    "question": "What should the team do?",
    "options": {"investigate": "Inspect logs.", "rollback": "Rollback."},
})
response.raise_for_status()
print(response.json())
```

Rules enforced by `app/schemas.py` (violations → `422`): non-blank state/question, 2–20 options, non-blank labels/descriptions, labels stripped and unique after stripping.

### `POST /v1/evaluate` — several typed questions, one shared state

```powershell
curl.exe -X POST http://localhost:8000/v1/evaluate `
  -H "Content-Type: application/json" `
  -d '{"model":"clef","state":"Checkout has been failing for every customer for the last hour.","questions":{"urgent":{"type":"noul","instructions":"Is this support request urgent?"},"team":{"type":"choice","instructions":"Which team should handle this request?","criteria":{"billing":"Payments, invoices, and refunds","technical":"Outages, errors, and configuration","sales":"Plans and upgrades"}},"severity":{"type":"score","instructions":"How severe is the customer impact?","criteria":["No impact","Minor","Major","Critical"]}}}'
```

| Type | Meaning | Response |
|---|---|---|
| `noul` / `bool` / `boolean` | yes/no judgment | `choice` is `yes` or `no` |
| `choice` | pick a key from `criteria` object | `choice` is the winning key |
| `score` | ordered rubric from `criteria` list | `choice` is the winning **label** (index mapped back), probabilities keyed by label |

Response shape: `{model, results: {questionId: {type, choice, confidence, probabilities}}, inference_ms}`.

### Status endpoints

- `GET /health` → `{status: ok|degraded, model_loaded, device}`
- `GET /v1/model` → `{model, backbone, hidden_size, decision_head_parameters, device}`

Unexpected failures return generic `500 {"error": "Model inference failed"}` without leaking tracebacks.

---

## LangChain / TypeSafe compatibility

`POST /v1/systemone` implements the TypeSafe wire protocol, so `langchain-typesafe` works against local Pai-1. Any bearer value is accepted locally:

```powershell
$env:TYPESAFE_BASE_URL="http://127.0.0.1:8000"
$env:TYPESAFE_API_KEY="local"
```

```python
from langchain.agents import create_agent
from langchain_typesafe.experimental.middleware import (
    ModelChoice,
    ModelRouterMiddleware,
)

router = ModelRouterMiddleware(
    choices={
        "fast": ModelChoice(
            model="openai:luna",
            criteria="Direct lookups, extraction, and localized changes.",
        ),
        "powerful": ModelChoice(
            model="openai:sol",
            criteria="Architecture and high-stakes decisions.",
        ),
    },
    instructions="Choose the least costly model that can complete the task.",
)

agent = create_agent("openai:gpt-5.6-luna", middleware=[router])
```

Mapping:

| LangChain / TypeSafe | Pai-1 `/v1/systemone` |
|---|---|
| `state: str \| object \| array \| messages` | accepted, stringified for the backbone |
| `Choice` | `Pai1.predict` over `criteria` keys → `choice` + `probabilities` + `confidence` |
| `Noul` | `yes`/`no` options (honors `criteria.true/false`) → `noul = P(yes)` |
| `Score` | expected value over rubric levels → `score` + `legend` + `probabilities` + `confidence` |
| `answers` + `usage` + `x-typesafe-request-id` | returned; `model` is the local checkpoint (e.g. `pai-1-0.5b`) |

Direct use:

```python
from langchain_typesafe import Choice, TypeSafeClassifier

clf = TypeSafeClassifier(base_url="http://127.0.0.1:8000", api_key="local")
res = clf.invoke({
    "state": "Checkout keeps crashing on pay.",
    "questions": {"team": Choice(
        instructions="Which team should handle this?",
        criteria={"billing": "payments", "technical": "outages"})},
})
print(res.choices["team"].choice)
```

---

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PAI1_MODEL_PATH` | `models/pai-1-0.5b` | Local export directory |
| `PAI1_BACKBONE` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face backbone |
| `PAI1_DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `PAI1_MAX_LENGTH` | `2048` | Tokenization truncation limit |
| `PAI1_CORS_ORIGINS` | `*` | Comma-separated allowed origins |

Legacy `PAICLEF_*` variables are still accepted as fallback when the corresponding `PAI1_*` variable is unset.

---

## Project structure

```text
pai-1-api/
├── app/
│   ├── main.py          # FastAPI app, lifespan, all endpoints, error handler
│   ├── model.py         # Pai1 loader + MiniPaiHead (PaiClef aliases kept)
│   ├── schemas.py       # /v1/decide + /v1/evaluate validation
│   ├── systemone.py     # TypeSafe-compatible /v1/systemone layer
│   ├── config.py        # Settings from PAI1_* env (PAICLEF_* fallback)
│   └── static/index.html# /ui/ workbench
├── models/pai-1-0.5b/   # export: config.json, decision_head.pt*, tokenizer/
├── tests/               # 69 mocked tests (FakeModel, no Qwen download)
├── test_api.py          # manual live-request script (not collected by pytest)
├── Makefile             # install/test/start/docker/clean
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

\* `decision_head.pt` is tracked via Git LFS — clones get it automatically (requires `git lfs install`).

---

## Testing

```powershell
make test
# .venv/Scripts/python.exe -m pytest -q → 69 passed
```

Tests inject a deterministic `FakeModel` and never run `lifespan`, so they need no GPU, no downloads, and finish in ~1s:

- `test_health.py` — root, health ok/degraded, model metadata
- `test_decide.py` — contract (confidence == winning probability, probs sum to 1, sorted desc), arg forwarding, whitespace stripping, 422 cases, 500 without leaks
- `test_evaluate.py` — `noul`/`bool` aliases, choice keys, score label remapping, validation and safe-500 paths
- `test_systemone.py` — router Choice shape, noul/score answers, object state + JSON criteria, bearer tolerance, 422 cases
- `test_schemas.py`, `test_config.py` (incl. legacy fallback), `test_model_head.py` (shapes, 1,414,657 param count, device/span helpers)

`pytest.ini` scopes collection to `tests/`; `test_api.py` is a manual script for hitting a live server.

---

## Docker

```powershell
docker compose up --build
```

API at `http://localhost:8000`; the `huggingface-cache` volume avoids re-downloading the backbone. CPU inference by default — for NVIDIA GPUs use a CUDA-enabled PyTorch base and launch with the NVIDIA Container Toolkit plus `PAI1_DEVICE=cuda`.

---

## Model files

Place the supplied export at `models/pai-1-0.5b/`:

```text
models/pai-1-0.5b/
├── config.json
├── decision_head.pt
└── tokenizer/
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── chat_template.jinja
```

`decision_head.pt` is versioned with Git LFS, so a normal clone includes it (run `git lfs install` first if you don't have LFS). The Qwen backbone downloads from Hugging Face on first startup and is cached.

---

## Limitations

- Long inputs are truncated at `PAI1_MAX_LENGTH`; fewer than 2 or more than 20 options/questions are rejected.
- Confidence is a model probability, not a guarantee — Pai-1 classifies, it does not verify facts or safety. Threshold high-stakes actions accordingly.
- `score` confidence is the peak rubric probability; `noul` has no separate confidence value by design.
