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

Pai-1 is a **classifier, not a chatbot**. One request in, one ranked decision out:

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

Per request (`app/model.py:116`):

1. Build one text block: `STATE: …`, `QUESTION: …`, `OPTION_1: …`, `OPTION_2: …`.
2. Backbone encodes it into hidden states.
3. Mean-pool the token spans for each section (via tokenizer offsets).
4. `MiniPaiHead` — `896→256` projections, 4-head attention, residual FFN, `768→256→1` scorer — emits one logit per option.
5. Softmax → probabilities sorted highest-first. `choice` is the winner, `confidence` its probability.

The head object is built **once** at startup (`lifespan`, `app/main.py:23`) and shared across requests. Loading uses `strict=True`, so a mismatched checkpoint fails fast instead of mis-scoring silently. CPU runs FP32; CUDA runs the backbone in FP16 with FP32 head inputs.

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
