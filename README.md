# Pai-1 API

Pai-1 is a local decision/classification API. It uses the frozen
`Qwen/Qwen2.5-0.5B-Instruct` backbone for semantic hidden states and an
exported, lightweight decision head for scoring a variable number of options.

## Architecture

```text
                Qwen2.5-0.5B-Instruct
                         │
                  semantic embeddings
                         │
                         ▼
                  Pai-1 Decision Head
                         │
              ┌──────────┴──────────┐
              │                     │
        option representations   context
              │                     │
              └──────────┬──────────┘
                         ▼
                    option logits
                         │
                      softmax
                         │
                         ▼
                decision + confidence
```

The head reconstructs three `896 -> 256` projections, four-head attention, a
`256 -> 512 -> 256` residual FFN, and a `768 -> 256 -> 1` scorer. The exported
checkpoint contains 1,414,657 parameters. The loader checks the complete
state-dict strictly so an incompatible export fails at startup rather than
producing silently incorrect decisions.

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

`decision_head.pt` is intentionally ignored by Git. Do not commit it unless
the repository is configured for Git LFS or uses a private artifact store.
The Qwen backbone is downloaded from Hugging Face on first startup and cached.

## Local installation

```powershell
cd pai-1-api
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open Swagger at <http://localhost:8000/docs>.

Open the manual testing workbench at <http://localhost:8000/ui/>. It lets you
edit the shared state, add typed `noul`, `choice`, and `score` questions, run
`/v1/evaluate`, and inspect both readable results and the raw JSON response.

CPU inference uses FP32. With a CUDA-enabled PyTorch installation, the
backbone uses FP16 and the decision head receives FP32 hidden states.

## API

```powershell
curl.exe -X POST http://localhost:8000/v1/decide `
  -H "Content-Type: application/json" `
  -d '{"state":"Production API is returning HTTP 500 errors.","question":"What should the engineering team do?","options":{"investigate":"Investigate logs and identify the root cause.","rollback":"Rollback the latest deployment.","escalate":"Escalate the incident."}}'
```

Response probabilities are sorted from highest to lowest:

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

Python client:

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

`GET /health` reports model load state and device. `GET /v1/model` reports the
backbone, hidden size, device, and dynamic head parameter count.

### Multiple typed questions

Use `POST /v1/evaluate` when several questions share the same state:

```powershell
curl.exe -X POST http://localhost:8000/v1/evaluate `
  -H "Content-Type: application/json" `
  -d '{"model":"clef","state":"Checkout has been failing for every customer for the last hour.","questions":{"urgent":{"type":"noul","instructions":"Is this support request urgent?"},"team":{"type":"choice","instructions":"Which team should handle this request?","criteria":{"billing":"Payments, invoices, and refunds","technical":"Outages, errors, and configuration","sales":"Plans and upgrades"}},"severity":{"type":"score","instructions":"How severe is the customer impact?","criteria":["No impact","Minor","Major","Critical"]}}}'
```

`noul` is supported as the boolean question type and produces `yes` or `no`.
`choice` uses the keys from its criteria object. `score` uses the supplied
criteria labels and returns the selected label rather than an internal index.
The response has a `results` object keyed by the question IDs.

## Docker

```powershell
docker compose up --build
```

The API is available at `http://localhost:8000`, and the `huggingface-cache`
volume prevents repeated backbone downloads. The image supports CPU inference.
For NVIDIA GPU inference, use an NVIDIA-enabled PyTorch base image or install
the CUDA-compatible PyTorch dependencies in the Dockerfile, then launch with
NVIDIA Container Toolkit and set `PAI1_DEVICE=cuda`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `PAI1_MODEL_PATH` | `models/pai-1-0.5b` | Local export directory |
| `PAI1_BACKBONE` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face backbone |
| `PAI1_DEVICE` | `auto` | `auto`, `cpu`, or `cuda` |
| `PAI1_MAX_LENGTH` | `2048` | Tokenization limit |
| `PAI1_CORS_ORIGINS` | `*` | Comma-separated allowed origins |

Legacy `PAICLEF_*` variables are still accepted as fallback when the
corresponding `PAI1_*` variable is unset.

## Tests and limitations

Run mocked API tests with `pytest`. They do not download Qwen. A full
integration test requires the model export, an installed PyTorch/Transformers
stack, and network access on first startup. Long inputs are truncated at the
configured maximum; the API rejects fewer than two or more than twenty
options. The model is a classifier, not a factual or safety verifier, so its
confidence is a model probability rather than a guarantee of correctness.
