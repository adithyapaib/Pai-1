"""FastAPI application for local Pai-1 inference."""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings
from .model import Pai1
from .schemas import DecideRequest, DecideResponse, ErrorResponse, EvaluateRequest, EvaluateResponse, QuestionResult
from .systemone import ClassifierResponse, SystemOneRequest, Usage, classify_questions, new_request_id, REQUEST_ID_HEADER

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
settings = Settings.from_environment()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = Pai1(settings)
    yield


app = FastAPI(title="Pai-1 API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"], allow_credentials=False)
app.mount("/ui", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="ui")


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    LOGGER.exception("Unhandled request error: %s", exc)
    return JSONResponse(status_code=500, content={"error": "Model inference failed", "detail": "The request could not be completed."})


@app.get("/", summary="API status")
async def root() -> dict[str, str]:
    return {"name": "Pai-1 API", "docs": "/docs"}


@app.get("/health", summary="Model health")
async def health(request: Request) -> dict[str, object]:
    model = getattr(request.app.state, "model", None)
    return {"status": "ok" if model else "degraded", "model_loaded": model is not None, "device": str(model.device) if model else None}


@app.get("/v1/model", summary="Loaded model metadata")
async def model_info(request: Request) -> dict[str, object]:
    model: Pai1 = request.app.state.model
    return {"model": model.model_name, "backbone": model.settings.backbone, "hidden_size": model.hidden_size, "decision_head_parameters": model.parameter_count, "device": str(model.device)}


@app.post("/v1/decide", response_model=DecideResponse, responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}, summary="Choose the most likely option")
async def decide(payload: DecideRequest, request: Request) -> DecideResponse:
    model: Pai1 = request.app.state.model
    started = time.perf_counter()
    result = model.predict(payload.state, payload.question, payload.options)
    result["model"] = model.model_name
    result["inference_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return DecideResponse(**result)


@app.post(
    "/v1/systemone",
    responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="TypeSafe-compatible classification (LangChain)",
)
async def systemone(payload: SystemOneRequest, request: Request) -> JSONResponse:
    """Classify state with Noul/Choice/Score questions (TypeSafe wire protocol).

    Compatible with ``langchain-typesafe`` ``TypeSafeClassifier`` and the
    experimental ``ModelRouterMiddleware`` / ``AutoModeMiddleware``. Point them
    at this server with ``TYPESAFE_BASE_URL=http://127.0.0.1:8000``. Any
    ``Authorization`` bearer value is accepted; auth is not enforced locally.
    """
    from fastapi.responses import JSONResponse as _JSONResponse

    model: Pai1 = request.app.state.model
    request_id = new_request_id()
    answers, input_tokens = classify_questions(model, payload)
    body = {
        "model": model.model_name,
        "answers": answers,
        "usage": {"input_tokens": input_tokens, "output_tokens": 0 if input_tokens is not None else None},
        "request_id": request_id,
    }
    # Validate shape before sending so malformed answers become 500, not junk.
    ClassifierResponse.model_validate(body)
    return _JSONResponse(content=body, headers={REQUEST_ID_HEADER: request_id})


@app.post("/v1/evaluate", response_model=EvaluateResponse, responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}}, summary="Evaluate multiple typed questions")
async def evaluate(payload: EvaluateRequest, request: Request) -> EvaluateResponse:
    """Answer boolean, choice, and score questions using one shared state."""
    model: Pai1 = request.app.state.model
    started = time.perf_counter()
    results: dict[str, QuestionResult] = {}
    for question_id, question in payload.questions.items():
        if question.type in {"noul", "bool", "boolean"}:
            options = {"yes": "The answer to this question is yes.", "no": "The answer to this question is no."}
        elif question.type == "choice":
            if not isinstance(question.criteria, dict):
                raise ValueError(f"question '{question_id}' requires criteria as an object")
            options = question.criteria
        else:
            if not isinstance(question.criteria, list):
                raise ValueError(f"question '{question_id}' requires criteria as a list")
            options = {str(index): criterion for index, criterion in enumerate(question.criteria)}

        prediction = model.predict(payload.state, question.instructions, options)
        if question.type == "score":
            prediction["choice"] = question.criteria[int(prediction["choice"])]
            prediction["probabilities"] = {
                question.criteria[int(label)]: probability
                for label, probability in prediction["probabilities"].items()
            }
        results[question_id] = QuestionResult(type=question.type, **prediction)

    return EvaluateResponse(
        model=model.model_name,
        results=results,
        inference_ms=round((time.perf_counter() - started) * 1000, 2),
    )