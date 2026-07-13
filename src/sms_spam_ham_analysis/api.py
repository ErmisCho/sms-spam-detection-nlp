"""HTTP serving layer for the trained SMS classifier."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Literal

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from sms_spam_ham_analysis.config import TFIDF_MODEL_PATH
from sms_spam_ham_analysis.predict import (
    InvalidMessageError,
    ModelUnavailableError,
    PredictionError,
    load_pipeline,
    predict_message,
)


LOGGER = logging.getLogger("sms_spam_ham_analysis.api")
MODEL_PATH_ENV = "SMS_SPAM_MODEL_PATH"
REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


class JsonFormatter(logging.Formatter):
    """Format API records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        for name in ("request_id", "method", "route", "status", "latency_ms"):
            value = getattr(record, name, None)
            if value is not None:
                payload[name] = value
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    text: str = Field(min_length=1, max_length=10_000, description="SMS text to classify")

    @field_validator("text")
    @classmethod
    def reject_whitespace_only_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("SMS text must not be empty")
        return value


class PredictionResponse(BaseModel):
    label: Literal["ham", "spam"]
    confidence: float = Field(ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_ready: bool


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]


def create_app(*, model_path: Path | None = None) -> FastAPI:
    """Create an application with an explicit, testable model artifact path."""

    resolved_model_path = model_path or Path(os.getenv(MODEL_PATH_ENV, str(TFIDF_MODEL_PATH)))
    application = FastAPI(
        title="SMS Spam Detection API",
        version="0.1.0",
        description="Classify an SMS as HAM or SPAM with a trained TF-IDF model.",
    )
    application.state.model_path = resolved_model_path

    @application.middleware("http")
    async def request_observability(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = _request_id(request.headers.get(REQUEST_ID_HEADER))
        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers[REQUEST_ID_HEADER] = request_id
            return response
        finally:
            LOGGER.info(
                "request_completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "route": request.url.path,
                    "status": status,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                },
            )

    @application.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        return HealthResponse(model_ready=_model_is_ready(application.state.model_path))

    @application.get("/health/live", response_model=HealthResponse, tags=["health"])
    async def liveness() -> HealthResponse:
        return HealthResponse(model_ready=_model_is_ready(application.state.model_path))

    @application.get(
        "/health/ready",
        response_model=ReadinessResponse,
        responses={503: {"model": ReadinessResponse}},
        tags=["health"],
    )
    async def readiness() -> ReadinessResponse | JSONResponse:
        if _model_is_ready(application.state.model_path):
            return ReadinessResponse(status="ready")
        return JSONResponse(status_code=503, content={"status": "not_ready"})

    @application.post("/predict", response_model=PredictionResponse, tags=["prediction"])
    async def predict(payload: PredictionRequest) -> PredictionResponse | JSONResponse:
        try:
            result = predict_message(payload.text, model_path=application.state.model_path)
        except InvalidMessageError:
            return JSONResponse(status_code=422, content={"detail": "SMS text must not be empty."})
        except ModelUnavailableError:
            return JSONResponse(status_code=503, content={"detail": "Prediction model is unavailable."})
        except PredictionError:
            return JSONResponse(status_code=500, content={"detail": "Prediction failed."})
        return PredictionResponse(label=result.label, confidence=result.confidence)

    return application


def configure_logging() -> None:
    """Configure structured service logs without overriding an embedding application's handlers."""

    if LOGGER.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


def _request_id(candidate: str | None) -> str:
    if candidate and REQUEST_ID_PATTERN.fullmatch(candidate):
        return candidate
    return uuid.uuid4().hex


def _model_is_ready(model_path: Path) -> bool:
    try:
        load_pipeline(model_path)
    except PredictionError:
        return False
    return True


def main() -> None:
    configure_logging()
    uvicorn.run(
        "sms_spam_ham_analysis.api:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        log_config=None,
    )


configure_logging()
app = create_app()


if __name__ == "__main__":
    main()
