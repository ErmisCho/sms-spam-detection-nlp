"""Classify an SMS with a previously trained local model artifact."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from sms_spam_ham_analysis.config import TFIDF_MODEL_PATH


class PredictionError(ValueError):
    """Raised when a local prediction cannot be completed."""


class InvalidMessageError(PredictionError):
    """Raised when an SMS cannot be classified because its input is invalid."""


class ModelUnavailableError(PredictionError):
    """Raised when the trained classifier artifact is absent or unusable."""


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


def predict_message(text: str, *, model_path: Path = TFIDF_MODEL_PATH) -> Prediction:
    """Load the cached trained artifact and predict one SMS label with confidence."""

    if not text.strip():
        raise InvalidMessageError("SMS text must not be empty.")

    pipeline = load_pipeline(model_path)
    try:
        label = str(pipeline.predict([text])[0])
        confidence = _prediction_confidence(pipeline, text, label)
    except PredictionError:
        raise
    except Exception as exc:
        raise PredictionError("The classifier could not produce a prediction.") from exc
    return Prediction(label=label, confidence=confidence)


def load_pipeline(model_path: Path = TFIDF_MODEL_PATH) -> Any:
    """Load and validate a model artifact, cached by path and file identity."""

    path = model_path.expanduser().resolve()
    try:
        stat = path.stat()
    except FileNotFoundError as exc:
        raise ModelUnavailableError(
            f"Trained model not found at {path}. "
            "Run the modeling step first: python -m sms_spam_ham_analysis.modeling"
        ) from exc
    except OSError as exc:
        raise ModelUnavailableError(f"Trained model is not readable at {path}.") from exc

    if not path.is_file():
        raise ModelUnavailableError(f"Trained model path is not a file: {path}.")
    return _load_pipeline_cached(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4)
def _load_pipeline_cached(path: str, modified_ns: int, size: int) -> Any:
    del modified_ns, size
    try:
        artifact: Any = joblib.load(path)
    except Exception as exc:
        raise ModelUnavailableError(f"Unable to load trained model artifact at {path}.") from exc

    pipeline = artifact.get("pipeline") if isinstance(artifact, dict) else artifact
    if pipeline is None or not hasattr(pipeline, "predict"):
        raise ModelUnavailableError(f"Invalid model artifact at {path}: classifier pipeline is missing.")
    if not hasattr(pipeline, "predict_proba") or not hasattr(pipeline, "classes_"):
        raise ModelUnavailableError(f"Invalid model artifact at {path}: probability metadata is missing.")
    return pipeline


def clear_model_cache() -> None:
    """Clear cached artifacts, primarily for model replacement and deterministic tests."""

    _load_pipeline_cached.cache_clear()


def _prediction_confidence(pipeline: Any, text: str, label: str) -> float:
    if not hasattr(pipeline, "predict_proba"):
        raise PredictionError("The trained classifier does not provide probability-based confidence.")
    probabilities = pipeline.predict_proba([text])[0]
    classes = [str(value) for value in pipeline.classes_]
    try:
        return float(probabilities[classes.index(label)])
    except (AttributeError, ValueError) as exc:
        raise PredictionError("The trained classifier has incompatible class metadata.") from exc


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify one SMS as HAM or SPAM.")
    parser.add_argument("text", help="SMS text to classify")
    parser.add_argument("--model", type=Path, default=TFIDF_MODEL_PATH, help="Path to trained joblib artifact")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = predict_message(args.text, model_path=args.model)
    except (PredictionError, OSError, ValueError) as exc:
        print(f"Prediction failed: {exc}", file=sys.stderr)
        return 2

    print(f"Prediction: {result.label.upper()}")
    print(f"Confidence: {result.confidence:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
