"""Classify an SMS with a previously trained local model artifact."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from sms_spam_ham_analysis.config import TFIDF_MODEL_PATH


class PredictionError(ValueError):
    """Raised when a local prediction cannot be completed."""


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float


def predict_message(text: str, *, model_path: Path = TFIDF_MODEL_PATH) -> Prediction:
    """Load a trained artifact and predict one SMS label with confidence."""

    if not text.strip():
        raise PredictionError("SMS text must not be empty.")
    if not model_path.is_file():
        raise PredictionError(
            f"Trained model not found at {model_path}. "
            "Run the modeling step first: python -m sms_spam_ham_analysis.modeling"
        )

    artifact: Any = joblib.load(model_path)
    pipeline = artifact.get("pipeline") if isinstance(artifact, dict) else artifact
    if pipeline is None or not hasattr(pipeline, "predict"):
        raise PredictionError(f"Invalid model artifact at {model_path}: classifier pipeline is missing.")

    label = str(pipeline.predict([text])[0])
    confidence = _prediction_confidence(pipeline, text, label)
    return Prediction(label=label, confidence=confidence)


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
