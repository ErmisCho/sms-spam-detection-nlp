"""Shared model utilities for SMS spam classification."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


RANDOM_STATE = 42
TEST_SIZE = 0.2
LABELS = ["ham", "spam"]


def build_tfidf_classifier() -> Pipeline:
    """Build the classical ML baseline pipeline."""

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    token_pattern=r"(?u)\b\w[\w']*\b",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                    solver="liblinear",
                ),
            ),
        ]
    )
