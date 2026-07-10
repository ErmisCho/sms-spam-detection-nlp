"""Train and evaluate the TF-IDF SMS spam classifier in one workflow."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from sms_spam_ham_analysis.config import (
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    ERROR_EXAMPLES_PATH,
    MODEL_METRICS_PATH,
    TEST_PREDICTIONS_PATH,
    TFIDF_MODEL_PATH,
    TRAINING_METADATA_PATH,
    VALIDATED_DATASET_PATH,
)
from sms_spam_ham_analysis.data import LABEL_COLUMN, ROW_ID_COLUMN, TEXT_COLUMN, load_validated_dataset
from sms_spam_ham_analysis.model import LABELS, RANDOM_STATE, TEST_SIZE, build_tfidf_classifier


class ModelingError(ValueError):
    """Raised when classifier training or evaluation cannot run."""


@dataclass(frozen=True)
class ModelingSummary:
    dataset: str
    model_output: str
    metadata_output: str
    metrics_output: str
    report_output: str
    confusion_matrix_output: str
    error_examples_output: str
    predictions_output: str
    train_rows: int
    test_rows: int
    accuracy: float
    spam_precision: float
    spam_recall: float
    spam_f1: float
    false_positives: int
    false_negatives: int


def train_and_evaluate_model(
    dataset_path: Path,
    *,
    model_output: Path = TFIDF_MODEL_PATH,
    metadata_output: Path = TRAINING_METADATA_PATH,
    metrics_output: Path = MODEL_METRICS_PATH,
    report_output: Path = CLASSIFICATION_REPORT_PATH,
    confusion_matrix_output: Path = CONFUSION_MATRIX_PATH,
    error_examples_output: Path = ERROR_EXAMPLES_PATH,
    predictions_output: Path = TEST_PREDICTIONS_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> ModelingSummary:
    """Train the baseline classifier and write evaluation artifacts."""

    df = load_validated_dataset(dataset_path)
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[LABEL_COLUMN],
    )
    duplicate_summary = _duplicate_summary(df, train_df, test_df)

    pipeline = build_tfidf_classifier()
    pipeline.fit(train_df[TEXT_COLUMN], train_df[LABEL_COLUMN])
    duplicate_safe = _duplicate_safe_evaluation(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    metadata = {
        "dataset": str(dataset_path),
        "model_output": str(model_output),
        "metadata_output": str(metadata_output),
        "model_type": "scikit-learn Pipeline",
        "representation": "TF-IDF word unigrams+bigrams, min_df=2, sublinear_tf=True",
        "classifier": "LogisticRegression(class_weight='balanced', solver='liblinear')",
        "random_state": random_state,
        "test_size": test_size,
        "total_rows": int(len(df)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "labels": LABELS,
        "class_counts": _counts(df),
        "train_class_counts": _counts(train_df),
        "test_class_counts": _counts(test_df),
        "duplicate_summary": duplicate_summary,
        "train_row_ids": sorted(train_df[ROW_ID_COLUMN].astype(int).tolist()),
        "test_row_ids": sorted(test_df[ROW_ID_COLUMN].astype(int).tolist()),
    }

    model_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "metadata": metadata}, model_output)
    metadata_output.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    evaluation = _evaluate_predictions(
        test_df.sort_values(ROW_ID_COLUMN),
        pipeline,
        metrics_output=metrics_output,
        report_output=report_output,
        confusion_matrix_output=confusion_matrix_output,
        error_examples_output=error_examples_output,
        predictions_output=predictions_output,
        duplicate_summary=duplicate_summary,
        duplicate_safe=duplicate_safe,
    )

    return ModelingSummary(
        dataset=str(dataset_path),
        model_output=str(model_output),
        metadata_output=str(metadata_output),
        metrics_output=str(metrics_output),
        report_output=str(report_output),
        confusion_matrix_output=str(confusion_matrix_output),
        error_examples_output=str(error_examples_output),
        predictions_output=str(predictions_output),
        train_rows=int(len(train_df)),
        test_rows=int(len(test_df)),
        accuracy=float(evaluation["accuracy"]),
        spam_precision=float(evaluation["spam"]["precision"]),
        spam_recall=float(evaluation["spam"]["recall"]),
        spam_f1=float(evaluation["spam"]["f1"]),
        false_positives=int(evaluation["errors"]["false_positives"]),
        false_negatives=int(evaluation["errors"]["false_negatives"]),
    )


def _evaluate_predictions(
    test_df: pd.DataFrame,
    pipeline,
    *,
    metrics_output: Path,
    report_output: Path,
    confusion_matrix_output: Path,
    error_examples_output: Path,
    predictions_output: Path,
    duplicate_summary: dict,
    duplicate_safe: dict,
) -> dict:
    if test_df.empty:
        raise ModelingError("No held-out test rows available for evaluation.")

    y_true = test_df[LABEL_COLUMN]
    y_pred = pipeline.predict(test_df[TEXT_COLUMN])
    spam_probability = _spam_probability(pipeline, test_df[TEXT_COLUMN])
    predictions = pd.DataFrame(
        {
            "row_id": test_df[ROW_ID_COLUMN].astype(int),
            "true_label": y_true,
            "predicted_label": y_pred,
            "spam_probability": spam_probability,
            "text": test_df[TEXT_COLUMN],
        }
    )
    predictions["is_correct"] = predictions["true_label"] == predictions["predicted_label"]
    predictions["error_type"] = predictions.apply(_error_type, axis=1)

    row_metrics = _build_metrics(y_true, y_pred, predictions)
    metrics = _metrics_payload(
        row_metrics,
        duplicate_summary=duplicate_summary,
        duplicate_safe=duplicate_safe["metrics"],
    )
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    report_text = classification_report(y_true, y_pred, labels=LABELS, zero_division=0)

    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(matrix, index=[f"true_{label}" for label in LABELS], columns=[f"pred_{label}" for label in LABELS]).to_csv(
        confusion_matrix_output
    )
    predictions.to_csv(predictions_output, index=False)
    _error_examples(predictions).to_csv(error_examples_output, index=False)
    report_output.write_text(
        _report_markdown(metrics, report_text, duplicate_safe_report_text=duplicate_safe["report_text"]),
        encoding="utf-8",
    )
    return metrics


def _duplicate_safe_evaluation(df: pd.DataFrame, *, test_size: float, random_state: int) -> dict:
    train_df, test_df = _grouped_train_test_split(df, test_size=test_size, random_state=random_state)
    pipeline = build_tfidf_classifier()
    pipeline.fit(train_df[TEXT_COLUMN], train_df[LABEL_COLUMN])

    y_true = test_df[LABEL_COLUMN]
    y_pred = pipeline.predict(test_df[TEXT_COLUMN])
    predictions = pd.DataFrame(
        {
            "row_id": test_df[ROW_ID_COLUMN].astype(int),
            "true_label": y_true,
            "predicted_label": y_pred,
        }
    )
    predictions["error_type"] = predictions.apply(_error_type, axis=1)
    metrics = _build_metrics(y_true, y_pred, predictions)
    metrics.update(
        {
            "evaluation_strategy": "group_shuffle_split_by_text",
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "train_unique_texts": int(train_df[TEXT_COLUMN].nunique()),
            "test_unique_texts": int(test_df[TEXT_COLUMN].nunique()),
            "text_overlap_count": int(
                len(set(train_df[TEXT_COLUMN].astype(str)).intersection(set(test_df[TEXT_COLUMN].astype(str))))
            ),
        }
    )
    return {
        "metrics": metrics,
        "report_text": classification_report(y_true, y_pred, labels=LABELS, zero_division=0),
    }


def _grouped_train_test_split(
    df: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = df[TEXT_COLUMN].astype(str)
    for offset in range(50):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state + offset)
        train_index, test_index = next(splitter.split(df, groups=groups))
        train_df = df.iloc[train_index].copy()
        test_df = df.iloc[test_index].copy()
        if _contains_all_labels(train_df) and _contains_all_labels(test_df):
            return train_df, test_df
    raise ModelingError(
        "Could not create a duplicate-safe split containing both labels in train and test sets."
    )


def _contains_all_labels(df: pd.DataFrame) -> bool:
    return set(df[LABEL_COLUMN].astype(str)) == set(LABELS)


def _duplicate_summary(df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    duplicated_text_mask = df[TEXT_COLUMN].duplicated(keep=False)
    train_texts = set(train_df[TEXT_COLUMN].astype(str))
    test_texts = set(test_df[TEXT_COLUMN].astype(str))
    overlapping_texts = train_texts.intersection(test_texts)
    return {
        "duplicate_message_rows": int(duplicated_text_mask.sum()),
        "duplicate_message_groups": int(df.loc[duplicated_text_mask, TEXT_COLUMN].nunique()),
        "row_split_text_overlap_count": int(len(overlapping_texts)),
        "row_split_test_rows_with_train_text": int(
            test_df[TEXT_COLUMN].astype(str).isin(overlapping_texts).sum()
        ),
    }


def _metrics_payload(row_metrics: dict, *, duplicate_summary: dict, duplicate_safe: dict) -> dict:
    payload = {
        **row_metrics,
        "evaluation_strategy": "row_stratified_split",
        "duplicate_summary": duplicate_summary,
        "duplicate_safe_evaluation": duplicate_safe,
    }
    return payload


def _counts(df: pd.DataFrame) -> dict[str, int]:
    counts = df[LABEL_COLUMN].value_counts().to_dict()
    return {label: int(counts.get(label, 0)) for label in LABELS}


def _spam_probability(pipeline, texts: pd.Series) -> list[float]:
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        classes = list(classifier.classes_)
        spam_index = classes.index("spam")
        return pipeline.predict_proba(texts)[:, spam_index].tolist()
    scores = pipeline.decision_function(texts)
    return [float(score) for score in scores]


def _error_type(row: pd.Series) -> str:
    if row["true_label"] == row["predicted_label"]:
        return ""
    if row["true_label"] == "ham" and row["predicted_label"] == "spam":
        return "false_positive_spam"
    if row["true_label"] == "spam" and row["predicted_label"] == "ham":
        return "false_negative_spam"
    return "other_error"


def _build_metrics(y_true: pd.Series, y_pred, predictions: pd.DataFrame) -> dict:
    spam_precision, spam_recall, spam_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=["spam"],
        average="binary",
        pos_label="spam",
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=LABELS,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "spam": {
            "precision": float(spam_precision),
            "recall": float(spam_recall),
            "f1": float(spam_f1),
        },
        "macro_average": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1": float(macro_f1),
        },
        "weighted_average": {
            "precision": float(weighted_precision),
            "recall": float(weighted_recall),
            "f1": float(weighted_f1),
        },
        "support": {label: int((y_true == label).sum()) for label in LABELS},
        "errors": {
            "false_positives": int((predictions["error_type"] == "false_positive_spam").sum()),
            "false_negatives": int((predictions["error_type"] == "false_negative_spam").sum()),
        },
    }


def _error_examples(predictions: pd.DataFrame) -> pd.DataFrame:
    errors = predictions[predictions["error_type"] != ""].copy()
    if errors.empty:
        return errors
    errors["confidence"] = errors.apply(
        lambda row: row["spam_probability"] if row["predicted_label"] == "spam" else 1 - row["spam_probability"],
        axis=1,
    )
    return errors.sort_values(["error_type", "confidence"], ascending=[True, False])


def _report_markdown(metrics: dict, report_text: str, *, duplicate_safe_report_text: str) -> str:
    duplicate_safe = metrics["duplicate_safe_evaluation"]
    duplicate_summary = metrics["duplicate_summary"]
    return "\n".join(
        [
            "# TF-IDF Classifier Evaluation",
            "",
            "## Summary",
            "",
            f"- Row-stratified accuracy: {metrics['accuracy']:.4f}",
            f"- Row-stratified SPAM precision: {metrics['spam']['precision']:.4f}",
            f"- Row-stratified SPAM recall: {metrics['spam']['recall']:.4f}",
            f"- Row-stratified SPAM F1: {metrics['spam']['f1']:.4f}",
            f"- Duplicate-safe SPAM F1: {duplicate_safe['spam']['f1']:.4f}",
            f"- Row-stratified false positives: {metrics['errors']['false_positives']}",
            f"- Row-stratified false negatives: {metrics['errors']['false_negatives']}",
            "",
            "## Duplicate-Safe Check",
            "",
            f"- Duplicate message rows retained in validated data: {duplicate_summary['duplicate_message_rows']}",
            f"- Duplicate message groups: {duplicate_summary['duplicate_message_groups']}",
            f"- Row-stratified split text overlap count: {duplicate_summary['row_split_text_overlap_count']}",
            f"- Row-stratified test rows with text also present in train: {duplicate_summary['row_split_test_rows_with_train_text']}",
            f"- Duplicate-safe strategy: {duplicate_safe['evaluation_strategy']}",
            f"- Duplicate-safe train/test rows: {duplicate_safe['train_rows']} / {duplicate_safe['test_rows']}",
            f"- Duplicate-safe train/test text overlap count: {duplicate_safe['text_overlap_count']}",
            f"- Duplicate-safe accuracy: {duplicate_safe['accuracy']:.4f}",
            f"- Duplicate-safe SPAM precision: {duplicate_safe['spam']['precision']:.4f}",
            f"- Duplicate-safe SPAM recall: {duplicate_safe['spam']['recall']:.4f}",
            f"- Duplicate-safe SPAM F1: {duplicate_safe['spam']['f1']:.4f}",
            "",
            "## Row-Stratified Classification Report",
            "",
            "```text",
            report_text,
            "```",
            "",
            "## Duplicate-Safe Classification Report",
            "",
            "```text",
            duplicate_safe_report_text,
            "```",
            "",
            "## Error Analysis Notes",
            "",
            "- False positives are HAM messages predicted as SPAM; these matter because they could block legitimate messages.",
            "- False negatives are SPAM messages predicted as HAM; these matter because spam reaches the inbox.",
            "- Accuracy should be interpreted alongside SPAM recall and precision because the dataset is imbalanced.",
            "- The duplicate-safe grouped split is the stricter score to cite when discussing potential duplicate leakage.",
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the TF-IDF SMS spam classifier.")
    parser.add_argument("--dataset", type=Path, default=VALIDATED_DATASET_PATH)
    parser.add_argument("--model-out", type=Path, default=TFIDF_MODEL_PATH)
    parser.add_argument("--metadata-out", type=Path, default=TRAINING_METADATA_PATH)
    parser.add_argument("--metrics-output", type=Path, default=MODEL_METRICS_PATH)
    parser.add_argument("--report-output", type=Path, default=CLASSIFICATION_REPORT_PATH)
    parser.add_argument("--confusion-matrix-output", type=Path, default=CONFUSION_MATRIX_PATH)
    parser.add_argument("--error-examples-output", type=Path, default=ERROR_EXAMPLES_PATH)
    parser.add_argument("--predictions-output", type=Path, default=TEST_PREDICTIONS_PATH)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = train_and_evaluate_model(
            args.dataset,
            model_output=args.model_out,
            metadata_output=args.metadata_out,
            metrics_output=args.metrics_output,
            report_output=args.report_output,
            confusion_matrix_output=args.confusion_matrix_output,
            error_examples_output=args.error_examples_output,
            predictions_output=args.predictions_output,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    except (ModelingError, ValueError) as exc:
        print(f"Modeling error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
