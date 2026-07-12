"""Generate compact figures and an artifact index for the SMS project."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "sms_spam_ham_analysis_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score, precision_recall_curve

from sms_spam_ham_analysis.config import (
    ARTIFACT_INDEX_PATH,
    CLASSIFICATION_REPORT_PATH,
    CLUSTER_PROVIDER_COMPARISON_PATH,
    CLUSTER_SUMMARY_PATH,
    CONFUSION_MATRIX_FIGURE_PATH,
    CONFUSION_MATRIX_PATH,
    ERROR_EXAMPLES_PATH,
    FIGURES_DIR,
    FREQUENT_BIGRAMS_PATH,
    FREQUENT_TRIGRAMS_PATH,
    FREQUENT_WORDS_CSV_PATH,
    HAM_TYPICAL_WORDS_PATH,
    MODEL_METRICS_PATH,
    MODEL_METRICS_COMPARISON_FIGURE_PATH,
    NGRAMS_FIGURE_PATH,
    PRECISION_RECALL_CURVE_FIGURE_PATH,
    SEMANTIC_CLUSTERS_FIGURE_PATH,
    SEMANTIC_CLUSTERS_PATH,
    SPAM_TYPICAL_WORDS_PATH,
    TOP_WORDS_FIGURE_PATH,
    TEST_PREDICTIONS_PATH,
    VOCABULARY_FIGURE_PATH,
    VOCABULARY_FINDINGS_PATH,
)


class VisualizationError(ValueError):
    """Raised when figures cannot be generated."""


@dataclass(frozen=True)
class VisualizationSummary:
    outputs_dir: str
    figures_dir: str
    artifact_index: str
    figures: list[str]


def run_visualizations(outputs_dir: Path, figures_dir: Path, artifact_index: Path) -> VisualizationSummary:
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    figures = [
        _plot_top_words(outputs_dir / FREQUENT_WORDS_CSV_PATH.name, figures_dir / TOP_WORDS_FIGURE_PATH.name),
        _plot_vocabulary(
            outputs_dir / SPAM_TYPICAL_WORDS_PATH.name,
            outputs_dir / HAM_TYPICAL_WORDS_PATH.name,
            figures_dir / VOCABULARY_FIGURE_PATH.name,
        ),
        _plot_ngrams(
            outputs_dir / FREQUENT_BIGRAMS_PATH.name,
            outputs_dir / FREQUENT_TRIGRAMS_PATH.name,
            figures_dir / NGRAMS_FIGURE_PATH.name,
        ),
        _plot_confusion_matrix(
            outputs_dir / CONFUSION_MATRIX_PATH.name,
            outputs_dir / MODEL_METRICS_PATH.name,
            figures_dir / CONFUSION_MATRIX_FIGURE_PATH.name,
        ),
        _plot_model_metrics_comparison(
            outputs_dir / MODEL_METRICS_PATH.name,
            figures_dir / MODEL_METRICS_COMPARISON_FIGURE_PATH.name,
        ),
        _plot_precision_recall_curve(
            outputs_dir / TEST_PREDICTIONS_PATH.name,
            outputs_dir / MODEL_METRICS_PATH.name,
            figures_dir / PRECISION_RECALL_CURVE_FIGURE_PATH.name,
        ),
        _plot_semantic_clusters(_semantic_clusters_input(outputs_dir), figures_dir / SEMANTIC_CLUSTERS_FIGURE_PATH.name),
    ]

    _write_provider_comparison(outputs_dir, outputs_dir / "clustering" / CLUSTER_PROVIDER_COMPARISON_PATH.name)
    artifact_index.write_text(_build_artifact_index(outputs_dir, figures), encoding="utf-8")
    return VisualizationSummary(
        outputs_dir=str(outputs_dir),
        figures_dir=str(figures_dir),
        artifact_index=str(artifact_index),
        figures=[str(path) for path in figures],
    )


def _plot_top_words(input_path: Path, output_path: Path) -> Path:
    df = _read_csv(input_path).head(20).sort_values("count")
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(df["token"], df["count"], color="#4c78a8")
    ax.set_title("Top 20 Frequent SMS Tokens")
    ax.set_xlabel("Count")
    ax.set_ylabel("Token")
    _save(fig, output_path)
    return output_path


def _plot_vocabulary(spam_path: Path, ham_path: Path, output_path: Path) -> Path:
    spam = _read_csv(spam_path).head(15).assign(label="spam", score=lambda df: df["log_rate_ratio_spam_vs_ham"])
    ham = _read_csv(ham_path).head(15).assign(label="ham", score=lambda df: df["log_rate_ratio_ham_vs_spam"])
    df = pd.concat([spam[["token", "score", "label"]], ham[["token", "score", "label"]]], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharex=False)
    for ax, label, color in zip(axes, ["spam", "ham"], ["#e45756", "#54a24b"], strict=True):
        subset = df[df["label"] == label].sort_values("score")
        ax.barh(subset["token"], subset["score"], color=color)
        ax.set_title(f"Top {label.upper()}-Typical Tokens")
        ax.set_xlabel("Log rate ratio")
        ax.set_ylabel("")
    _save(fig, output_path)
    return output_path


def _plot_ngrams(bigram_path: Path, trigram_path: Path, output_path: Path) -> Path:
    bigrams = _read_csv(bigram_path).head(10).assign(kind="bigram")
    trigrams = _read_csv(trigram_path).head(10).assign(kind="trigram")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, df, title, color in zip(
        axes,
        [bigrams, trigrams],
        ["Top Bigrams", "Top Trigrams"],
        ["#f58518", "#72b7b2"],
        strict=True,
    ):
        subset = df.sort_values("count")
        ax.barh(subset["ngram"], subset["count"], color=color)
        ax.set_title(title)
        ax.set_xlabel("Count")
        ax.set_ylabel("")
    _save(fig, output_path)
    return output_path


def _plot_confusion_matrix(input_path: Path, metrics_path: Path, output_path: Path) -> Path:
    row_matrix = pd.read_csv(input_path, index_col=0).to_numpy(dtype=int)
    if row_matrix.shape != (2, 2):
        raise VisualizationError(f"Expected a 2x2 confusion matrix, got {row_matrix.shape} from {input_path}")

    metrics = _read_json(metrics_path)
    duplicate_safe = metrics["duplicate_safe_evaluation"]
    safe_false_positives = int(duplicate_safe["errors"]["false_positives"])
    safe_false_negatives = int(duplicate_safe["errors"]["false_negatives"])
    safe_matrix = np.array(
        [
            [int(duplicate_safe["support"]["ham"]) - safe_false_positives, safe_false_positives],
            [safe_false_negatives, int(duplicate_safe["support"]["spam"]) - safe_false_negatives],
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharex=True, sharey=True)
    specifications = [
        (
            axes[0],
            row_matrix,
            "Row-stratified split",
            f"Text overlap: {metrics['duplicate_summary']['row_split_text_overlap_count']} | n={int(row_matrix.sum()):,}",
        ),
        (
            axes[1],
            safe_matrix,
            "Duplicate-safe split",
            f"Text overlap: {duplicate_safe['text_overlap_count']} | n={int(safe_matrix.sum()):,}",
        ),
    ]
    for ax, matrix, title, subtitle in specifications:
        row_totals = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, row_totals, where=row_totals != 0)
        annotations = np.array(
            [
                [f"{matrix[row, column]:,}\n{normalized[row, column]:.1%}" for column in range(2)]
                for row in range(2)
            ]
        )
        sns.heatmap(
            normalized,
            annot=annotations,
            fmt="",
            cmap="Blues",
            vmin=0,
            vmax=1,
            cbar=False,
            square=True,
            linewidths=2,
            linecolor="white",
            xticklabels=["HAM", "SPAM"],
            yticklabels=["HAM", "SPAM"],
            ax=ax,
        )
        ax.set_title(f"{title}\n{subtitle}", fontsize=15)
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label" if ax is axes[0] else "")
        ax.tick_params(axis="both", rotation=0)
    fig.suptitle("TF-IDF Classifier: Conventional vs Leakage-Aware Evaluation", fontsize=19, y=1.03)
    _save(fig, output_path)
    return output_path


def _plot_model_metrics_comparison(metrics_path: Path, output_path: Path) -> Path:
    metrics = _read_json(metrics_path)
    duplicate_safe = metrics["duplicate_safe_evaluation"]
    metric_names = ["Precision", "Recall", "F1"]
    row_values = [metrics["spam"][name.lower()] for name in metric_names]
    safe_values = [duplicate_safe["spam"][name.lower()] for name in metric_names]

    positions = np.arange(len(metric_names))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 6))
    row_bars = ax.bar(positions - width / 2, row_values, width, label="Row-stratified", color="#4c78a8")
    safe_bars = ax.bar(positions + width / 2, safe_values, width, label="Duplicate-safe", color="#f58518")
    for bars in (row_bars, safe_bars):
        ax.bar_label(bars, labels=[f"{bar.get_height():.1%}" for bar in bars], padding=4, fontsize=11)
    ax.set_title("SPAM Classification Performance by Evaluation Strategy")
    ax.set_ylabel("Score")
    ax.set_xticks(positions, metric_names)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower center", ncol=2, fontsize=11)
    ax.text(
        0.5,
        -0.14,
        "Duplicate-safe evaluation keeps identical SMS text in only one split (train/test overlap = 0).",
        transform=ax.transAxes,
        ha="center",
        fontsize=10,
    )
    _save(fig, output_path)
    return output_path


def _plot_precision_recall_curve(predictions_path: Path, metrics_path: Path, output_path: Path) -> Path:
    predictions = _read_csv(predictions_path)
    required = {"true_label", "spam_probability"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise VisualizationError(f"Test predictions missing columns: {missing}")

    y_true = predictions["true_label"].eq("spam").astype(int)
    y_score = predictions["spam_probability"].astype(float)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)
    prevalence = float(y_true.mean())
    metrics = _read_json(metrics_path)

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot(recall, precision, color="#4c78a8", linewidth=3, label=f"PR curve (AP = {average_precision:.3f})")
    ax.axhline(prevalence, color="#9c755f", linestyle="--", linewidth=2, label=f"Spam prevalence = {prevalence:.1%}")
    ax.scatter(
        [metrics["spam"]["recall"]],
        [metrics["spam"]["precision"]],
        color="#e45756",
        s=110,
        zorder=3,
        label="Current decision threshold",
    )
    ax.set_title("SPAM Precision–Recall Tradeoff (Row-Stratified Test Split)")
    ax.set_xlabel("Recall — share of spam detected")
    ax.set_ylabel("Precision — share of spam predictions correct")
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0, 1.03)
    ax.legend(loc="lower left")
    _save(fig, output_path)
    return output_path


def _plot_semantic_clusters(input_path: Path, output_path: Path) -> Path:
    df = _read_csv(input_path)
    required = {"x", "y", "cluster_id", "label"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise VisualizationError(f"Semantic cluster output missing columns: {missing}")

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=df,
        x="x",
        y="y",
        hue="cluster_id",
        style="label",
        palette="tab10",
        alpha=0.75,
        s=35,
        ax=ax,
    )
    ax.set_title("Semantic SMS Clusters")
    ax.set_xlabel("Embedding projection X")
    ax.set_ylabel("Embedding projection Y")
    ax.legend(title="Cluster / label", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    _save(fig, output_path)
    return output_path


def _build_artifact_index(outputs_dir: Path, figures: list[Path]) -> str:
    outputs_dir = outputs_dir.resolve()
    artifacts = [
        outputs_dir / "dataset_validation.json",
        outputs_dir / "frequent_words.md",
        outputs_dir / VOCABULARY_FINDINGS_PATH.name,
        outputs_dir / "ngram_findings.md",
        outputs_dir / CLASSIFICATION_REPORT_PATH.name,
        outputs_dir / MODEL_METRICS_PATH.name,
        outputs_dir / CONFUSION_MATRIX_PATH.name,
        outputs_dir / ERROR_EXAMPLES_PATH.name,
        outputs_dir / CLUSTER_SUMMARY_PATH.name,
    ]
    comparison_path = outputs_dir / "clustering" / CLUSTER_PROVIDER_COMPARISON_PATH.name
    if comparison_path.exists():
        artifacts.append(comparison_path)
    for provider in ("local", "azure"):
        provider_dir = outputs_dir / "clustering" / provider
        if provider_dir.exists():
            artifacts.extend(
                [
                    provider_dir / "cluster_summary.md",
                    provider_dir / "embeddings" / "metadata.json",
                ]
            )
    lines = [
        "# Artifact Index",
        "",
        "## Tables, Reports, And Models",
        "",
    ]
    print("Indexing generated tables, reports, and models...", file=sys.stderr)
    for path in artifacts:
        status = "present" if path.exists() else "missing"
        display_path = _display_path(path)
        print(f"  [{status}] {display_path}", file=sys.stderr)
        lines.append(f"- `{display_path}` - {status}")
    lines.extend(["", "## Figures", ""])
    print("Indexing generated figures...", file=sys.stderr)
    for path in figures:
        status = "present" if path.exists() else "missing"
        display_path = _display_path(path)
        print(f"  [{status}] {display_path}", file=sys.stderr)
        lines.append(f"- `{display_path}` - {status}")
    lines.append("")
    return "\n".join(lines)


def _write_provider_comparison(outputs_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_build_provider_comparison(outputs_dir), encoding="utf-8")


def _build_provider_comparison(outputs_dir: Path) -> str:
    provider_specs = [
        (
            "local",
            "Local reproducibility fallback",
            "No API key, no external service, deterministic review path.",
        ),
        (
            "azure",
            "GenAI embedding path",
            "Uses Azure OpenAI / Azure AI Foundry embeddings; requires `.env` and may incur cost.",
        ),
    ]
    rows = []
    available = []
    for folder_name, role, tradeoff in provider_specs:
        metadata_path = outputs_dir / "clustering" / folder_name / "embeddings" / "metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            available.append((folder_name, metadata))
            rows.append(
                "| "
                + " | ".join(
                    [
                        folder_name,
                        "generated",
                        str(metadata.get("embedding_provider", "")),
                        str(metadata.get("embedding_model", "")),
                        str(metadata.get("embedding_dimensions", "")),
                        str(metadata.get("total_messages", "")),
                        str(metadata.get("cluster_count", "")),
                        _format_optional_float(metadata.get("silhouette_score")),
                        role,
                    ]
                )
                + " |"
            )
        else:
            rows.append(
                "| "
                + " | ".join([folder_name, "not generated", "-", "-", "-", "-", "-", "-", role])
                + " |"
            )

    lines = [
        "# Local vs Azure Clustering Provider Comparison",
        "",
        "| output folder | status | embedding provider | model | dimensions | messages | clusters | silhouette | role |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        *rows,
        "",
        "## Interpretation",
        "",
        "- The local provider is the no-key reproducibility path and is suitable for local execution without cloud credentials.",
        "- The Azure provider is the GenAI embedding path expected for semantic clustering when credentials are configured.",
        "- The silhouette score is an unsupervised separation signal, not a classifier metric; compare it with the cluster themes before claiming one provider is better.",
        "- Cost, privacy, latency, and company data policy matter for Azure because SMS text is sent to the configured embedding service.",
    ]
    if len(available) < 2:
        lines.extend(
            [
                "",
                "## Missing Provider Runs",
                "",
                "- Azure comparison artifacts are absent until an Azure clustering command is run, for example `bash scripts/run_pipeline.sh --use-azure --azure-sample-size 250 --azure-clusters 6`.",
            ]
        )
    else:
        best = max(available, key=lambda item: item[1].get("silhouette_score") or float("-inf"))
        lines.extend(
            [
                "",
                "## Metric Note",
                "",
                f"- By silhouette score alone, `{best[0]}` is currently higher. This should be treated as one diagnostic, not a final quality verdict.",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def _format_optional_float(value: object) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "-"


def _display_path(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        return path.name
    return relative.as_posix()


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise VisualizationError(f"Required output not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise VisualizationError(f"Required output is empty: {path}")
    return df


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise VisualizationError(f"Required output not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _semantic_clusters_input(outputs_dir: Path) -> Path:
    return outputs_dir / SEMANTIC_CLUSTERS_PATH.name


def _save(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate project figures and an artifact index.")
    parser.add_argument("--outputs", type=Path, default=FIGURES_DIR.parent)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    parser.add_argument("--artifact-index", type=Path, default=ARTIFACT_INDEX_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_visualizations(args.outputs, args.figures_dir, args.artifact_index)
    except (VisualizationError, ValueError) as exc:
        print(f"Visualization error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
