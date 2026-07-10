"""Cluster SMS messages by semantic similarity."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from sms_spam_ham_analysis.config import (
    CLUSTER_SUMMARY_PATH,
    EMBEDDINGS_ARRAY_PATH,
    EMBEDDINGS_METADATA_PATH,
    SEMANTIC_CLUSTERS_PATH,
    VALIDATED_DATASET_PATH,
)
from sms_spam_ham_analysis.data import LABEL_COLUMN, ROW_ID_COLUMN, TEXT_COLUMN, load_validated_dataset
from sms_spam_ham_analysis.embeddings import EmbeddingError, build_embeddings


class ClusteringError(ValueError):
    """Raised when semantic clustering cannot run."""


PROVIDER_OUTPUT_DIRS = {
    "sklearn-svd": "local",
    "azure-openai": "azure",
}


@dataclass(frozen=True)
class ClusteringSummary:
    dataset: str
    assignments_output: str
    summary_output: str
    embeddings_output: str
    metadata_output: str
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    embedding_notes: str
    embedding_provider_role: str
    clustering_algorithm: str
    cluster_count: int
    total_messages: int
    silhouette_score: float | None
    random_state: int


def run_clustering(
    dataset_path: Path,
    assignments_output: Path,
    summary_output: Path,
    embeddings_output: Path,
    metadata_output: Path,
    *,
    clusters: str,
    provider: str,
    embedding_model: str | None,
    dimensions: int,
    random_state: int,
    representative_count: int,
    sample_size: int | None = None,
) -> ClusteringSummary:
    df = load_validated_dataset(dataset_path)
    if sample_size is not None:
        if sample_size < 2:
            raise ClusteringError("sample_size must be at least 2.")
        df = df.sample(n=min(sample_size, len(df)), random_state=random_state).sort_values(ROW_ID_COLUMN)

    texts = df[TEXT_COLUMN].astype(str).tolist()
    embedding_result = build_embeddings(
        texts,
        provider=provider,
        model_name=embedding_model,
        dimensions=dimensions,
    )
    vectors = embedding_result.vectors

    cluster_count, score = _select_cluster_count(vectors, clusters=clusters, random_state=random_state)
    model = KMeans(n_clusters=cluster_count, random_state=random_state, n_init=20)
    cluster_ids = model.fit_predict(vectors)
    distances = model.transform(vectors)
    assigned_distances = distances[np.arange(len(cluster_ids)), cluster_ids]
    coordinates = _project_to_2d(vectors)

    assignments = pd.DataFrame(
        {
            "row_id": df[ROW_ID_COLUMN].astype(int),
            "label": df[LABEL_COLUMN],
            "text": df[TEXT_COLUMN],
            "cluster_id": cluster_ids.astype(int),
            "distance_to_centroid": assigned_distances.astype(float),
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
        }
    ).sort_values(["cluster_id", "distance_to_centroid", "row_id"])

    summary = ClusteringSummary(
        dataset=str(dataset_path),
        assignments_output=str(assignments_output),
        summary_output=str(summary_output),
        embeddings_output=str(embeddings_output),
        metadata_output=str(metadata_output),
        embedding_provider=embedding_result.provider,
        embedding_model=embedding_result.model_name,
        embedding_dimensions=embedding_result.dimensions,
        embedding_notes=embedding_result.notes,
        embedding_provider_role=_embedding_provider_role(embedding_result.provider),
        clustering_algorithm="KMeans",
        cluster_count=cluster_count,
        total_messages=int(len(assignments)),
        silhouette_score=score,
        random_state=random_state,
    )
    _write_clustering_artifacts(
        assignments=assignments,
        vectors=vectors,
        summary=summary,
        representative_count=representative_count,
    )
    provider_summary = _provider_specific_summary(summary, provider=embedding_result.provider, outputs_dir=summary_output.parent)
    _write_clustering_artifacts(
        assignments=assignments,
        vectors=vectors,
        summary=provider_summary,
        representative_count=representative_count,
    )
    return summary


def _write_clustering_artifacts(
    *,
    assignments: pd.DataFrame,
    vectors: np.ndarray,
    summary: ClusteringSummary,
    representative_count: int,
) -> None:
    assignments_output = Path(summary.assignments_output)
    summary_output = Path(summary.summary_output)
    embeddings_output = Path(summary.embeddings_output)
    metadata_output = Path(summary.metadata_output)

    assignments_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    embeddings_output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    assignments.to_csv(assignments_output, index=False)
    np.save(embeddings_output, vectors)
    metadata_output.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
    summary_output.write_text(
        _build_cluster_summary(summary, assignments, representative_count=representative_count),
        encoding="utf-8",
    )


def _provider_specific_summary(summary: ClusteringSummary, *, provider: str, outputs_dir: Path) -> ClusteringSummary:
    provider_dir = outputs_dir / "clustering" / _provider_output_dir_name(provider)
    return replace(
        summary,
        assignments_output=str(provider_dir / "semantic_clusters.csv"),
        summary_output=str(provider_dir / "cluster_summary.md"),
        embeddings_output=str(provider_dir / "embeddings" / "sms_embeddings.npy"),
        metadata_output=str(provider_dir / "embeddings" / "metadata.json"),
    )


def _provider_output_dir_name(provider: str) -> str:
    return PROVIDER_OUTPUT_DIRS.get(provider, provider.replace("-", "_"))


def _select_cluster_count(vectors: np.ndarray, *, clusters: str, random_state: int) -> tuple[int, float | None]:
    if clusters != "auto":
        cluster_count = int(clusters)
        if cluster_count < 2:
            raise ClusteringError("Cluster count must be at least 2.")
        if cluster_count >= len(vectors):
            raise ClusteringError("Cluster count must be smaller than the number of messages.")
        return cluster_count, _silhouette(vectors, cluster_count, random_state)

    max_clusters = min(8, len(vectors) - 1)
    candidates = range(3, max_clusters + 1)
    best_count = 3
    best_score: float | None = None
    for candidate in candidates:
        score = _silhouette(vectors, candidate, random_state)
        if best_score is None or score > best_score:
            best_count = candidate
            best_score = score
    return best_count, best_score


def _silhouette(vectors: np.ndarray, cluster_count: int, random_state: int) -> float:
    labels = KMeans(n_clusters=cluster_count, random_state=random_state, n_init=10).fit_predict(vectors)
    return float(silhouette_score(vectors, labels, metric="cosine"))


def _project_to_2d(vectors: np.ndarray) -> np.ndarray:
    if vectors.shape[1] == 2:
        return vectors.astype(float)
    return PCA(n_components=2, random_state=42).fit_transform(vectors).astype(float)


def _build_cluster_summary(
    summary: ClusteringSummary,
    assignments: pd.DataFrame,
    *,
    representative_count: int,
) -> str:
    lines = [
        "# Semantic SMS Clusters",
        "",
        "## Method",
        "",
        f"- Embedding provider: `{summary.embedding_provider}`",
        f"- Embedding model: `{summary.embedding_model}`",
        f"- Embedding dimensions: {summary.embedding_dimensions}",
        f"- Embedding notes: {summary.embedding_notes}",
        f"- Provider role: {summary.embedding_provider_role}",
        f"- Clustering algorithm: {summary.clustering_algorithm}",
        f"- Cluster count: {summary.cluster_count}",
        f"- Random state: {summary.random_state}",
        f"- Silhouette score: {_format_optional_float(summary.silhouette_score)}",
        "",
        "## Cluster Overview",
        "",
        "| cluster | size | ham | spam | spam_rate | representative theme |",
        "| ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for cluster_id, cluster_df in assignments.groupby("cluster_id", sort=True):
        label_counts = cluster_df["label"].value_counts()
        ham = int(label_counts.get("ham", 0))
        spam = int(label_counts.get("spam", 0))
        spam_rate = spam / len(cluster_df)
        theme = _infer_theme(cluster_df)
        lines.append(f"| {cluster_id} | {len(cluster_df)} | {ham} | {spam} | {spam_rate:.3f} | {theme} |")

    for cluster_id, cluster_df in assignments.groupby("cluster_id", sort=True):
        lines.extend(["", f"## Cluster {cluster_id} Representatives", ""])
        representatives = cluster_df.sort_values("distance_to_centroid").head(representative_count)
        for row in representatives.itertuples(index=False):
            text = str(row.text).replace("|", "/")
            lines.append(f"- `{row.label}` row `{row.row_id}`: {text}")

    lines.append("")
    return "\n".join(lines)


def _embedding_provider_role(provider: str) -> str:
    if provider == "azure-openai":
        return "GenAI semantic embedding path."
    return "Local reproducibility fallback; this is not an LLM embedding model."


def _infer_theme(cluster_df: pd.DataFrame) -> str:
    spam_rate = (cluster_df["label"] == "spam").mean()
    sample_text = " ".join(cluster_df.sort_values("distance_to_centroid").head(20)["text"].astype(str)).lower()
    if spam_rate >= 0.65:
        if any(term in sample_text for term in ("free", "claim", "prize", "win", "won")):
            return "promotional prize or claim spam"
        if any(term in sample_text for term in ("txt", "text", "stop", "reply")):
            return "text/reply campaign spam"
        return "spam-heavy promotional messages"
    if any(term in sample_text for term in ("call", "later", "home", "going", "come")):
        return "personal coordination messages"
    if any(term in sample_text for term in ("love", "dear", "happy", "sorry")):
        return "personal relationship messages"
    return "mixed conversational messages"


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "not calculated"
    return f"{value:.4f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cluster SMS messages using text embeddings.")
    parser.add_argument("--dataset", type=Path, default=VALIDATED_DATASET_PATH)
    parser.add_argument("--assignments-output", type=Path, default=SEMANTIC_CLUSTERS_PATH)
    parser.add_argument("--summary-output", type=Path, default=CLUSTER_SUMMARY_PATH)
    parser.add_argument("--embeddings-output", type=Path, default=EMBEDDINGS_ARRAY_PATH)
    parser.add_argument("--metadata-output", type=Path, default=EMBEDDINGS_METADATA_PATH)
    parser.add_argument("--clusters", default="auto", help="Number of clusters or 'auto'.")
    parser.add_argument(
        "--provider",
        choices=["sklearn-svd", "azure-openai"],
        default="sklearn-svd",
    )
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--dimensions", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--representatives", type=int, default=5)
    parser.add_argument("--sample-size", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_clustering(
            args.dataset,
            args.assignments_output,
            args.summary_output,
            args.embeddings_output,
            args.metadata_output,
            clusters=args.clusters,
            provider=args.provider,
            embedding_model=args.embedding_model,
            dimensions=args.dimensions,
            random_state=args.random_state,
            representative_count=args.representatives,
            sample_size=args.sample_size,
        )
    except (ClusteringError, EmbeddingError, ValueError) as exc:
        print(f"Clustering error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
