"""Build text embeddings for semantic clustering."""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import numpy as np
from scipy.sparse import vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from sms_spam_ham_analysis.config import PROJECT_ROOT


DEFAULT_SKLEARN_MODEL = "tfidf-word-bigram-truncated-svd"
DEFAULT_AZURE_API_VERSION = "2024-02-01"
DEFAULT_AZURE_AI_INFERENCE_API_VERSION = "2024-05-01-preview"
DEFAULT_AZURE_BATCH_SIZE = 64
DEFAULT_AZURE_CONCURRENCY = 4
DEFAULT_AZURE_MAX_RETRIES = 4
DEFAULT_LOCAL_CONCURRENCY = max(1, min(32, os.cpu_count() or 1))
DEFAULT_LOCAL_PROGRESS_BATCH_SIZE = 256
REQUIRED_AZURE_EMBEDDING_ENV_VARS = (
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_BATCH_SIZE",
    "AZURE_OPENAI_CONCURRENCY",
)


class EmbeddingError(ValueError):
    """Raised when embeddings cannot be created."""


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: np.ndarray
    provider: str
    model_name: str
    dimensions: int
    notes: str


def build_embeddings(
    texts: list[str],
    *,
    provider: str = "sklearn-svd",
    model_name: str | None = None,
    dimensions: int = 100,
) -> EmbeddingResult:
    """Return dense message embeddings.

    The default is a deterministic local latent semantic analysis embedding so
    the project runs without API keys or model downloads. Azure OpenAI is the
    optional GenAI provider for stronger semantic embeddings.
    """

    if not texts:
        raise EmbeddingError("At least one text is required to build embeddings.")

    provider = provider.lower().strip()
    if provider == "sklearn-svd":
        return _build_sklearn_svd_embeddings(texts, dimensions=dimensions)
    if provider == "azure-openai":
        return _build_azure_openai_embeddings(texts, model_name=model_name)
    raise EmbeddingError("Unknown embedding provider. Use 'sklearn-svd' or 'azure-openai'.")


def _build_sklearn_svd_embeddings(texts: list[str], *, dimensions: int) -> EmbeddingResult:
    if dimensions < 2:
        raise EmbeddingError("SVD embedding dimensions must be at least 2.")

    prepared_texts = _prepare_local_embedding_texts(texts)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=5000,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w[\w']*\b",
    )
    print("Fitting local TF-IDF vocabulary and IDF on full corpus...", file=sys.stderr, flush=True)
    vectorizer.fit(prepared_texts)
    print(
        f"Local TF-IDF vocabulary ready: {len(vectorizer.vocabulary_)} features.",
        file=sys.stderr,
        flush=True,
    )
    matrix = _transform_local_tfidf_batches(vectorizer, prepared_texts)
    print(
        f"TF-IDF matrix ready: {matrix.shape[0]} messages x {matrix.shape[1]} features.",
        file=sys.stderr,
        flush=True,
    )
    if matrix.shape[1] < 2:
        raise EmbeddingError("Not enough TF-IDF features to build SVD embeddings.")

    n_components = min(dimensions, matrix.shape[1] - 1, len(texts) - 1)
    print(f"Fitting TruncatedSVD with {n_components} components...", file=sys.stderr, flush=True)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    vectors = svd.fit_transform(matrix)
    print("Normalizing local embeddings...", file=sys.stderr, flush=True)
    vectors = normalize(vectors).astype("float32")
    print(
        f"Completed local SVD embeddings: {vectors.shape[0]} messages x {vectors.shape[1]} dimensions.",
        file=sys.stderr,
        flush=True,
    )
    return EmbeddingResult(
        vectors=vectors,
        provider="sklearn-svd",
        model_name=DEFAULT_SKLEARN_MODEL,
        dimensions=int(vectors.shape[1]),
        notes=(
            "Local latent semantic analysis embeddings from TF-IDF plus TruncatedSVD. "
            "This is deterministic and requires no external service."
        ),
    )


def _build_azure_openai_embeddings(texts: list[str], model_name: str | None) -> EmbeddingResult:
    validate_azure_embedding_config()
    endpoint = _required_env("AZURE_OPENAI_ENDPOINT").rstrip("/")
    api_key = _required_env("AZURE_OPENAI_API_KEY")
    deployment = model_name or _required_env("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    api_version = _azure_api_version(endpoint)
    batch_size = _positive_int_env("AZURE_OPENAI_BATCH_SIZE", DEFAULT_AZURE_BATCH_SIZE)
    concurrency = _positive_int_env("AZURE_OPENAI_CONCURRENCY", DEFAULT_AZURE_CONCURRENCY)

    batches = [texts[start : start + batch_size] for start in range(0, len(texts), batch_size)]
    print(
        f"Embedding {len(texts)} messages with Azure in {len(batches)} batches "
        f"(batch size {batch_size}, concurrency {concurrency})...",
        file=sys.stderr,
        flush=True,
    )
    batch_vectors = _post_azure_embedding_batches(
        batches,
        endpoint=endpoint,
        api_key=api_key,
        deployment=deployment,
        api_version=api_version,
        concurrency=concurrency,
    )
    vectors = [embedding for batch in batch_vectors for embedding in batch]

    array = normalize(np.asarray(vectors, dtype="float32"))
    return EmbeddingResult(
        vectors=array,
        provider="azure-openai",
        model_name=deployment,
        dimensions=int(array.shape[1]),
        notes=(
            "Azure embeddings API. Requires AZURE_OPENAI_ENDPOINT, "
            "AZURE_OPENAI_API_KEY, and AZURE_OPENAI_EMBEDDING_DEPLOYMENT. "
            f"API version: {api_version}. Batch size: {batch_size}. Concurrency: {concurrency}."
        ),
    )


def validate_azure_embedding_config() -> None:
    """Validate Azure embedding configuration without calling Azure."""

    _load_project_env_file()
    missing = [name for name in REQUIRED_AZURE_EMBEDDING_ENV_VARS if not os.getenv(name)]
    placeholder_values = [
        name
        for name in REQUIRED_AZURE_EMBEDDING_ENV_VARS
        if name not in missing and _looks_like_azure_placeholder(name, os.getenv(name, ""))
    ]

    problems = []
    if missing:
        problems.append("missing " + ", ".join(missing))
    if placeholder_values:
        problems.append("placeholder values in " + ", ".join(placeholder_values))
    if problems:
        raise EmbeddingError(
            "Azure embedding configuration is incomplete: "
            + "; ".join(problems)
            + ". Copy .env.example to .env and fill in every AZURE_OPENAI_* value before using Azure mode."
        )

    _positive_int_env("AZURE_OPENAI_BATCH_SIZE", DEFAULT_AZURE_BATCH_SIZE)
    _positive_int_env("AZURE_OPENAI_CONCURRENCY", DEFAULT_AZURE_CONCURRENCY)


def _looks_like_azure_placeholder(name: str, value: str) -> bool:
    cleaned = value.strip().lower()
    if not cleaned:
        return True
    if cleaned in {"...", "todo", "tbd", "changeme", "change-me"}:
        return True
    if "replace-with" in cleaned:
        return True
    if name == "AZURE_OPENAI_ENDPOINT" and "your-resource-name" in cleaned:
        return True
    if name == "AZURE_OPENAI_API_KEY" and ("your-key" in cleaned or "api-key" in cleaned):
        return True
    return False


def _prepare_local_embedding_texts(texts: list[str]) -> list[str]:
    return [str(text) for text in texts]


def _transform_local_tfidf_batches(vectorizer: TfidfVectorizer, texts: list[str]):
    batch_size = DEFAULT_LOCAL_PROGRESS_BATCH_SIZE
    total_batches = (len(texts) + batch_size - 1) // batch_size
    workers = min(DEFAULT_LOCAL_CONCURRENCY, total_batches)
    print(
        f"Embedding {len(texts)} messages locally with TF-IDF/SVD in {total_batches} TF-IDF batches "
        f"(batch size {batch_size}, concurrency {workers})...",
        file=sys.stderr,
        flush=True,
    )

    batches = [texts[start : start + batch_size] for start in range(0, len(texts), batch_size)]
    if workers <= 1:
        matrices = []
        for index, batch in enumerate(batches):
            _print_local_tfidf_batch_submitted(index + 1, total_batches, len(batch))
            matrices.append(vectorizer.transform(batch))
            _print_local_tfidf_batch_completed(index + 1, total_batches, index)
        return vstack(matrices, format="csr")

    results = [None] * total_batches
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for index, batch in enumerate(batches):
            _print_local_tfidf_batch_submitted(index + 1, total_batches, len(batch))
            futures.append(executor.submit(_transform_local_tfidf_batch, vectorizer, index, batch))
        for future in as_completed(futures):
            index, matrix = future.result()
            results[index] = matrix
            completed += 1
            _print_local_tfidf_batch_completed(completed, total_batches, index)

    if any(matrix is None for matrix in results):
        raise EmbeddingError("Local TF-IDF vectorization completed with missing batch results.")
    return vstack([matrix for matrix in results if matrix is not None], format="csr")


def _transform_local_tfidf_batch(vectorizer: TfidfVectorizer, batch_index: int, batch: list[str]):
    return batch_index, vectorizer.transform(batch)


def _print_local_tfidf_batch_submitted(submitted: int, total_batches: int, message_count: int) -> None:
    print(
        f"Submitted local TF-IDF batch {submitted}/{total_batches} ({message_count} messages)...",
        file=sys.stderr,
        flush=True,
    )


def _print_local_tfidf_batch_completed(completed: int, total_batches: int, batch_index: int) -> None:
    print(
        f"Completed local TF-IDF batch {completed}/{total_batches} "
        f"(batch {batch_index + 1}/{total_batches})...",
        file=sys.stderr,
        flush=True,
    )


def _post_azure_embedding_batches(
    batches: list[list[str]],
    *,
    endpoint: str,
    api_key: str,
    deployment: str,
    api_version: str,
    concurrency: int,
) -> list[list[list[float]]]:
    total_batches = len(batches)
    workers = min(concurrency, len(batches))
    if workers <= 1:
        results = []
        for index, batch in enumerate(batches):
            _print_azure_batch_submitted(index + 1, total_batches, len(batch))
            _, vectors = _post_azure_embedding_batch(
                index,
                total_batches,
                batch,
                endpoint=endpoint,
                api_key=api_key,
                deployment=deployment,
                api_version=api_version,
            )
            results.append(vectors)
            print(f"Completed Azure batch {index + 1}/{total_batches}...", file=sys.stderr, flush=True)
        return results

    results: list[list[list[float]] | None] = [None] * len(batches)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = []
        for index, batch in enumerate(batches):
            _print_azure_batch_submitted(index + 1, total_batches, len(batch))
            futures.append(
                executor.submit(
                    _post_azure_embedding_batch,
                    index,
                    total_batches,
                    batch,
                    endpoint=endpoint,
                    api_key=api_key,
                    deployment=deployment,
                    api_version=api_version,
                )
            )
        for future in as_completed(futures):
            index, vectors = future.result()
            results[index] = vectors
            completed += 1
            print(
                f"Completed Azure batch {completed}/{total_batches} "
                f"(batch {index + 1}/{total_batches})...",
                file=sys.stderr,
                flush=True,
            )

    if any(batch is None for batch in results):
        raise EmbeddingError("Azure embeddings completed with missing batch results.")
    return [batch for batch in results if batch is not None]


def _print_azure_batch_submitted(submitted: int, total_batches: int, message_count: int) -> None:
    print(
        f"Submitted Azure batch {submitted}/{total_batches} ({message_count} messages)...",
        file=sys.stderr,
        flush=True,
    )


def _post_azure_embedding_batch(
    batch_index: int,
    total_batches: int,
    batch: list[str],
    *,
    endpoint: str,
    api_key: str,
    deployment: str,
    api_version: str,
) -> tuple[int, list[list[float]]]:
    response = _post_azure_embeddings(
        endpoint=endpoint,
        api_key=api_key,
        deployment=deployment,
        api_version=api_version,
        texts=batch,
    )
    data = response.get("data")
    if not isinstance(data, list) or len(data) != len(batch):
        raise EmbeddingError("Azure OpenAI embedding response did not match the requested batch size.")
    sorted_data = sorted(data, key=lambda item: item["index"])
    return batch_index, [item["embedding"] for item in sorted_data]


def _post_azure_embeddings(
    *,
    endpoint: str,
    api_key: str,
    deployment: str,
    api_version: str,
    texts: list[str],
) -> dict:
    attempted_errors = []
    for route in _azure_embedding_routes(endpoint=endpoint, deployment=deployment, api_version=api_version):
        payload = _azure_embedding_payload(texts=texts, deployment=deployment, include_model=route.include_model)
        request = Request(
            route.url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "api-key": api_key,
            },
            method="POST",
        )
        try:
            return _urlopen_json_with_retries(request)
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            attempted_errors.append(f"{route.url} -> HTTP {exc.code}: {details}")
            if exc.code == 400 and "unavailable_model" in details:
                continue
            if exc.code != 404:
                raise EmbeddingError(f"Azure embeddings request failed with HTTP {exc.code}: {details}") from exc
        except URLError as exc:
            raise EmbeddingError(f"Azure embeddings request failed: {exc.reason}") from exc
        except json.JSONDecodeError as exc:
            raise EmbeddingError("Azure embeddings response was not valid JSON.") from exc

    raise EmbeddingError("Azure embeddings request failed. Attempted routes: " + " | ".join(attempted_errors))


def _urlopen_json_with_retries(request: Request) -> dict:
    last_url_error: URLError | None = None
    for attempt in range(DEFAULT_AZURE_MAX_RETRIES + 1):
        try:
            with urlopen(request, timeout=90) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code in {408, 409, 425, 429, 500, 502, 503, 504} and attempt < DEFAULT_AZURE_MAX_RETRIES:
                _sleep_before_retry(attempt)
                continue
            raise
        except URLError as exc:
            last_url_error = exc
            if attempt < DEFAULT_AZURE_MAX_RETRIES:
                _sleep_before_retry(attempt)
                continue
            raise

    raise last_url_error or EmbeddingError("Azure embeddings request failed after retries.")


def _sleep_before_retry(attempt: int) -> None:
    time.sleep(min(2**attempt, 8))


@dataclass(frozen=True)
class _AzureEmbeddingRoute:
    url: str
    include_model: bool


def _azure_embedding_payload(*, texts: list[str], deployment: str, include_model: bool) -> dict:
    payload = {"input": texts}
    if include_model:
        payload["model"] = deployment
    return payload


def _azure_embeddings_url(*, endpoint: str, deployment: str, api_version: str) -> str:
    return _azure_embedding_routes(endpoint=endpoint, deployment=deployment, api_version=api_version)[0].url


def _azure_embedding_routes(*, endpoint: str, deployment: str, api_version: str) -> list[_AzureEmbeddingRoute]:
    deployment_path = quote(deployment, safe="")
    routes = [
        _AzureEmbeddingRoute(url=f"{endpoint}/openai/v1/embeddings", include_model=True),
    ]
    if _is_azure_ai_inference_endpoint(endpoint):
        routes.extend(
            [
                _AzureEmbeddingRoute(
                    url=f"{endpoint}/openai/deployments/{deployment_path}/embeddings?api-version={api_version}",
                    include_model=False,
                ),
                _AzureEmbeddingRoute(
                    url=f"{endpoint}/models/{deployment_path}/embeddings?api-version={api_version}",
                    include_model=False,
                ),
                _AzureEmbeddingRoute(url=f"{endpoint}/models/embeddings?api-version={api_version}", include_model=True),
                _AzureEmbeddingRoute(url=f"{endpoint}/embeddings?api-version={api_version}", include_model=False),
            ]
        )
        return routes

    routes.append(
        _AzureEmbeddingRoute(
            url=f"{endpoint}/openai/deployments/{deployment_path}/embeddings?api-version={api_version}",
            include_model=False,
        )
    )
    return routes


def _azure_api_version(endpoint: str) -> str:
    configured = os.getenv("AZURE_OPENAI_API_VERSION")
    if _is_azure_ai_inference_endpoint(endpoint) and configured in (None, "", DEFAULT_AZURE_API_VERSION):
        return DEFAULT_AZURE_AI_INFERENCE_API_VERSION
    return configured or DEFAULT_AZURE_API_VERSION


def _is_azure_ai_inference_endpoint(endpoint: str) -> bool:
    host = urlparse(endpoint).hostname or ""
    return host.endswith(".services.ai.azure.com")


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise EmbeddingError(f"Missing required environment variable: {name}")
    return value


def _positive_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise EmbeddingError(f"{name} must be an integer, got {value!r}") from exc
    if parsed < 1:
        raise EmbeddingError(f"{name} must be at least 1, got {parsed}")
    return parsed


def _load_project_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _clean_env_value(value)
        if key.startswith("AZURE_OPENAI_"):
            os.environ[key] = value
        elif key and key not in os.environ:
            os.environ[key] = value


def _clean_env_value(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        return cleaned[1:-1]
    return cleaned
