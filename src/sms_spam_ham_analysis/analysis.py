"""Run the exploratory SMS text analyses in one workflow."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from sms_spam_ham_analysis.config import (
    FREQUENT_BIGRAMS_PATH,
    FREQUENT_TRIGRAMS_PATH,
    FREQUENT_WORDS_CSV_PATH,
    FREQUENT_WORDS_MARKDOWN_PATH,
    HAM_TYPICAL_WORDS_PATH,
    NGRAM_FINDINGS_PATH,
    SPAM_TYPICAL_WORDS_PATH,
    VALIDATED_DATASET_PATH,
    VOCABULARY_BY_LABEL_PATH,
    VOCABULARY_FINDINGS_PATH,
)
from sms_spam_ham_analysis.data import EXPECTED_LABELS, LABEL_COLUMN, TEXT_COLUMN, load_validated_dataset
from sms_spam_ham_analysis.text import make_ngrams, tokenize_text


class TextAnalysisError(ValueError):
    """Raised when exploratory text analysis cannot run."""


@dataclass(frozen=True)
class TextAnalysisSummary:
    dataset: str
    total_messages: int
    total_tokens: int
    unique_tokens: int
    ham_messages: int
    spam_messages: int
    frequent_words_output: str
    vocabulary_output: str
    ngram_outputs: list[str]


def run_text_analysis(
    dataset_path: Path,
    *,
    frequent_words_output: Path = FREQUENT_WORDS_CSV_PATH,
    frequent_words_markdown: Path = FREQUENT_WORDS_MARKDOWN_PATH,
    vocabulary_output: Path = VOCABULARY_BY_LABEL_PATH,
    spam_output: Path = SPAM_TYPICAL_WORDS_PATH,
    ham_output: Path = HAM_TYPICAL_WORDS_PATH,
    vocabulary_findings: Path = VOCABULARY_FINDINGS_PATH,
    bigrams_output: Path = FREQUENT_BIGRAMS_PATH,
    trigrams_output: Path = FREQUENT_TRIGRAMS_PATH,
    ngram_findings: Path = NGRAM_FINDINGS_PATH,
    top_words: int = 50,
    top_typical_words: int = 30,
    top_ngrams: int = 30,
    min_count: int = 1,
    smoothing: float = 0.5,
    remove_stopwords: bool = False,
    min_token_length: int = 1,
) -> TextAnalysisSummary:
    """Generate all text-analysis outputs from the validated dataset."""

    if top_words < 1 or top_typical_words < 1 or top_ngrams < 1:
        raise TextAnalysisError("top counts must be at least 1")
    if min_count < 1:
        raise TextAnalysisError("min_count must be at least 1")
    if smoothing <= 0:
        raise TextAnalysisError("smoothing must be greater than 0")

    df = load_validated_dataset(dataset_path)
    tokenized = [
        tokenize_text(text, remove_stopwords=remove_stopwords, min_token_length=min_token_length)
        for text in df[TEXT_COLUMN]
    ]

    frequent_words, token_counts, document_counts = _frequent_words(df, tokenized, top_words)
    vocabulary_by_label, spam_typical, ham_typical, vocabulary_stats = _vocabulary_tables(
        df,
        tokenized,
        top_n=top_typical_words,
        min_count=min_count,
        smoothing=smoothing,
    )
    ngram_tables = _ngram_tables(df, tokenized, top_n=top_ngrams, min_count=min_count)

    _write_csv(frequent_words_output, frequent_words)
    _write_csv(vocabulary_output, vocabulary_by_label)
    _write_csv(spam_output, spam_typical)
    _write_csv(ham_output, ham_typical)
    _write_csv(bigrams_output, ngram_tables[2])
    _write_csv(trigrams_output, ngram_tables[3])

    total_tokens = sum(token_counts.values())
    summary = TextAnalysisSummary(
        dataset=str(dataset_path),
        total_messages=len(df),
        total_tokens=total_tokens,
        unique_tokens=len(token_counts),
        ham_messages=vocabulary_stats["ham_messages"],
        spam_messages=vocabulary_stats["spam_messages"],
        frequent_words_output=str(frequent_words_output),
        vocabulary_output=str(vocabulary_output),
        ngram_outputs=[str(bigrams_output), str(trigrams_output)],
    )

    frequent_words_markdown.parent.mkdir(parents=True, exist_ok=True)
    frequent_words_markdown.write_text(
        _frequent_words_markdown(
            frequent_words,
            dataset_path=dataset_path,
            total_messages=len(df),
            messages_with_tokens=sum(1 for tokens in tokenized if tokens),
            total_tokens=total_tokens,
            unique_tokens=len(token_counts),
            top_n=top_words,
            remove_stopwords=remove_stopwords,
            min_token_length=min_token_length,
        ),
        encoding="utf-8",
    )
    vocabulary_findings.parent.mkdir(parents=True, exist_ok=True)
    vocabulary_findings.write_text(
        _vocabulary_markdown(vocabulary_stats, spam_typical, ham_typical, smoothing=smoothing, min_count=min_count),
        encoding="utf-8",
    )
    ngram_findings.parent.mkdir(parents=True, exist_ok=True)
    ngram_findings.write_text(
        _ngram_markdown(
            ngram_tables,
            dataset_path=dataset_path,
            total_messages=len(df),
            top_n=top_ngrams,
            min_count=min_count,
            remove_stopwords=remove_stopwords,
            min_token_length=min_token_length,
        ),
        encoding="utf-8",
    )
    return summary


def _frequent_words(
    df: pd.DataFrame,
    tokenized: list[list[str]],
    top_n: int,
) -> tuple[pd.DataFrame, Counter[str], Counter[str]]:
    token_counts: Counter[str] = Counter()
    document_counts: Counter[str] = Counter()
    for tokens in tokenized:
        token_counts.update(tokens)
        document_counts.update(set(tokens))
    if not token_counts:
        raise TextAnalysisError("No tokens found after preprocessing. Relax filtering settings.")

    total_tokens = sum(token_counts.values())
    rows = []
    for rank, (token, count) in enumerate(token_counts.most_common(top_n), start=1):
        doc_count = document_counts[token]
        rows.append(
            {
                "rank": rank,
                "token": token,
                "count": count,
                "percentage": round((count / total_tokens) * 100, 4),
                "document_count": doc_count,
                "document_percentage": round((doc_count / len(df)) * 100, 4),
            }
        )
    return pd.DataFrame(rows), token_counts, document_counts


def _vocabulary_tables(
    df: pd.DataFrame,
    tokenized: list[list[str]],
    *,
    top_n: int,
    min_count: int,
    smoothing: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    token_counts: dict[str, Counter[str]] = {label: Counter() for label in EXPECTED_LABELS}
    document_counts: dict[str, Counter[str]] = {label: Counter() for label in EXPECTED_LABELS}
    message_counts = {label: 0 for label in EXPECTED_LABELS}

    for label, tokens in zip(df[LABEL_COLUMN].astype(str), tokenized, strict=True):
        label = label.lower()
        message_counts[label] += 1
        token_counts[label].update(tokens)
        document_counts[label].update(set(tokens))

    totals = {label: sum(token_counts[label].values()) for label in EXPECTED_LABELS}
    if totals["ham"] == 0 or totals["spam"] == 0:
        raise TextAnalysisError("Both HAM and SPAM classes need at least one token after preprocessing.")

    vocabulary = sorted(set(token_counts["ham"]).union(token_counts["spam"]))
    by_label = _vocabulary_by_label(token_counts, document_counts, totals, message_counts)
    scored = _score_terms(token_counts, totals, vocabulary, min_count=min_count, smoothing=smoothing)
    spam_typical = _with_rank(
        scored.sort_values(["log_rate_ratio_spam_vs_ham", "spam_count", "token"], ascending=[False, False, True]).head(
            top_n
        )
    )
    ham_typical = _with_rank(
        scored.sort_values(["log_rate_ratio_ham_vs_spam", "ham_count", "token"], ascending=[False, False, True]).head(
            top_n
        )
    )

    stats = {
        "ham_messages": message_counts["ham"],
        "spam_messages": message_counts["spam"],
        "ham_tokens": totals["ham"],
        "spam_tokens": totals["spam"],
        "total_vocabulary_size": len(vocabulary),
        "shared_vocabulary_size": len(set(token_counts["ham"]).intersection(token_counts["spam"])),
    }
    return by_label, spam_typical, ham_typical, stats


def _vocabulary_by_label(
    token_counts: dict[str, Counter[str]],
    document_counts: dict[str, Counter[str]],
    totals: dict[str, int],
    message_counts: dict[str, int],
) -> pd.DataFrame:
    rows = []
    for label in EXPECTED_LABELS:
        for token, count in token_counts[label].most_common():
            rows.append(
                {
                    "label": label,
                    "token": token,
                    "count": count,
                    "percentage": round((count / totals[label]) * 100, 4),
                    "document_count": document_counts[label][token],
                    "document_percentage": round((document_counts[label][token] / message_counts[label]) * 100, 4),
                }
            )
    return pd.DataFrame(rows)


def _score_terms(
    token_counts: dict[str, Counter[str]],
    totals: dict[str, int],
    vocabulary: list[str],
    *,
    min_count: int,
    smoothing: float,
) -> pd.DataFrame:
    vocab_size = len(vocabulary)
    rows = []
    for token in vocabulary:
        ham_count = token_counts["ham"][token]
        spam_count = token_counts["spam"][token]
        total_count = ham_count + spam_count
        if total_count < min_count:
            continue
        ham_rate = (ham_count + smoothing) / (totals["ham"] + smoothing * vocab_size)
        spam_rate = (spam_count + smoothing) / (totals["spam"] + smoothing * vocab_size)
        spam_vs_ham = math.log(spam_rate / ham_rate)
        rows.append(
            {
                "token": token,
                "spam_count": spam_count,
                "ham_count": ham_count,
                "total_count": total_count,
                "spam_percentage": round((spam_count / totals["spam"]) * 100, 4),
                "ham_percentage": round((ham_count / totals["ham"]) * 100, 4),
                "smoothed_spam_rate": spam_rate,
                "smoothed_ham_rate": ham_rate,
                "log_rate_ratio_spam_vs_ham": spam_vs_ham,
                "log_rate_ratio_ham_vs_spam": -spam_vs_ham,
            }
        )
    if not rows:
        raise TextAnalysisError("No vocabulary terms remain after min_count filtering.")
    return pd.DataFrame(rows)


def _ngram_tables(
    df: pd.DataFrame,
    tokenized: list[list[str]],
    *,
    top_n: int,
    min_count: int,
) -> dict[int, pd.DataFrame]:
    tables = {}
    for size in (2, 3):
        counts: Counter[str] = Counter()
        document_counts: Counter[str] = Counter()
        for tokens in tokenized:
            ngrams = make_ngrams(tokens, size)
            counts.update(ngrams)
            document_counts.update(set(ngrams))

        total = sum(counts.values())
        rows = []
        rank = 1
        for ngram, count in counts.most_common():
            if count < min_count:
                continue
            rows.append(
                {
                    "rank": rank,
                    "n": size,
                    "ngram": ngram,
                    "label": "all",
                    "count": count,
                    "percentage": round((count / total) * 100, 4) if total else 0,
                    "document_count": document_counts[ngram],
                    "document_percentage": round((document_counts[ngram] / len(df)) * 100, 4),
                }
            )
            rank += 1
            if len(rows) >= top_n:
                break
        tables[size] = pd.DataFrame(
            rows,
            columns=[
                "rank",
                "n",
                "ngram",
                "label",
                "count",
                "percentage",
                "document_count",
                "document_percentage",
            ],
        )
    return tables


def _with_rank(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.reset_index(drop=True).copy()
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _frequent_words_markdown(
    result: pd.DataFrame,
    *,
    dataset_path: Path,
    total_messages: int,
    messages_with_tokens: int,
    total_tokens: int,
    unique_tokens: int,
    top_n: int,
    remove_stopwords: bool,
    min_token_length: int,
) -> str:
    lines = [
        "# Frequent Words",
        "",
        "## Settings",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Total messages: {total_messages}",
        f"- Messages with tokens: {messages_with_tokens}",
        f"- Total tokens: {total_tokens}",
        f"- Unique tokens: {unique_tokens}",
        f"- Top N: {top_n}",
        f"- Remove stopwords: {remove_stopwords}",
        f"- Minimum token length: {min_token_length}",
        "",
        "## Top Tokens",
        "",
        "| rank | token | count | percentage | document_count | document_percentage |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result.itertuples(index=False):
        lines.append(
            f"| {row.rank} | {row.token} | {row.count} | {row.percentage:.4f} | "
            f"{row.document_count} | {row.document_percentage:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def _vocabulary_markdown(
    stats: dict[str, int],
    spam_typical: pd.DataFrame,
    ham_typical: pd.DataFrame,
    *,
    smoothing: float,
    min_count: int,
) -> str:
    lines = [
        "# HAM vs SPAM Vocabulary",
        "",
        "## Method",
        "",
        "Terms are ranked by a smoothed log rate ratio, not by raw count alone.",
        f"Smoothing value: {smoothing}. Minimum total token count: {min_count}.",
        "",
        "## Corpus",
        "",
        f"- HAM messages: {stats['ham_messages']}",
        f"- SPAM messages: {stats['spam_messages']}",
        f"- HAM tokens: {stats['ham_tokens']}",
        f"- SPAM tokens: {stats['spam_tokens']}",
        f"- Total vocabulary size: {stats['total_vocabulary_size']}",
        f"- Shared vocabulary size: {stats['shared_vocabulary_size']}",
        "",
        "## SPAM-Typical Words",
        "",
    ]
    lines.extend(_typical_words_table(spam_typical, "log_rate_ratio_spam_vs_ham"))
    lines.extend(["", "## HAM-Typical Words", ""])
    lines.extend(_typical_words_table(ham_typical, "log_rate_ratio_ham_vs_spam"))
    lines.append("")
    return "\n".join(lines)


def _typical_words_table(df: pd.DataFrame, score_column: str) -> list[str]:
    lines = [
        "| rank | token | spam_count | ham_count | score |",
        "| ---: | --- | ---: | ---: | ---: |",
    ]
    for row in df.itertuples(index=False):
        score = getattr(row, score_column)
        lines.append(f"| {row.rank} | {row.token} | {row.spam_count} | {row.ham_count} | {score:.4f} |")
    return lines


def _ngram_markdown(
    results: dict[int, pd.DataFrame],
    *,
    dataset_path: Path,
    total_messages: int,
    top_n: int,
    min_count: int,
    remove_stopwords: bool,
    min_token_length: int,
) -> str:
    lines = [
        "# Frequent Bigrams And Trigrams",
        "",
        "## Settings",
        "",
        f"- Dataset: `{dataset_path}`",
        f"- Total messages: {total_messages}",
        "- N-gram sizes: 2, 3",
        f"- Top N: {top_n}",
        f"- Minimum count: {min_count}",
        f"- Remove stopwords: {remove_stopwords}",
        f"- Minimum token length: {min_token_length}",
    ]
    for size in (2, 3):
        lines.extend(["", f"## Top {size}-grams", ""])
        df = results[size]
        if df.empty:
            lines.append("No n-grams met the configured threshold.")
            continue
        lines.extend(
            [
                "| rank | ngram | count | percentage | document_count | document_percentage |",
                "| ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in df.itertuples(index=False):
            lines.append(
                f"| {row.rank} | {row.ngram} | {row.count} | {row.percentage:.4f} | "
                f"{row.document_count} | {row.document_percentage:.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SMS word, vocabulary, and n-gram analyses.")
    parser.add_argument("--dataset", type=Path, default=VALIDATED_DATASET_PATH)
    parser.add_argument("--top-words", type=int, default=50)
    parser.add_argument("--top-typical-words", type=int, default=30)
    parser.add_argument("--top-ngrams", type=int, default=30)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--remove-stopwords", action="store_true")
    parser.add_argument("--min-token-length", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = run_text_analysis(
            args.dataset,
            top_words=args.top_words,
            top_typical_words=args.top_typical_words,
            top_ngrams=args.top_ngrams,
            min_count=args.min_count,
            remove_stopwords=args.remove_stopwords,
            min_token_length=args.min_token_length,
        )
    except (TextAnalysisError, ValueError) as exc:
        print(f"Text analysis error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
