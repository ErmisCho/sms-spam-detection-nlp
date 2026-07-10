"""Load and validate the SMS spam detection dataset."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from sms_spam_ham_analysis.config import (
    IGNORED_RAW_FILENAMES,
    RAW_DATA_DIR,
    VALIDATED_DATASET_PATH,
    VALIDATION_SUMMARY_PATH,
)


LABEL_COLUMN_CANDIDATES = ("label", "class", "target", "category", "v1")
TEXT_COLUMN_CANDIDATES = ("text", "message", "sms", "body", "v2")

ROW_ID_COLUMN = "row_id"
LABEL_COLUMN = "label"
TEXT_COLUMN = "text"
EXPECTED_LABELS = ["ham", "spam"]

LABEL_MAP = {
    "ham": "ham",
    "spam": "spam",
    "0": "ham",
    "1": "spam",
    "false": "ham",
    "true": "spam",
}

SUPPORTED_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1")
SUPPORTED_DELIMITERS = ("\t", ",", ";")


class DatasetValidationError(ValueError):
    """Raised when the raw SMS dataset cannot be loaded or validated."""


@dataclass(frozen=True)
class DatasetSummary:
    input_file: str
    output_file: str
    summary_file: str
    detected_columns: list[str]
    label_column: str
    text_column: str
    delimiter: str
    encoding: str
    raw_rows: int
    validated_rows: int
    ham_count: int
    spam_count: int
    missing_label_count: int
    missing_text_count: int
    empty_text_count: int
    invalid_label_count: int
    duplicate_message_count: int
    duplicate_message_groups: int
    exact_duplicate_row_count: int
    duplicate_messages_removed: int


def load_validated_dataset(dataset_path: Path) -> pd.DataFrame:
    """Load the normalized dataset used by downstream analysis/modeling code."""

    if not dataset_path.exists():
        raise DatasetValidationError(
            f"Validated dataset not found: {dataset_path}. "
            "Run dataset validation first to create outputs/validated_sms_dataset.csv."
        )

    df = pd.read_csv(dataset_path, dtype=str, keep_default_na=False)
    required_columns = (ROW_ID_COLUMN, LABEL_COLUMN, TEXT_COLUMN)
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise DatasetValidationError(
            f"Validated dataset is missing required columns {missing}. "
            f"Detected columns: {df.columns.tolist()}"
        )
    if df.empty:
        raise DatasetValidationError(f"Validated dataset is empty: {dataset_path}")

    df = df.copy()
    try:
        df[ROW_ID_COLUMN] = df[ROW_ID_COLUMN].astype(int)
    except ValueError as exc:
        raise DatasetValidationError(f"Column '{ROW_ID_COLUMN}' must contain integer row ids.") from exc

    labels = sorted(df[LABEL_COLUMN].astype(str).str.lower().unique().tolist())
    if labels != EXPECTED_LABELS:
        raise DatasetValidationError(f"Expected labels {EXPECTED_LABELS}, got {labels}.")

    return df


def discover_dataset_file(input_path: Path) -> Path:
    """Return the raw dataset file from a file path or single-file directory."""

    if not input_path.exists():
        raise DatasetValidationError(
            f"Input path does not exist: {input_path}. Place the provided dataset under data/raw/."
        )

    if input_path.is_file():
        return input_path

    if not input_path.is_dir():
        raise DatasetValidationError(f"Input path is neither a file nor a directory: {input_path}")

    candidates = [
        path
        for path in sorted(input_path.iterdir())
        if path.is_file()
        and path.name not in IGNORED_RAW_FILENAMES
        and not path.name.startswith(".")
    ]

    if not candidates:
        raise DatasetValidationError(
            f"No SMS dataset file found under {input_path}. "
            "Add the provided raw dataset file to data/raw/ or pass a file path with --input."
        )

    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise DatasetValidationError(
            f"Multiple dataset files found under {input_path}: {names}. "
            "Pass the intended file path with --input."
        )

    return candidates[0]


def load_and_validate_dataset(
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    *,
    drop_duplicate_messages: bool = False,
) -> DatasetSummary:
    """Load the raw dataset, validate it, and write normalized outputs."""

    dataset_file = discover_dataset_file(input_path)
    raw_df, metadata = _read_raw_dataset(dataset_file)
    label_column, text_column = _detect_required_columns(raw_df)

    label_values = raw_df[label_column]
    text_values = raw_df[text_column]

    missing_label_mask = label_values.isna() | label_values.astype(str).str.strip().eq("")
    missing_text_mask = text_values.isna() | text_values.astype(str).str.strip().eq("")

    normalized_labels = label_values.map(normalize_label)
    invalid_label_mask = normalized_labels.isna() & ~missing_label_mask

    errors = []
    if missing_label_mask.any():
        errors.append(f"{int(missing_label_mask.sum())} rows have missing labels")
    if missing_text_mask.any():
        errors.append(f"{int(missing_text_mask.sum())} rows have missing text values")
    if invalid_label_mask.any():
        invalid_values = sorted(
            {
                str(value).strip()
                for value in label_values[invalid_label_mask].dropna().unique().tolist()
            }
        )
        errors.append(f"unexpected label values: {invalid_values}")
    if errors:
        raise DatasetValidationError("; ".join(errors))

    text_clean = text_values.astype(str).str.strip()
    validated = pd.DataFrame(
        {
            ROW_ID_COLUMN: range(1, len(raw_df) + 1),
            LABEL_COLUMN: normalized_labels.astype(str),
            TEXT_COLUMN: text_clean,
        }
    )

    duplicate_message_count = int(validated[TEXT_COLUMN].duplicated(keep=False).sum())
    duplicate_message_groups = int(
        validated.loc[validated[TEXT_COLUMN].duplicated(keep=False), TEXT_COLUMN].nunique()
    )
    exact_duplicate_row_count = int(validated.duplicated(subset=[LABEL_COLUMN, TEXT_COLUMN], keep=False).sum())

    duplicate_messages_removed = 0
    if drop_duplicate_messages:
        before = len(validated)
        validated = validated.drop_duplicates(subset=[TEXT_COLUMN], keep="first").reset_index(drop=True)
        validated[ROW_ID_COLUMN] = range(1, len(validated) + 1)
        duplicate_messages_removed = before - len(validated)

    label_counts = validated[LABEL_COLUMN].value_counts().to_dict()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    validated.to_csv(output_path, index=False)

    summary = DatasetSummary(
        input_file=str(dataset_file),
        output_file=str(output_path),
        summary_file=str(summary_path),
        detected_columns=[str(column) for column in raw_df.columns.tolist()],
        label_column=str(label_column),
        text_column=str(text_column),
        delimiter=_display_delimiter(metadata["delimiter"]),
        encoding=metadata["encoding"],
        raw_rows=int(len(raw_df)),
        validated_rows=int(len(validated)),
        ham_count=int(label_counts.get("ham", 0)),
        spam_count=int(label_counts.get("spam", 0)),
        missing_label_count=int(missing_label_mask.sum()),
        missing_text_count=int(missing_text_mask.sum()),
        empty_text_count=int(missing_text_mask.sum()),
        invalid_label_count=int(invalid_label_mask.sum()),
        duplicate_message_count=duplicate_message_count,
        duplicate_message_groups=duplicate_message_groups,
        exact_duplicate_row_count=exact_duplicate_row_count,
        duplicate_messages_removed=int(duplicate_messages_removed),
    )
    summary_path.write_text(json.dumps(asdict(summary), indent=2) + "\n", encoding="utf-8")
    return summary


def normalize_label(value: object) -> str | None:
    normalized = str(value).strip().lower()
    return LABEL_MAP.get(normalized)


def _read_raw_dataset(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    last_error: Exception | None = None

    for encoding in SUPPORTED_ENCODINGS:
        try:
            sample = path.read_text(encoding=encoding, errors="strict")[:8192]
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

        delimiter = _detect_delimiter(sample)
        has_header = _detect_header(sample, delimiter)

        try:
            if has_header:
                df = pd.read_csv(path, sep=delimiter, dtype=str, keep_default_na=False, encoding=encoding)
            else:
                df = pd.read_csv(
                    path,
                    sep=delimiter,
                    header=None,
                    names=["label", "text"],
                    usecols=[0, 1],
                    dtype=str,
                    keep_default_na=False,
                    encoding=encoding,
                )
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
            last_error = exc
            continue

        if df.shape[1] < 2:
            last_error = DatasetValidationError(
                f"Dataset file {path} must contain at least two columns: label and text."
            )
            continue

        return df, {"encoding": encoding, "delimiter": delimiter}

    if last_error is None:
        raise DatasetValidationError(f"Unable to read dataset file: {path}")
    raise DatasetValidationError(f"Unable to parse dataset file {path}: {last_error}") from last_error


def _detect_required_columns(df: pd.DataFrame) -> tuple[str, str]:
    normalized_columns = {_normalize_column_name(column): column for column in df.columns}

    label_column = _first_matching_column(normalized_columns, LABEL_COLUMN_CANDIDATES)
    text_column = _first_matching_column(normalized_columns, TEXT_COLUMN_CANDIDATES)

    if label_column is None or text_column is None:
        if df.shape[1] >= 2 and _series_looks_like_labels(df.iloc[:, 0]):
            return str(df.columns[0]), str(df.columns[1])

        raise DatasetValidationError(
            "Could not detect required label/text columns. "
            f"Detected columns: {[str(column) for column in df.columns.tolist()]}. "
            f"Accepted label names: {list(LABEL_COLUMN_CANDIDATES)}. "
            f"Accepted text names: {list(TEXT_COLUMN_CANDIDATES)}."
        )

    if label_column == text_column:
        raise DatasetValidationError("Detected label and text columns resolve to the same column.")

    return str(label_column), str(text_column)


def _detect_delimiter(sample: str) -> str:
    if not sample.strip():
        raise DatasetValidationError("Dataset file is empty.")

    try:
        dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
        if dialect.delimiter in SUPPORTED_DELIMITERS:
            return dialect.delimiter
    except csv.Error:
        pass

    counts = {delimiter: sample.count(delimiter) for delimiter in SUPPORTED_DELIMITERS}
    delimiter, count = max(counts.items(), key=lambda item: item[1])
    if count == 0:
        raise DatasetValidationError("Could not detect CSV/TSV delimiter. Expected tab, comma, or semicolon.")
    return delimiter


def _detect_header(sample: str, delimiter: str) -> bool:
    first_row = _first_row(sample, delimiter)
    if not first_row:
        raise DatasetValidationError("Dataset file has no readable rows.")

    first_cell = _normalize_column_name(first_row[0])
    if normalize_label(first_row[0]) is not None:
        return False
    if first_cell in {_normalize_column_name(name) for name in LABEL_COLUMN_CANDIDATES}:
        return True

    try:
        return bool(csv.Sniffer().has_header(sample))
    except csv.Error:
        return False


def _first_row(sample: str, delimiter: str) -> list[str]:
    rows = csv.reader(sample.splitlines(), delimiter=delimiter)
    for row in rows:
        if row and any(cell.strip() for cell in row):
            return row
    return []


def _first_matching_column(columns_by_normalized_name: dict[str, object], candidates: Iterable[str]) -> object | None:
    for candidate in candidates:
        column = columns_by_normalized_name.get(_normalize_column_name(candidate))
        if column is not None:
            return column
    return None


def _normalize_column_name(column: object) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(column).strip().lower())


def _series_looks_like_labels(series: pd.Series) -> bool:
    non_empty = series.dropna().astype(str).str.strip()
    if non_empty.empty:
        return False
    return non_empty.map(normalize_label).notna().all()


def _display_delimiter(delimiter: str) -> str:
    if delimiter == "\t":
        return "\\t"
    return delimiter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load and validate the SMS spam detection dataset.")
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DATA_DIR,
        help="Raw dataset file or directory containing exactly one dataset file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=VALIDATED_DATASET_PATH,
        help="Path for the normalized validated CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=VALIDATION_SUMMARY_PATH,
        help="Path for the validation summary JSON.",
    )
    parser.add_argument(
        "--drop-duplicate-messages",
        action="store_true",
        help="Drop duplicate message texts after reporting duplicate counts.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        summary = load_and_validate_dataset(
            args.input,
            args.output,
            args.summary_output,
            drop_duplicate_messages=args.drop_duplicate_messages,
        )
    except DatasetValidationError as exc:
        print(f"Dataset validation error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(asdict(summary), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
