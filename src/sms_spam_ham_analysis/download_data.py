"""Download the public UCI SMS Spam Collection dataset."""

from __future__ import annotations

import argparse
import io
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.request import urlopen

from sms_spam_ham_analysis.config import RAW_DATA_DIR


UCI_SMS_SPAM_COLLECTION_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
UCI_SMS_SPAM_COLLECTION_PAGE = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
DATASET_FILENAME = "SMSSpamCollection"
DATASET_CITATION = (
    "Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. "
    "UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84"
)


class DatasetDownloadError(ValueError):
    """Raised when the public dataset cannot be downloaded or extracted."""


@dataclass(frozen=True)
class DatasetDownloadSummary:
    source_url: str
    dataset_page: str
    output_file: str
    bytes_written: int
    citation: str


def download_public_dataset(
    output_dir: Path,
    *,
    source_url: str = UCI_SMS_SPAM_COLLECTION_URL,
    force: bool = False,
) -> DatasetDownloadSummary:
    """Download and extract the public UCI SMS Spam Collection file."""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / DATASET_FILENAME
    if output_file.exists() and not force:
        return DatasetDownloadSummary(
            source_url=source_url,
            dataset_page=UCI_SMS_SPAM_COLLECTION_PAGE,
            output_file=str(output_file),
            bytes_written=output_file.stat().st_size,
            citation=DATASET_CITATION,
        )

    try:
        with urlopen(source_url, timeout=60) as response:
            archive_bytes = response.read()
    except OSError as exc:
        raise DatasetDownloadError(f"Could not download dataset from {source_url}: {exc}") from exc

    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
            member = _find_dataset_member(archive)
            dataset_bytes = archive.read(member)
    except zipfile.BadZipFile as exc:
        raise DatasetDownloadError(f"Downloaded dataset archive is not a valid zip file: {source_url}") from exc
    except KeyError as exc:
        raise DatasetDownloadError(f"Dataset file {DATASET_FILENAME!r} was not found in the archive.") from exc

    output_file.write_bytes(dataset_bytes)
    return DatasetDownloadSummary(
        source_url=source_url,
        dataset_page=UCI_SMS_SPAM_COLLECTION_PAGE,
        output_file=str(output_file),
        bytes_written=len(dataset_bytes),
        citation=DATASET_CITATION,
    )


def _find_dataset_member(archive: zipfile.ZipFile) -> str:
    for member in archive.namelist():
        if Path(member).name == DATASET_FILENAME:
            return member
    raise KeyError(DATASET_FILENAME)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download the public UCI SMS Spam Collection dataset.")
    parser.add_argument("--output-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--source-url", default=UCI_SMS_SPAM_COLLECTION_URL)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing data/raw/SMSSpamCollection file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = download_public_dataset(args.output_dir, source_url=args.source_url, force=args.force)
    except DatasetDownloadError as exc:
        print(f"Dataset download error: {exc}", file=sys.stderr)
        return 1

    print("Downloaded public SMS Spam Collection dataset:")
    for key, value in asdict(summary).items():
        print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
