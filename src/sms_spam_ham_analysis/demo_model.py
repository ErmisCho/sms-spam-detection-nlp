"""Build and verify the small, synthetic classifier bundled in the demo image.

The artifact is created inside the trusted container build rather than downloaded or
committed as a pickle. It is intentionally for product demonstrations, not for
production filtering or benchmark reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import joblib

from sms_spam_ham_analysis.model import RANDOM_STATE, build_tfidf_classifier


MANIFEST_SCHEMA = "sms-spam-demo-model/v1"
DEMO_MESSAGES: tuple[tuple[str, str], ...] = (
    ("ham", "Are we still meeting for dinner tonight"),
    ("ham", "Call me when you arrive at the office"),
    ("ham", "Can you pick up milk on your way home"),
    ("ham", "The team meeting starts at ten tomorrow"),
    ("ham", "Thanks for your help see you later"),
    ("ham", "I am running late but will be there soon"),
    ("ham", "Happy birthday hope you have a lovely day"),
    ("ham", "Please send me the notes from class"),
    ("ham", "Lunch at the usual place sounds good"),
    ("ham", "Your appointment is confirmed for Monday"),
    ("ham", "I left the keys on the kitchen table"),
    ("ham", "The train arrives at platform four"),
    ("spam", "Free cash prize claim now text WIN"),
    ("spam", "Congratulations you won a reward claim today"),
    ("spam", "Urgent claim your free voucher now"),
    ("spam", "Win money today reply CLAIM to this message"),
    ("spam", "Limited offer click now for a free phone"),
    ("spam", "You have won cash call this premium number"),
    ("spam", "Exclusive prize waiting text WIN now"),
    ("spam", "Claim your free ringtone subscription today"),
    ("spam", "Selected winner collect your cash reward now"),
    ("spam", "Act now to receive free tickets and prizes"),
    ("spam", "Final notice claim the jackpot before midnight"),
    ("spam", "Reply YES for your guaranteed cash bonus"),
)


class DemoModelError(ValueError):
    """Raised when a demo artifact cannot be built or verified."""


def build_demo_model(output: Path, manifest_output: Path) -> dict[str, object]:
    """Train the fixed synthetic corpus and write an artifact plus provenance."""

    pipeline = build_tfidf_classifier()
    pipeline.fit(
        [text for _, text in DEMO_MESSAGES],
        [label for label, _ in DEMO_MESSAGES],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "metadata": {
                "purpose": "trusted synthetic product demo",
                "training_corpus_sha256": _corpus_sha256(),
                "random_state": RANDOM_STATE,
            },
        },
        output,
    )

    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "purpose": "Product demonstration only; not a production spam filter or benchmark artifact.",
        "artifact": output.name,
        "artifact_sha256": _file_sha256(output),
        "training_data": "Embedded synthetic HAM/SPAM examples; no private or downloaded data.",
        "training_rows": len(DEMO_MESSAGES),
        "training_corpus_sha256": _corpus_sha256(),
        "algorithm": "TF-IDF word unigrams/bigrams plus balanced Logistic Regression",
        "random_state": RANDOM_STATE,
        "dependencies": {
            "joblib": version("joblib"),
            "scikit-learn": version("scikit-learn"),
        },
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def verify_demo_model(artifact: Path, manifest_path: Path) -> dict[str, object]:
    """Verify provenance and checksum before an image is published."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DemoModelError(f"Cannot read demo model manifest: {manifest_path}") from exc
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise DemoModelError("Demo model manifest schema is not supported.")
    if manifest.get("training_corpus_sha256") != _corpus_sha256():
        raise DemoModelError("Demo model training corpus provenance does not match this source tree.")
    if manifest.get("artifact_sha256") != _file_sha256(artifact):
        raise DemoModelError("Demo model artifact checksum does not match its manifest.")
    return manifest


def _corpus_sha256() -> str:
    payload = json.dumps(DEMO_MESSAGES, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise DemoModelError(f"Cannot read demo model artifact: {path}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build or verify the trusted synthetic demo classifier.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--artifact", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.command == "build":
        build_demo_model(args.output, args.manifest)
    else:
        verify_demo_model(args.artifact, args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
