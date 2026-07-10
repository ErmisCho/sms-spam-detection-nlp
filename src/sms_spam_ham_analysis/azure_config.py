"""Validate Azure embedding configuration for wrapper scripts."""

from __future__ import annotations

import sys

from sms_spam_ham_analysis.embeddings import EmbeddingError, validate_azure_embedding_config


def main() -> int:
    try:
        validate_azure_embedding_config()
    except EmbeddingError as exc:
        print(f"Azure configuration error: {exc}", file=sys.stderr)
        return 1

    print("Azure embedding configuration is complete.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
