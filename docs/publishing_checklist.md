# Publishing Checklist

Use this checklist before pushing the project to a public GitHub repository.

## Include

- `README.md`
- `requirements.txt`
- `pyproject.toml`
- `uv.lock`
- `.env.example`
- `src/`
- `scripts/`
- `tests/`
- `docs/project_notes.md`
- `docs/publishing_checklist.md`
- `data/raw/.gitkeep`
- selected generated summaries and figures from `outputs/`, if desired:
  - `artifact_index.md`
  - `classification_report.md`
  - `cluster_summary.md`
  - `clustering/provider_comparison.md`
  - `dataset_validation.json`
  - `frequent_words.md`
  - `vocabulary_findings.md`
  - `ngram_findings.md`
  - `model_metrics.json`
  - `confusion_matrix.csv`
  - `error_examples.csv`
  - `figures/`

## Exclude

- `.env` and any real API keys
- virtual environments, caches, and editor settings
- local notes, draft materials, or company-specific documents
- raw dataset files unless you intentionally redistribute them with UCI attribution
- `outputs/models/` model binaries
- `outputs/embeddings/` embedding arrays
- `outputs/clustering/**/embeddings/*.npy` provider-specific embedding arrays
- `outputs/clustering/**/semantic_clusters.csv` full provider-specific cluster assignment tables unless you intentionally want to publish all SMS text rows with attribution

## Before Publishing

1. Rotate any cloud keys that were used during development.
2. Confirm `.env` has never been committed. If it has, create a fresh clean repository or rewrite history before publishing.
3. Run:

```bash
python -m unittest discover -s tests
python -m compileall -q src
```

4. Inspect tracked files:

```bash
git status --short
git ls-files
```

5. Verify the README dataset citation remains visible.
6. Verify the downloader still points to the public UCI dataset page/source.
