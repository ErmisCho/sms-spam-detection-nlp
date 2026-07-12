# SMS Spam Detection NLP Pipeline

[![CI](https://github.com/ErmisCho/sms-spam-detection-nlp/actions/workflows/ci.yml/badge.svg)](https://github.com/ErmisCho/sms-spam-detection-nlp/actions/workflows/ci.yml)
[![Python 3.10–3.13](https://img.shields.io/badge/python-3.10%E2%80%933.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

End-to-end NLP pipeline for SMS spam detection using the public SMS Spam Collection dataset. The project validates the raw SMS file, explores text patterns, trains a classical spam classifier, and clusters messages by embedding similarity. It includes a fully local path for reproducibility and an optional Azure OpenAI / Azure AI Foundry embedding path for semantic clustering.

This is an independent portfolio project.

![Conventional and duplicate-safe SMS spam classifier confusion matrices](outputs/figures/confusion_matrix.png)

*The conventional split makes the model's 6 false positives and 7 false negatives reviewable, while the duplicate-safe matrix shows performance after eliminating identical train/test message overlap.*

## Portfolio Highlights

- End-to-end NLP workflow: dataset download, validation, exploration, modeling, clustering, visualization, and generated reports.
- Reproducible local baseline using TF-IDF, Logistic Regression, and TF-IDF/SVD clustering with no API key required.
- Optional GenAI path using Azure embeddings for semantic clustering, with fail-fast credential checks and provider-specific artifacts.
- Stricter duplicate-safe evaluation to reduce overly optimistic scores from repeated SMS text.
- Cross-platform automation for Windows PowerShell and Linux/WSL, plus smoke tests for the core pipeline.

## Headline Results

- Row-stratified classifier accuracy: 98.83%.
- Row-stratified SPAM F1: 95.62%.
- Duplicate-safe SPAM F1: 94.16%.
- Duplicate-safe test split text overlap with training: 0.
- Row-stratified errors: 6 false positives and 7 false negatives on the test split.

## Model Evaluation Visuals

![SPAM precision, recall, and F1 comparison](outputs/figures/model_metrics_comparison.png)

The duplicate-safe scores are the more credible generalization estimate. Their modest reduction relative to the conventional split shows that the classifier remains strong after removing exact-text leakage.

![SPAM precision-recall curve](outputs/figures/precision_recall_curve.png)

The precision-recall curve shows the tradeoff available when changing the classification threshold. It is more informative than accuracy alone because only 13.4% of messages are SPAM; the marked operating point is the classifier's current decision threshold.

## 30-Second Demo

After completing [Setup](#setup) and [downloading the dataset](#dataset), run the complete local workflow with no API key:

```bash
bash scripts/run_pipeline.sh
```

On Windows PowerShell, use `.\scripts\run_pipeline.ps1` instead. The workflow validates the raw data, analyzes its vocabulary, trains and evaluates the classifier, clusters messages, and refreshes the reviewable artifacts in `outputs/`.

For a quick results tour, open:

- [Classification report](outputs/classification_report.md) for row-stratified and duplicate-safe metrics.
- [Error examples](outputs/error_examples.csv) to inspect false positives and false negatives.
- [Cluster summary](outputs/cluster_summary.md) for discovered message themes and label mix.
- [Artifact index](outputs/artifact_index.md) for the complete generated deliverable set.

## What It Demonstrates

- Data validation for a raw text classification dataset.
- Exploratory NLP analysis: frequent words, spam-vs-legitimate vocabulary differences, bigrams, and trigrams.
- Classical ML classification with TF-IDF and Logistic Regression.
- Evaluation with metrics, confusion matrix, representative errors, and duplicate-safe grouped validation.
- Semantic clustering with either local TF-IDF/SVD embeddings or Azure embedding vectors.
- Reproducible Windows and Linux/WSL scripts, tests, and reviewable output artifacts.

## Dataset

The project uses the public SMS Spam Collection dataset from the UCI Machine Learning Repository. The repository includes only an empty placeholder directory, not the raw dataset file.

Download it after setup:

```powershell
.\scripts\download_dataset.ps1
```

```bash
bash scripts/download_dataset.sh
```

Both commands download the public UCI archive and extract `SMSSpamCollection` into `data/raw/`.

Expected file format:

```text
ham<TAB>Ok lar... Joking wif u oni...
spam<TAB>Free entry in 2 a wkly comp...
```

Dataset citation:

> Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84

The UCI dataset page states that the dataset is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0), allowing sharing and adaptation with appropriate credit.

Source: https://archive.ics.uci.edu/dataset/228/sms+spam+collection

## Setup

Use Python 3.10 or newer from the repository root.

Native Windows PowerShell:

```powershell
python -m venv .venv-win
.\.venv-win\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux or WSL:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`requirements.txt` installs the third-party libraries and the local `src/` package.

## Run

Recommended local run on Windows:

```powershell
.\scripts\run_pipeline.ps1
```

Recommended local run on Linux or WSL:

```bash
bash scripts/run_pipeline.sh
```

Both commands run dataset validation, text analysis, TF-IDF model training/evaluation, local semantic clustering, and output generation. The default local clustering path uses TF-IDF plus TruncatedSVD embeddings, so it needs no API key or model download.

Azure embedding variants:

```powershell
# Azure OpenAI sample for a quick semantic clustering check
.\scripts\run_pipeline.ps1 -UseAzure -AzureSampleSize 250 -AzureClusters 6

# Full-dataset Azure OpenAI embeddings
.\scripts\run_pipeline.ps1 -FullAzure -AzureClusters 8
```

```bash
# Azure OpenAI sample for a quick semantic clustering check
bash scripts/run_pipeline.sh --use-azure --azure-sample-size 250 --azure-clusters 6

# Full-dataset Azure OpenAI embeddings
bash scripts/run_pipeline.sh --full-azure --azure-clusters 8
```

The Azure path is the GenAI embedding implementation. The local path creates classical TF-IDF/SVD vectors for reproducibility; it is not an LLM embedding model.

Manual module commands after activating the virtual environment:

```bash
python -m sms_spam_ham_analysis.data --input data/raw --output outputs/validated_sms_dataset.csv
python -m sms_spam_ham_analysis.analysis --dataset outputs/validated_sms_dataset.csv
python -m sms_spam_ham_analysis.modeling --dataset outputs/validated_sms_dataset.csv --model-out outputs/models/tfidf_classifier.joblib
python -m sms_spam_ham_analysis.clustering --dataset outputs/validated_sms_dataset.csv --clusters auto
python -m sms_spam_ham_analysis.visualize --outputs outputs
```

Try the trained classifier on a message (the local pipeline must have been run first):

```bash
python -m sms_spam_ham_analysis.predict "Congratulations, claim your free prize now"
```

The command prints the predicted `HAM`/`SPAM` label and the classifier's confidence.

If the dataset has not been downloaded yet, run:

```bash
python -m sms_spam_ham_analysis.download_data
```

## Azure Embeddings

The default local run does not need `.env`. For Azure semantic clustering, copy `.env.example` to `.env` and fill in every `AZURE_OPENAI_*` value before running an Azure command. Azure pipeline variants fail before dataset validation if required values are missing or still contain template placeholders.

Azure sends SMS text to the configured embedding deployment, so consider cost, privacy, and data handling policy before using it. If Azure returns rate-limit or transient service errors, lower `AZURE_OPENAI_CONCURRENCY` first.

## Outputs

Important generated files:

- `outputs/validated_sms_dataset.csv`: normalized dataset with `row_id`, `label`, and `text`.
- `outputs/dataset_validation.json`: raw-file validation summary.
- `outputs/frequent_words.md`: most frequent tokens.
- `outputs/vocabulary_findings.md`: HAM-typical and SPAM-typical words.
- `outputs/ngram_findings.md`: frequent bigrams and trigrams.
- `outputs/model_metrics.json`: classifier metrics.
- `outputs/classification_report.md`: row-stratified and duplicate-safe classifier reports.
- `outputs/error_examples.csv`: false positives and false negatives.
- `outputs/semantic_clusters.csv`: current clustering assignments.
- `outputs/cluster_summary.md`: current cluster sizes, label mix, themes, and representatives.
- `outputs/clustering/local/` and `outputs/clustering/azure/`: provider-specific clustering copies when those runs have been executed.
- `outputs/clustering/provider_comparison.md`: local-vs-Azure clustering comparison based on generated provider metadata.
- `outputs/artifact_index.md`: generated tables, reports, models, and figures.

## Architecture

- `data.py`: raw-file discovery, format checks, label/text validation, duplicate reporting, and normalized CSV output.
- `analysis.py`: frequent words, spam-vs-legitimate vocabulary differences, and n-grams.
- `model.py` and `modeling.py`: TF-IDF representation, Logistic Regression, deterministic evaluation, duplicate-safe grouped evaluation, metrics, and error examples.
- `embeddings.py`: local TF-IDF/SVD embeddings and optional Azure embedding API integration with batching, retries, concurrency, and fail-fast config validation.
- `clustering.py`: KMeans clustering over embedding vectors, cluster summaries, provider-specific output folders, and silhouette diagnostics.
- `visualize.py`: compact figures, artifact index, and provider comparison.

## Engineering Decisions

- **Start with an interpretable baseline.** TF-IDF and Logistic Regression are fast, reproducible, and easy to inspect. They establish a strong benchmark before adding more expensive model complexity.
- **Evaluate duplicate-safe performance.** Repeated SMS text can leak across a conventional random split and inflate results. The grouped evaluation keeps normalized duplicate messages together and reports zero train/test text overlap.
- **Keep the default workflow local.** TF-IDF/SVD clustering makes the full pipeline runnable without credentials, network calls, or per-request cost. Azure embeddings remain an explicit opt-in path for comparing semantic representations.
- **Treat errors as deliverables.** Aggregate metrics are paired with a confusion matrix and exported misclassifications so reviewers can examine failure modes—especially spam false negatives—rather than relying on accuracy alone.
- **Separate generated artifacts by provider.** Local and Azure clustering outputs retain their own metadata and summaries, making comparisons traceable without silently overwriting the provenance of a result.

## Verification

Run the automated smoke tests without the real dataset:

```powershell
.\.venv-win\Scripts\Activate.ps1
python -m unittest discover -s tests
python -m compileall -q src
```

Linux or WSL:

```bash
source .venv/bin/activate
python -m unittest discover -s tests
python -m compileall -q src
```

## Limitations

- The SMS dataset is old, so modern spam patterns may differ.
- The classifier is a strong baseline, not a production spam filter.
- Clustering is exploratory. Silhouette score is a useful separation diagnostic, but provider choice should also consider representative messages, label mix, business usefulness, cost, privacy, and reproducibility.
- Azure embeddings require approved data handling, credentials, and cost awareness.
- A production version would add Docker, monitoring, retraining strategy, threshold tuning, and a serving/integration path.
