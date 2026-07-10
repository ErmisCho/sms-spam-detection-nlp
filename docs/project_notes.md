# Project Notes

## Short Summary

This project turns the public UCI SMS Spam Collection dataset into a reproducible text analytics and machine-learning pipeline. It downloads the public archive on demand, validates the raw SMS file, produces word/ngram/vocabulary findings, trains a TF-IDF Logistic Regression spam classifier, and clusters messages using embedding similarity.

The local path is fully reproducible with open-source Python libraries. The optional Azure path uses Azure OpenAI / Azure AI Foundry embeddings for semantic clustering.

## Key Results To Inspect

- `outputs/frequent_words.md`: most frequent tokens.
- `outputs/vocabulary_findings.md`: HAM-typical and SPAM-typical vocabulary.
- `outputs/ngram_findings.md`: frequent bigrams and trigrams.
- `outputs/classification_report.md`: row-stratified and duplicate-safe classifier evaluation.
- `outputs/model_metrics.json`: structured classifier metrics.
- `outputs/error_examples.csv`: representative false positives and false negatives.
- `outputs/cluster_summary.md`: current semantic cluster summary.
- `outputs/clustering/provider_comparison.md`: local-vs-Azure clustering comparison.

## Design Choices

- The raw dataset is treated as untrusted input. Downstream code reads only the normalized `outputs/validated_sms_dataset.csv`.
- TF-IDF plus Logistic Regression is used for classification because it is strong for short text, fast, deterministic, and explainable.
- Evaluation includes a duplicate-safe grouped split because duplicate SMS texts can otherwise appear in both train and test and inflate metrics.
- Semantic clustering is implemented as: embed SMS messages, cluster vectors with KMeans, then summarize label mix and representative messages.
- Azure embeddings are the cloud/GenAI semantic path. Local TF-IDF/SVD embeddings are a no-key fallback for reproducible review and development.
- Silhouette score is displayed as a clustering separation diagnostic, not as a final business-quality metric.

## Common Questions

### Why TF-IDF and Logistic Regression?

SMS spam has strong lexical signals. TF-IDF captures those word and phrase patterns, and Logistic Regression gives a stable, interpretable baseline without unnecessary model complexity.

### Why not only accuracy?

The dataset is imbalanced. Accuracy can hide weak spam recall. Precision, recall, F1, confusion matrix, and error examples give a more useful view.

### Why duplicate-safe evaluation?

The dataset contains repeated SMS texts. If identical messages appear in both train and test, held-out metrics become optimistic. Grouping by message text gives a stricter check.

### Why embeddings rather than chat completions?

The clustering task is a semantic similarity problem. Embeddings are the right representation for vector similarity and KMeans clustering. Chat completions would be more appropriate for follow-up tasks such as naming clusters or generating a natural-language report.

### Is the local embedding path an LLM model?

No. The local path creates classical TF-IDF/SVD vectors. The Azure path uses a neural embedding model and is the GenAI semantic representation.

### What does silhouette score mean?

Silhouette measures how much closer points are to their own cluster than to the nearest other cluster. Scores near 1 indicate clean separation, near 0 indicate overlapping clusters, and negative scores suggest poor assignments. In this project, the scores are low, so the clusters should be treated as exploratory groupings.

### Which clustering provider is better?

Do not choose only by silhouette score. The local provider may produce sharper keyword-driven separation, while Azure embeddings may capture broader semantic similarity. Provider choice should consider cluster representatives, label mix, operational constraints, cost, privacy, and whether the grouping is useful for the task.

### What would make this production-ready?

A production system would need newer data, privacy review, monitoring, retraining, threshold tuning, CI, Docker, deployment scripts, and serving/integration tests.

## Privacy And Cloud Notes

The Azure path sends SMS text to the configured embedding deployment. Use it only with appropriate data handling approval and cost controls. The default local path avoids external services.
