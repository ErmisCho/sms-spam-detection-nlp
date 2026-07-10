# Local vs Azure Clustering Provider Comparison

| output folder | status | embedding provider | model | dimensions | messages | clusters | silhouette | role |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| local | generated | sklearn-svd | tfidf-word-bigram-truncated-svd | 100 | 5572 | 8 | 0.0928 | Local reproducibility fallback |
| azure | generated | azure-openai | text-embedding-3-small | 1536 | 5572 | 8 | 0.0262 | GenAI embedding path |

## Interpretation

- The local provider is the no-key reproducibility path and is suitable for local execution without cloud credentials.
- The Azure provider is the GenAI embedding path expected for semantic clustering when credentials are configured.
- The silhouette score is an unsupervised separation signal, not a classifier metric; compare it with the cluster themes before claiming one provider is better.
- Cost, privacy, latency, and company data policy matter for Azure because SMS text is sent to the configured embedding service.

## Metric Note

- By silhouette score alone, `local` is currently higher. This should be treated as one diagnostic, not a final quality verdict.
