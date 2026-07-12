"""Project paths and default output locations."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

VALIDATED_DATASET_PATH = OUTPUTS_DIR / "validated_sms_dataset.csv"
VALIDATION_SUMMARY_PATH = OUTPUTS_DIR / "dataset_validation.json"
FREQUENT_WORDS_CSV_PATH = OUTPUTS_DIR / "frequent_words.csv"
FREQUENT_WORDS_MARKDOWN_PATH = OUTPUTS_DIR / "frequent_words.md"
VOCABULARY_BY_LABEL_PATH = OUTPUTS_DIR / "vocabulary_by_label.csv"
SPAM_TYPICAL_WORDS_PATH = OUTPUTS_DIR / "spam_typical_words.csv"
HAM_TYPICAL_WORDS_PATH = OUTPUTS_DIR / "ham_typical_words.csv"
VOCABULARY_FINDINGS_PATH = OUTPUTS_DIR / "vocabulary_findings.md"
FREQUENT_BIGRAMS_PATH = OUTPUTS_DIR / "frequent_bigrams.csv"
FREQUENT_TRIGRAMS_PATH = OUTPUTS_DIR / "frequent_trigrams.csv"
NGRAM_FINDINGS_PATH = OUTPUTS_DIR / "ngram_findings.md"
MODELS_DIR = OUTPUTS_DIR / "models"
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_classifier.joblib"
TRAINING_METADATA_PATH = MODELS_DIR / "training_metadata.json"
MODEL_METRICS_PATH = OUTPUTS_DIR / "model_metrics.json"
CLASSIFICATION_REPORT_PATH = OUTPUTS_DIR / "classification_report.md"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.csv"
ERROR_EXAMPLES_PATH = OUTPUTS_DIR / "error_examples.csv"
TEST_PREDICTIONS_PATH = OUTPUTS_DIR / "test_predictions.csv"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
EMBEDDINGS_ARRAY_PATH = EMBEDDINGS_DIR / "sms_embeddings.npy"
EMBEDDINGS_METADATA_PATH = EMBEDDINGS_DIR / "metadata.json"
SEMANTIC_CLUSTERS_PATH = OUTPUTS_DIR / "semantic_clusters.csv"
CLUSTER_SUMMARY_PATH = OUTPUTS_DIR / "cluster_summary.md"
CLUSTER_PROVIDER_COMPARISON_PATH = OUTPUTS_DIR / "clustering" / "provider_comparison.md"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TOP_WORDS_FIGURE_PATH = FIGURES_DIR / "top_words.png"
VOCABULARY_FIGURE_PATH = FIGURES_DIR / "vocabulary_comparison.png"
NGRAMS_FIGURE_PATH = FIGURES_DIR / "ngrams.png"
CONFUSION_MATRIX_FIGURE_PATH = FIGURES_DIR / "confusion_matrix.png"
MODEL_METRICS_COMPARISON_FIGURE_PATH = FIGURES_DIR / "model_metrics_comparison.png"
PRECISION_RECALL_CURVE_FIGURE_PATH = FIGURES_DIR / "precision_recall_curve.png"
SEMANTIC_CLUSTERS_FIGURE_PATH = FIGURES_DIR / "semantic_clusters.png"
ARTIFACT_INDEX_PATH = OUTPUTS_DIR / "artifact_index.md"

IGNORED_RAW_FILENAMES = frozenset({".gitkeep", ".DS_Store", "readme", "README", "README.txt", "readme.txt"})
