from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import zipfile
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sms_spam_ham_analysis.analysis import run_text_analysis
from sms_spam_ham_analysis.clustering import run_clustering
from sms_spam_ham_analysis.data import load_and_validate_dataset
from sms_spam_ham_analysis.download_data import download_public_dataset
from sms_spam_ham_analysis.embeddings import (
    EmbeddingError,
    EmbeddingResult,
    build_embeddings,
    validate_azure_embedding_config,
)
from sms_spam_ham_analysis.modeling import train_and_evaluate_model
from sms_spam_ham_analysis.visualize import run_visualizations


SAMPLE_MESSAGES = [
    ("ham", "Are we still meeting for dinner tonight"),
    ("ham", "I will call you when I get home"),
    ("ham", "Can you pick up milk on your way home"),
    ("ham", "Lunch was good see you tomorrow"),
    ("ham", "Please call me when you are free"),
    ("ham", "I am running late but I will be there"),
    ("ham", "Happy birthday hope you have a good day"),
    ("ham", "Can we move the meeting to tomorrow"),
    ("ham", "I left the keys at home"),
    ("ham", "See you later at the train station"),
    ("ham", "Thanks for calling me back today"),
    ("ham", "Dinner at home sounds good"),
    ("spam", "Free prize claim now text WIN to 8000"),
    ("spam", "You have won cash prize claim now"),
    ("spam", "Claim your free ringtone text TONE now"),
    ("spam", "Win money now reply CLAIM to this message"),
    ("spam", "Free entry in prize draw text WIN"),
    ("spam", "Urgent claim your cash reward now"),
    ("spam", "Congratulations you won free tickets claim today"),
    ("spam", "Text STOP to end free prize alerts"),
    ("spam", "Win a free phone claim your reward"),
    ("spam", "Limited offer claim cash prize now"),
    ("spam", "Reply WIN for free vouchers today"),
    ("spam", "Your prize is ready claim free cash"),
]


class EndToEndPipelineTest(unittest.TestCase):
    def test_download_public_dataset_extracts_sms_collection_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "sms_spam_collection.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("SMSSpamCollection", "ham\thello there\nspam\tclaim prize now\n")

            output_dir = root / "raw"
            summary = download_public_dataset(output_dir, source_url=archive_path.as_uri())

            output_file = output_dir / "SMSSpamCollection"
            self.assertTrue(output_file.exists())
            self.assertEqual(output_file.read_text(encoding="utf-8"), "ham\thello there\nspam\tclaim prize now\n")
            self.assertEqual(summary.output_file, str(output_file))
            self.assertEqual(summary.bytes_written, output_file.stat().st_size)

    def test_azure_openai_provider_requires_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("sms_spam_ham_analysis.embeddings.PROJECT_ROOT", Path(tmp)):
                with patch.dict("os.environ", {}, clear=True):
                    with self.assertRaisesRegex(EmbeddingError, "AZURE_OPENAI_ENDPOINT"):
                        build_embeddings(["hello"], provider="azure-openai")

    def test_azure_config_validation_requires_all_submitted_details(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("sms_spam_ham_analysis.embeddings.PROJECT_ROOT", Path(tmp)):
                with patch.dict(
                    "os.environ",
                    {
                        "AZURE_OPENAI_ENDPOINT": "https://example.services.ai.azure.com",
                        "AZURE_OPENAI_API_KEY": "key",
                        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-3-small",
                    },
                    clear=True,
                ):
                    with self.assertRaisesRegex(EmbeddingError, "AZURE_OPENAI_API_VERSION"):
                        validate_azure_embedding_config()

    def test_azure_config_validation_rejects_example_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("sms_spam_ham_analysis.embeddings.PROJECT_ROOT", Path(tmp)):
                with patch.dict(
                    "os.environ",
                    {
                        "AZURE_OPENAI_ENDPOINT": "https://your-resource-name.openai.azure.com",
                        "AZURE_OPENAI_API_KEY": "replace-with-your-key",
                        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-3-small",
                        "AZURE_OPENAI_API_VERSION": "2024-05-01-preview",
                        "AZURE_OPENAI_BATCH_SIZE": "256",
                        "AZURE_OPENAI_CONCURRENCY": "22",
                    },
                    clear=True,
                ):
                    with self.assertRaisesRegex(EmbeddingError, "placeholder values"):
                        validate_azure_embedding_config()

    def test_azure_embedding_batches_preserve_order_with_concurrency(self) -> None:
        def fake_post_azure_embeddings(**kwargs):
            texts = kwargs["texts"]
            return {
                "data": [
                    {"index": index, "embedding": [float(text), 1.0]}
                    for index, text in enumerate(texts)
                ]
            }

        with patch("sms_spam_ham_analysis.embeddings._load_project_env_file"):
            with patch("sms_spam_ham_analysis.embeddings._post_azure_embeddings", side_effect=fake_post_azure_embeddings):
                with patch.dict(
                    "os.environ",
                    {
                        "AZURE_OPENAI_ENDPOINT": "https://example.services.ai.azure.com",
                        "AZURE_OPENAI_API_KEY": "key",
                        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT": "text-embedding-3-small",
                        "AZURE_OPENAI_API_VERSION": "2024-05-01-preview",
                        "AZURE_OPENAI_BATCH_SIZE": "2",
                        "AZURE_OPENAI_CONCURRENCY": "3",
                    },
                    clear=True,
                ):
                    stderr = io.StringIO()
                    with redirect_stderr(stderr):
                        result = build_embeddings(["1", "2", "3", "4", "5"], provider="azure-openai")

        first_dimensions = result.vectors[:, 0].tolist()
        self.assertEqual(first_dimensions, sorted(first_dimensions))
        self.assertIn("Concurrency: 3", result.notes)
        progress = stderr.getvalue()
        self.assertIn("Submitted Azure batch 1/3 (2 messages)", progress)
        self.assertIn("Submitted Azure batch 2/3 (2 messages)", progress)
        self.assertIn("Submitted Azure batch 3/3 (1 messages)", progress)
        self.assertIn("Completed Azure batch 1/3", progress)
        self.assertIn("Completed Azure batch 2/3", progress)
        self.assertIn("Completed Azure batch 3/3", progress)

    def test_local_embeddings_print_batch_and_phase_progress(self) -> None:
        texts = [
            "team meeting tomorrow project update",
            "team meeting today project plan",
            "free prize claim text now",
            "free cash prize claim now",
            "call me after dinner today",
            "call me tomorrow after work",
        ]

        stderr = io.StringIO()
        with patch("sms_spam_ham_analysis.embeddings.DEFAULT_LOCAL_PROGRESS_BATCH_SIZE", 2):
            with patch("sms_spam_ham_analysis.embeddings.DEFAULT_LOCAL_CONCURRENCY", 2):
                with redirect_stderr(stderr):
                    result = build_embeddings(texts, provider="sklearn-svd", dimensions=3)

        self.assertEqual(result.provider, "sklearn-svd")
        self.assertEqual(result.vectors.shape, (6, 3))
        progress = stderr.getvalue()
        self.assertIn("Fitting local TF-IDF vocabulary and IDF on full corpus", progress)
        self.assertIn(
            "Embedding 6 messages locally with TF-IDF/SVD in 3 TF-IDF batches "
            "(batch size 2, concurrency 2)",
            progress,
        )
        self.assertIn("Submitted local TF-IDF batch 1/3 (2 messages)", progress)
        self.assertIn("Submitted local TF-IDF batch 2/3 (2 messages)", progress)
        self.assertIn("Submitted local TF-IDF batch 3/3 (2 messages)", progress)
        self.assertIn("Completed local TF-IDF batch 1/3", progress)
        self.assertIn("Completed local TF-IDF batch 2/3", progress)
        self.assertIn("Completed local TF-IDF batch 3/3", progress)
        self.assertIn("Fitting TruncatedSVD with 3 components", progress)
        self.assertIn("Completed local SVD embeddings: 6 messages x 3 dimensions", progress)

        sequential_stderr = io.StringIO()
        with patch("sms_spam_ham_analysis.embeddings.DEFAULT_LOCAL_PROGRESS_BATCH_SIZE", 2):
            with patch("sms_spam_ham_analysis.embeddings.DEFAULT_LOCAL_CONCURRENCY", 1):
                with redirect_stderr(sequential_stderr):
                    sequential_result = build_embeddings(texts, provider="sklearn-svd", dimensions=3)

        self.assertTrue(np.allclose(result.vectors, sequential_result.vectors))

    def test_pipeline_generates_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "data" / "raw"
            outputs = root / "outputs"
            raw_dir.mkdir(parents=True)
            outputs.mkdir()

            raw_file = raw_dir / "SMSSpamCollection"
            raw_file.write_text(
                "\n".join(f"{label}\t{text}" for label, text in SAMPLE_MESSAGES) + "\n",
                encoding="utf-8",
            )

            validated_dataset = outputs / "validated_sms_dataset.csv"
            validation_summary = outputs / "dataset_validation.json"
            dataset_summary = load_and_validate_dataset(raw_dir, validated_dataset, validation_summary)
            self.assertEqual(dataset_summary.validated_rows, len(SAMPLE_MESSAGES))
            self.assertEqual(dataset_summary.ham_count, 12)
            self.assertEqual(dataset_summary.spam_count, 12)

            run_text_analysis(
                validated_dataset,
                frequent_words_output=outputs / "frequent_words.csv",
                frequent_words_markdown=outputs / "frequent_words.md",
                vocabulary_output=outputs / "vocabulary_by_label.csv",
                spam_output=outputs / "spam_typical_words.csv",
                ham_output=outputs / "ham_typical_words.csv",
                vocabulary_findings=outputs / "vocabulary_findings.md",
                bigrams_output=outputs / "frequent_bigrams.csv",
                trigrams_output=outputs / "frequent_trigrams.csv",
                ngram_findings=outputs / "ngram_findings.md",
                top_words=20,
                top_typical_words=10,
                top_ngrams=10,
                min_count=1,
                remove_stopwords=False,
                min_token_length=1,
                smoothing=0.5,
            )

            model_path = outputs / "models" / "tfidf_classifier.joblib"
            evaluation = train_and_evaluate_model(
                validated_dataset,
                model_output=model_path,
                metadata_output=outputs / "models" / "training_metadata.json",
                metrics_output=outputs / "model_metrics.json",
                report_output=outputs / "classification_report.md",
                confusion_matrix_output=outputs / "confusion_matrix.csv",
                error_examples_output=outputs / "error_examples.csv",
                predictions_output=outputs / "test_predictions.csv",
                test_size=0.25,
                random_state=42,
            )
            self.assertEqual(evaluation.test_rows, 6)
            self.assertGreaterEqual(evaluation.accuracy, 0.5)

            clustering = run_clustering(
                validated_dataset,
                outputs / "semantic_clusters.csv",
                outputs / "cluster_summary.md",
                outputs / "embeddings" / "sms_embeddings.npy",
                outputs / "embeddings" / "metadata.json",
                clusters="3",
                provider="sklearn-svd",
                embedding_model=None,
                dimensions=8,
                random_state=42,
                representative_count=2,
                sample_size=None,
            )
            self.assertEqual(clustering.cluster_count, 3)
            self.assertEqual(clustering.total_messages, len(SAMPLE_MESSAGES))
            local_clustering_dir = outputs / "clustering" / "local"
            self.assertTrue((local_clustering_dir / "semantic_clusters.csv").exists())
            self.assertTrue((local_clustering_dir / "cluster_summary.md").exists())
            self.assertTrue((local_clustering_dir / "embeddings" / "sms_embeddings.npy").exists())
            local_metadata_path = local_clustering_dir / "embeddings" / "metadata.json"
            self.assertTrue(local_metadata_path.exists())
            local_metadata = json.loads(local_metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(local_metadata["embedding_provider"], "sklearn-svd")
            self.assertIn("outputs/clustering/local", Path(local_metadata["metadata_output"]).as_posix())

            visualization = run_visualizations(
                outputs,
                outputs / "figures",
                outputs / "artifact_index.md",
            )
            self.assertEqual(len(visualization.figures), 7)

            required_files = [
                "validated_sms_dataset.csv",
                "dataset_validation.json",
                "frequent_words.csv",
                "vocabulary_findings.md",
                "frequent_bigrams.csv",
                "frequent_trigrams.csv",
                "classification_report.md",
                "confusion_matrix.csv",
                "semantic_clusters.csv",
                "cluster_summary.md",
                "artifact_index.md",
                "clustering/provider_comparison.md",
                "figures/top_words.png",
                "figures/vocabulary_comparison.png",
                "figures/ngrams.png",
                "figures/confusion_matrix.png",
                "figures/model_metrics_comparison.png",
                "figures/precision_recall_curve.png",
                "figures/semantic_clusters.png",
            ]
            for relative_path in required_files:
                self.assertTrue((outputs / relative_path).exists(), relative_path)
            provider_comparison = (outputs / "clustering" / "provider_comparison.md").read_text(encoding="utf-8")
            self.assertIn("| local | generated | sklearn-svd |", provider_comparison)
            self.assertIn("| azure | not generated |", provider_comparison)

            metrics = json.loads((outputs / "model_metrics.json").read_text(encoding="utf-8"))
            self.assertIn("accuracy", metrics)
            self.assertEqual(set(metrics["support"]), {"ham", "spam"})
            self.assertIn("duplicate_summary", metrics)
            self.assertIn("duplicate_safe_evaluation", metrics)
            self.assertEqual(metrics["duplicate_safe_evaluation"]["text_overlap_count"], 0)
            self.assertEqual(
                metrics["duplicate_safe_evaluation"]["evaluation_strategy"],
                "group_shuffle_split_by_text",
            )

            cluster_assignments = pd.read_csv(outputs / "semantic_clusters.csv")
            self.assertEqual(len(cluster_assignments), len(SAMPLE_MESSAGES))
            self.assertEqual(
                {"row_id", "label", "text", "cluster_id", "distance_to_centroid", "x", "y"},
                set(cluster_assignments.columns),
            )

    def test_azure_clustering_writes_provider_specific_outputs(self) -> None:
        vectors = np.vstack(
            [
                np.tile([1.0, 0.0, 0.0], (8, 1)),
                np.tile([0.0, 1.0, 0.0], (8, 1)),
                np.tile([0.0, 0.0, 1.0], (8, 1)),
            ]
        )

        def fake_build_embeddings(*args, **kwargs):
            return EmbeddingResult(
                vectors=vectors,
                provider="azure-openai",
                model_name="text-embedding-3-small",
                dimensions=3,
                notes="Mocked Azure embeddings for provider-specific output testing.",
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs = root / "outputs"
            outputs.mkdir()
            dataset = outputs / "validated_sms_dataset.csv"
            pd.DataFrame(
                {
                    "row_id": range(1, len(SAMPLE_MESSAGES) + 1),
                    "label": [label for label, _ in SAMPLE_MESSAGES],
                    "text": [text for _, text in SAMPLE_MESSAGES],
                }
            ).to_csv(dataset, index=False)

            with patch("sms_spam_ham_analysis.clustering.build_embeddings", side_effect=fake_build_embeddings):
                summary = run_clustering(
                    dataset,
                    outputs / "semantic_clusters.csv",
                    outputs / "cluster_summary.md",
                    outputs / "embeddings" / "sms_embeddings.npy",
                    outputs / "embeddings" / "metadata.json",
                    clusters="3",
                    provider="azure-openai",
                    embedding_model=None,
                    dimensions=8,
                    random_state=42,
                    representative_count=2,
                    sample_size=None,
                )

            self.assertEqual(summary.embedding_provider, "azure-openai")
            azure_clustering_dir = outputs / "clustering" / "azure"
            self.assertTrue((azure_clustering_dir / "semantic_clusters.csv").exists())
            self.assertTrue((azure_clustering_dir / "cluster_summary.md").exists())
            self.assertTrue((azure_clustering_dir / "embeddings" / "sms_embeddings.npy").exists())
            azure_metadata = json.loads(
                (azure_clustering_dir / "embeddings" / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(azure_metadata["embedding_provider"], "azure-openai")
            self.assertIn("outputs/clustering/azure", Path(azure_metadata["metadata_output"]).as_posix())

    def test_duplicate_safe_evaluation_groups_duplicate_texts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "data" / "raw"
            outputs = root / "outputs"
            raw_dir.mkdir(parents=True)
            outputs.mkdir()

            duplicate_messages = SAMPLE_MESSAGES + [
                SAMPLE_MESSAGES[0],
                SAMPLE_MESSAGES[12],
            ]
            raw_file = raw_dir / "SMSSpamCollection"
            raw_file.write_text(
                "\n".join(f"{label}\t{text}" for label, text in duplicate_messages) + "\n",
                encoding="utf-8",
            )

            validated_dataset = outputs / "validated_sms_dataset.csv"
            load_and_validate_dataset(raw_dir, validated_dataset, outputs / "dataset_validation.json")
            train_and_evaluate_model(
                validated_dataset,
                model_output=outputs / "models" / "tfidf_classifier.joblib",
                metadata_output=outputs / "models" / "training_metadata.json",
                metrics_output=outputs / "model_metrics.json",
                report_output=outputs / "classification_report.md",
                confusion_matrix_output=outputs / "confusion_matrix.csv",
                error_examples_output=outputs / "error_examples.csv",
                predictions_output=outputs / "test_predictions.csv",
                test_size=0.25,
                random_state=42,
            )

            metrics = json.loads((outputs / "model_metrics.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(metrics["duplicate_summary"]["duplicate_message_rows"], 4)
            self.assertEqual(metrics["duplicate_safe_evaluation"]["text_overlap_count"], 0)


if __name__ == "__main__":
    unittest.main()
