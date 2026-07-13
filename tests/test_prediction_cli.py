from __future__ import annotations

import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sms_spam_ham_analysis.model import build_tfidf_classifier
from sms_spam_ham_analysis.predict import ModelUnavailableError, PredictionError, main, predict_message


class PredictionCliTest(unittest.TestCase):
    def _write_model(self, path: Path) -> None:
        pipeline = build_tfidf_classifier()
        texts = [
            "meeting at home tonight",
            "call me when you arrive home",
            "free cash prize claim now",
            "win free reward claim now",
        ]
        pipeline.fit(texts, ["ham", "ham", "spam", "spam"])
        joblib.dump({"pipeline": pipeline, "metadata": {}}, path)

    def test_predict_message_returns_label_and_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "classifier.joblib"
            self._write_model(model_path)

            result = predict_message("free prize claim now", model_path=model_path)

            self.assertEqual(result.label, "spam")
            self.assertGreaterEqual(result.confidence, 0.5)
            self.assertLessEqual(result.confidence, 1.0)

    def test_missing_model_error_explains_how_to_train(self) -> None:
        missing = Path("does-not-exist.joblib")
        with self.assertRaisesRegex(ModelUnavailableError, "Run the modeling step first"):
            predict_message("hello", model_path=missing)

    def test_corrupt_model_is_reported_as_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "classifier.joblib"
            model_path.write_bytes(b"not a model")

            with self.assertRaisesRegex(ModelUnavailableError, "Unable to load trained model artifact"):
                predict_message("hello", model_path=model_path)

    def test_cli_prints_readable_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "classifier.joblib"
            self._write_model(model_path)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = main(["free cash prize", "--model", str(model_path)])

            self.assertEqual(exit_code, 0)
            self.assertIn("Prediction: SPAM", stdout.getvalue())
            self.assertIn("Confidence:", stdout.getvalue())

    def test_cli_reports_missing_model_without_traceback(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            exit_code = main(["hello", "--model", "missing.joblib"])

        self.assertEqual(exit_code, 2)
        self.assertIn("Trained model not found", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
