from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sms_spam_ham_analysis.demo_model import DemoModelError, build_demo_model, verify_demo_model
from sms_spam_ham_analysis.predict import clear_model_cache, predict_message


class DemoModelTest(unittest.TestCase):
    def tearDown(self) -> None:
        clear_model_cache()

    def test_builds_self_contained_model_with_verifiable_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "demo.joblib"
            manifest_path = Path(tmp) / "manifest.json"

            manifest = build_demo_model(artifact, manifest_path)
            verified = verify_demo_model(artifact, manifest_path)

            self.assertEqual(verified, manifest)
            self.assertEqual(manifest["training_rows"], 24)
            self.assertIn("demonstration only", str(manifest["purpose"]).lower())
            self.assertEqual(len(str(manifest["artifact_sha256"])), 64)
            self.assertEqual(predict_message("claim free cash prize now", model_path=artifact).label, "spam")
            self.assertEqual(predict_message("team meeting tomorrow", model_path=artifact).label, "ham")

    def test_verification_rejects_a_tampered_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "demo.joblib"
            manifest_path = Path(tmp) / "manifest.json"
            build_demo_model(artifact, manifest_path)
            artifact.write_bytes(artifact.read_bytes() + b"tampered")

            with self.assertRaisesRegex(DemoModelError, "checksum"):
                verify_demo_model(artifact, manifest_path)

    def test_verification_rejects_changed_corpus_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "demo.joblib"
            manifest_path = Path(tmp) / "manifest.json"
            build_demo_model(artifact, manifest_path)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["training_corpus_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(DemoModelError, "provenance"):
                verify_demo_model(artifact, manifest_path)
