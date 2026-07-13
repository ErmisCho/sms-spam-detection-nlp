from __future__ import annotations

import asyncio
import io
import json
import logging
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import httpx
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sms_spam_ham_analysis.api import JsonFormatter, LOGGER, create_app
from sms_spam_ham_analysis.model import build_tfidf_classifier
from sms_spam_ham_analysis.predict import clear_model_cache


class ApiClient:
    """Small synchronous facade over httpx's supported in-process ASGI transport."""

    def __init__(self, app: Any) -> None:
        self.app = app

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        async def send() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", path, **kwargs)

    def close(self) -> None:
        return None


class PredictionApiTest(unittest.TestCase):
    def setUp(self) -> None:
        clear_model_cache()
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.model_path = self.root / "classifier.joblib"
        self._write_model(self.model_path)
        self.client = ApiClient(create_app(model_path=self.model_path))

    def tearDown(self) -> None:
        self.client.close()
        clear_model_cache()
        self.temporary_directory.cleanup()

    @staticmethod
    def _write_model(path: Path) -> None:
        pipeline = build_tfidf_classifier()
        pipeline.fit(
            [
                "team meeting at the office tomorrow",
                "please call me when you get home",
                "office meeting agenda for tomorrow",
                "call home after the team meeting",
                "claim your free cash prize now",
                "winner claim free reward immediately",
                "free prize cash offer claim today",
                "urgent winner reward claim now",
            ],
            ["ham", "ham", "ham", "ham", "spam", "spam", "spam", "spam"],
        )
        joblib.dump({"pipeline": pipeline, "metadata": {}}, path)

    def test_health_distinguishes_liveness_and_readiness(self) -> None:
        health = self.client.get("/health")
        live = self.client.get("/health/live")
        ready = self.client.get("/health/ready")

        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok", "model_ready": True})
        self.assertEqual(live.status_code, 200)
        self.assertTrue(live.json()["model_ready"])
        self.assertEqual(ready.status_code, 200)
        self.assertEqual(ready.json(), {"status": "ready"})

    def test_predicts_ham_and_spam_with_contract_fields(self) -> None:
        ham = self.client.post("/predict", json={"text": "team meeting at office tomorrow"})
        spam = self.client.post("/predict", json={"text": "claim free cash prize now"})

        self.assertEqual(ham.status_code, 200)
        self.assertEqual(ham.json()["label"], "ham")
        self.assertGreaterEqual(ham.json()["confidence"], 0.5)
        self.assertEqual(spam.status_code, 200)
        self.assertEqual(spam.json()["label"], "spam")
        self.assertGreaterEqual(spam.json()["confidence"], 0.5)
        self.assertEqual(set(spam.json()), {"label", "confidence"})

    def test_rejects_empty_extra_and_malformed_requests(self) -> None:
        empty = self.client.post("/predict", json={"text": "   "})
        extra = self.client.post("/predict", json={"text": "hello", "phone": "+431"})
        malformed = self.client.post(
            "/predict",
            content=b'{"text":',
            headers={"content-type": "application/json"},
        )

        self.assertEqual(empty.status_code, 422)
        self.assertEqual(extra.status_code, 422)
        self.assertEqual(malformed.status_code, 422)

    def test_missing_and_corrupt_models_fail_without_leaking_paths(self) -> None:
        missing_path = self.root / "private" / "missing.joblib"
        missing_client = ApiClient(create_app(model_path=missing_path))
        missing_health = missing_client.get("/health")
        missing_ready = missing_client.get("/health/ready")
        missing_prediction = missing_client.post("/predict", json={"text": "hello"})
        missing_client.close()

        self.assertEqual(missing_health.json(), {"status": "ok", "model_ready": False})
        self.assertEqual(missing_ready.status_code, 503)
        self.assertEqual(missing_prediction.status_code, 503)
        self.assertNotIn(str(missing_path), missing_prediction.text)

        corrupt_path = self.root / "corrupt.joblib"
        corrupt_path.write_bytes(b"not a trusted joblib artifact")
        corrupt_client = ApiClient(create_app(model_path=corrupt_path))
        corrupt_prediction = corrupt_client.post("/predict", json={"text": "hello"})
        corrupt_client.close()

        self.assertEqual(corrupt_prediction.status_code, 503)
        self.assertEqual(corrupt_prediction.json(), {"detail": "Prediction model is unavailable."})
        self.assertNotIn(str(corrupt_path), corrupt_prediction.text)

    def test_openapi_documents_prediction_contract(self) -> None:
        schema = self.client.get("/openapi.json").json()

        self.assertIn("/predict", schema["paths"])
        self.assertIn("/api/v1/predict", schema["paths"])
        self.assertTrue(schema["paths"]["/predict"]["post"]["deprecated"])
        self.assertIn("PredictionRequest", schema["components"]["schemas"])
        self.assertIn("PredictionResponse", schema["components"]["schemas"])

    def test_versioned_prediction_route_matches_legacy_contract(self) -> None:
        payload = {"text": "claim free cash prize now"}

        versioned = self.client.post("/api/v1/predict", json=payload)
        legacy = self.client.post("/predict", json=payload)

        self.assertEqual(versioned.status_code, 200)
        self.assertEqual(versioned.json(), legacy.json())

    def test_serves_compiled_frontend_without_shadowing_api_docs(self) -> None:
        frontend_dist = self.root / "frontend"
        assets = frontend_dist / "assets"
        assets.mkdir(parents=True)
        (frontend_dist / "index.html").write_text(
            '<!doctype html><div id="root">MessageGuard</div>', encoding="utf-8"
        )
        (assets / "app.js").write_text("console.log('demo')", encoding="utf-8")
        client = ApiClient(create_app(model_path=self.model_path, frontend_dist=frontend_dist))

        home = client.get("/")
        asset = client.get("/assets/app.js")
        docs = client.get("/docs")
        prediction = client.post("/api/v1/predict", json={"text": "team meeting tomorrow"})
        client.close()

        self.assertEqual(home.status_code, 200)
        self.assertIn("MessageGuard", home.text)
        self.assertEqual(asset.status_code, 200)
        self.assertIn("console.log", asset.text)
        self.assertEqual(docs.status_code, 200)
        self.assertEqual(prediction.status_code, 200)

    def test_request_logs_are_structured_and_do_not_capture_sms_text(self) -> None:
        stream = io.StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())
        original_handlers = list(LOGGER.handlers)
        LOGGER.handlers = [handler]
        try:
            response = self.client.post(
                "/predict",
                json={"text": "PRIVATE MESSAGE CONTENT"},
                headers={"X-Request-ID": "interview-demo-123"},
            )
        finally:
            LOGGER.handlers = original_handlers

        record = json.loads(stream.getvalue().strip())
        self.assertEqual(response.headers["X-Request-ID"], "interview-demo-123")
        self.assertEqual(record["event"], "request_completed")
        self.assertEqual(record["request_id"], "interview-demo-123")
        self.assertEqual(record["route"], "/predict")
        self.assertEqual(record["status"], 200)
        self.assertIn("latency_ms", record)
        self.assertNotIn("PRIVATE MESSAGE CONTENT", stream.getvalue())

    def test_invalid_request_id_is_replaced(self) -> None:
        response = self.client.get("/health", headers={"X-Request-ID": "invalid id with spaces"})

        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(response.headers["X-Request-ID"], "invalid id with spaces")
        self.assertRegex(response.headers["X-Request-ID"], r"^[a-f0-9]{32}$")


if __name__ == "__main__":
    unittest.main()
