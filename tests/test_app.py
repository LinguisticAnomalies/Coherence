"""
Content Warning: The unit tests are AI-generated.

Tests for the CCC FastAPI endpoints in scripts/app.py.

Strategy
--------
* The heavy model-loading startup event is intentionally NOT run (the
  TestClient is used without its context manager), so ``state.ready`` stays
  unset and no embeddings/models are loaded.
* ``state.ready`` and the expensive feature pipeline (``_compute_features``)
  are managed/stubbed directly, isolating the HTTP/routing/response logic
  from the actual ML pipeline, which is far too slow for unit tests.

Run with (no pytest required):
    python -m unittest discover -s tests    # from repo root
    python -m unittest tests.test_app
    python -m pytest tests/test_app.py      # if pytest is installed
"""

import os
import sys
import unittest

# Make `src` importable (matches package-dir in pyproject.toml).
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pandas as pd
from fastapi.testclient import TestClient

from coherencecalculator.scripts import app as app_mod
from coherencecalculator.scripts.app import app


class BaseEndpointTest(unittest.TestCase):
    """Shared client setup. Subclasses get a fresh client + clean ready state."""

    def setUp(self) -> None:
        # No context manager => the model-loading startup event does not run.
        self.client = TestClient(app)
        self.addCleanup(self.client.close)
        # Guard against a leftover ready state from a prior test.
        app_mod.state.ready.clear()

    def tearDown(self) -> None:
        app_mod.state.ready.clear()


class HealthTests(BaseEndpointTest):
    def test_health_ok_when_ready(self) -> None:
        app_mod.state.ready.set()
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_health_loading_when_not_ready(self) -> None:
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.json(), {"status": "loading"})


class MetaTests(BaseEndpointTest):
    def test_meta_server(self) -> None:
        r = self.client.get("/meta/server")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), app_mod.META_DATA)

    def test_meta_feature(self) -> None:
        r = self.client.get("/meta/feature")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), app_mod.FEATURE_DATA)


class RootTests(BaseEndpointTest):
    def test_root(self) -> None:
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["service"], "ccc")
        self.assertIn("/health", body["endpoints"])
        self.assertIn("/infer", body["endpoints"])


class InferTests(BaseEndpointTest):
    """Tests /infer. The real feature pipeline is stubbed out.

    The actual pipeline (VecLoader + timeseries + agg) is replaced here so
    routing, input validation, and response shaping can be tested quickly
    and deterministically.
    """

    @staticmethod
    def _fake_compute(input_df: pd.DataFrame, state) -> pd.DataFrame:
        """Stand-in for _compute_features: one agg-style row per input row."""
        return pd.DataFrame(
            {
                "filename": [str(f) for f in input_df["filename"]],
                "wordCoherenceSeq": [0.5] * len(input_df),
            }
        )

    def setUp(self) -> None:
        super().setUp()
        self._orig_compute = app_mod._compute_features
        app_mod._compute_features = self._fake_compute
        app_mod.state.ready.set()

    def tearDown(self) -> None:
        app_mod._compute_features = self._orig_compute
        super().tearDown()

    def test_infer_503_when_not_ready(self) -> None:
        app_mod.state.ready.clear()
        r = self.client.post("/infer", json={"filename": "a.txt", "text": "hi"})
        self.assertEqual(r.status_code, 503)

    def test_infer_400_invalid_json(self) -> None:
        r = self.client.post("/infer", content=b"this is not json", headers={"Content-Type": "text/plain"})
        self.assertEqual(r.status_code, 400)

    def test_infer_422_missing_required_fields(self) -> None:
        r = self.client.post("/infer", json={"filename": "a.txt"})  # no "text"
        self.assertEqual(r.status_code, 422)

    def test_infer_single_object(self) -> None:
        r = self.client.post("/infer", json={"filename": "a.txt", "text": "hello world"})
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["n"], 1)
        self.assertEqual(body["rows"][0]["filename"], "a.txt")

    def test_infer_list(self) -> None:
        payload = [
            {"filename": "a.txt", "text": "first"},
            {"filename": "b.txt", "text": "second"},
        ]
        r = self.client.post("/infer", json=payload)
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["n"], 2)
        self.assertEqual([row["filename"] for row in body["rows"]], ["a.txt", "b.txt"])

    def test_infer_items_wrapper(self) -> None:
        payload = {"items": [{"filename": "a.txt", "text": "one"}]}
        r = self.client.post("/infer", json=payload)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["n"], 1)

    def test_infer_returns_n0_when_empty_result(self) -> None:
        app_mod._compute_features = lambda input_df, state: pd.DataFrame()
        r = self.client.post("/infer", json={"filename": "a.txt", "text": "hi"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"n": 0, "rows": []})

    def test_infer_drops_text_column_from_response(self) -> None:
        r = self.client.post("/infer", json={"filename": "a.txt", "text": "hi"})
        row = r.json()["rows"][0]
        self.assertNotIn("text", row)


if __name__ == "__main__":
    unittest.main(verbosity=2)
