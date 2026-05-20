from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import reranking  # noqa: E402
import vector_store  # noqa: E402


class RetrievalConfigTests(unittest.TestCase):
    def test_neural_rerank_is_noop_when_disabled(self) -> None:
        previous = os.environ.get("RERANKER_ENABLED")
        os.environ["RERANKER_ENABLED"] = "false"
        try:
            results = [
                SimpleNamespace(text="primo", score=0.2, rank=1, metadata={}),
                SimpleNamespace(text="secondo", score=0.1, rank=2, metadata={}),
            ]

            reranked = reranking.neural_rerank("query", results, top_k=1)
        finally:
            if previous is None:
                os.environ.pop("RERANKER_ENABLED", None)
            else:
                os.environ["RERANKER_ENABLED"] = previous

        self.assertEqual(reranked, results[:1])
        self.assertNotIn("reranker_score", results[0].metadata)

    def test_vectorstore_config_mismatches_detects_stale_dense_profile(self) -> None:
        stats = vector_store.expected_vectorstore_config()
        stats["dense_index_profile"] = "all"

        mismatches = vector_store.vectorstore_config_mismatches(stats)

        self.assertIn(
            "dense_index_profile='all' (atteso 'core')",
            mismatches,
        )


if __name__ == "__main__":
    unittest.main()
