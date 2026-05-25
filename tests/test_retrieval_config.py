"""
Test per configurazione retrieval, reranking e vector store.

Testa solo funzionalità che esistono ancora dopo la semplificazione del retrieval.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import reranking  # noqa: E402
import vector_store  # noqa: E402
from reranking import reranker_passage  # noqa: E402


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

    def test_reranker_passage_includes_metadata_context(self) -> None:
        result = SimpleNamespace(
            text="### Strumentazione\n2 x robot antropomorfi",
            metadata={
                "title": "Dipartimento | Strutture",
                "content_title": "Dipartimento | Robotica",
                "section_heading": "Strumentazione",
                "source_url": "https://www.diem.unisa.it/dipartimento/strutture?id=2",
            },
        )

        passage = reranker_passage(result)

        self.assertIn("Titolo contenuto: Dipartimento | Robotica", passage)
        self.assertIn("Sezione: Strumentazione", passage)
        self.assertIn("https://www.diem.unisa.it/dipartimento/strutture?id=2", passage)

    def test_cross_encoder_score_keeps_probability_outputs(self) -> None:
        self.assertEqual(reranking.cross_encoder_relevance_score(0.8), 0.8)
        self.assertGreater(reranking.cross_encoder_relevance_score(2.0), 0.8)
        self.assertLess(reranking.cross_encoder_relevance_score(-2.0), 0.2)

    def test_vectorstore_config_mismatches_detects_stale_dense_profile(self) -> None:
        stats = vector_store.expected_vectorstore_config()
        stats["dense_index_profile"] = "all"

        mismatches = vector_store.vectorstore_config_mismatches(stats)

        self.assertIn(
            "dense_index_profile='all' (atteso 'core')",
            mismatches,
        )

    def test_vectorstore_config_mismatches_detects_stale_chunk_schema(self) -> None:
        stats = vector_store.expected_vectorstore_config()
        stats["chunk_metadata_schema_version"] = 2

        mismatches = vector_store.vectorstore_config_mismatches(stats)

        self.assertIn(
            "chunk_metadata_schema_version=2 (atteso 4)",
            mismatches,
        )

    def test_dense_policy_keeps_high_value_study_plan_pdf(self) -> None:
        chunk = {
            "chunk_id": "study_plan",
            "text": "CORSO DI LAUREA IN INGEGNERIA INFORMATICA L-8\n1° ANNO\nAnalisi Matematica I",
            "retrieval_metadata": {
                "source": "pdf",
                "source_url": "https://corsi.unisa.it/uploads/rescue/__piano-studi-cds/2018/IE127.pdf",
                "title": "Piano degli studi",
                "chunk_kind": "study_plan",
            },
        }

        self.assertTrue(vector_store.should_index_dense(chunk))

    def test_dense_policy_can_skip_low_value_old_pdf(self) -> None:
        chunk = {
            "chunk_id": "old_pdf",
            "text": "Documento storico senza segnali didattici o amministrativi rilevanti.",
            "retrieval_metadata": {
                "source": "pdf",
                "source_url": "https://www.diem.unisa.it/uploads/rescue/2010/documento-generico.pdf",
                "title": "Documento generico",
                "chunk_kind": "text",
            },
        }

        self.assertFalse(vector_store.should_index_dense(chunk))

    def test_chunk_to_document_flattens_structured_chunk_metadata(self) -> None:
        document = vector_store.chunk_to_document(
            {
                "chunk_id": "structured",
                "document_hash": "doc",
                "chunk_index": 0,
                "chunk_count": 1,
                "retrieval_metadata": {
                    "source": "pdf",
                    "source_url": "https://corsi.unisa.it/uploads/rescue/__almalaurea/2024/x.pdf",
                    "title": "AlmaLaurea",
                    "document_type": "almalaurea",
                    "document_years": [2024],
                },
                "provenance": {
                    "last_crawled": "2026-05-23T00:00:00+00:00",
                },
                "debug": {
                    "clean_status": "ok",
                },
                "text": "contenuto",
                "text_for_embedding": "contenuto",
                "text_for_display": "[CONTESTO]\ncontenuto",
                "text_hash": "hash",
                "chars": 8,
            }
        )

        self.assertEqual(document.metadata["source"], "pdf")
        self.assertEqual(document.metadata["document_type"], "almalaurea")
        self.assertEqual(document.metadata["document_years"], "[2024]")
        self.assertEqual(document.metadata["clean_status"], "ok")
        # Verifica che text_for_display sia nei metadata
        self.assertIn("text_for_display", document.metadata)

    def test_chunk_to_document_can_build_locator_document(self) -> None:
        document = vector_store.chunk_to_document(
            {
                "chunk_id": "study_plan",
                "document_hash": "doc",
                "chunk_index": 0,
                "chunk_count": 1,
                "retrieval_metadata": {
                    "source_url": "https://unisa.coursecatalogue.cineca.it/corsi/2025/500853",
                    "title": "INGEGNERIA INFORMATICA",
                    "chunk_kind": "study_plan",
                    "course_name": "INGEGNERIA INFORMATICA",
                    "course_year": 2,
                    "curriculum": "SOFTWARE",
                },
                "locator_text": "corso=INGEGNERIA INFORMATICA; anno_corso=2; curriculum=SOFTWARE",
                "body_text": "Algoritmi e Strutture Dati",
                "text": "display",
            },
            representation="locator",
        )

        self.assertIn("anno_corso=2", document.page_content)
        self.assertEqual(document.metadata["vector_representation"], "locator")


if __name__ == "__main__":
    unittest.main()
