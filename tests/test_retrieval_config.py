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
from retrieval import (  # noqa: E402
    RetrievalResult,
    deduplicate_for_query,
    normalize_url_for_dedup,
    route_relevance_multiplier,
)


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

    def test_jina_backend_uses_model_rerank_scores(self) -> None:
        class FakeJinaReranker:
            def rerank(self, query, documents, top_n=None):
                self.query = query
                self.documents = documents
                self.top_n = top_n
                return [
                    {"index": 1, "relevance_score": 0.95},
                    {"index": 0, "relevance_score": 0.10},
                ]

        previous_backend = reranking.RERANKER_BACKEND
        previous_enabled = os.environ.get("RERANKER_ENABLED")
        fake_model = FakeJinaReranker()

        try:
            reranking.RERANKER_BACKEND = "jina"
            os.environ["RERANKER_ENABLED"] = "true"
            results = [
                SimpleNamespace(text="primo", score=0.1, rank=1, metadata={}),
                SimpleNamespace(text="secondo", score=0.9, rank=2, metadata={}),
            ]

            with patch.object(reranking, "get_reranker", return_value=fake_model):
                reranked = reranking.neural_rerank("query", results, top_k=2)
        finally:
            reranking.RERANKER_BACKEND = previous_backend
            if previous_enabled is None:
                os.environ.pop("RERANKER_ENABLED", None)
            else:
                os.environ["RERANKER_ENABLED"] = previous_enabled

        self.assertEqual(fake_model.query, "query")
        self.assertEqual(fake_model.top_n, 2)
        self.assertEqual(reranked[0].text, "secondo")
        self.assertEqual(results[1].metadata["reranker_backend"], "jina")
        self.assertEqual(results[1].metadata["reranker_score"], 0.95)

    def test_vectorstore_config_mismatches_detects_stale_dense_profile(self) -> None:
        stats = vector_store.expected_vectorstore_config()
        stats["dense_index_profile"] = "all"

        mismatches = vector_store.vectorstore_config_mismatches(stats)

        self.assertIn(
            "dense_index_profile='all' (atteso 'core')",
            mismatches,
        )

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
                "text_hash": "hash",
                "chars": 8,
            }
        )

        self.assertEqual(document.metadata["source"], "pdf")
        self.assertEqual(document.metadata["document_type"], "almalaurea")
        self.assertEqual(document.metadata["document_years"], "[2024]")
        self.assertEqual(document.metadata["clean_status"], "ok")

    def test_normalize_url_for_dedup_preserves_structure_detail_id(self) -> None:
        first = normalize_url_for_dedup("https://www.diem.unisa.it/dipartimento/strutture?id=2")
        second = normalize_url_for_dedup("https://www.diem.unisa.it/dipartimento/strutture?id=23")

        self.assertNotEqual(first, second)
        self.assertTrue(first.endswith("/dipartimento/strutture?id=2"))

    def test_route_relevance_boosts_labrob_equipment_chunk(self) -> None:
        labrob = RetrievalResult(
            chunk_id="labrob",
            text="Titolo contenuto: Dipartimento | Robotica\nSezione: Strumentazione\n2 x robot antropomorfi",
            metadata={
                "source_url": "https://www.diem.unisa.it/dipartimento/strutture?id=2",
                "title": "Dipartimento | Strutture",
                "content_title": "Dipartimento | Robotica",
                "section_heading": "Strumentazione",
            },
            source="test",
            rank=1,
            score=1.0,
        )
        teti = RetrievalResult(
            chunk_id="teti",
            text="Sezione: Strumentazione\n10 calcolatori e 1 analizzatore di spettro",
            metadata={
                "source_url": "https://www.diem.unisa.it/dipartimento/strutture?id=23",
                "title": "Dipartimento | Strutture",
                "content_title": "Dipartimento | Telecomunicazioni e Teoria dell'Informazione",
                "section_heading": "Strumentazione",
            },
            source="test",
            rank=2,
            score=1.0,
        )
        robot_pepper = RetrievalResult(
            chunk_id="pepper",
            text="Sezione: Strumentazione\nRobot Pepper umanoide",
            metadata={
                "source_url": "https://www.diem.unisa.it/dipartimento/strutture?id=212",
                "title": "Dipartimento | Strutture",
                "content_title": "Dipartimento | Strutture",
                "section_heading": "Strumentazione",
            },
            source="test",
            rank=3,
            score=1.0,
        )

        query = "Qual è la strumentazione del laboratorio di robotica?"
        labrob_score = route_relevance_multiplier(labrob, query)

        self.assertGreater(labrob_score, route_relevance_multiplier(teti, query))
        self.assertGreater(labrob_score, route_relevance_multiplier(robot_pepper, query))

    def test_route_relevance_boosts_almalaurea_year_and_course(self) -> None:
        result = RetrievalResult(
            chunk_id="almalaurea",
            text="Testo link sorgente: Livello di soddisfazione dei laureandi (profilo dei laureati)",
            metadata={
                "source_url": "https://corsi.unisa.it/uploads/rescue/__almalaurea/2024/0650107303300003.pdf",
                "discovered_from": "https://corsi.unisa.it/information-Engineering-for-digital-medicine/statistiche",
                "link_text": "AlmaLaurea 2024",
                "document_type": "almalaurea",
                "document_years": [2024],
            },
            source="pdf",
            rank=1,
            score=1.0,
        )

        self.assertGreater(
            route_relevance_multiplier(
                result,
                "Valutazione laureati 2024 corso Information Engineering for Digital Medicine",
            ),
            4.0,
        )

    def test_teacher_publication_dedup_uses_publication_id_not_url_only(self) -> None:
        results = [
            RetrievalResult(
                chunk_id="pub_header",
                text="Fabio POSTIGLIONE Pubblicazioni Anno: Tutti",
                metadata={
                    "source_url": "https://docenti.unisa.it/003735/ricerca/pubblicazioni?anno=0",
                    "title": "Fabio POSTIGLIONE | Pubblicazioni",
                    "chunk_kind": "teacher_publications_page",
                },
                source="test",
                rank=1,
                score=1.0,
            ),
            RetrievalResult(
                chunk_id="pub_2026",
                text="Titolo pubblicazione: Recent work",
                metadata={
                    "source_url": "https://docenti.unisa.it/003735/ricerca/pubblicazioni?anno=0",
                    "title": "Fabio POSTIGLIONE | Pubblicazioni",
                    "chunk_kind": "publication_summary",
                    "publication_id": "435220",
                    "publication_year": 2026,
                },
                source="test",
                rank=2,
                score=0.9,
            ),
            RetrievalResult(
                chunk_id="pub_2025",
                text="Titolo pubblicazione: Previous work",
                metadata={
                    "source_url": "https://docenti.unisa.it/003735/ricerca/pubblicazioni?anno=0",
                    "title": "Fabio POSTIGLIONE | Pubblicazioni",
                    "chunk_kind": "publication_summary",
                    "publication_id": "435219",
                    "publication_year": 2025,
                },
                source="test",
                rank=3,
                score=0.8,
            ),
            RetrievalResult(
                chunk_id="pub_2025_dup",
                text="Titolo pubblicazione: Previous work duplicate",
                metadata={
                    "source_url": "https://docenti.unisa.it/003735/ricerca/pubblicazioni?anno=0",
                    "title": "Fabio POSTIGLIONE | Pubblicazioni",
                    "chunk_kind": "publication_summary",
                    "publication_id": "435219",
                    "publication_year": 2025,
                },
                source="test",
                rank=4,
                score=0.7,
            ),
        ]

        deduped = deduplicate_for_query(
            results,
            "quali sono le recenti pubblicazioni del professore Fabio Postiglione?",
        )
        chunk_ids = [result.chunk_id for result in deduped]

        self.assertIn("pub_header", chunk_ids)
        self.assertIn("pub_2026", chunk_ids)
        self.assertIn("pub_2025", chunk_ids)
        self.assertNotIn("pub_2025_dup", chunk_ids)

    def test_route_relevance_boosts_recent_publication_summary(self) -> None:
        summary = RetrievalResult(
            chunk_id="summary",
            text="Titolo pubblicazione: Recent work",
            metadata={
                "source_url": "https://docenti.unisa.it/003735/ricerca/pubblicazioni?anno=0",
                "title": "Fabio POSTIGLIONE | Pubblicazioni",
                "chunk_kind": "publication_summary",
                "publication_id": "435220",
                "publication_year": 2026,
            },
            source="test",
            rank=1,
            score=1.0,
        )
        header = RetrievalResult(
            chunk_id="header",
            text="Fabio POSTIGLIONE Pubblicazioni Anno: Tutti",
            metadata={
                "source_url": "https://docenti.unisa.it/003735/ricerca/pubblicazioni?anno=0",
                "title": "Fabio POSTIGLIONE | Pubblicazioni",
                "chunk_kind": "teacher_publications_page",
            },
            source="test",
            rank=2,
            score=1.0,
        )

        query = "quali sono le recenti pubblicazioni del professore Fabio Postiglione?"

        self.assertGreater(
            route_relevance_multiplier(summary, query),
            route_relevance_multiplier(header, query),
        )

    def test_final_exam_query_penalizes_immatricolazioni(self) -> None:
        final_exam = RetrievalResult(
            chunk_id="final_exam",
            text="Domanda Conseguimento Titolo e sedute di laurea. Prova finale.",
            metadata={
                "source_url": "https://corsi.unisa.it/ingegneria-informatica/didattica/esame-finale",
                "title": "Ingegneria Informatica | Esame Finale",
                "breadcrumb": "Ingegneria Informatica > Didattica > Esame Finale",
                "chunk_kind": "course_info",
            },
            source="test",
            rank=1,
            score=1.0,
        )
        admission = RetrievalResult(
            chunk_id="admission",
            text="Verifica dei requisiti TOLC OFA.",
            metadata={
                "source_url": "https://corsi.unisa.it/ingegneria-informatica/immatricolazioni",
                "title": "Ingegneria Informatica | Modalità di accesso - Immatricolazioni",
                "breadcrumb": "Ingegneria Informatica > Modalità di accesso - Immatricolazioni",
                "chunk_kind": "course_info",
            },
            source="test",
            rank=2,
            score=1.0,
        )
        query = "Quali sono i criteri di ammissione alla seduta di laurea triennale?"

        self.assertGreater(
            route_relevance_multiplier(final_exam, query),
            route_relevance_multiplier(admission, query),
        )


if __name__ == "__main__":
    unittest.main()
