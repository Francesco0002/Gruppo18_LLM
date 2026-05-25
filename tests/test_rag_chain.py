from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rag_chain  # noqa: E402
from rag_chain import (  # noqa: E402
    RagResponse,
    answer_question_as_text,
    build_retrieval_question,
    build_sources,
    build_direct_publications_answer,
    parse_contextualizer_payload,
    clean_display_url,
    parse_model_answer,
    parse_used_source_indexes,
    strip_model_thinking,
)
from retrieval import RetrievalResult  # noqa: E402


class RagChainTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_contextualizer_enabled = os.environ.get("RAG_CONTEXTUALIZER_ENABLED")
        os.environ["RAG_CONTEXTUALIZER_ENABLED"] = "false"

    def tearDown(self) -> None:
        if self.previous_contextualizer_enabled is None:
            os.environ.pop("RAG_CONTEXTUALIZER_ENABLED", None)
        else:
            os.environ["RAG_CONTEXTUALIZER_ENABLED"] = self.previous_contextualizer_enabled

    def test_strip_model_thinking_removes_qwen_reasoning_block(self) -> None:
        answer = """
<think>
Devo ragionare internamente prima di rispondere.
</think>

La risposta finale.
FONTI_USATE: [1]
"""

        self.assertEqual(
            strip_model_thinking(answer),
            "La risposta finale.\nFONTI_USATE: [1]",
        )

    def test_parse_used_source_indexes_after_thinking_cleanup(self) -> None:
        clean_answer, indexes = parse_used_source_indexes(
            strip_model_thinking(
                """
<think>Ragionamento interno.</think>
Risposta basata sul documento.
FONTI_USATE: [2, 3]
"""
            )
        )

        self.assertEqual(clean_answer, "Risposta basata sul documento.")
        self.assertEqual(indexes, [2, 3])

    def test_parse_model_answer_reads_json_mode(self) -> None:
        clean_answer, indexes = parse_model_answer(
            """
{
  "answer": "Risposta con fonte [1].",
  "used_sources": [1],
  "inline_citations": [1],
  "no_answer_reason": ""
}
"""
        )

        self.assertEqual(clean_answer, "Risposta con fonte [1].")
        self.assertEqual(indexes, [1])

    def test_parse_contextualizer_payload_rewrites_only_when_needed(self) -> None:
        rewritten = parse_contextualizer_payload(
            '{"needs_context": true, "standalone_question": "Quali sono i requisiti del programma Buddy?", "reason": "ellissi"}',
            "quali sono i requisiti?",
        )
        unchanged = parse_contextualizer_payload(
            '{"needs_context": false, "standalone_question": "Quali sono i requisiti del programma Buddy?", "reason": "nuovo topic"}',
            "Quali sono i requisiti di accesso a Ingegneria Informatica?",
        )

        self.assertEqual(rewritten, "Quali sono i requisiti del programma Buddy?")
        self.assertEqual(unchanged, "Quali sono i requisiti di accesso a Ingegneria Informatica?")

    def test_build_retrieval_question_prefers_llm_contextualizer_when_enabled(self) -> None:
        os.environ["RAG_CONTEXTUALIZER_ENABLED"] = "true"
        history = [
            {"role": "user", "content": "Parlami del programma Buddy dell'Ateneo"},
            {"role": "assistant", "content": "Il programma Buddy supporta gli studenti internazionali."},
        ]

        with patch.object(
            rag_chain,
            "contextualize_question_with_llm",
            return_value="Quali sono i requisiti del programma Buddy dell'Ateneo?",
        ):
            retrieval_question = build_retrieval_question("quali sono i requisiti?", history)

        self.assertEqual(
            retrieval_question,
            "Quali sono i requisiti del programma Buddy dell'Ateneo?",
        )

    def test_build_retrieval_question_expands_labrob_followup(self) -> None:
        history = [
            {
                "role": "user",
                "content": "Quali laboratori di ricerca ci sono nel DIEM?",
            },
            {
                "role": "assistant",
                "content": "Nel DIEM sono presenti diversi laboratori, tra cui LabROB.",
            },
        ]

        retrieval_question = build_retrieval_question(
            "il labROB che strumenti possiede?",
            history,
        )

        self.assertIn("Laboratorio di Robotica LabROB del DIEM", retrieval_question)

    def test_build_retrieval_question_keeps_autonomous_question(self) -> None:
        question = "Quali progetti finanziati sull'intelligenza artificiale svolge il DIEM dal 2024?"

        self.assertEqual(build_retrieval_question(question, []), question)

    def test_build_retrieval_question_contextualizes_erasmus_followup(self) -> None:
        history = [
            {
                "role": "user",
                "content": "Quali accordi Erasmus per studio sono disponibili al DIEM?",
            },
            {
                "role": "assistant",
                "content": "Gli accordi Erasmus del DIEM sono organizzati per mobilità per studio.",
            },
        ]

        retrieval_question = build_retrieval_question("quando scade?", history)

        self.assertIn("Contesto della domanda precedente", retrieval_question)
        self.assertIn("Quali accordi Erasmus per studio", retrieval_question)

    def test_build_retrieval_question_contextualizes_research_project_followup(self) -> None:
        history = [
            {
                "role": "user",
                "content": "Che progetti sull'intelligenza artificiale svolge il DIEM dal 2024?",
            },
            {
                "role": "assistant",
                "content": "Il DIEM svolge progetti di ricerca finanziati su intelligenza artificiale generativa.",
            },
        ]

        retrieval_question = build_retrieval_question("quanto dura?", history)

        self.assertIn("Contesto della domanda precedente", retrieval_question)
        self.assertIn("Che progetti sull'intelligenza artificiale", retrieval_question)

    def test_build_retrieval_question_contextualizes_course_followup(self) -> None:
        history = [
            {
                "role": "user",
                "content": "Quali corsi di laurea offre il DIEM?",
            },
            {
                "role": "assistant",
                "content": "Il DIEM presenta corsi di laurea e laurea magistrale nell'offerta formativa.",
            },
        ]

        retrieval_question = build_retrieval_question("come si accede?", history)

        self.assertIn("Contesto della domanda precedente", retrieval_question)
        self.assertIn("Quali corsi di laurea offre il DIEM", retrieval_question)

    def test_build_retrieval_question_contextualizes_teacher_followup_without_prof_prefix(self) -> None:
        history = [
            {
                "role": "user",
                "content": "Quali sono le recenti pubblicazioni di Fabio Postiglione?",
            },
            {
                "role": "assistant",
                "content": "Le pubblicazioni più recenti di Fabio POSTIGLIONE che ho trovato sono: ...",
            },
        ]

        retrieval_question = build_retrieval_question("dove si trova il suo studio?", history)

        self.assertIn("Contesto della domanda precedente", retrieval_question)
        self.assertIn("Fabio Postiglione", retrieval_question)

    def test_build_retrieval_question_does_not_use_teacher_history_for_final_exam_question(self) -> None:
        history = [
            {
                "role": "user",
                "content": "Quali sono le recenti pubblicazioni di Fabio Postiglione?",
            },
            {
                "role": "assistant",
                "content": "Le pubblicazioni più recenti di Fabio POSTIGLIONE che ho trovato sono: ...",
            },
        ]
        question = "Quali sono i criteri di ammissione alla seduta di laurea triennale dei percorsi offerti dal DIEM?"

        self.assertEqual(build_retrieval_question(question, history), question)

    def test_build_retrieval_question_uses_recent_turn_for_generic_followup_fallback(self) -> None:
        history = [
            {
                "role": "user",
                "content": "Parlami del programma Buddy dell'Ateneo",
            },
            {
                "role": "assistant",
                "content": "Il programma Buddy supporta gli studenti internazionali.",
            },
        ]

        retrieval_question = build_retrieval_question("come funziona?", history)

        self.assertIn("Contesto della domanda precedente", retrieval_question)
        self.assertIn("programma Buddy", retrieval_question)

    def test_build_retrieval_question_uses_recent_turn_for_requirements_fallback(self) -> None:
        history = [
            {
                "role": "user",
                "content": "Parlami del programma Buddy dell'Ateneo",
            },
            {
                "role": "assistant",
                "content": "Il programma Buddy supporta gli studenti internazionali.",
            },
        ]

        retrieval_question = build_retrieval_question("quali sono i requisiti?", history)

        self.assertIn("Contesto della domanda precedente", retrieval_question)
        self.assertIn("programma Buddy", retrieval_question)

    def test_build_retrieval_question_blocks_incompatible_recent_topic(self) -> None:
        os.environ["RAG_CONTEXTUALIZER_ENABLED"] = "true"
        history = [
            {
                "role": "user",
                "content": "Quali sono le recenti pubblicazioni di Fabio Postiglione?",
            },
            {
                "role": "assistant",
                "content": "Le pubblicazioni più recenti di Fabio POSTIGLIONE che ho trovato sono: ...",
            },
            {
                "role": "user",
                "content": "Quali sono i laboratori del DIEM?",
            },
            {
                "role": "assistant",
                "content": "I laboratori del DIEM includono varie strutture.",
            },
        ]
        question = "dove si trova il suo studio?"

        with patch.object(rag_chain, "contextualize_question_with_llm", return_value=question):
            self.assertEqual(build_retrieval_question(question, history), question)

    def test_build_retrieval_question_does_not_attach_incompatible_teacher_topic(self) -> None:
        os.environ["RAG_CONTEXTUALIZER_ENABLED"] = "true"
        history = [
            {
                "role": "user",
                "content": "Quali sono le recenti pubblicazioni di Fabio Postiglione?",
            },
            {
                "role": "assistant",
                "content": "Le pubblicazioni più recenti di Fabio POSTIGLIONE che ho trovato sono: ...",
            },
        ]
        question = "quali sono i requisiti?"

        with patch.object(rag_chain, "contextualize_question_with_llm", return_value=question):
            self.assertEqual(build_retrieval_question(question, history), question)

    def test_build_retrieval_question_does_not_guess_without_subject(self) -> None:
        question = "che strumenti possiede?"
        history = [
            {"role": "user", "content": "Ciao"},
            {"role": "assistant", "content": "Ciao, come posso aiutarti?"},
        ]

        self.assertEqual(build_retrieval_question(question, history), question)

    def test_build_retrieval_question_does_not_force_history_for_new_subject(self) -> None:
        history = [
            {
                "role": "user",
                "content": "Quali accordi Erasmus sono disponibili?",
            },
            {
                "role": "assistant",
                "content": "Gli accordi Erasmus sono organizzati per mobilità.",
            },
        ]
        question = "Quali sono i laboratori del DIEM?"

        self.assertEqual(build_retrieval_question(question, history), question)

    def test_build_direct_publications_answer_filters_wrong_teachers(self) -> None:
        from rag_chain import build_direct_publications_answer

        results = [
            RetrievalResult(
                chunk_id="foggia",
                text="Titolo pubblicazione: Wrong paper",
                metadata={
                    "entity_name": "Pasquale FOGGIA",
                    "title": "Pasquale FOGGIA | Pubblicazioni",
                    "publication_title": "Wrong paper",
                },
                source="hybrid",
                rank=1,
                score=1.0,
            ),
            RetrievalResult(
                chunk_id="greco",
                text="Titolo pubblicazione: Right paper",
                metadata={
                    "entity_name": "ANTONIO GRECO",
                    "title": "ANTONIO GRECO | Pubblicazioni",
                    "publication_title": "Right paper",
                },
                source="hybrid",
                rank=2,
                score=0.9,
            ),
        ]

        answer = build_direct_publications_answer(
            "quali sono le pubblicazioni recenti del prof Antonio Greco",
            results,
        )

        self.assertIsNotNone(answer)
        self.assertIn("Right paper", answer)
        self.assertNotIn("Wrong paper", answer)

    def test_build_retrieval_question_uses_history_for_fragment_followup(self) -> None:
        history = [
            {
                "role": "user",
                "content": "Quali accordi Erasmus sono disponibili?",
            },
            {
                "role": "assistant",
                "content": "Gli accordi Erasmus sono organizzati per mobilità.",
            },
        ]

        retrieval_question = build_retrieval_question("E per il prossimo semestre?", history)

        self.assertIn("Contesto della domanda precedente", retrieval_question)
        self.assertIn("Quali accordi Erasmus sono disponibili", retrieval_question)

    def test_clean_display_url_removes_query_and_fragment(self) -> None:
        self.assertEqual(
            clean_display_url("https://www.diem.unisa.it/dipartimento/strutture?id=2#x"),
            "https://www.diem.unisa.it/dipartimento/strutture",
        )

    def test_build_sources_deduplicates_by_clean_display_url(self) -> None:
        results = [
            RetrievalResult(
                chunk_id="chunk_1",
                text="Uno",
                metadata={
                    "title": "Dipartimento | Strutture",
                    "source_url": "https://www.diem.unisa.it/dipartimento/strutture?id=2",
                    "breadcrumb": ["Dipartimento", "Strutture"],
                    "chunk_id": "chunk_1",
                },
                source="test",
                rank=1,
                score=1.0,
            ),
            RetrievalResult(
                chunk_id="chunk_2",
                text="Due",
                metadata={
                    "title": "Dipartimento | Strutture",
                    "source_url": "https://www.diem.unisa.it/dipartimento/strutture?id=725#top",
                    "breadcrumb": ["Dipartimento", "Strutture"],
                    "chunk_id": "chunk_2",
                },
                source="test",
                rank=2,
                score=0.9,
            ),
        ]

        sources = build_sources(results)

        self.assertEqual(len(sources), 1)
        self.assertEqual(
            sources[0].url,
            "https://www.diem.unisa.it/dipartimento/strutture",
        )

    def test_build_sources_uses_filename_when_title_is_missing(self) -> None:
        results = [
            RetrievalResult(
                chunk_id="chunk_pdf",
                text="PDF",
                metadata={
                    "title": None,
                    "source_url": "https://corsi.unisa.it/uploads/rescue/499/1391/volantino-2026-tolc-diem-ofa.pdf",
                    "chunk_id": "chunk_pdf",
                },
                source="test",
                rank=1,
                score=1.0,
            )
        ]

        sources = build_sources(results)

        self.assertEqual(sources[0].title, "Volantino 2026 TOLC DIEM OFA")

    def test_build_sources_ignores_placeholder_title(self) -> None:
        results = [
            RetrievalResult(
                chunk_id="chunk_pdf",
                text="PDF",
                metadata={
                    "title": "N/D",
                    "source_url": "https://corsi.unisa.it/uploads/rescue/499/1391/info-ofa-diem-25-26.pdf",
                    "chunk_id": "chunk_pdf",
                },
                source="test",
                rank=1,
                score=1.0,
            )
        ]

        sources = build_sources(results)

        self.assertEqual(sources[0].title, "Info OFA DIEM 25 26")

    def test_build_sources_uses_breadcrumb_when_title_is_missing(self) -> None:
        results = [
            RetrievalResult(
                chunk_id="chunk_breadcrumb",
                text="Pagina",
                metadata={
                    "title": "",
                    "source_url": "https://www.diem.unisa.it/didattica/bandi",
                    "breadcrumb": ["DIEM", "Didattica", "Bandi"],
                    "chunk_id": "chunk_breadcrumb",
                },
                source="test",
                rank=1,
                score=1.0,
            )
        ]

        sources = build_sources(results)

        self.assertEqual(sources[0].title, "Bandi")

    def test_build_direct_publications_answer_lists_recent_summaries(self) -> None:
        results = [
            RetrievalResult(
                chunk_id="pub_2026",
                text="Titolo pubblicazione: Recent work",
                metadata={
                    "title": "Fabio POSTIGLIONE | Pubblicazioni",
                    "source_url": "https://docenti.unisa.it/003735/ricerca/pubblicazioni?anno=0",
                    "chunk_kind": "publication_summary",
                    "entity_name": "Fabio POSTIGLIONE",
                    "publication_id": "435220",
                    "publication_title": "Recent work on networks",
                    "publication_year": 2026,
                    "publication_type": "Articolo in rivista",
                    "publication_venue": "JOURNAL OF NETWORKS",
                    "publication_doi": "10.1234/example.2026",
                },
                source="test",
                rank=1,
                score=1.0,
            ),
            RetrievalResult(
                chunk_id="pub_2025",
                text="Titolo pubblicazione: Previous work",
                metadata={
                    "title": "Fabio POSTIGLIONE | Pubblicazioni",
                    "source_url": "https://docenti.unisa.it/003735/ricerca/pubblicazioni?anno=0",
                    "chunk_kind": "publication_summary",
                    "entity_name": "Fabio POSTIGLIONE",
                    "publication_id": "435219",
                    "publication_title": "Previous work on signals",
                    "publication_year": 2025,
                    "publication_type": "Contributo in Atti di convegno",
                },
                source="test",
                rank=2,
                score=0.9,
            ),
        ]

        answer = build_direct_publications_answer(
            "quali sono le recenti pubblicazioni del professore Fabio Postiglione?",
            results,
        )

        self.assertIsNotNone(answer)
        self.assertIn("Fabio POSTIGLIONE", answer or "")
        self.assertIn("2026: Recent work on networks", answer or "")
        self.assertIn("DOI: 10.1234/example.2026", answer or "")
        self.assertIn("2025: Previous work on signals", answer or "")

    def test_answer_question_as_text_accepts_conversation_history(self) -> None:
        history = [{"role": "user", "content": "Quali laboratori ci sono?"}]

        with patch("rag_chain.answer_question") as mocked_answer_question:
            mocked_answer_question.return_value = RagResponse(
                question="che strumenti possiede?",
                answer="Risposta",
                sources=[],
                retrieved_chunks=[],
            )

            answer = answer_question_as_text(
                "che strumenti possiede?",
                conversation_history=history,
            )

        self.assertEqual(answer, "Risposta")
        self.assertEqual(
            mocked_answer_question.call_args.kwargs["conversation_history"],
            history,
        )


if __name__ == "__main__":
    unittest.main()
