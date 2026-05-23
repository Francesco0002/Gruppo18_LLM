from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag_chain import (  # noqa: E402
    RagResponse,
    answer_question_as_text,
    build_retrieval_question,
    build_sources,
    clean_display_url,
    parse_model_answer,
    parse_used_source_indexes,
    strip_model_thinking,
)
from retrieval import RetrievalResult  # noqa: E402


class RagChainTests(unittest.TestCase):
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

        self.assertIn("Erasmus e mobilità internazionale del DIEM", retrieval_question)

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

        self.assertIn("progetti di ricerca e progetti finanziati del DIEM", retrieval_question)

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

        self.assertIn("offerta formativa e corsi di laurea del DIEM", retrieval_question)

    def test_build_retrieval_question_does_not_guess_without_subject(self) -> None:
        question = "che strumenti possiede?"
        history = [
            {"role": "user", "content": "Ciao"},
            {"role": "assistant", "content": "Ciao, come posso aiutarti?"},
        ]

        self.assertEqual(build_retrieval_question(question, history), question)

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
