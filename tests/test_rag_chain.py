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
    GroqPayloadTooLargeError,
    GroqReasoningOnlyError,
    RagResponse,
    adaptive_context_budgets,
    answer_question_as_text,
    build_groq_request_kwargs,
    build_prompt,
    build_relevance_compacted_context,
    build_retrieval_question,
    build_sources,
    call_groq_with_adaptive_context,
    build_direct_publications_answer,
    is_payload_too_large_error,
    parse_contextualizer_payload,
    clean_display_url,
    parse_model_answer,
    parse_used_source_indexes,
    should_retry_groq_error,
    sort_retrieved_chunks_for_generation,
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

    def test_strip_model_thinking_removes_truncated_reasoning_block(self) -> None:
        answer = "<think> Okay, let's see. The user is asking..."

        self.assertEqual(strip_model_thinking(answer), "")

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

    def test_payload_too_large_errors_are_detected_without_retrying_blindly(self) -> None:
        self.assertTrue(
            is_payload_too_large_error(
                RuntimeError('HTTP Request failed: "413 Payload Too Large"')
            )
        )

    def test_bad_request_and_rate_limit_are_not_retried_by_tenacity(self) -> None:
        self.assertFalse(should_retry_groq_error(RuntimeError("Error code: 400")))
        self.assertFalse(should_retry_groq_error(RuntimeError("Error code: 429")))

    def test_groq_request_uses_hidden_reasoning_and_completion_token_cap(self) -> None:
        request_kwargs = build_groq_request_kwargs(
            prompt="Rispondi in JSON",
            model="qwen/qwen3-32b",
        )

        self.assertEqual(request_kwargs["reasoning_format"], "hidden")
        self.assertNotIn("reasoning_effort", request_kwargs)
        self.assertIn("max_completion_tokens", request_kwargs)
        self.assertNotIn("max_tokens", request_kwargs)

    def test_adaptive_context_budgets_reduce_only_after_payload_errors(self) -> None:
        budgets = adaptive_context_budgets(12000)

        self.assertEqual(budgets[0], 12000)
        self.assertGreater(budgets[0], budgets[-1])
        self.assertGreaterEqual(budgets[-1], rag_chain.RAG_MIN_CONTEXT_CHARS_ON_PAYLOAD_RETRY)

    def test_prompt_requires_complete_official_lists_from_context(self) -> None:
        prompt = build_prompt(
            question="Quali sono i laboratori disponibili nel DIEM?",
            context="[DOCUMENTO 1]\nTabella ufficiale con molti laboratori.",
        )

        self.assertIn("riporta tutte le voci pertinenti presenti nel contesto", prompt)
        self.assertIn("usa la tabella indice per decidere quali voci elencare", prompt)

    def test_call_groq_with_adaptive_context_retries_with_smaller_context(self) -> None:
        results = [
            RetrievalResult(
                chunk_id="large",
                text="x" * 5000,
                metadata={"title": "Documento"},
                source="test",
                rank=1,
                score=1.0,
            )
        ]

        with patch.object(
            rag_chain,
            "call_groq",
            side_effect=[
                GroqPayloadTooLargeError("payload too large"),
                (
                    '{"answer": "ok [1]", "used_sources": [1], '
                    '"inline_citations": [1], "no_answer_reason": ""}'
                ),
            ],
        ):
            answer, trace = call_groq_with_adaptive_context(
                question="Domanda",
                retrieved_chunks=results,
                conversation_context="",
                max_context_chars=4000,
                model="test-model",
            )

        self.assertIn("ok", answer)
        self.assertEqual(trace["attempts"][0]["status"], "payload_too_large")
        self.assertEqual(trace["attempts"][1]["status"], "ok")
        self.assertEqual(trace["attempts"][1]["context_mode"], "full")
        self.assertLess(
            trace["attempts"][1]["context_budget_chars"],
            trace["attempts"][0]["context_budget_chars"],
        )

    def test_call_groq_with_adaptive_context_uses_relevance_compaction_for_formula_queries(self) -> None:
        results = [
            RetrievalResult(
                chunk_id="noise",
                text="""
[DOCUMENTO 1]
Titolo: Scheda insegnamento
[CONTENUTO]
programma del corso e obiettivi formativi.
""",
                metadata={"title": "Scheda insegnamento", "source_url": "https://example.test/course"},
                source="test",
                rank=1,
                score=1.0,
            ),
            RetrievalResult(
                chunk_id="formula",
                text="""
[DOCUMENTO 2]
Titolo: Regolamento esame finale lauree magistrali
[CONTENUTO]
Nella valutazione conclusiva si tiene conto delle valutazioni negli esami di profitto
attraverso il calcolo del Voto_Base = (4,1*Media_pesata_sui_crediti - 8,8)/110.
""",
                metadata={
                    "title": "Regolamento esame finale lauree magistrali",
                    "source_url": "https://corsi.unisa.it/regolamento.pdf",
                    "document_type": "regolamento",
                    "chunk_kind": "official_document",
                },
                source="test",
                rank=2,
                score=0.9,
            ),
        ]

        with patch.object(
            rag_chain,
            "call_groq",
            side_effect=[
                GroqPayloadTooLargeError("payload too large"),
                '{"answer": "ok [1]", "used_sources": [1], "inline_citations": [1]}',
            ],
        ):
            answer, trace = call_groq_with_adaptive_context(
                question="con media esami 28.2 quale sara il voto di laurea finale?",
                retrieved_chunks=results,
                conversation_context="",
                max_context_chars=4000,
                model="test-model",
            )

        self.assertIn("ok", answer)
        self.assertEqual(trace["attempts"][0]["status"], "payload_too_large")
        self.assertEqual(trace["attempts"][1]["status"], "ok")
        self.assertEqual(trace["attempts"][1]["context_mode"], "relevance_compacted")

    def test_relevance_compacted_context_keeps_formula_evidence_after_retry(self) -> None:
        results = [
            RetrievalResult(
                chunk_id="noise",
                text=(
                    "[CONTESTO DOCUMENTO]\nTitolo: Scheda insegnamento\nFonte: test\n\n"
                    "[CONTENUTO]\n" + ("programma del corso e obiettivi formativi. " * 120)
                ),
                metadata={"title": "Scheda insegnamento", "source_url": "https://example.test/course"},
                source="test",
                rank=1,
                score=1.0,
            ),
            RetrievalResult(
                chunk_id="formula",
                text=(
                    "[CONTESTO DOCUMENTO]\n"
                    "Titolo: Regolamento esame finale lauree magistrali\n"
                    "Fonte: https://corsi.unisa.it/regolamento.pdf\n"
                    "Tipo documento: regolamento\n\n"
                    "[CONTENUTO]\n"
                    "Nella valutazione conclusiva si tiene conto delle valutazioni negli esami "
                    "di profitto attraverso il calcolo del Voto_Base = "
                    "(4,1*Media_pesata_sui crediti - 8,8)/110, approssimandolo "
                    "all'intero piu vicino.\n"
                    "|Fattore|Punti aggiuntivi|\n"
                    "|Svolgimento di attivita formative all'estero|2 centodecimi|"
                ),
                metadata={
                    "title": "Regolamento esame finale lauree magistrali",
                    "source_url": "https://corsi.unisa.it/regolamento.pdf",
                    "document_type": "regolamento",
                    "chunk_kind": "official_document",
                },
                source="test",
                rank=2,
                score=0.9,
            ),
        ]

        context = build_relevance_compacted_context(
            question="con media esami 28.2 quale sara il voto di laurea finale?",
            results=results,
            max_context_chars=1200,
        )

        self.assertLessEqual(len(context), 1200)
        self.assertIn("[DOCUMENTO 2]", context)
        self.assertIn("Voto_Base", context)
        self.assertIn("Media_pesata", context)
        self.assertNotIn("obiettivi formativi. programma del corso", context)

    def test_call_groq_with_adaptive_context_retries_reasoning_only_output(self) -> None:
        results = [
            RetrievalResult(
                chunk_id="doc",
                text="testo",
                metadata={"title": "Documento"},
                source="test",
                rank=1,
                score=1.0,
            )
        ]

        with patch.object(
            rag_chain,
            "call_groq",
            side_effect=[
                GroqReasoningOnlyError("reasoning only"),
                '{"answer": "ok [1]", "used_sources": [1]}',
            ],
        ):
            answer, trace = call_groq_with_adaptive_context(
                question="Domanda",
                retrieved_chunks=results,
                conversation_context="",
                max_context_chars=4000,
                model="test-model",
            )

        self.assertIn("ok", answer)
        self.assertEqual(trace["attempts"][0]["status"], "reasoning_only")
        self.assertEqual(trace["attempts"][1]["status"], "ok")

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

    def test_sort_retrieved_chunks_for_generation_orders_publications_for_llm(self) -> None:
        results = [
            RetrievalResult(
                chunk_id="pub_2024",
                text="Titolo pubblicazione: Older work",
                metadata={
                    "title": "Fabio POSTIGLIONE | Pubblicazioni",
                    "source_url": "https://docenti.unisa.it/003735/ricerca/pubblicazioni?anno=0",
                    "chunk_kind": "publication_summary",
                    "entity_name": "Fabio POSTIGLIONE",
                    "publication_id": "435218",
                    "publication_title": "Older work",
                    "publication_year": 2024,
                    "publication_order": 3,
                },
                source="test",
                rank=1,
                score=1.0,
            ),
            RetrievalResult(
                chunk_id="pub_2026",
                text="Titolo pubblicazione: Newer work",
                metadata={
                    "title": "Fabio POSTIGLIONE | Pubblicazioni",
                    "source_url": "https://docenti.unisa.it/003735/ricerca/pubblicazioni?anno=0",
                    "chunk_kind": "publication_summary",
                    "entity_name": "Fabio POSTIGLIONE",
                    "publication_id": "435220",
                    "publication_title": "Newer work",
                    "publication_year": 2026,
                    "publication_order": 1,
                },
                source="test",
                rank=2,
                score=0.9,
            ),
        ]

        ordered = sort_retrieved_chunks_for_generation(
            "quali sono le recenti pubblicazioni del professore Fabio Postiglione?",
            results,
        )

        self.assertEqual([result.chunk_id for result in ordered], ["pub_2026", "pub_2024"])

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
