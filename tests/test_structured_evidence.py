from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from query_planner import build_retrieval_query, plan_query  # noqa: E402
from structured_evidence import structured_retrieve  # noqa: E402


class StructuredEvidenceTests(unittest.TestCase):
    def test_final_grade_query_is_planned_as_official_document_search(self) -> None:
        query = "con una media voto esami di 28.2 alla magistrale in ingegneria informatica, quale sara il voto di laurea?"
        plan = plan_query(query)
        retrieval_query = build_retrieval_query(query, plan)

        self.assertEqual(plan.task_type, "official_docs")
        self.assertEqual(plan.filters.get("course_level"), "magistrale")
        self.assertIn("regolamento esame finale", retrieval_query)
        self.assertIn("voto base media pesata crediti", retrieval_query)

    def test_final_grade_query_prefers_current_regulation_evidence(self) -> None:
        plan = plan_query(
            "con una media voto esami di 28.2 alla magistrale in ingegneria informatica, quale sara il voto di laurea?"
        )
        chunks = (
            {
                "chunk_id": "syllabus",
                "text": "Scheda insegnamento con modalita esame, voto finale e programma del corso.",
                "retrieval_metadata": {
                    "source_url": "https://unisa.coursecatalogue.cineca.it/corsi/2025/500854",
                    "source_family": "course_catalogue",
                    "chunk_kind": "course_syllabus",
                    "topic_family": "didattica",
                    "title": "Scheda insegnamento",
                },
            },
            {
                "chunk_id": "undergraduate_regulation",
                "text": (
                    "REGOLAMENTO DIDATTICO DEL CORSO DI LAUREA IN INGEGNERIA INFORMATICA "
                    "(classe L-8). Il voto di laurea e calcolato con una formula."
                ),
                "retrieval_metadata": {
                    "source_url": "https://corsi.unisa.it/uploads/rescue/__regolamenti-cds/2025/IE127.pdf",
                    "source_family": "course",
                    "chunk_kind": "official_document",
                    "document_type": "regolamento",
                    "title": "Regolamento didattico laurea triennale",
                    "year": 2025,
                    "document_years": [2025],
                },
            },
            {
                "chunk_id": "old_regulation",
                "text": (
                    "La valutazione conclusiva usa il calcolo del Voto_Base = "
                    "(4,1*Media_pesata_sui crediti - 7,8)/110."
                ),
                "retrieval_metadata": {
                    "source_url": "https://corsi.unisa.it/uploads/rescue/__regolamenti-cds/2014/06127.pdf",
                    "source_family": "course",
                    "chunk_kind": "official_document",
                    "document_type": "regolamento",
                    "title": "Regolamento didattico",
                    "year": 2014,
                    "document_years": [2014],
                },
            },
            {
                "chunk_id": "current_regulation",
                "text": (
                    "Regolamento per l'attribuzione del voto dell'Esame Finale. "
                    "Nella valutazione conclusiva si tiene conto delle valutazioni "
                    "negli esami di profitto attraverso il calcolo del Voto_Base = "
                    "(4,1*Media_pesata_sui crediti - 8,8)/110."
                ),
                "retrieval_metadata": {
                    "source_url": "https://corsi.unisa.it/uploads/rescue/499/6513/regolamento-esame-finale-lauree-magistrali-2023-24.pdf",
                    "source_family": "course",
                    "chunk_kind": "official_document",
                    "document_type": "regolamento",
                    "title": "Regolamento esame finale lauree magistrali 2023/2024",
                    "year": 2023,
                    "document_years": [2023],
                },
            },
        )

        evidence = structured_retrieve(
            "con una media voto esami di 28.2 alla magistrale in ingegneria informatica, quale sara il voto di laurea?",
            plan,
            chunks,
            limit=5,
        )

        self.assertEqual(evidence[0].chunk["chunk_id"], "current_regulation")
        self.assertNotIn("syllabus", [item.chunk["chunk_id"] for item in evidence])

    def test_lab_catalog_query_prefers_department_lab_sources(self) -> None:
        plan = plan_query("che laboratori possiede il diem?")
        chunks = (
            {
                "chunk_id": "course_stats",
                "text": "Valutazione attrezzature per attività didattiche e laboratori.",
                "retrieval_metadata": {
                    "source_url": "https://corsi.unisa.it/uploads/rescue/__almalaurea/2026/report.pdf",
                    "source_family": "course",
                    "topic_family": "didattica",
                    "chunk_kind": "lab_equipment",
                    "title": "Statistiche corso",
                },
            },
            {
                "chunk_id": "official_structures",
                "text": "#### Laboratori\nRobotica LabROB\nTecnologie Elettriche per la Digital Energy TE4DE",
                "retrieval_metadata": {
                    "source_url": "https://www.diem.unisa.it/dipartimento/strutture",
                    "source_family": "diem",
                    "topic_family": "laboratori",
                    "chunk_kind": "text",
                    "title": "Dipartimento | Strutture",
                },
            },
            {
                "chunk_id": "lab_detail",
                "text": "Mission del laboratorio Robotica LabROB con molte attività di dettaglio.",
                "retrieval_metadata": {
                    "source_url": "https://www.diem.unisa.it/dipartimento/strutture?id=2",
                    "source_family": "diem",
                    "topic_family": "laboratori",
                    "chunk_kind": "text",
                    "section_heading": "Mission",
                    "title": "Dipartimento | Robotica",
                },
            },
        )

        evidence = structured_retrieve(
            "che laboratori possiede il diem?",
            plan,
            chunks,
            limit=5,
        )

        self.assertEqual([item.chunk["chunk_id"] for item in evidence], ["official_structures"])

    def test_teacher_publications_are_scored_by_recency(self) -> None:
        plan = plan_query("quali sono le recenti pubblicazioni del professore Fabio Postiglione?")
        chunks = (
            {
                "chunk_id": "old_pub",
                "text": "Docente: Fabio POSTIGLIONE\nTitolo pubblicazione: Older\nAnno: 2013",
                "retrieval_metadata": {
                    "source_url": "https://docenti.unisa.it/003735/ricerca/pubblicazioni?anno=0",
                    "source_family": "teacher_profile",
                    "chunk_kind": "publication_summary",
                    "entity_name": "Fabio POSTIGLIONE",
                    "publication_title": "Older",
                    "publication_year": 2013,
                },
            },
            {
                "chunk_id": "new_pub",
                "text": "Docente: Fabio POSTIGLIONE\nTitolo pubblicazione: Newer\nAnno: 2025",
                "retrieval_metadata": {
                    "source_url": "https://docenti.unisa.it/003735/ricerca/pubblicazioni?anno=0",
                    "source_family": "teacher_profile",
                    "chunk_kind": "publication_summary",
                    "entity_name": "Fabio POSTIGLIONE",
                    "publication_title": "Newer",
                    "publication_year": 2025,
                },
            },
        )

        evidence = structured_retrieve(
            "quali sono le recenti pubblicazioni del professore Fabio Postiglione?",
            plan,
            chunks,
            limit=5,
        )

        self.assertEqual([item.chunk["chunk_id"] for item in evidence], ["new_pub", "old_pub"])


if __name__ == "__main__":
    unittest.main()
