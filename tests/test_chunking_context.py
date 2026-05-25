from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chunk_metadata import flatten_chunk_metadata  # noqa: E402
from chunking import (  # noqa: E402
    build_context_header,
    classify_chunk_kind,
    chunk_document,
    infer_document_type,
    years_from_record,
)


class ChunkingContextTests(unittest.TestCase):
    def test_context_header_adds_html_content_and_section_titles(self) -> None:
        record = {
            "title": "Dipartimento | Strutture",
            "url": "https://www.diem.unisa.it/dipartimento/strutture?id=2",
            "breadcrumb": ["Dipartimento", "Strutture"],
        }

        header = build_context_header(
            record,
            document_content_title="Dipartimento | Robotica",
            section_heading="Strumentazione",
        )

        self.assertIn("Titolo: Dipartimento | Strutture", header)
        self.assertIn("Titolo contenuto: Dipartimento | Robotica", header)
        self.assertIn("Sezione: Strumentazione", header)

    def test_context_header_adds_pdf_parent_signal(self) -> None:
        record = {
            "title": None,
            "url": "https://corsi.unisa.it/uploads/rescue/__almalaurea/2024/0650107303300003.pdf",
            "discovered_from": "https://corsi.unisa.it/information-Engineering-for-digital-medicine/statistiche",
            "link_text": "Livello di soddisfazione dei laureandi (profilo dei laureati) / Condizione occupazionale dei laureati",
        }

        header = build_context_header(record, section_heading="Profilo dei laureati")

        self.assertIn("Fonte originaria: https://corsi.unisa.it/information-Engineering-for-digital-medicine/statistiche", header)
        self.assertIn("Testo link sorgente: Livello di soddisfazione dei laureandi", header)
        self.assertIn("Tipo documento: almalaurea", header)
        self.assertIn("Anni documento: 2024", header)
        self.assertEqual(infer_document_type(record), "almalaurea")
        self.assertEqual(years_from_record(record), [2024])

    def test_classify_course_catalogue_syllabus_chunk(self) -> None:
        record = {
            "url": "https://unisa.coursecatalogue.cineca.it/corsi/2025/500854/insegnamenti/2025/521541/2025/10000",
            "title": "MACHINE LEARNING",
        }

        chunk_kind = classify_chunk_kind(
            "## Contenuti\nUNITÀ DIDATTICA 1: concetti fondamentali.",
            "Contenuti",
            record,
        )

        self.assertEqual(chunk_kind, "course_syllabus")

    def test_classify_study_plan_pdf_chunk(self) -> None:
        record = {
            "source": "pdf",
            "url": "https://corsi.unisa.it/uploads/rescue/__piano-studi-cds/2025/IE127.pdf",
            "title": "Piano degli studi",
        }

        chunk_kind = classify_chunk_kind(
            "CORSO DI LAUREA IN INGEGNERIA INFORMATICA L-8\n1° ANNO\nAnalisi Matematica I",
            "1° ANNO - 2025/26",
            record,
        )

        self.assertEqual(chunk_kind, "study_plan")

    def test_course_catalogue_study_plan_children_inherit_metadata(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            markdown_path = Path(temp_dir) / "catalogue.md"
            markdown_path.write_text(
                "# INGEGNERIA INFORMATICA\n\n"
                "## Piano di studi\n\n"
                "## Percorso: SOFTWARE - COORTE 2025\n\n"
                "### 1 anno\n\n"
                "| Insegnamento | CFU |\n"
                "| --- | --- |\n"
                "| Analisi Matematica 1 | 9 |\n\n"
                "### 2 anno\n\n"
                "| Insegnamento | CFU |\n"
                "| --- | --- |\n"
                "| Algoritmi e Strutture Dati | 9 |\n\n"
                "### 3 anno\n\n"
                "| Insegnamento | CFU |\n"
                "| --- | --- |\n"
                "| Ingegneria del Software | 9 |\n",
                encoding="utf-8",
            )

            record = {
                "hash": "course123",
                "content_hash": "content_course",
                "source": "course_catalogue",
                "url": "https://unisa.coursecatalogue.cineca.it/corsi/2025/500853?annoOrdinamento=2025",
                "title": "INGEGNERIA INFORMATICA",
                "breadcrumb": ["Offerta formativa", "INGEGNERIA INFORMATICA"],
                "index_markdown_path": str(markdown_path.relative_to(ROOT)),
                "last_crawled": "2026-05-23T00:00:00+00:00",
                "clean_status": "ok",
            }

            chunks = chunk_document(record)

        study_chunks = [
            chunk
            for chunk in chunks
            if chunk["retrieval_metadata"].get("chunk_kind") == "study_plan"
        ]

        self.assertGreaterEqual(len(study_chunks), 3)
        years = {
            chunk["retrieval_metadata"].get("course_year")
            for chunk in study_chunks
        }
        self.assertTrue({1, 2, 3}.issubset(years))
        for chunk in study_chunks:
            metadata = chunk["retrieval_metadata"]
            self.assertEqual(metadata["course_name"], "INGEGNERIA INFORMATICA")
            self.assertEqual(metadata["curriculum"], "SOFTWARE")
            self.assertEqual(metadata["cohort"], 2025)
            self.assertIn("locator_text", chunk)
            self.assertNotIn("[CONTESTO DOCUMENTO]", chunk["body_text"])

    def test_tutorato_page_is_not_classified_as_study_plan(self) -> None:
        record = {
            "source": "html",
            "url": "https://corsi.unisa.it/ingegneria-informatica/attivita-e-servizi/tutorato",
            "title": "Ingegneria Informatica | Orientamento e Tutorato in Itinere",
            "breadcrumb": ["Ingegneria Informatica", "Attività e Servizi", "Orientamento e Tutorato in Itinere"],
        }

        chunk_kind = classify_chunk_kind(
            "assistenza alla elaborazione del piano di studio e alla scelta della tesi",
            "Ingegneria Informatica Orientamento e Tutorato in Itinere",
            record,
        )

        self.assertNotEqual(chunk_kind, "study_plan")

    def test_chunk_document_separates_metadata_levels(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            markdown_path = Path(temp_dir) / "lab.md"
            markdown_path.write_text(
                "# Dipartimento | Robotica\n\n"
                "### Strumentazione\n\n"
                "2 x robot antropomorfi Comau Smart-Six montati su slitta attuata.",
                encoding="utf-8",
            )

            record = {
                "hash": "doc123",
                "content_hash": "content123",
                "source": "html",
                "url": "https://www.diem.unisa.it/dipartimento/strutture?id=2",
                "title": "Dipartimento | Strutture",
                "breadcrumb": ["Dipartimento", "Strutture"],
                "index_markdown_path": str(markdown_path.relative_to(ROOT)),
                "last_crawled": "2026-05-23T00:00:00+00:00",
                "clean_status": "ok",
            }

            chunk = chunk_document(record)[0]

        self.assertIn("retrieval_metadata", chunk)
        self.assertIn("provenance", chunk)
        self.assertIn("debug", chunk)
        self.assertNotIn("source_url", chunk)
        self.assertEqual(
            chunk["retrieval_metadata"]["source_url"],
            "https://www.diem.unisa.it/dipartimento/strutture?id=2",
        )
        self.assertEqual(chunk["retrieval_metadata"]["section_heading"], "Dipartimento | Robotica")
        self.assertEqual(chunk["provenance"]["last_crawled"], "2026-05-23T00:00:00+00:00")

        flattened = flatten_chunk_metadata(chunk)
        self.assertEqual(flattened["source_url"], "https://www.diem.unisa.it/dipartimento/strutture?id=2")
        self.assertEqual(flattened["clean_status"], "ok")

    def test_chunk_document_preserves_short_office_hours_section(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            markdown_path = Path(temp_dir) / "antonio_greco.md"
            markdown_path.write_text(
                "# ANTONIO GRECO | Home\n\n"
                "## ANTONIO GRECO Home\n\n"
                "| Professore Associato |\n"
                "| --- |\n"
                "| Dipartimento di Ingegneria dell'Informazione ed Elettrica e Matematica applicata/DIEM |\n"
                "| 3003 |\n\n"
                "| [Campus di Fisciano, Edificio E, Piano Terzo, Stanza 005](https://docenti.unisa.it#026557) |\n"
                "| [Campus di Fisciano, Edificio E, Piano Primo, Stanza 036](https://docenti.unisa.it#026557_lab) |\n\n"
                "### Orario di Ricevimento\n\n"
                "| **Martedì** | **9:00 - 11:00** | Ufficio 365 - Edificio E |\n"
                "| --- | --- | --- |\n",
                encoding="utf-8",
            )

            record = {
                "hash": "greco123",
                "content_hash": "content_greco",
                "source": "html",
                "url": "https://docenti.unisa.it/026557/home",
                "title": "ANTONIO GRECO | Home",
                "breadcrumb": ["Docenti", "GRECO Antonio", "Home"],
                "index_markdown_path": str(markdown_path.relative_to(ROOT)),
                "last_crawled": "2026-05-23T00:00:00+00:00",
                "clean_status": "ok",
            }

            chunks = chunk_document(record)

        combined_text = "\n".join(chunk["text"] for chunk in chunks)
        chunk_kinds = {
            chunk["retrieval_metadata"].get("chunk_kind")
            for chunk in chunks
        }

        self.assertIn("Orario di Ricevimento", combined_text)
        self.assertIn("Martedì", combined_text)
        self.assertIn("9:00 - 11:00", combined_text)
        self.assertIn("Ufficio 365 - Edificio E", combined_text)
        self.assertIn("office_hours", chunk_kinds)

    def test_chunk_document_creates_publication_summary_chunks(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            markdown_path = Path(temp_dir) / "postiglione_pubblicazioni.md"
            markdown_path.write_text(
                "# Fabio POSTIGLIONE | Pubblicazioni\n\n"
                "## Fabio POSTIGLIONE Pubblicazioni\n\n"
                "####  435220[Recent work on networks](https://docenti.unisa.it#pubblicazione-collapse-435220)\n"
                "| 2026 |\n"
                "| --- |\n"
                "| Articolo in rivista |\n"
                "| **Recent work on networks** JOURNAL OF NETWORKS. Vol. 12. Pag.1-10 ISSN:1234-5678. |\n"
                "| Rossi, Mario; Postiglione, Fabio |\n"
                "| Digital Object Identifier (DOI): [10.1234/example.2026](http://dx.doi.org/10.1234/example.2026) |\n"
                "| [Visualizza sul Database dei Prodotti (IRIS)](http://hdl.handle.net/11386/9999999) |\n",
                encoding="utf-8",
            )

            record = {
                "hash": "postiglione123",
                "content_hash": "content_postiglione",
                "source": "html",
                "url": "https://docenti.unisa.it/003735/ricerca/pubblicazioni?anno=0",
                "title": "Fabio POSTIGLIONE | Pubblicazioni",
                "breadcrumb": ["Docenti", "POSTIGLIONE Fabio", "Ricerca", "Pubblicazioni"],
                "index_markdown_path": str(markdown_path.relative_to(ROOT)),
                "last_crawled": "2026-05-23T00:00:00+00:00",
                "clean_status": "ok",
            }

            chunks = chunk_document(record)

        publication_chunks = [
            chunk for chunk in chunks
            if chunk["retrieval_metadata"].get("chunk_kind") == "publication_summary"
        ]

        self.assertEqual(len(publication_chunks), 1)
        metadata = publication_chunks[0]["retrieval_metadata"]
        self.assertEqual(metadata["entity_type"], "teacher")
        self.assertEqual(metadata["teacher_id"], "003735")
        self.assertEqual(metadata["publication_id"], "435220")
        self.assertEqual(metadata["publication_year"], 2026)
        self.assertEqual(metadata["publication_title"], "Recent work on networks")
        self.assertIn("DOI: 10.1234/example.2026", publication_chunks[0]["text"])

    def test_publication_summary_compacts_very_long_author_lists(self) -> None:
        authors = "; ".join(f"Autore {index}" for index in range(1, 180))

        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            markdown_path = Path(temp_dir) / "guida_pubblicazioni.md"
            markdown_path.write_text(
                "# MICHELE GUIDA | Pubblicazioni\n\n"
                "####  123456[Large collaboration paper](https://docenti.unisa.it#pubblicazione-collapse-123456)\n"
                "| 2025 |\n"
                "| --- |\n"
                "| Articolo in rivista |\n"
                "| **Large collaboration paper** JOURNAL OF LARGE SYSTEMS. Vol. 1. Pag.1-10. |\n"
                f"| {authors} |\n"
                "| Digital Object Identifier (DOI): [10.1234/large.2025](http://dx.doi.org/10.1234/large.2025) |\n",
                encoding="utf-8",
            )

            record = {
                "hash": "guida123",
                "content_hash": "content_guida",
                "source": "html",
                "url": "https://docenti.unisa.it/001295/ricerca/pubblicazioni?anno=0",
                "title": "MICHELE GUIDA | Pubblicazioni",
                "breadcrumb": ["Docenti", "GUIDA Michele", "Ricerca", "Pubblicazioni"],
                "index_markdown_path": str(markdown_path.relative_to(ROOT)),
                "last_crawled": "2026-05-23T00:00:00+00:00",
                "clean_status": "ok",
            }

            chunks = chunk_document(record)

        publication_chunks = [
            chunk for chunk in chunks
            if chunk["retrieval_metadata"].get("chunk_kind") == "publication_summary"
        ]

        self.assertEqual(len(publication_chunks), 1)
        self.assertLess(publication_chunks[0]["chars"], 2500)
        self.assertIn("Autori: Autore 1", publication_chunks[0]["text"])
        self.assertIn("[...]", publication_chunks[0]["text"])
        self.assertIn("DOI: 10.1234/large.2025", publication_chunks[0]["text"])

    def test_chunk_document_splits_long_lab_equipment_summaries(self) -> None:
        equipment_lines = [
            f"- Strumentazione avanzata numero {index}: sistema di misura, sensori e unità di controllo dedicate."
            for index in range(1, 80)
        ]

        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            markdown_path = Path(temp_dir) / "mivia.md"
            markdown_path.write_text(
                "# Dipartimento | Macchine Intelligenti\n\n"
                "### Strumentazione del MIVIA\n\n"
                + "\n".join(equipment_lines),
                encoding="utf-8",
            )

            record = {
                "hash": "mivia123",
                "content_hash": "content_mivia",
                "source": "html",
                "url": "https://www.diem.unisa.it/dipartimento/strutture?id=15",
                "title": "Dipartimento | Strutture",
                "breadcrumb": ["Dipartimento", "Strutture"],
                "index_markdown_path": str(markdown_path.relative_to(ROOT)),
                "last_crawled": "2026-05-23T00:00:00+00:00",
                "clean_status": "ok",
            }

            chunks = chunk_document(record)

        lab_chunks = [
            chunk for chunk in chunks
            if chunk["retrieval_metadata"].get("chunk_kind") == "lab_equipment"
        ]
        synthetic_lab_chunks = [
            chunk for chunk in lab_chunks
            if "[SCHEDA STRUMENTAZIONE LABORATORIO]" in chunk["text"]
        ]

        self.assertGreater(len(synthetic_lab_chunks), 1)
        self.assertLess(max(chunk["chars"] for chunk in lab_chunks), 2600)
        self.assertTrue(
            all(chunk["retrieval_metadata"].get("chunk_part_count") for chunk in synthetic_lab_chunks)
        )
        self.assertIn("strumentazione avanzata numero 79", "\n".join(chunk["text"].lower() for chunk in synthetic_lab_chunks))

    def test_chunk_document_does_not_mark_course_content_as_lab_equipment(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            markdown_path = Path(temp_dir) / "elettrotecnica.md"
            markdown_path.write_text(
                "# ELETTROTECNICA\n\n"
                "### Contenuti\n\n"
                "Il corso introduce l'uso della strumentazione di laboratorio "
                "per misure elettriche e attrezzature di base.",
                encoding="utf-8",
            )

            record = {
                "hash": "course123",
                "content_hash": "content_course",
                "source": "html",
                "url": "https://unisa.coursecatalogue.cineca.it/corsi/2024/500189/insegnamenti/2025/507092/2022/10002",
                "title": "ELETTROTECNICA",
                "breadcrumb": ["CourseCatalogue", "Ingegneria Informatica", "Insegnamenti"],
                "index_markdown_path": str(markdown_path.relative_to(ROOT)),
                "last_crawled": "2026-05-23T00:00:00+00:00",
                "clean_status": "ok",
            }

            chunks = chunk_document(record)

        chunk_kinds = {
            chunk["retrieval_metadata"].get("chunk_kind")
            for chunk in chunks
        }
        self.assertNotIn("lab_equipment", chunk_kinds)

    def test_flatten_chunk_metadata_supports_legacy_chunks(self) -> None:
        flattened = flatten_chunk_metadata(
            {
                "chunk_id": "legacy",
                "source_url": "https://www.diem.unisa.it/legacy",
                "title": "Legacy",
                "clean_status": "ok",
            }
        )

        self.assertEqual(flattened["source_url"], "https://www.diem.unisa.it/legacy")
        self.assertEqual(flattened["title"], "Legacy")
        self.assertEqual(flattened["clean_status"], "ok")


if __name__ == "__main__":
    unittest.main()
