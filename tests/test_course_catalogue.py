from __future__ import annotations

import sys
import unittest
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from course_catalogue import (  # noqa: E402
    collect_course_teachings,
    render_course_markdown,
    render_teaching_markdown,
    settings_from_config,
    teaching_api_url,
    teaching_url,
)


def sample_config() -> dict:
    return {
        "crawler": {"timeout": 20},
        "course_catalogue": {
            "enabled": True,
            "domain": "unisa.coursecatalogue.cineca.it",
            "course_entries": [
                {"year": "2024", "course_id": "500639"},
                {"year": "2025", "course_id": "500838"},
            ],
            "fetch_teaching_details": True,
        },
        "scope": {},
    }


def sample_course() -> dict:
    return {
        "aa": "2025",
        "cod": "500838",
        "cdsCod": "IE128",
        "codicione": "0650106200800003",
        "des_it": "INGEGNERIA DELL'INFORMAZIONE PER LA MEDICINA DIGITALE",
        "tipo_corso_des_it": "CORSO DI LAUREA",
        "classe_it": "Ingegneria dell'informazione",
        "crediti_it": "180",
        "durata_it": "3 anni",
        "ruoli_it": [
            {
                "caricaNome": "Ada",
                "caricaCognome": "Lovelace",
                "carica": "Presidente del Consiglio didattico",
                "caricaMatricola": "000001",
            }
        ],
        "programma_testi_obiettivi_it": [
            {
                "carattTitolo": "Obiettivi formativi specifici.",
                "carattTesto": "Forma competenze di ingegneria dell'informazione.",
            }
        ],
        "percorsi": [
            {
                "des_it": "STANDARD - COORTE 2025",
                "schemaId": 19391,
                "anni": [
                    {
                        "anno": 1,
                        "insegnamenti": [
                            {
                                "label_it": "Attivita obbligatorie",
                                "attivita": [
                                    {
                                        "cod": "521891",
                                        "adCod": "IE12800001",
                                        "des_it": "ANALISI MATEMATICA 1",
                                        "crediti": 9,
                                        "ore": 72,
                                        "aa": "2025",
                                        "ordinamento_aa": 2025,
                                        "corso_cod": "500838",
                                        "cdsCod": "IE128",
                                        "af_percorso_id": "9999",
                                        "corso_percorso_id": "10002",
                                        "af_percorso_cod": "PDS0-2025",
                                        "periodo_didattico_it": "PRIMO SEMESTRE",
                                        "docenti": [
                                            {
                                                "des": "ZAMPOLI VITTORIO",
                                                "matricola": "020763",
                                                "cod": 502629,
                                            }
                                        ],
                                    }
                                ],
                            }
                        ],
                    },
                    {"anno": 2, "insegnamenti": []},
                    {"anno": 3, "insegnamenti": []},
                ],
            }
        ],
    }


class CourseCatalogueTests(unittest.TestCase):
    def test_settings_read_configured_ids(self) -> None:
        settings = settings_from_config(sample_config())

        self.assertTrue(settings.enabled)
        self.assertEqual(
            settings.course_entries,
            (("2024", "500639"), ("2025", "500838")),
        )

    def test_project_config_covers_course_catalogue_academic_years_2023_2026(self) -> None:
        config = yaml.safe_load((ROOT / "config.example.yaml").read_text(encoding="utf-8"))
        settings = settings_from_config(config)
        expected = {
            ("2023", "500189"),
            ("2023", "500639"),
            ("2023", "500191"),
            ("2023", "500648"),
            ("2023", "500679"),
            ("2024", "500189"),
            ("2024", "500639"),
            ("2024", "500191"),
            ("2024", "500648"),
            ("2024", "500679"),
            ("2025", "500853"),
            ("2025", "500838"),
            ("2025", "500854"),
            ("2025", "500878"),
            ("2025", "500879"),
        }

        self.assertEqual(set(settings.course_entries), expected)

    def test_settings_keep_backward_compatible_year_id_cross_product(self) -> None:
        config = sample_config()
        config["course_catalogue"].pop("course_entries")
        config["course_catalogue"]["years"] = ["2024", "2025"]
        config["course_catalogue"]["course_ids"] = ["500838", "500853"]

        settings = settings_from_config(config)

        self.assertEqual(
            settings.course_entries,
            (
                ("2024", "500838"),
                ("2024", "500853"),
                ("2025", "500838"),
                ("2025", "500853"),
            ),
        )

    def test_course_markdown_includes_years_roles_and_teachings(self) -> None:
        markdown = render_course_markdown(sample_course())

        self.assertIn("# INGEGNERIA DELL'INFORMAZIONE PER LA MEDICINA DIGITALE", markdown)
        self.assertIn("Presidente del Consiglio didattico", markdown)
        self.assertIn("### 1 anno", markdown)
        self.assertIn("### 2 anno", markdown)
        self.assertIn("### 3 anno", markdown)
        self.assertIn("ANALISI MATEMATICA 1", markdown)
        self.assertIn("ZAMPOLI VITTORIO", markdown)

    def test_teaching_urls_match_coursecatalogue_routes(self) -> None:
        settings = settings_from_config(sample_config())
        teaching = collect_course_teachings(sample_course())[0]

        self.assertEqual(
            teaching_api_url(settings, teaching),
            "https://unisa.coursecatalogue.cineca.it/api/v1/insegnamento-offerta/"
            "2025/521891/2025/10002/500838",
        )
        self.assertEqual(
            teaching_url(settings, teaching),
            "https://unisa.coursecatalogue.cineca.it/corsi/2025/500838/"
            "insegnamenti/2025/521891/2025/10002?coorte=2025&schemaid=19391",
        )

    def test_teaching_markdown_includes_syllabus_and_teacher_ids(self) -> None:
        teaching = collect_course_teachings(sample_course())[0]
        teaching.update(
            {
                "corso_des_it": sample_course()["des_it"],
                "testiTotali": [
                    {
                        "obiettivi_formativi_it": "Acquisire tecniche di calcolo.",
                        "contenuti_it": "Limiti, derivate e integrali.",
                        "verifica_apprendimento_it": "Prova scritta e orale.",
                    }
                ],
            }
        )

        markdown = render_teaching_markdown(teaching)

        self.assertIn("# ANALISI MATEMATICA 1", markdown)
        self.assertIn("matricola 020763", markdown)
        self.assertIn("## Obiettivi formativi", markdown)
        self.assertIn("Limiti, derivate e integrali.", markdown)


if __name__ == "__main__":
    unittest.main()
