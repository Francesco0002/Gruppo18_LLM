from __future__ import annotations

import sys
import unittest
import asyncio
from collections import Counter, deque
from datetime import UTC, datetime
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discover import (  # noqa: E402
    create_initial_state,
    diem_bandi_archive_urls,
    discovery_stop_reason,
    enqueue_links,
    fair_bfs_order,
    missing_diem_bandi_archive_items,
    take_batch,
    update_expansion_backlog,
)
from discovery_io import load_discovery_state, save_discovery_state  # noqa: E402
from discovery_models import CrawlItem, CrawlState, PersistentDiscoveryState  # noqa: E402
from discovery_processor import make_linked_pdf_records  # noqa: E402
from html_utils import extract_link_entries_from_soup  # noqa: E402
from url_filters import can_index_url, can_traverse_url  # noqa: E402


def base_config() -> dict:
    return {
        "crawler": {
            "allowed_domains": ["www.diem.unisa.it"],
            "pdf_allowed_domains": ["www.diem.unisa.it"],
            "max_total_urls": 10,
            "max_depth": 4,
            "max_concurrent_requests": 5,
            "refresh_after_days": 7,
            "per_domain_limits": {"www.diem.unisa.it": 10},
        },
        "scope": {
            "diem_domain": "www.diem.unisa.it",
            "teacher_domain": "docenti.unisa.it",
            "course_domain": "corsi.unisa.it",
            "allowed_course_paths": [],
            "allowed_course_codes": [],
        },
    }


class DiscoveryControlTests(unittest.TestCase):
    def test_bootstrap_items_track_their_origin_seed(self) -> None:
        seed_a = "https://www.diem.unisa.it/a"
        seed_b = "https://www.diem.unisa.it/b"

        state = create_initial_state([seed_a, seed_b], [])

        self.assertEqual(
            [(item.url, item.origin_seed) for item in state.queue],
            [(seed_a, seed_a), (seed_b, seed_b)],
        )

    def test_configured_course_numeric_aliases_are_traversable(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("corsi.unisa.it")
        config["crawler"]["per_domain_limits"]["corsi.unisa.it"] = 10
        config["scope"]["allowed_course_numeric_ids"] = ["0650106200800001"]

        self.assertTrue(
            can_traverse_url("https://corsi.unisa.it/0650106200800001", config)[0]
        )
        self.assertTrue(
            can_traverse_url(
                "https://corsi.unisa.it/0650106200800001/didattica/orari",
                config,
            )[0]
        )
        self.assertFalse(
            can_traverse_url("https://corsi.unisa.it/0650106201000001", config)[0]
        )

    def test_course_alias_links_from_in_scope_course_pages_are_traversable(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("corsi.unisa.it")
        config["crawler"]["per_domain_limits"]["corsi.unisa.it"] = 10
        config["scope"]["allowed_course_paths"] = ["ingegneria-informatica"]

        ok, reason = can_traverse_url(
            "https://corsi.unisa.it/0650106200800001/didattica/orari",
            config,
            {"discovered_from": "https://corsi.unisa.it/ingegneria-informatica"},
        )

        self.assertTrue(ok, reason)

    def test_short_numeric_course_codes_configured_by_full_code_are_traversable(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("corsi.unisa.it")
        config["crawler"]["per_domain_limits"]["corsi.unisa.it"] = 10
        config["scope"]["allowed_course_codes"] = ["06128L-8", "06233LM-28"]

        for url in (
            "https://corsi.unisa.it/06128/immatricolazioni",
            "https://corsi.unisa.it/06233",
        ):
            ok, reason = can_traverse_url(url, config)
            self.assertTrue(ok, reason)

    def test_course_sitemap_query_is_traversable_but_not_indexable(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("corsi.unisa.it")
        config["crawler"]["per_domain_limits"]["corsi.unisa.it"] = 10
        config["scope"]["allowed_course_paths"] = ["ingegneria-informatica"]
        url = "https://corsi.unisa.it/ingegneria-informatica?sitemap"

        traverse_ok, traverse_reason = can_traverse_url(url, config)
        index_ok, index_reason = can_index_url(url, config)

        self.assertTrue(traverse_ok, traverse_reason)
        self.assertFalse(index_ok)
        self.assertEqual(index_reason, "noisy_query")

    def test_course_rescue_detail_is_traversable_when_encoded_course_is_allowed(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("corsi.unisa.it")
        config["crawler"]["per_domain_limits"]["corsi.unisa.it"] = 10
        config["scope"]["allowed_course_paths"] = [
            "ingegneria-dell-informazione-per-la-medicina-digitale"
        ]

        ok, reason = can_traverse_url(
            "https://corsi.unisa.it/unisa-rescue-page/dettaglio/"
            "url/L2luZ2VnbmVyaWEtZGVsbC1pbmZvcm1hemlvbmUtcGVyLWxhLW1lZGljaW5hLWRpZ2l0YWxl/"
            "id/1540/module/501/row/29862",
            config,
        )

        self.assertTrue(ok, reason)

    def test_course_rescue_detail_with_percent_encoded_padding_is_traversable(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("corsi.unisa.it")
        config["crawler"]["per_domain_limits"]["corsi.unisa.it"] = 10
        config["scope"]["allowed_course_numeric_ids"] = ["0650106200800001"]

        ok, reason = can_traverse_url(
            "https://corsi.unisa.it/unisa-rescue-page/dettaglio/"
            "url/LzA2NTAxMDYyMDA4MDAwMDE%3D/id/1540/module/501/row/29903",
            config,
        )

        self.assertTrue(ok, reason)

    def test_malformed_course_rescue_detail_from_relative_menu_is_blocked(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("corsi.unisa.it")
        config["crawler"]["per_domain_limits"]["corsi.unisa.it"] = 10
        config["scope"]["allowed_course_paths"] = [
            "ingegneria-dell-informazione-per-la-medicina-digitale"
        ]
        config["scope"]["allowed_course_numeric_ids"] = ["0650106200800003"]

        ok, reason = can_traverse_url(
            "https://corsi.unisa.it/unisa-rescue-page/dettaglio/"
            "url/L2luZ2VnbmVyaWEtZGVsbC1pbmZvcm1hemlvbmUtcGVyLWxhLW1lZGljaW5hLWRpZ2l0YWxl/"
            "id/1540/module/0650106200800003/0650106200800003/0650106200800003/contatti",
            config,
        )

        self.assertFalse(ok)
        self.assertEqual(reason, "malformed_course_rescue")

    def test_malformed_course_rescue_search_from_relative_menu_is_blocked(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("corsi.unisa.it")
        config["crawler"]["per_domain_limits"]["corsi.unisa.it"] = 10
        config["scope"]["allowed_course_paths"] = ["ingegneria-informatica"]
        config["scope"]["allowed_course_numeric_ids"] = ["0650107303300001"]

        ok, reason = can_traverse_url(
            "https://corsi.unisa.it/unisa-rescue-page/search/id/2364/"
            "url/LzA2MjI3L2VuL3RlYWNoaW5nLWZhY2lsaXRpZXM=/"
            "calendario-occupazione-spazi/calendario-occupazione-spazi/"
            "0650107303300001/contatti",
            config,
        )

        self.assertFalse(ok)
        self.assertEqual(reason, "malformed_course_rescue")

    def test_bandi_structure_filter_keeps_only_diem_structure(self) -> None:
        config = base_config()

        ok, reason = can_traverse_url(
            "https://www.diem.unisa.it/home/bandi?anno=2026&modulo=139&struttura=300638",
            config,
        )
        self.assertTrue(ok, reason)

        ok, reason = can_traverse_url(
            "https://www.diem.unisa.it/home/bandi?anno=2026&modulo=139&struttura=300400",
            config,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "blocked_query")

    def test_ricerca_focus_id_is_traversable(self) -> None:
        config = base_config()

        ok, reason = can_traverse_url(
            "https://www.diem.unisa.it/ricerca/focus?id=1234",
            config,
        )
        self.assertTrue(ok, reason)

        ok, reason = can_traverse_url(
            "https://www.diem.unisa.it/ricerca/focus?anno=2026",
            config,
        )
        self.assertTrue(ok, reason)

    def test_conto_terzi_progetto_is_traversable(self) -> None:
        config = base_config()

        ok, reason = can_traverse_url(
            "https://www.diem.unisa.it/terza-missione/trasferimento-tecnologico/conto-terzi?progetto=66770",
            config,
        )
        self.assertTrue(ok, reason)

        # progetto su altri path generici (es: /home) deve rimanere bloccato
        ok, reason = can_traverse_url(
            "https://www.diem.unisa.it/home?progetto=66770",
            config,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "blocked_query")

    def test_conto_terzi_status_filters_are_indexable(self) -> None:
        config = base_config()

        for url in (
            "https://www.diem.unisa.it/terza-missione/trasferimento-tecnologico/conto-terzi?stato=1",
            "https://www.diem.unisa.it/terza-missione/trasferimento-tecnologico/conto-terzi?stato=0",
        ):
            with self.subTest(url=url):
                traverse_ok, traverse_reason = can_traverse_url(url, config)
                index_ok, index_reason = can_index_url(url, config)

                self.assertTrue(traverse_ok, traverse_reason)
                self.assertTrue(index_ok, index_reason)

    def test_diem_spin_off_incubator_filters_are_indexable(self) -> None:
        config = base_config()

        for url in (
            "https://www.diem.unisa.it/terza-missione/trasferimento-tecnologico/spin-off?incubatore=1",
            "https://www.diem.unisa.it/terza-missione/trasferimento-tecnologico/spin-off?incubatore=0",
        ):
            with self.subTest(url=url):
                traverse_ok, traverse_reason = can_traverse_url(url, config)
                index_ok, index_reason = can_index_url(url, config)

                self.assertTrue(traverse_ok, traverse_reason)
                self.assertTrue(index_ok, index_reason)

        ok, reason = can_traverse_url(
            "https://www.diem.unisa.it/terza-missione/trasferimento-tecnologico/spin-off?incubatore=2",
            config,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "blocked_query")

    def test_diem_international_status_filters_are_indexable_for_diem_structure(self) -> None:
        config = base_config()
        urls = [
            "https://www.diem.unisa.it/international/accordi-doppio-titolo?anno=&stato=attivi&struttura=300638",
            "https://www.diem.unisa.it/international/accordi-erasmus-plus/traineeship?anno=&stato=scaduti&struttura=300638",
            "https://www.diem.unisa.it/international/cooperazione-internazionale?anno=&stato=tutti&struttura=300638",
        ]

        for url in urls:
            with self.subTest(url=url):
                traverse_ok, traverse_reason = can_traverse_url(url, config)
                index_ok, index_reason = can_index_url(url, config)

                self.assertTrue(traverse_ok, traverse_reason)
                self.assertTrue(index_ok, index_reason)

        ok, reason = can_traverse_url(
            "https://www.diem.unisa.it/international/accordi-doppio-titolo?anno=&stato=attivi&struttura=300400",
            config,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "blocked_query")

    def test_doctoral_course_news_archive_is_traversable(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("corsi.unisa.it")
        config["crawler"]["per_domain_limits"]["corsi.unisa.it"] = 10
        config["scope"]["allowed_course_paths"] = ["DOT18CK8F9"]

        ok, reason = can_traverse_url(
            "https://corsi.unisa.it/DOT18CK8F9/news?archive=1",
            config,
        )

        self.assertTrue(ok, reason)

    def test_english_pages_are_filtered_from_italian_corpus(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("corsi.unisa.it")
        config["crawler"]["per_domain_limits"]["corsi.unisa.it"] = 10
        config["scope"]["allowed_course_paths"] = ["ingegneria-informatica"]

        for url in (
            "https://www.diem.unisa.it/en/department",
            "https://corsi.unisa.it/ingegneria-informatica/en/news",
            "https://corsi.unisa.it/0650107302900001/en/news",
            "https://corsi.unisa.it/DOT18CK8F9/en/news",
            "https://corsi.unisa.it/ingegneria-informatica/news?lang=en",
        ):
            ok, reason = can_traverse_url(url, config)
            self.assertFalse(ok)
            self.assertEqual(reason, "language")

    def test_allowed_teacher_profile_remains_traversable_without_source_context(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("docenti.unisa.it")
        config["crawler"]["per_domain_limits"]["docenti.unisa.it"] = 10
        context = {
            "discovered_from": "seed",
            "allowed_teacher_profiles": {"058553"},
        }

        ok, reason = can_traverse_url(
            "https://docenti.unisa.it/058553/home",
            config,
            context,
        )

        self.assertTrue(ok, reason)

    def test_teacher_numeric_subpages_are_limited_to_same_profile(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("docenti.unisa.it")
        config["crawler"]["per_domain_limits"]["docenti.unisa.it"] = 10

        same_profile_context = {
            "discovered_from": "https://docenti.unisa.it/058553/home",
            "allowed_teacher_profiles": {"antonio.parziale"},
        }
        ok, reason = can_traverse_url(
            "https://docenti.unisa.it/058553/curriculum",
            config,
            same_profile_context,
        )
        self.assertTrue(ok, reason)

        other_profile_context = {
            "discovered_from": "https://docenti.unisa.it/058553/home",
            "allowed_teacher_profiles": {"antonio.parziale"},
        }
        ok, reason = can_traverse_url(
            "https://docenti.unisa.it/004491/curriculum",
            config,
            other_profile_context,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "scope_teacher")

    def test_authorized_rubrica_link_uses_numeric_teacher_profile(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].extend(
            ["rubrica.unisa.it", "docenti.unisa.it"]
        )
        config["crawler"]["per_domain_limits"]["rubrica.unisa.it"] = 10
        config["crawler"]["per_domain_limits"]["docenti.unisa.it"] = 10
        config["scope"]["directory_domain"] = "rubrica.unisa.it"
        html = '<a href="https://docenti.unisa.it/diego.gragnaniello">WEB</a>'

        links = extract_link_entries_from_soup(
            BeautifulSoup(html, "lxml"),
            "https://rubrica.unisa.it/persone?matricola=058553",
            config,
            allowed_teacher_profiles=set(),
            authorized_directory_bridge=True,
        )

        self.assertEqual(
            [link["url"] for link in links],
            ["https://docenti.unisa.it/058553/home"],
        )

    def test_teacher_query_sections_are_not_traversed(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("docenti.unisa.it")
        config["crawler"]["per_domain_limits"]["docenti.unisa.it"] = 10
        context = {
            "discovered_from": "https://docenti.unisa.it/058553/ricerca/pubblicazioni",
            "allowed_teacher_profiles": {"058553"},
        }

        ok, reason = can_traverse_url(
            "https://docenti.unisa.it/058553/ricerca/pubblicazioni?anno=2025&tip=262",
            config,
            context,
        )

        self.assertFalse(ok)
        self.assertEqual(reason, "scope_teacher")

    def test_teacher_publications_all_years_query_is_indexable(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("docenti.unisa.it")
        config["crawler"]["per_domain_limits"]["docenti.unisa.it"] = 10
        context = {
            "discovered_from": "https://docenti.unisa.it/058553/ricerca/pubblicazioni",
            "allowed_teacher_profiles": {"058553"},
        }

        traverse_ok, traverse_reason = can_traverse_url(
            "https://docenti.unisa.it/058553/ricerca/pubblicazioni?anno=0",
            config,
            context,
        )
        index_ok, index_reason = can_index_url(
            "https://docenti.unisa.it/058553/ricerca/pubblicazioni?anno=0",
            config,
            context,
        )

        self.assertTrue(traverse_ok, traverse_reason)
        self.assertTrue(index_ok, index_reason)

    def test_teacher_historical_didactics_query_is_indexable(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("docenti.unisa.it")
        config["crawler"]["per_domain_limits"]["docenti.unisa.it"] = 10
        context = {
            "discovered_from": "https://docenti.unisa.it/027233/didattica",
            "allowed_teacher_profiles": {"027233"},
        }

        traverse_ok, traverse_reason = can_traverse_url(
            "https://docenti.unisa.it/027233/didattica?anno=2021",
            config,
            context,
        )
        index_ok, index_reason = can_index_url(
            "https://docenti.unisa.it/027233/didattica?anno=2021",
            config,
            context,
        )

        self.assertTrue(traverse_ok, traverse_reason)
        self.assertTrue(index_ok, index_reason)

    def test_teacher_lesson_schedule_is_indexable(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("docenti.unisa.it")
        config["crawler"]["per_domain_limits"]["docenti.unisa.it"] = 10
        context = {
            "discovered_from": "https://docenti.unisa.it/027233/didattica",
            "allowed_teacher_profiles": {"027233"},
        }

        traverse_ok, traverse_reason = can_traverse_url(
            "https://docenti.unisa.it/027233/didattica/orari?include=docente",
            config,
            context,
        )
        index_ok, index_reason = can_index_url(
            "https://docenti.unisa.it/027233/didattica/orari?include=docente",
            config,
            context,
        )

        self.assertTrue(traverse_ok, traverse_reason)
        self.assertTrue(index_ok, index_reason)

    def test_teacher_project_detail_is_indexable(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("docenti.unisa.it")
        config["crawler"]["per_domain_limits"]["docenti.unisa.it"] = 10
        context = {
            "discovered_from": "https://docenti.unisa.it/027233/ricerca/progetti",
            "allowed_teacher_profiles": {"027233"},
        }

        traverse_ok, traverse_reason = can_traverse_url(
            "https://docenti.unisa.it/027233/ricerca/progetti?progetto=59185",
            config,
            context,
        )
        index_ok, index_reason = can_index_url(
            "https://docenti.unisa.it/027233/ricerca/progetti?progetto=59185",
            config,
            context,
        )

        self.assertTrue(traverse_ok, traverse_reason)
        self.assertTrue(index_ok, index_reason)

    def test_teacher_spin_off_detail_is_indexable(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("docenti.unisa.it")
        config["crawler"]["per_domain_limits"]["docenti.unisa.it"] = 10
        context = {
            "discovered_from": "https://docenti.unisa.it/027233/ricerca/spin-off",
            "allowed_teacher_profiles": {"027233"},
        }

        traverse_ok, traverse_reason = can_traverse_url(
            "https://docenti.unisa.it/027233/ricerca/spin-off?id=101",
            config,
            context,
        )
        index_ok, index_reason = can_index_url(
            "https://docenti.unisa.it/027233/ricerca/spin-off?id=101",
            config,
            context,
        )

        self.assertTrue(traverse_ok, traverse_reason)
        self.assertTrue(index_ok, index_reason)

    def test_teacher_laboratories_are_indexable(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("docenti.unisa.it")
        config["crawler"]["per_domain_limits"]["docenti.unisa.it"] = 10
        context = {
            "discovered_from": "https://docenti.unisa.it/027233/ricerca",
            "allowed_teacher_profiles": {"027233"},
        }

        traverse_ok, traverse_reason = can_traverse_url(
            "https://docenti.unisa.it/027233/ricerca/laboratori",
            config,
            context,
        )
        index_ok, index_reason = can_index_url(
            "https://docenti.unisa.it/027233/ricerca/laboratori",
            config,
            context,
        )

        self.assertTrue(traverse_ok, traverse_reason)
        self.assertTrue(index_ok, index_reason)

    def test_teacher_laboratory_detail_is_indexable(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("docenti.unisa.it")
        config["crawler"]["per_domain_limits"]["docenti.unisa.it"] = 10
        context = {
            "discovered_from": "https://docenti.unisa.it/001366/ricerca/laboratori",
            "allowed_teacher_profiles": {"001366"},
        }
        url = "https://docenti.unisa.it/001366/ricerca/laboratori?id=722"

        traverse_ok, traverse_reason = can_traverse_url(url, config, context)
        index_ok, index_reason = can_index_url(url, config, context)

        self.assertTrue(traverse_ok, traverse_reason)
        self.assertTrue(index_ok, index_reason)

    def test_teacher_patents_focus_and_research_awards_are_indexable(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("docenti.unisa.it")
        config["crawler"]["per_domain_limits"]["docenti.unisa.it"] = 10
        context = {
            "discovered_from": "https://docenti.unisa.it/004294/ricerca",
            "allowed_teacher_profiles": {"004294"},
        }
        urls = [
            "https://docenti.unisa.it/004294/ricerca/brevetti",
            "https://docenti.unisa.it/004294/ricerca/brevetti?id=123",
            "https://docenti.unisa.it/004294/ricerca/premi-ricerca",
            "https://docenti.unisa.it/004294/ricerca/premi-ricerca?anno=2015",
            "https://docenti.unisa.it/004294/ricerca/focus",
            "https://docenti.unisa.it/004294/ricerca/focus?id=42",
        ]

        for url in urls:
            with self.subTest(url=url):
                traverse_ok, traverse_reason = can_traverse_url(url, config, context)
                index_ok, index_reason = can_index_url(url, config, context)

                self.assertTrue(traverse_ok, traverse_reason)
                self.assertTrue(index_ok, index_reason)

    def test_teacher_projects_all_roles_query_is_indexable(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("docenti.unisa.it")
        config["crawler"]["per_domain_limits"]["docenti.unisa.it"] = 10
        context = {
            "discovered_from": "https://docenti.unisa.it/058553/ricerca/progetti",
            "allowed_teacher_profiles": {"058553"},
        }

        traverse_ok, traverse_reason = can_traverse_url(
            "https://docenti.unisa.it/058553/ricerca/progetti?ruolo=tutti",
            config,
            context,
        )
        index_ok, index_reason = can_index_url(
            "https://docenti.unisa.it/058553/ricerca/progetti?ruolo=tutti",
            config,
            context,
        )

        self.assertTrue(traverse_ok, traverse_reason)
        self.assertTrue(index_ok, index_reason)

    def test_teacher_project_role_and_status_filters_are_indexable(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("docenti.unisa.it")
        config["crawler"]["per_domain_limits"]["docenti.unisa.it"] = 10
        context = {
            "discovered_from": "https://docenti.unisa.it/003495/ricerca/progetti",
            "allowed_teacher_profiles": {"003495"},
        }
        urls = [
            "https://docenti.unisa.it/003495/ricerca/progetti?ruolo=responsabile",
            "https://docenti.unisa.it/003495/ricerca/progetti?ruolo=componente",
            "https://docenti.unisa.it/003495/ricerca/progetti?ruolo=responsabile&stato=1",
            "https://docenti.unisa.it/003495/ricerca/progetti?ruolo=responsabile&stato=0",
        ]

        for url in urls:
            with self.subTest(url=url):
                traverse_ok, traverse_reason = can_traverse_url(url, config, context)
                index_ok, index_reason = can_index_url(url, config, context)

                self.assertTrue(traverse_ok, traverse_reason)
                self.assertTrue(index_ok, index_reason)

    def test_teacher_publications_landing_is_traversable_but_not_indexable(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("docenti.unisa.it")
        config["crawler"]["per_domain_limits"]["docenti.unisa.it"] = 10
        context = {
            "discovered_from": "https://docenti.unisa.it/058553/ricerca",
            "allowed_teacher_profiles": {"058553"},
        }

        traverse_ok, traverse_reason = can_traverse_url(
            "https://docenti.unisa.it/058553/ricerca/pubblicazioni",
            config,
            context,
        )
        index_ok, index_reason = can_index_url(
            "https://docenti.unisa.it/058553/ricerca/pubblicazioni",
            config,
            context,
        )

        self.assertTrue(traverse_ok, traverse_reason)
        self.assertFalse(index_ok)
        self.assertEqual(index_reason, "teacher_publications_landing")

    def test_teacher_publication_year_archives_are_not_indexable(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("docenti.unisa.it")
        config["crawler"]["per_domain_limits"]["docenti.unisa.it"] = 10
        context = {
            "discovered_from": "https://docenti.unisa.it/003495/home",
            "allowed_teacher_profiles": {"003495"},
        }

        traverse_ok, traverse_reason = can_traverse_url(
            "https://docenti.unisa.it/003495/ricerca/pubblicazioni?anno=2025",
            config,
            context,
        )
        index_ok, index_reason = can_index_url(
            "https://docenti.unisa.it/003495/ricerca/pubblicazioni?anno=2025",
            config,
            context,
        )

        self.assertFalse(traverse_ok)
        self.assertEqual(traverse_reason, "scope_teacher")
        self.assertFalse(index_ok)
        self.assertEqual(index_reason, "scope_teacher")

    def test_teacher_teaching_details_are_indexable(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("docenti.unisa.it")
        config["crawler"]["per_domain_limits"]["docenti.unisa.it"] = 10
        context = {
            "discovered_from": "https://docenti.unisa.it/003495/home",
            "allowed_teacher_profiles": {"003495"},
        }
        urls = [
            "https://docenti.unisa.it/003495/didattica?anno=2023&id=515713",
            "https://docenti.unisa.it/003495/didattica/orari",
        ]

        for url in urls:
            with self.subTest(url=url):
                traverse_ok, traverse_reason = can_traverse_url(url, config, context)
                index_ok, index_reason = can_index_url(url, config, context)

                self.assertTrue(traverse_ok, traverse_reason)
                self.assertTrue(index_ok, index_reason)

    def test_teacher_resources_notices_and_international_sections_are_indexable(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("docenti.unisa.it")
        config["crawler"]["per_domain_limits"]["docenti.unisa.it"] = 10
        context = {
            "discovered_from": "https://docenti.unisa.it/004294/home",
            "allowed_teacher_profiles": {"004294"},
        }
        urls = [
            "https://docenti.unisa.it/004294",
            "https://docenti.unisa.it/004294/home?avvisi=1",
            "https://docenti.unisa.it/004294/home?avviso=47893",
            "https://docenti.unisa.it/004294/risorse?categoria=348",
            "https://docenti.unisa.it/004294/risorse?categoria=336&risorsa=2",
            "https://docenti.unisa.it/004294/international/erasmus",
            "https://docenti.unisa.it/004294/international/traineeship",
            "https://docenti.unisa.it/004294/international/dottorato-con-tesi-in-cotutela",
            "https://docenti.unisa.it/004294/international/cooperazione-internazionale",
            "https://docenti.unisa.it/004294/international/doppio-titolo",
        ]

        for url in urls:
            with self.subTest(url=url):
                traverse_ok, traverse_reason = can_traverse_url(url, config, context)
                index_ok, index_reason = can_index_url(url, config, context)

                self.assertTrue(traverse_ok, traverse_reason)
                self.assertTrue(index_ok, index_reason)

    def test_diem_bandi_department_lists_are_indexable(self) -> None:
        config = base_config()
        url = "https://www.diem.unisa.it/home/bandi?anno=2026&modulo=139&struttura=300638"

        traverse_ok, traverse_reason = can_traverse_url(url, config)
        index_ok, index_reason = can_index_url(url, config)

        self.assertTrue(traverse_ok, traverse_reason)
        self.assertTrue(index_ok, index_reason)

    def test_historical_diem_bandi_department_module_is_indexable(self) -> None:
        config = base_config()
        url = "https://www.diem.unisa.it/home/bandi?anno=2023&struttura=300638&modulo=316"

        traverse_ok, traverse_reason = can_traverse_url(url, config)
        index_ok, index_reason = can_index_url(url, config)

        self.assertTrue(traverse_ok, traverse_reason)
        self.assertTrue(index_ok, index_reason)

    def test_diem_bandi_cds_structure_modules_are_indexable(self) -> None:
        config = base_config()
        urls = [
            "https://www.diem.unisa.it/home/bandi?modulo=293&cdsStruttura=300638",
            "https://www.diem.unisa.it/home/bandi?anno=2025&modulo=293&cdsStruttura=300638",
            "https://www.diem.unisa.it/home/bandi?anno=2024&modulo=505&struttura=300638",
        ]

        for url in urls:
            with self.subTest(url=url):
                traverse_ok, traverse_reason = can_traverse_url(url, config)
                index_ok, index_reason = can_index_url(url, config)

                self.assertTrue(traverse_ok, traverse_reason)
                self.assertTrue(index_ok, index_reason)

    def test_diem_bandi_dottorati_lists_are_traversable_but_not_indexable(self) -> None:
        config = base_config()
        urls = [
            "https://www.diem.unisa.it/home/bandi?modulo=226",
            "https://www.diem.unisa.it/home/bandi?anno=2025&modulo=226",
            "https://www.diem.unisa.it/home/bandi?anno=2025&bando=13933&modulo=226",
        ]

        for url in urls:
            with self.subTest(url=url):
                traverse_ok, traverse_reason = can_traverse_url(url, config)
                index_ok, index_reason = can_index_url(url, config)

                self.assertTrue(traverse_ok, traverse_reason)
                self.assertFalse(index_ok)
                self.assertEqual(index_reason, "noisy_bandi_query")

    def test_diem_bandi_dottorati_detail_is_indexable_only_for_diem_concorso(self) -> None:
        config = base_config()

        allowed_url = "https://www.diem.unisa.it/home/bandi?bando=13933&anno=2025&modulo=226&idConcorso=9443"
        other_department_url = "https://www.diem.unisa.it/home/bandi?bando=13933&anno=2025&modulo=226&idConcorso=9438"

        for url, expected_indexable in (
            (allowed_url, True),
            (other_department_url, False),
        ):
            with self.subTest(url=url):
                traverse_ok, traverse_reason = can_traverse_url(url, config)
                index_ok, index_reason = can_index_url(url, config)

                self.assertTrue(traverse_ok, traverse_reason)
                self.assertEqual(index_ok, expected_indexable)
                if not expected_indexable:
                    self.assertEqual(index_reason, "noisy_bandi_query")

    def test_diem_doctorate_detail_links_are_extracted_only_for_diem_codes(self) -> None:
        config = base_config()
        config["scope"]["allowed_course_paths"] = ["DOT18CK8F9", "DOT229NLP9"]
        soup = BeautifulSoup(
            """
            <a href="/home/bandi?bando=13933&anno=2025&modulo=226&idConcorso=9443">
              DOT18CK8F9 INGEGNERIA DELL'INFORMAZIONE
            </a>
            <a href="/home/bandi?bando=13933&anno=2025&modulo=226&idConcorso=9438">
              DOT227NLPN DATA SCIENCE, ACCOUNTING & MANAGEMENT
            </a>
            <a href="/home/bandi?anno=2025&modulo=226">Archivio 2025</a>
            """,
            "html.parser",
        )

        links = extract_link_entries_from_soup(
            soup,
            "https://www.diem.unisa.it/home/bandi?anno=2025&modulo=226",
            config,
        )

        self.assertEqual(
            [link["url"] for link in links],
            [
                "https://www.diem.unisa.it/home/bandi?anno=2025&bando=13933&idConcorso=9443&modulo=226",
                "https://www.diem.unisa.it/home/bandi?anno=2025&modulo=226",
            ],
        )

    def test_diem_bandi_detail_pages_remain_noisy_for_structured_modules(self) -> None:
        config = base_config()
        url = "https://www.diem.unisa.it/home/bandi?anno=2025&bando=13301&idConcorso=9089&modulo=139"

        traverse_ok, traverse_reason = can_traverse_url(url, config)
        index_ok, index_reason = can_index_url(url, config)

        self.assertTrue(traverse_ok, traverse_reason)
        self.assertFalse(index_ok)
        self.assertEqual(index_reason, "noisy_bandi_query")

    def test_diem_bandi_category_pages_are_indexable_for_diem_structure(self) -> None:
        config = base_config()
        urls = [
            "https://www.diem.unisa.it/home/bandi?anno=2026&categoria=258&cdsStruttura=300638&modulo=293",
            "https://www.diem.unisa.it/home/bandi?anno=2026&categoria=176&modulo=67&struttura=300638",
        ]

        for url in urls:
            with self.subTest(url=url):
                traverse_ok, traverse_reason = can_traverse_url(url, config)
                index_ok, index_reason = can_index_url(url, config)

                self.assertTrue(traverse_ok, traverse_reason)
                self.assertTrue(index_ok, index_reason)

    def test_diem_bandi_category_pages_for_other_structures_are_blocked(self) -> None:
        config = base_config()
        url = "https://www.diem.unisa.it/home/bandi?anno=2026&categoria=176&modulo=67&struttura=300389"

        traverse_ok, traverse_reason = can_traverse_url(url, config)

        self.assertFalse(traverse_ok)
        self.assertEqual(traverse_reason, "blocked_query")

    def test_noisy_diem_bandi_queries_are_traversable_but_not_indexable(self) -> None:
        config = base_config()
        noisy_urls = [
            "https://www.diem.unisa.it/home/bandi?anno=2026&bando=14332&modulo=293",
            "https://www.diem.unisa.it/home/bandi?struttura=300638",
            "https://www.diem.unisa.it/home/bandi?modulo=139&struttura=300638&page=2",
        ]

        for url in noisy_urls:
            with self.subTest(url=url):
                traverse_ok, traverse_reason = can_traverse_url(url, config)
                index_ok, index_reason = can_index_url(url, config)

                self.assertTrue(traverse_ok, traverse_reason)
                self.assertFalse(index_ok)
                self.assertEqual(index_reason, "noisy_bandi_query")

    def test_historical_diem_sections_are_indexable(self) -> None:
        config = base_config()
        urls = [
            "https://www.diem.unisa.it/didattica/offerta-formativa?anno=2020",
            "https://www.diem.unisa.it/home/dati-di-monitoraggio?anno=2020",
        ]

        for url in urls:
            with self.subTest(url=url):
                traverse_ok, traverse_reason = can_traverse_url(url, config)
                index_ok, index_reason = can_index_url(url, config)

                self.assertTrue(traverse_ok, traverse_reason)
                self.assertTrue(index_ok, index_reason)

    def test_bandi_other_department_lists_are_blocked(self) -> None:
        config = base_config()
        url = "https://www.diem.unisa.it/home/bandi?anno=2026&modulo=139&struttura=300400"

        traverse_ok, traverse_reason = can_traverse_url(url, config)
        index_ok, index_reason = can_index_url(url, config)

        self.assertFalse(traverse_ok)
        self.assertEqual(traverse_reason, "blocked_query")
        self.assertFalse(index_ok)
        self.assertEqual(index_reason, "blocked_query")

    def test_course_room_calendar_queries_are_traversable_but_not_indexable(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("corsi.unisa.it")
        config["crawler"]["per_domain_limits"]["corsi.unisa.it"] = 10
        config["scope"]["allowed_course_paths"] = ["ingegneria-informatica"]
        url = (
            "https://corsi.unisa.it/ingegneria-informatica/"
            "strutture-didattiche/calendario-occupazione-spazi?aula=E&sede=F"
        )

        traverse_ok, traverse_reason = can_traverse_url(url, config)
        index_ok, index_reason = can_index_url(url, config)

        self.assertTrue(traverse_ok, traverse_reason)
        self.assertFalse(index_ok)
        self.assertEqual(index_reason, "noisy_room_calendar_query")

    def test_diem_directory_contact_is_indexable_only_from_personnel_page(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("rubrica.unisa.it")
        config["crawler"]["per_domain_limits"]["rubrica.unisa.it"] = 10
        url = "https://rubrica.unisa.it/persone?matricola=004491"
        personnel_context = {
            "discovered_from": "https://www.diem.unisa.it/dipartimento/personale"
        }

        traverse_ok, traverse_reason = can_traverse_url(url, config, personnel_context)
        index_ok, index_reason = can_index_url(url, config, personnel_context)

        self.assertTrue(traverse_ok, traverse_reason)
        self.assertTrue(index_ok, index_reason)

        course_context = {
            "discovered_from": "https://cd.unisa.it/ingegneria-informatica/commissioni"
        }
        traverse_ok, traverse_reason = can_traverse_url(url, config, course_context)
        index_ok, index_reason = can_index_url(url, config, course_context)

        self.assertFalse(traverse_ok)
        self.assertEqual(traverse_reason, "scope_directory")
        self.assertFalse(index_ok)
        self.assertEqual(index_reason, "scope_directory")

    def test_teaching_council_pages_are_scoped_to_configured_diem_courses(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("cd.unisa.it")
        config["crawler"]["per_domain_limits"]["cd.unisa.it"] = 10
        config["scope"]["allowed_course_paths"] = [
            "ingegneria-informatica",
            "electrical-engineering-for-digital-energy",
        ]

        for url in (
            "https://cd.unisa.it/ingegneria-informatica",
            "https://cd.unisa.it/ingegneria-informatica/commissioni",
            "https://cd.unisa.it/ingegneria-informatica/delegati",
            "https://cd.unisa.it/electrical-engineering-for-digital-energy/commissioni",
            "https://cd.unisa.it/electrical-engineering-for-digital-energy/delegati",
        ):
            traverse_ok, traverse_reason = can_traverse_url(url, config)
            index_ok, index_reason = can_index_url(url, config)
            self.assertTrue(traverse_ok, traverse_reason)
            self.assertTrue(index_ok, index_reason)

        ok, reason = can_traverse_url(
            "https://cd.unisa.it/giurisprudenza/commissioni",
            config,
        )
        self.assertFalse(ok)
        self.assertEqual(reason, "scope_teaching_council")

    def test_course_rescue_detail_linked_from_allowed_course_is_traversable(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("corsi.unisa.it")
        config["crawler"]["per_domain_limits"]["corsi.unisa.it"] = 10
        config["scope"]["allowed_course_paths"] = ["ingegneria-informatica"]

        ok, reason = can_traverse_url(
            "https://corsi.unisa.it/unisa-rescue-page/dettaglio/id/1540/module/501/row/29862",
            config,
            {"discovered_from": "https://corsi.unisa.it/ingegneria-informatica"},
        )

        self.assertTrue(ok, reason)

    def test_fair_bfs_order_round_robins_seed_branches_within_depth(self) -> None:
        seed_a = "https://www.diem.unisa.it/a"
        seed_b = "https://www.diem.unisa.it/b"
        items = [
            CrawlItem(f"{seed_a}/1", 1, seed_a, origin_seed=seed_a),
            CrawlItem(f"{seed_a}/2", 1, seed_a, origin_seed=seed_a),
            CrawlItem(f"{seed_a}/3", 1, seed_a, origin_seed=seed_a),
            CrawlItem(f"{seed_b}/1", 1, seed_b, origin_seed=seed_b),
            CrawlItem(f"{seed_b}/2", 1, seed_b, origin_seed=seed_b),
        ]

        self.assertEqual(
            [item.url for item in fair_bfs_order(items)],
            [
                f"{seed_a}/1",
                f"{seed_b}/1",
                f"{seed_a}/2",
                f"{seed_b}/2",
                f"{seed_a}/3",
            ],
        )

    def test_fair_bfs_order_never_promotes_deeper_items_before_lower_depths(self) -> None:
        seed = "https://www.diem.unisa.it/"
        items = [
            CrawlItem(f"{seed}deep", 3, seed, origin_seed=seed),
            CrawlItem(f"{seed}shallow", 2, seed, origin_seed=seed),
        ]

        self.assertEqual(
            [item.depth for item in fair_bfs_order(items)],
            [2, 3],
        )

    def test_high_value_pdf_exception_is_limited_to_central_diem_sections(self) -> None:
        class DenyRobots:
            async def can_fetch(self, _url: str) -> bool:
                return False

        records = asyncio.run(
            make_linked_pdf_records(
                [
                    {
                        "url": "https://www.diem.unisa.it/uploads/documento-123.pdf",
                        "text": "Regolamento didattico del corso",
                    }
                ],
                "https://www.diem.unisa.it/didattica/offerta-formativa",
                1,
                DenyRobots(),
                base_config(),
            )
        )

        self.assertEqual(records[0]["status"], "pending_download")
        self.assertEqual(records[0]["pdf_download_decision"], "allowed_stable_document")
        self.assertEqual(records[0]["pdf_source_section"], "didattica")
        self.assertEqual(records[0]["pdf_match_keywords"], ["regolamento"])
        self.assertTrue(records[0]["robots_txt_denied"])

    def test_high_value_pdf_exception_allows_in_scope_course_didactics(self) -> None:
        class DenyRobots:
            async def can_fetch(self, _url: str) -> bool:
                return False

        config = base_config()
        config["crawler"]["allowed_domains"].append("corsi.unisa.it")
        config["crawler"]["pdf_allowed_domains"].append("corsi.unisa.it")
        config["crawler"]["per_domain_limits"]["corsi.unisa.it"] = 10
        records = asyncio.run(
            make_linked_pdf_records(
                [
                    {
                        "url": "https://corsi.unisa.it/uploads/rescue/__piano-studi-cds/2025/IE233.pdf",
                        "text": "Piano di Studi",
                    }
                ],
                "https://corsi.unisa.it/electrical-engineering-for-digital-energy/didattica/piano-di-studi",
                2,
                DenyRobots(),
                config,
            )
        )

        self.assertEqual(records[0]["status"], "pending_download")
        self.assertEqual(records[0]["pdf_source_section"], "didattica")
        self.assertEqual(records[0]["pdf_download_decision"], "allowed_stable_document")

    def test_main_opportunity_pdf_from_bandi_is_allowed_but_attachments_are_not(self) -> None:
        class DenyRobots:
            async def can_fetch(self, _url: str) -> bool:
                return False

        records = asyncio.run(
            make_linked_pdf_records(
                [
                    {
                        "url": "https://www.diem.unisa.it/uploads/bando-premio-tesi-2026.pdf",
                        "text": "PDF",
                    },
                    {
                        "url": "https://www.diem.unisa.it/uploads/graduatoria-premio-tesi-2026.pdf",
                        "text": "PDF",
                    },
                ],
                "https://www.diem.unisa.it/home/bandi",
                1,
                DenyRobots(),
                base_config(),
            )
        )

        self.assertEqual(records[0]["status"], "pending_download")
        self.assertEqual(records[0]["pdf_download_decision"], "allowed_opportunity_document")
        self.assertEqual(records[1]["status"], "robots_denied")

    def test_diem_rescue_upload_bandi_from_diem_structure_are_in_scope(self) -> None:
        config = base_config()
        parent = "https://www.diem.unisa.it/home/bandi?anno=2023&struttura=300638&modulo=316"
        urls = [
            "https://www.diem.unisa.it/uploads/rescue/504/14120/50-2026-nomina-commissione-bando-tutorato-2026-fondi-aggiuntivi-disabili-2025-1-.pdf",
            "https://www.diem.unisa.it/uploads/rescue/292/14172/bando-rep.-n.-69-del-13.02.2026.pdf",
            "https://www.diem.unisa.it/uploads/rescue/292/14149/rep-196-prot-46039-bando-borsa-savarese-dipmed-2026-bs07.pdf",
            "https://www.diem.unisa.it/uploads/rescue/292/14148/rep-299-prot-67529-d-comm-borsa-dipmed-2026-bs03.pdf",
            "https://www.diem.unisa.it/uploads/rescue/292/14285/bs11-bando-fraternali..pdf",
        ]

        for url in urls:
            ok, reason = can_traverse_url(url, config, {"discovered_from": parent})
            self.assertTrue(ok, reason)

    def test_diem_rescue_upload_bandi_from_other_structures_are_out_of_scope(self) -> None:
        config = base_config()
        parent = "https://www.diem.unisa.it/home/bandi?anno=2023&struttura=300400&modulo=316"
        url = "https://www.diem.unisa.it/uploads/rescue/292/14172/bando-rep.-n.-69-del-13.02.2026.pdf"

        ok, reason = can_traverse_url(url, config, {"discovered_from": parent})

        self.assertFalse(ok)
        self.assertEqual(reason, "diem_rescue_upload_untrusted")

    def test_diem_rescue_upload_bandi_records_are_kept_for_diem_structure(self) -> None:
        class DenyRobots:
            async def can_fetch(self, _url: str) -> bool:
                return False

        records = asyncio.run(
            make_linked_pdf_records(
                [
                    {
                        "url": "https://www.diem.unisa.it/uploads/rescue/292/14149/rep-196-prot-46039-bando-borsa-savarese-dipmed-2026-bs07.pdf",
                        "text": "Bando",
                    }
                ],
                "https://www.diem.unisa.it/home/bandi?anno=2023&struttura=300638&modulo=316",
                1,
                DenyRobots(),
                base_config(),
            )
        )

        self.assertEqual(records[0]["status"], "pending_download")
        self.assertEqual(records[0]["pdf_source_section"], "home_bandi")
        self.assertEqual(records[0]["pdf_download_decision"], "allowed_opportunity_document")

    def test_all_pdfs_from_explicit_diem_bandi_pages_are_kept(self) -> None:
        class DenyRobots:
            async def can_fetch(self, _url: str) -> bool:
                return False

        records = asyncio.run(
            make_linked_pdf_records(
                [
                    {
                        "url": "https://www.diem.unisa.it/uploads/rescue/293/14107/locandina-viii-bando-tesi-di-laurea-2026.pdf",
                        "text": "Locandina",
                    },
                    {
                        "url": "https://www.diem.unisa.it/uploads/rescue/226/13933/bando-di-concorso-xli-ciclo.pdf",
                        "text": "Bando",
                    },
                ],
                "https://www.diem.unisa.it/home/bandi?anno=2025&cdsStruttura=300638&modulo=293",
                1,
                DenyRobots(),
                base_config(),
            )
        )
        dottorati_records = asyncio.run(
            make_linked_pdf_records(
                [
                    {
                        "url": "https://www.diem.unisa.it/uploads/rescue/226/13933/allegato-dottorati.pdf",
                        "text": "Allegato",
                    },
                ],
                "https://www.diem.unisa.it/home/bandi?anno=2025&modulo=226",
                1,
                DenyRobots(),
                base_config(),
            )
        )

        self.assertEqual([record["status"] for record in records], ["pending_download"] * 2)
        self.assertEqual(records[0]["pdf_download_decision"], "allowed_bandi_document")
        self.assertEqual(records[1]["pdf_download_decision"], "allowed_opportunity_document")
        self.assertEqual(dottorati_records[0]["status"], "out_of_scope")
        self.assertEqual(
            dottorati_records[0]["skip_reason"],
            "diem_rescue_upload_untrusted",
        )

    def test_diem_doctorate_detail_pdfs_are_blocked_for_other_departments(self) -> None:
        class DenyRobots:
            async def can_fetch(self, _url: str) -> bool:
                return False

        records = asyncio.run(
            make_linked_pdf_records(
                [
                    {
                        "url": "https://www.diem.unisa.it/uploads/rescue/151/9438/verbale-preliminare.pdf",
                        "text": "Verbale preliminare",
                    },
                ],
                "https://www.diem.unisa.it/home/bandi?bando=13933&anno=2025&modulo=226&idConcorso=9438",
                1,
                DenyRobots(),
                base_config(),
            )
        )

        self.assertEqual(records[0]["status"], "out_of_scope")
        self.assertEqual(
            records[0]["skip_reason"],
            "diem_rescue_upload_untrusted",
        )

    def test_diem_bandi_dottorati_detail_records_keep_pdfs(self) -> None:
        class DenyRobots:
            async def can_fetch(self, _url: str) -> bool:
                return False

        records = asyncio.run(
            make_linked_pdf_records(
                [
                    {
                        "url": "https://www.diem.unisa.it/uploads/rescue/226/13933/bando-di-concorso-xli-ciclo-regione-campania-loghi.pdf",
                        "text": "Bando di concorso",
                    },
                ],
                "https://www.diem.unisa.it/home/bandi?bando=13933&anno=2025&modulo=226&idConcorso=9443",
                1,
                DenyRobots(),
                base_config(),
            )
        )

        self.assertEqual(records[0]["status"], "pending_download")
        self.assertEqual(records[0]["pdf_source_section"], "home_bandi")
        self.assertEqual(records[0]["pdf_download_decision"], "allowed_opportunity_document")

    def test_bandi_moduli_and_presentations_do_not_pass_as_main_opportunities(self) -> None:
        class DenyRobots:
            async def can_fetch(self, _url: str) -> bool:
                return False

        records = asyncio.run(
            make_linked_pdf_records(
                [
                    {
                        "url": "https://www.diem.unisa.it/uploads/modulo-premio-2026.pdf",
                        "text": "PDF",
                    },
                    {
                        "url": "https://www.diem.unisa.it/uploads/presentazione-concorso-2026.pdf",
                        "text": "PDF",
                    },
                    {
                        "url": "https://www.diem.unisa.it/uploads/comunicato-borsa-2026.pdf",
                        "text": "PDF",
                    },
                ],
                "https://www.diem.unisa.it/home/bandi",
                1,
                DenyRobots(),
                base_config(),
            )
        )

        self.assertEqual([record["status"] for record in records], ["robots_denied"] * 3)

    def test_bandi_verbal_and_schedule_attachments_do_not_pass_as_main_opportunities(self) -> None:
        class DenyRobots:
            async def can_fetch(self, _url: str) -> bool:
                return False

        records = asyncio.run(
            make_linked_pdf_records(
                [
                    {
                        "url": "https://www.diem.unisa.it/uploads/verbale-colloquio-bando.pdf",
                        "text": "PDF",
                    },
                    {
                        "url": "https://www.diem.unisa.it/uploads/avviso-differimento-colloquio-borsa.pdf",
                        "text": "PDF",
                    },
                    {
                        "url": "https://www.diem.unisa.it/uploads/elenco-progetti-con-borsa.pdf",
                        "text": "PDF",
                    },
                    {
                        "url": "https://www.diem.unisa.it/uploads/dr-scorrimento-informatica-ii-bando.pdf",
                        "text": "PDF",
                    },
                ],
                "https://www.diem.unisa.it/home/bandi",
                1,
                DenyRobots(),
                base_config(),
            )
        )

        self.assertEqual([record["status"] for record in records], ["robots_denied"] * 4)

    def test_teaching_operations_pdf_is_allowed_from_didactics(self) -> None:
        class DenyRobots:
            async def can_fetch(self, _url: str) -> bool:
                return False

        config = base_config()
        config["crawler"]["allowed_domains"].append("corsi.unisa.it")
        config["crawler"]["pdf_allowed_domains"].append("corsi.unisa.it")
        config["crawler"]["per_domain_limits"]["corsi.unisa.it"] = 10
        records = asyncio.run(
            make_linked_pdf_records(
                [
                    {
                        "url": "https://corsi.unisa.it/uploads/calendario-attivita-didattiche.pdf",
                        "text": "PDF",
                    }
                ],
                "https://corsi.unisa.it/ingegneria-informatica/didattica/calendari",
                2,
                DenyRobots(),
                config,
            )
        )

        self.assertEqual(records[0]["status"], "pending_download")
        self.assertEqual(
            records[0]["pdf_download_decision"],
            "allowed_teaching_operations_document",
        )

    def test_informative_rescue_calendar_is_allowed_but_generic_rescue_calendar_is_not(self) -> None:
        class DenyRobots:
            async def can_fetch(self, _url: str) -> bool:
                return False

        records = asyncio.run(
            make_linked_pdf_records(
                [
                    {
                        "url": "https://www.diem.unisa.it/uploads/calendario-prove-in-itinere.pdf",
                        "text": "PDF",
                    }
                ],
                (
                    "https://www.diem.unisa.it/unisa-rescue-page/dettaglio/id/1402/"
                    "module/475/row/29463/calendario-prove-in-itinere-diem"
                ),
                2,
                DenyRobots(),
                base_config(),
            )
        )
        generic_records = asyncio.run(
            make_linked_pdf_records(
                [
                    {
                        "url": "https://www.diem.unisa.it/uploads/calendario-evento.pdf",
                        "text": "PDF",
                    }
                ],
                (
                    "https://www.diem.unisa.it/unisa-rescue-page/dettaglio/id/1413/"
                    "module/487/row/3850/incontro-interdisciplinare"
                ),
                2,
                DenyRobots(),
                base_config(),
            )
        )

        self.assertEqual(records[0]["status"], "pending_download")
        self.assertEqual(
            records[0]["pdf_download_decision"],
            "allowed_informative_calendar_document",
        )
        self.assertEqual(generic_records[0]["status"], "robots_denied")

    def test_didactic_focus_pdf_attachments_are_allowed_when_they_match_parent_id(self) -> None:
        class DenyRobots:
            async def can_fetch(self, _url: str) -> bool:
                return False

        records = asyncio.run(
            make_linked_pdf_records(
                [
                    {
                        "url": "https://www.diem.unisa.it/uploads/rescue/502/1439/ai-applications.pdf",
                        "text": "AI APPLICATIONS PDF Altri formati",
                    },
                    {
                        "url": "https://www.diem.unisa.it/uploads/rescue/502/9999/ai-applications.pdf",
                        "text": "AI APPLICATIONS PDF Altri formati",
                    },
                ],
                "https://www.diem.unisa.it/didattica/focus?id=1439",
                2,
                DenyRobots(),
                base_config(),
            )
        )

        self.assertEqual(records[0]["status"], "pending_download")
        self.assertEqual(
            records[0]["pdf_download_decision"],
            "allowed_didactic_focus_document",
        )
        self.assertEqual(records[1]["status"], "out_of_scope")
        self.assertEqual(records[1]["skip_reason"], "diem_rescue_upload_untrusted")

    def test_international_program_pdf_is_allowed_from_international_section(self) -> None:
        class DenyRobots:
            async def can_fetch(self, _url: str) -> bool:
                return False

        records = asyncio.run(
            make_linked_pdf_records(
                [
                    {
                        "url": "https://www.diem.unisa.it/uploads/accordi-erasmus.pdf",
                        "text": "Accordi Erasmus",
                    }
                ],
                "https://www.diem.unisa.it/international",
                1,
                DenyRobots(),
                base_config(),
            )
        )

        self.assertEqual(records[0]["status"], "pending_download")
        self.assertEqual(
            records[0]["pdf_download_decision"],
            "allowed_international_program_document",
        )

    def test_international_rescue_pdf_endpoint_is_allowed_from_international_section(self) -> None:
        class DenyRobots:
            async def can_fetch(self, _url: str) -> bool:
                return False

        config = base_config()
        parent = "https://www.diem.unisa.it/international/accordi-erasmus-plus"
        pdf_url = (
            "https://www.diem.unisa.it/unisa-rescue-page/pdf/id/1463/module/209"
            "?paese=&struttura=300638&matricola="
        )

        traverse_ok, traverse_reason = can_traverse_url(
            pdf_url,
            config,
            {"discovered_from": parent},
        )
        records = asyncio.run(
            make_linked_pdf_records(
                [{"url": pdf_url, "text": "PDF"}],
                parent,
                1,
                DenyRobots(),
                config,
            )
        )

        self.assertTrue(traverse_ok, traverse_reason)
        self.assertEqual(records[0]["status"], "pending_download")
        self.assertEqual(records[0]["pdf_source_section"], "international")
        self.assertEqual(
            records[0]["pdf_download_decision"],
            "allowed_international_program_document",
        )

    def test_course_evidence_pdf_is_allowed_only_from_in_scope_course_pages(self) -> None:
        class DenyRobots:
            async def can_fetch(self, _url: str) -> bool:
                return False

        config = base_config()
        config["crawler"]["allowed_domains"].append("corsi.unisa.it")
        config["crawler"]["pdf_allowed_domains"].append("corsi.unisa.it")
        config["crawler"]["pdf_allowed_domains"].append("www.unisa.it")
        config["crawler"]["per_domain_limits"]["corsi.unisa.it"] = 10
        records = asyncio.run(
            make_linked_pdf_records(
                [
                    {
                        "url": "https://www.unisa.it/uploads/rescue/__schede-sua/2025/0650.pdf",
                        "text": "Scheda completa SUA-CDS",
                    },
                    {
                        "url": "https://corsi.unisa.it/uploads/rescue/__almalaurea/2025/0650.pdf",
                        "text": "AlmaLaurea",
                    },
                ],
                "https://corsi.unisa.it/ingegneria-informatica/qualita",
                2,
                DenyRobots(),
                config,
            )
        )

        self.assertEqual([record["status"] for record in records], ["pending_download"] * 2)
        self.assertEqual(
            [record["pdf_download_decision"] for record in records],
            ["allowed_course_evidence_document", "allowed_course_evidence_document"],
        )

    def test_central_decree_requires_descriptive_context(self) -> None:
        class DenyRobots:
            async def can_fetch(self, _url: str) -> bool:
                return False

        records = asyncio.run(
            make_linked_pdf_records(
                [
                    {
                        "url": "https://www.diem.unisa.it/uploads/decreto.pdf",
                        "text": "Decreto",
                    },
                    {
                        "url": "https://www.diem.unisa.it/uploads/decreto-regolamento-didattico.pdf",
                        "text": "Decreto approvazione regolamento didattico",
                    },
                ],
                "https://www.diem.unisa.it/didattica",
                1,
                DenyRobots(),
                base_config(),
            )
        )

        self.assertEqual(records[0]["status"], "robots_denied")
        self.assertEqual(records[1]["status"], "pending_download")
        self.assertEqual(records[1]["pdf_download_decision"], "allowed_stable_document")

    def test_pdf_exception_keeps_bandi_and_decreti_blocked(self) -> None:
        class DenyRobots:
            async def can_fetch(self, _url: str) -> bool:
                return False

        bandi_records = asyncio.run(
            make_linked_pdf_records(
                [
                    {
                        "url": "https://www.diem.unisa.it/uploads/guida.pdf",
                        "text": "Guida",
                    }
                ],
                "https://www.diem.unisa.it/home/bandi",
                1,
                DenyRobots(),
                base_config(),
            )
        )
        didattica_decreto = asyncio.run(
            make_linked_pdf_records(
                [
                    {
                        "url": "https://www.diem.unisa.it/uploads/decreto.pdf",
                        "text": "Decreto",
                    }
                ],
                "https://www.diem.unisa.it/didattica",
                1,
                DenyRobots(),
                base_config(),
            )
        )

        self.assertEqual(bandi_records[0]["status"], "robots_denied")
        self.assertEqual(bandi_records[0]["pdf_source_section"], "home_bandi")
        self.assertEqual(didattica_decreto[0]["status"], "robots_denied")
        self.assertEqual(didattica_decreto[0]["pdf_match_keywords"], ["decreto"])

    def test_no_eligible_urls_wins_after_queue_is_drained(self) -> None:
        state = CrawlState(
            queue=deque(),
            queued=set(),
            visited=set(),
            seen_documents=set(),
            domain_counts=Counter(),
        )

        self.assertEqual(
            discovery_stop_reason(state, base_config(), exhausted_without_batch=True),
            "no_eligible_urls",
        )

    def test_recent_known_urls_are_counted_when_batch_is_empty(self) -> None:
        item = CrawlItem("https://www.diem.unisa.it/", 0, "seed")
        state = CrawlState(
            queue=deque([item]),
            queued={item.url},
            visited=set(),
            seen_documents=set(),
            domain_counts=Counter(),
            known_urls={item.url: "2026-05-16T00:00:00+00:00"},
            known_documents={item.url: "2026-05-16T00:00:00+00:00"},
        )
        skip_counts: Counter[str] = Counter()

        batch = take_batch(
            state,
            base_config(),
            reference_time=datetime(2026, 5, 16, tzinfo=UTC),
            skip_counts=skip_counts,
        )

        self.assertEqual(batch, [])
        self.assertEqual(skip_counts, {"recently_known": 1})

    def test_recent_known_url_without_document_can_be_retried(self) -> None:
        item = CrawlItem("https://www.diem.unisa.it/retry", 1, "parent")
        state = CrawlState(
            queue=deque([item]),
            queued={item.url},
            visited=set(),
            seen_documents=set(),
            domain_counts=Counter(),
            known_urls={item.url: "2026-05-16T00:00:00+00:00"},
            known_documents={},
        )

        batch = take_batch(
            state,
            base_config(),
            reference_time=datetime(2026, 5, 16, tzinfo=UTC),
        )

        self.assertEqual(batch, [item])

    def test_recent_known_document_is_not_requeued_from_links(self) -> None:
        parent = CrawlItem("https://www.diem.unisa.it/parent", 0, "seed")
        known = "https://www.diem.unisa.it/known"
        retry = "https://www.diem.unisa.it/retry"
        state = CrawlState(
            queue=deque(),
            queued=set(),
            visited=set(),
            seen_documents=set(),
            domain_counts=Counter(),
            known_urls={
                known: "2026-05-16T00:00:00+00:00",
                retry: "2026-05-16T00:00:00+00:00",
            },
            known_documents={known: "2026-05-16T00:00:00+00:00"},
        )

        enqueue_links(parent, [known, retry], state, base_config())

        self.assertEqual([item.url for item in state.queue], [retry])

    def test_domain_limited_urls_remain_in_frontier_for_future_runs(self) -> None:
        item = CrawlItem("https://www.diem.unisa.it/pending", 4, "parent")
        state = CrawlState(
            queue=deque([item]),
            queued={item.url},
            visited=set(),
            seen_documents=set(),
            domain_counts=Counter({"www.diem.unisa.it": 10}),
        )
        skip_counts: Counter[str] = Counter()
        skip_counts_by_depth: Counter[tuple[str, int]] = Counter()

        batch = take_batch(
            state,
            base_config(),
            skip_counts=skip_counts,
            skip_counts_by_depth=skip_counts_by_depth,
        )

        self.assertEqual(batch, [])
        self.assertEqual(list(state.queue), [item])
        self.assertEqual(state.queued, {item.url})
        self.assertEqual(skip_counts, {"domain_limit": 1})
        self.assertEqual(skip_counts_by_depth, {("domain_limit", 4): 1})

    def test_terminal_forced_reexpansion_is_removed_from_backlog(self) -> None:
        item = CrawlItem(
            "https://www.diem.unisa.it/home/bandi?struttura=000000",
            1,
            "https://www.diem.unisa.it/home/bandi",
            force_revisit=True,
        )
        state = CrawlState(
            queue=deque([item]),
            queued={item.url},
            visited=set(),
            seen_documents=set(),
            domain_counts=Counter(),
            expansion_backlog={item.url: item},
        )
        skip_counts: Counter[str] = Counter()
        skip_counts_by_depth: Counter[tuple[str, int]] = Counter()

        batch = take_batch(
            state,
            base_config(),
            skip_counts=skip_counts,
            skip_counts_by_depth=skip_counts_by_depth,
        )

        self.assertEqual(batch, [])
        self.assertEqual(list(state.queue), [])
        self.assertEqual(state.queued, set())
        self.assertEqual(state.expansion_backlog, {})
        self.assertEqual(skip_counts, {"blocked_query": 1})
        self.assertEqual(skip_counts_by_depth, {("blocked_query", 1): 1})

    def test_diem_bandi_archive_urls_cover_all_target_sections_and_years(self) -> None:
        config = base_config()
        config["crawler"]["diem_bandi_archive_start_year"] = 2025
        reference_time = datetime(2026, 5, 22, tzinfo=UTC)

        urls = diem_bandi_archive_urls(config, reference_time)

        expected = {
            "https://www.diem.unisa.it/home/bandi?modulo=139&struttura=300638",
            "https://www.diem.unisa.it/home/bandi?anno=2025&modulo=139&struttura=300638",
            "https://www.diem.unisa.it/home/bandi?modulo=504&struttura=300638",
            "https://www.diem.unisa.it/home/bandi?modulo=316&struttura=300638",
            "https://www.diem.unisa.it/home/bandi?modulo=226",
            "https://www.diem.unisa.it/home/bandi?anno=2026&modulo=226",
            "https://www.diem.unisa.it/home/bandi?modulo=67&struttura=300638",
            "https://www.diem.unisa.it/home/bandi?modulo=292&struttura=300638",
            "https://www.diem.unisa.it/home/bandi?cdsStruttura=300638&modulo=293",
            "https://www.diem.unisa.it/home/bandi?anno=2025&cdsStruttura=300638&modulo=293",
            "https://www.diem.unisa.it/home/bandi?modulo=505&struttura=300638",
        }

        self.assertTrue(expected <= set(urls))

    def test_missing_diem_bandi_archives_are_bootstrapped_without_readding_recent_documents(self) -> None:
        config = base_config()
        config["crawler"]["diem_bandi_archive_start_year"] = 2026
        known = "https://www.diem.unisa.it/home/bandi?anno=2026&modulo=139&struttura=300638"
        persistent = PersistentDiscoveryState(
            frontier=deque(),
            known_urls={known: "2026-05-20T00:00:00+00:00"},
            known_documents={known: "2026-05-20T00:00:00+00:00"},
        )

        items = missing_diem_bandi_archive_items(
            persistent,
            config,
            datetime(2026, 5, 22, tzinfo=UTC),
        )
        urls = [item.url for item in items]

        self.assertNotIn(known, urls)
        self.assertIn(
            "https://www.diem.unisa.it/home/bandi?anno=2026&modulo=226",
            urls,
        )
        self.assertIn(
            "https://www.diem.unisa.it/home/bandi?anno=2026&cdsStruttura=300638&modulo=293",
            urls,
        )

    def test_depth_increase_requeues_boundary_pages_for_forced_revisit(self) -> None:
        boundary = CrawlItem("https://www.diem.unisa.it/a", 1, "parent")
        persistent = PersistentDiscoveryState(
            frontier=deque(),
            known_urls={boundary.url: "2026-05-16T00:00:00+00:00"},
            known_documents={},
            expansion_backlog=deque([boundary]),
        )

        same_depth = create_initial_state([], [], persistent, max_depth=1)
        deeper = create_initial_state([], [], persistent, max_depth=2)

        self.assertEqual(list(same_depth.queue), [])
        self.assertEqual(len(deeper.queue), 1)
        self.assertTrue(deeper.queue[0].force_revisit)

    def test_recent_seed_is_not_requeued_while_persistent_work_exists(self) -> None:
        seed = "https://www.diem.unisa.it/"
        pending = CrawlItem("https://www.diem.unisa.it/pending", 3, "parent")
        persistent = PersistentDiscoveryState(
            frontier=deque([pending]),
            known_urls={seed: "2026-05-16T00:00:00+00:00"},
            known_documents={seed: "2026-05-16T00:00:00+00:00"},
        )

        state = create_initial_state(
            [seed],
            [],
            persistent,
            max_depth=3,
            config=base_config(),
            reference_time=datetime(2026, 5, 16, tzinfo=UTC),
        )

        self.assertEqual(list(state.queue), [pending])

    def test_recent_seed_without_document_is_requeued(self) -> None:
        seed = "https://www.diem.unisa.it/"
        persistent = PersistentDiscoveryState(
            frontier=deque(),
            known_urls={seed: "2026-05-16T00:00:00+00:00"},
            known_documents={},
        )

        state = create_initial_state(
            [seed],
            [],
            persistent,
            max_depth=3,
            config=base_config(),
            reference_time=datetime(2026, 5, 16, tzinfo=UTC),
        )

        self.assertEqual([item.url for item in state.queue], [seed])

    def test_missing_teacher_profiles_from_known_rubrica_are_bootstrapped(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].extend(
            ["rubrica.unisa.it", "docenti.unisa.it"]
        )
        config["crawler"]["per_domain_limits"]["rubrica.unisa.it"] = 10
        config["crawler"]["per_domain_limits"]["docenti.unisa.it"] = 10
        config["scope"]["directory_domain"] = "rubrica.unisa.it"
        directory_url = "https://rubrica.unisa.it/persone?matricola=058553"
        profile_url = "https://docenti.unisa.it/058553/home"
        persistent = PersistentDiscoveryState(
            frontier=deque(),
            known_urls={
                directory_url: "2026-05-16T00:00:00+00:00",
                profile_url: "2026-05-16T00:00:00+00:00",
            },
            known_documents={directory_url: "2026-05-16T00:00:00+00:00"},
        )

        state = create_initial_state(
            [],
            [],
            persistent,
            max_depth=4,
            config=config,
            reference_time=datetime(2026, 5, 16, tzinfo=UTC),
        )

        self.assertEqual([item.url for item in state.queue], [profile_url])
        self.assertIn("058553", state.allowed_teacher_profiles)

    def test_known_teacher_document_is_not_bootstrapped_from_rubrica(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].extend(
            ["rubrica.unisa.it", "docenti.unisa.it"]
        )
        config["scope"]["directory_domain"] = "rubrica.unisa.it"
        directory_url = "https://rubrica.unisa.it/persone?matricola=058553"
        profile_url = "https://docenti.unisa.it/058553/home"
        persistent = PersistentDiscoveryState(
            frontier=deque(),
            known_urls={directory_url: "2026-05-16T00:00:00+00:00"},
            known_documents={
                directory_url: "2026-05-16T00:00:00+00:00",
                profile_url: "2026-05-16T00:00:00+00:00",
            },
        )

        state = create_initial_state([], [], persistent, max_depth=4, config=config)

        self.assertEqual(list(state.queue), [])

    def test_forced_revisit_bypasses_refresh_window(self) -> None:
        item = CrawlItem("https://www.diem.unisa.it/a", 1, "parent", force_revisit=True)
        state = CrawlState(
            queue=deque([item]),
            queued={item.url},
            visited=set(),
            seen_documents=set(),
            domain_counts=Counter(),
            known_urls={item.url: "2026-05-16T00:00:00+00:00"},
        )

        batch = take_batch(
            state,
            base_config(),
            reference_time=datetime(2026, 5, 16, tzinfo=UTC),
        )

        self.assertEqual(batch, [item])

    def test_discovery_state_preserves_expansion_backlog(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"paths": {"discovery_state_file": str(Path(temp_dir) / "state.json")}}
            boundary = CrawlItem("https://www.diem.unisa.it/a", 1, "parent")
            state = CrawlState(
                queue=deque(),
                queued=set(),
                visited=set(),
                seen_documents=set(),
                domain_counts=Counter(),
                expansion_backlog={boundary.url: boundary},
            )

            save_discovery_state(state, config)
            loaded = load_discovery_state(config)

        self.assertEqual(list(loaded.expansion_backlog), [boundary])

    def test_discovery_state_preserves_allowed_teacher_profiles(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            config = {"paths": {"discovery_state_file": str(Path(temp_dir) / "state.json")}}
            state = CrawlState(
                queue=deque(),
                queued=set(),
                visited=set(),
                seen_documents=set(),
                domain_counts=Counter(),
                allowed_teacher_profiles={"mario.rossi"},
            )

            save_discovery_state(state, config)
            loaded = load_discovery_state(config)

        self.assertEqual(loaded.allowed_teacher_profiles, {"mario.rossi"})

    def test_personnel_page_registers_teacher_profile(self) -> None:
        config = base_config()
        config["crawler"]["allowed_domains"].append("docenti.unisa.it")
        config["crawler"]["per_domain_limits"]["docenti.unisa.it"] = 10
        parent = CrawlItem(
            "https://www.diem.unisa.it/dipartimento/personale",
            1,
            "https://www.diem.unisa.it/dipartimento",
        )
        teacher = "https://docenti.unisa.it/mario.rossi/home"
        state = CrawlState(
            queue=deque(),
            queued=set(),
            visited=set(),
            seen_documents=set(),
            domain_counts=Counter(),
        )

        enqueue_links(parent, [teacher], state, config)

        self.assertEqual(state.allowed_teacher_profiles, {"mario.rossi"})
        self.assertEqual(list(state.queue), [CrawlItem(teacher, 2, parent.url)])

    def test_enqueue_links_propagates_origin_seed_for_new_runs(self) -> None:
        seed = "https://www.diem.unisa.it/"
        parent = CrawlItem(
            "https://www.diem.unisa.it/dipartimento",
            1,
            seed,
            origin_seed=seed,
        )
        child = "https://www.diem.unisa.it/dipartimento/personale"
        state = CrawlState(
            queue=deque(),
            queued=set(),
            visited=set(),
            seen_documents=set(),
            domain_counts=Counter(),
        )

        enqueue_links(parent, [child], state, base_config())

        self.assertEqual(
            list(state.queue),
            [CrawlItem(child, 2, parent.url, origin_seed=seed)],
        )

    def test_boundary_page_moves_in_and_out_of_expansion_backlog(self) -> None:
        item = CrawlItem("https://www.diem.unisa.it/a", 1, "parent")
        state = CrawlState(
            queue=deque(),
            queued=set(),
            visited=set(),
            seen_documents=set(),
            domain_counts=Counter(),
        )
        record = {"type": "html", "status": "ok"}

        update_expansion_backlog(item, record, expand_links=True, expanded=False, state=state)
        self.assertEqual(state.expansion_backlog[item.url], item)

        revisit = CrawlItem(item.url, item.depth, item.discovered_from, force_revisit=True)
        update_expansion_backlog(revisit, record, expand_links=True, expanded=True, state=state)
        self.assertEqual(state.expansion_backlog, {})


if __name__ == "__main__":
    unittest.main()
