from __future__ import annotations

import sys
import unittest
import asyncio
from collections import Counter, deque
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discover import (  # noqa: E402
    create_initial_state,
    discovery_stop_reason,
    enqueue_links,
    fair_bfs_order,
    take_batch,
    update_expansion_backlog,
)
from discovery_io import load_discovery_state, save_discovery_state  # noqa: E402
from discovery_models import CrawlItem, CrawlState, PersistentDiscoveryState  # noqa: E402
from discovery_processor import make_linked_pdf_records  # noqa: E402


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

        self.assertEqual(list(state.queue), [pending])

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
