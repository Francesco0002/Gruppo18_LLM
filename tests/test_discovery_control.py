from __future__ import annotations

import sys
import unittest
from collections import Counter, deque
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discover import (  # noqa: E402
    create_initial_state,
    discovery_stop_reason,
    take_batch,
    update_expansion_backlog,
)
from discovery_io import load_discovery_state, save_discovery_state  # noqa: E402
from discovery_models import CrawlItem, CrawlState, PersistentDiscoveryState  # noqa: E402


def base_config() -> dict:
    return {
        "crawler": {
            "allowed_domains": ["www.diem.unisa.it"],
            "max_total_urls": 10,
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
