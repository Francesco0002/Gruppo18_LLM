from __future__ import annotations

import sys
import unittest
from collections import Counter, deque
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from discover import discovery_stop_reason, take_batch  # noqa: E402
from discovery_models import CrawlItem, CrawlState  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
