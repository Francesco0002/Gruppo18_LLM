from __future__ import annotations

"""Test del verdetto di copertura BFS per depth e della sua evoluzione tra run."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingest import build_depth_coverage  # noqa: E402


class DepthCoverageTests(unittest.TestCase):
    def test_pending_depth_is_incomplete(self) -> None:
        coverage = build_depth_coverage(
            max_depth=2,
            frontier_by_depth={"2": 3},
            checkpoint_status="completed",
            stop_reason="max_total_urls",
        )

        self.assertEqual(coverage["verdict"], "incomplete")
        self.assertFalse(coverage["depths"][2]["complete"])
        self.assertEqual(coverage["depths"][2]["pending_at_depth"], 3)

    def test_child_depth_stays_unsealed_while_parent_is_pending(self) -> None:
        coverage = build_depth_coverage(
            max_depth=2,
            frontier_by_depth={"1": 2},
            checkpoint_status="completed",
            stop_reason="max_total_urls",
        )

        self.assertFalse(coverage["depths"][2]["sealed"])
        self.assertFalse(coverage["depths"][2]["complete"])
        self.assertEqual(coverage["depths"][2]["pending_below_depth"], 2)

    def test_drained_depths_are_complete(self) -> None:
        coverage = build_depth_coverage(
            max_depth=2,
            frontier_by_depth={},
            checkpoint_status="completed",
            stop_reason="queue_exhausted",
        )

        self.assertEqual(coverage["verdict"], "complete")
        self.assertTrue(all(depth["complete"] for depth in coverage["depths"]))

    def test_no_eligible_urls_is_incomplete(self) -> None:
        coverage = build_depth_coverage(
            max_depth=2,
            frontier_by_depth={},
            checkpoint_status="completed",
            stop_reason="no_eligible_urls",
        )

        self.assertEqual(coverage["verdict"], "incomplete")
        self.assertEqual(coverage["blocking_reasons"], ["stop_reason:no_eligible_urls"])

    def test_depth_progresses_across_simulated_runs(self) -> None:
        run_one = build_depth_coverage(
            max_depth=2,
            frontier_by_depth={"2": 5},
            checkpoint_status="completed",
            stop_reason="max_total_urls",
        )
        run_two = build_depth_coverage(
            max_depth=2,
            frontier_by_depth={"2": 2},
            checkpoint_status="completed",
            stop_reason="max_total_urls",
        )
        run_three = build_depth_coverage(
            max_depth=2,
            frontier_by_depth={},
            checkpoint_status="completed",
            stop_reason="queue_exhausted",
        )

        self.assertEqual(
            [
                run_one["depths"][2]["pending_at_depth"],
                run_two["depths"][2]["pending_at_depth"],
                run_three["depths"][2]["pending_at_depth"],
            ],
            [5, 2, 0],
        )
        self.assertEqual(
            [run_one["verdict"], run_two["verdict"], run_three["verdict"]],
            ["incomplete", "incomplete", "complete"],
        )


if __name__ == "__main__":
    unittest.main()
