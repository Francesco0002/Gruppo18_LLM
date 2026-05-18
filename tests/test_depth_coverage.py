from __future__ import annotations

"""Test del verdetto di copertura BFS per depth e della sua evoluzione tra run."""

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ingest import (  # noqa: E402
    build_depth_coverage,
    build_extraction_stats,
    build_pdf_coverage,
    compact_seed_rows,
    compact_depth_rows,
    discovery_failures_by_kind,
)
from extract_pdf import build_manifest_record  # noqa: E402
from markdown_cleaner import clean_markdown, clean_markdown_body  # noqa: E402


class DepthCoverageTests(unittest.TestCase):
    def test_empty_structured_pdf_is_warned_even_when_it_is_long_enough(self) -> None:
        raw = "\n".join(
            [
                "## **Visiting Professors - Elenco degli Accordi**",
                "",
                "|||||Ultimo Aggiornamento: 0 0000|Ultimo Aggiornamento: 0 0000||||",
                "|---|---|---|---|---|---|---|---|---|",
                "|**Dipartimento**|**Visiting Professor**|**Attività**|**Paese**||**Università di**|**Host Professor**|**Data Inizio**|**Data Fine**|",
                "||||||**Provenienza**||||",
            ]
        )

        clean_body, _, quality = clean_markdown(raw, source="pdf", metadata={})

        self.assertGreaterEqual(len(clean_body), 100)
        self.assertEqual(quality["clean_status"], "warning")
        self.assertIn("empty_structured_pdf", quality["clean_warnings"])

    def test_structured_pdf_with_a_real_data_row_stays_meaningful(self) -> None:
        raw = "\n".join(
            [
                "## **Visiting Professors - Elenco degli Accordi**",
                "",
                "|**Dipartimento**|**Visiting Professor**|**Paese**|",
                "|---|---|---|",
                "|DIEM|Mario Rossi|Italia|",
            ]
        )

        _, _, quality = clean_markdown(raw, source="pdf", metadata={})

        self.assertNotIn("empty_structured_pdf", quality["clean_warnings"])

    def test_empty_structured_pdf_is_not_indexable_in_manifest(self) -> None:
        raw = "\n".join(
            [
                "## **Staff Teaching - Elenco degli Accordi**",
                "",
                "Ultimo Aggiornamento: 0 0000",
                "",
                "|**Dipartimento**|**Docente**|**Paese**|",
                "|---|---|---|",
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            record = build_manifest_record(
                {
                    "record": {
                        "url": "https://www.diem.unisa.it/example.pdf",
                        "hash": "abc123",
                    },
                    "status": "downloaded",
                    "raw_pdf_path": "data/raw_pdf/ab/abc123.pdf",
                },
                markdown=raw,
                config={
                    "paths": {
                        "processed_raw_markdown_dir": str(base / "markdown_raw"),
                        "processed_markdown_dir": str(base / "markdown"),
                    }
                },
            )

        self.assertFalse(record["text_extracted"])
        self.assertFalse(record["indexable"])
        self.assertIn("empty_structured_pdf", record["clean_warnings"])

    def test_pdf_cleaning_does_not_truncate_after_university_contact_line(self) -> None:
        raw = "\n".join(
            [
                "# Titolo",
                "Università degli Studi di Salerno Via Giovanni Paolo II",
                "## Art. 2",
                "Contenuto che deve restare nel PDF.",
            ]
        )

        cleaned, warnings = clean_markdown_body(raw, "pdf")

        self.assertIn("## Art. 2", cleaned)
        self.assertIn("Contenuto che deve restare nel PDF.", cleaned)
        self.assertEqual(warnings, [])

    def test_html_cleaning_still_truncates_known_footer(self) -> None:
        raw = "\n".join(
            [
                "# Titolo",
                "Università degli Studi di Salerno",
                "Riga di footer",
            ]
        )

        cleaned, warnings = clean_markdown_body(raw, "html")

        self.assertEqual(cleaned, "# Titolo")
        self.assertEqual(warnings, ["removed_boilerplate_lines:2"])

    def test_html_cleaning_removes_english_navigation_and_footer(self) -> None:
        raw = "\n".join(
            [
                "Share",
                "# Title",
                "University of Salerno - Via Giovanni Paolo II",
                "Footer row",
            ]
        )

        cleaned, warnings = clean_markdown_body(raw, "html")

        self.assertEqual(cleaned, "# Title")
        self.assertEqual(warnings, ["removed_boilerplate_lines:3"])

    def test_compact_depth_rows_merge_run_and_coverage_metrics(self) -> None:
        coverage = build_depth_coverage(
            max_depth=1,
            frontier_by_depth={"1": 2},
            checkpoint_status="completed",
            stop_reason="max_total_urls",
        )

        self.assertEqual(
            compact_depth_rows(
                coverage["depths"],
                html_found_by_depth={"1": 3},
                pdf_found_by_depth={"1": 2},
                visited_html_by_depth={"1": 3},
            ),
            [
                {
                    "depth": 0,
                    "html_found_in_run": 0,
                    "pdf_found_in_run": 0,
                    "visited_html_in_run": 0,
                    "pending_new": 0,
                    "pending_reexpansion": 0,
                    "pending_from_lower_depths": 0,
                    "complete": True,
                    "visited_complete": True,
                },
                {
                    "depth": 1,
                    "html_found_in_run": 3,
                    "pdf_found_in_run": 2,
                    "visited_html_in_run": 3,
                    "pending_new": 2,
                    "pending_reexpansion": 0,
                    "pending_from_lower_depths": 0,
                    "complete": False,
                    "visited_complete": False,
                },
            ],
        )

    def test_compact_seed_rows_shows_branch_balance(self) -> None:
        self.assertEqual(
            compact_seed_rows(
                {"seed-a": {"0": 1, "1": 3}, "seed-b": {"0": 1, "1": 1}},
                {"seed-a": {"1": 1}},
                {"seed-a": {"0": 1, "1": 3}, "seed-b": {"0": 1, "1": 1}},
                {"seed-a": {"1": 1}},
            ),
            [
                {
                    "seed": "seed-a",
                    "html_found_in_run_by_depth": {"0": 1, "1": 3},
                    "pdf_found_in_run_by_depth": {"1": 1},
                    "visited_html_in_run_by_depth": {"0": 1, "1": 3},
                    "pending_new_by_depth": {"1": 1},
                },
                {
                    "seed": "seed-b",
                    "html_found_in_run_by_depth": {"0": 1, "1": 1},
                    "pdf_found_in_run_by_depth": {},
                    "visited_html_in_run_by_depth": {"0": 1, "1": 1},
                    "pending_new_by_depth": {},
                },
            ],
        )

    def test_extraction_stats_keep_only_readable_run_counters(self) -> None:
        self.assertEqual(
            build_extraction_stats(
                {
                    "scrape": {
                        "html_candidates": 4,
                        "processed_ok": 3,
                        "failed": 1,
                        "manifest": "ignored",
                    },
                    "extract_pdf": {
                        "pdf_pending_discovered": 5,
                        "skipped_recent": 1,
                        "pdf_candidates": 4,
                        "ready_for_extraction": 4,
                        "extracted_ok": 3,
                        "failed": 1,
                        "too_large": 0,
                        "manifest": "ignored",
                    },
                }
            ),
            {
                "html": {
                    "candidates": 4,
                    "processed_ok": 3,
                    "failed": 1,
                },
                "pdf": {
                    "summary": {
                        "pending_found": 5,
                        "selected_for_processing": 4,
                        "extracted_ok": 3,
                        "failed": 1,
                        "too_large": 0,
                    },
                    "volume": {
                        "skipped_recent": 1,
                        "ready_for_extraction": 4,
                        "network_downloaded": 0,
                        "reused_raw": 0,
                        "downloaded_bytes": 0,
                    },
                    "performance": {
                        "elapsed_seconds": 0,
                        "download_seconds": 0,
                        "extraction_seconds": 0,
                        "throughput_pdf_per_minute": 0,
                        "executor_backend": None,
                    },
                    "errors": {
                        "failure_kinds": {},
                    },
                },
            },
        )

    def test_pdf_coverage_summarizes_denied_sections_and_keywords(self) -> None:
        records = [
            {
                "type": "pdf",
                "status": "robots_denied",
                "url": "https://www.diem.unisa.it/uploads/bando-graduatoria.pdf",
                "discovered_from": "https://www.diem.unisa.it/home/bandi?anno=2024",
            },
            {
                "type": "pdf",
                "status": "robots_denied",
                "url": "https://www.diem.unisa.it/uploads/regolamento.pdf",
                "discovered_from": "https://www.diem.unisa.it/didattica",
            },
            {
                "type": "pdf",
                "status": "pending_download",
                "url": "https://www.diem.unisa.it/uploads/guida.pdf",
                "discovered_from": "https://www.diem.unisa.it/didattica",
                "pdf_source_section": "didattica",
                "pdf_download_decision": "allowed_stable_document",
                "pdf_match_keywords": ["guida"],
            },
            {
                "type": "pdf",
                "status": "pending_download",
                "url": "https://www.diem.unisa.it/uploads/verbale-bando.pdf",
                "discovered_from": "https://www.diem.unisa.it/home/bandi?anno=2024",
                "pdf_source_section": "home_bandi",
                "pdf_download_decision": "allowed_opportunity_document",
                "pdf_match_keywords": ["bando"],
            },
        ]

        self.assertEqual(
            build_pdf_coverage(records),
            {
                "summary": {
                    "found": 4,
                    "allowed_by_policy": 2,
                    "blocked_by_robots": 2,
                    "review_candidates": 1,
                    "intentionally_excluded": 1,
                },
                "allowed_by_policy": {
                    "by_section": {
                        "didattica": 1,
                        "home_bandi": 1,
                    },
                    "by_keyword": {
                        "bando": 1,
                        "guida": 1,
                    },
                },
                "blocked_by_robots": {
                    "records": 2,
                    "unique_pdfs": 2,
                    "by_section": {
                        "home_bandi": 1,
                        "didattica": 1,
                    },
                    "by_keyword": {
                        "bando": 1,
                        "graduatoria": 1,
                        "regolamento": 1,
                    },
                },
                "allowed_suspicious_attachments": {
                    "count": 1,
                    "by_section": {
                        "home_bandi": 1,
                    },
                    "by_hint": {
                        "verbale": 1,
                    },
                },
                "blocked_review_candidates": {
                    "count": 1,
                    "by_section": {
                        "didattica": 1,
                    },
                    "by_keyword": {
                        "regolamento": 1,
                    },
                },
                "blocked_intentionally_excluded": {
                    "count": 1,
                    "by_section": {
                        "home_bandi": 1,
                    },
                    "by_keyword": {
                        "bando": 1,
                        "graduatoria": 1,
                    },
                },
                "blocked_other": {
                    "count": 0,
                    "by_section": {},
                    "by_keyword": {},
                },
            },
        )

    def test_discovery_failures_are_grouped_by_kind(self) -> None:
        records = [
            {"status": "failed", "error": "Client error '404 Not Found' for url x"},
            {"status": "failed", "error": "ReadTimeout timed out"},
            {"status": "failed", "error": "Server error '503 Service Unavailable' for url x"},
            {"status": "failed", "error": "Client error '403 Forbidden' for url x"},
            {"status": "failed", "error": "ConnectError"},
            {"status": "ok"},
        ]

        self.assertEqual(
            discovery_failures_by_kind(records),
            {
                "http_404": 1,
                "http_5xx": 1,
                "other": 1,
                "other_http": 1,
                "timeout": 1,
            },
        )

    def test_pending_depth_is_incomplete(self) -> None:
        coverage = build_depth_coverage(
            max_depth=2,
            frontier_by_depth={"2": 3},
            checkpoint_status="completed",
            stop_reason="max_total_urls",
        )

        self.assertEqual(coverage["verdict"], "incomplete")
        self.assertFalse(coverage["depths"][2]["complete"])
        self.assertEqual(coverage["depths"][2]["pending_new_at_depth"], 3)
        self.assertEqual(coverage["depths"][2]["pending_total_at_depth"], 3)

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

    def test_closed_without_visit_keeps_depth_not_visited_complete(self) -> None:
        coverage = build_depth_coverage(
            max_depth=2,
            frontier_by_depth={},
            checkpoint_status="completed",
            stop_reason="queue_exhausted",
            closed_without_visit_by_depth={"2": 3},
        )

        self.assertTrue(coverage["depths"][2]["complete"])
        self.assertFalse(coverage["depths"][2]["visited_complete"])
        self.assertEqual(coverage["depths"][2]["closed_without_visit"], 3)
        self.assertEqual(coverage["blocking_reasons"], ["closed_without_visit"])

    def test_reexpansion_backlog_keeps_next_depth_unsealed(self) -> None:
        coverage = build_depth_coverage(
            max_depth=2,
            frontier_by_depth={},
            checkpoint_status="completed",
            stop_reason="queue_exhausted",
            reexpansion_by_depth={"1": 2},
        )

        self.assertTrue(coverage["depths"][1]["complete"])
        self.assertFalse(coverage["depths"][2]["sealed"])
        self.assertEqual(coverage["depths"][1]["pending_reexpansion_at_depth"], 2)
        self.assertEqual(coverage["depths"][1]["pending_total_at_depth"], 2)
        self.assertEqual(coverage["blocking_reasons"], ["reexpansion_not_drained"])

    def test_reexpansions_do_not_double_count_pending_below_depth(self) -> None:
        coverage = build_depth_coverage(
            max_depth=3,
            frontier_by_depth={},
            checkpoint_status="completed",
            stop_reason="max_total_urls",
            reexpansion_by_depth={"2": 4},
        )

        self.assertEqual(coverage["depths"][2]["pending_new_at_depth"], 0)
        self.assertEqual(coverage["depths"][2]["pending_reexpansion_at_depth"], 4)
        self.assertEqual(coverage["depths"][2]["pending_total_at_depth"], 4)
        self.assertEqual(coverage["depths"][3]["pending_below_depth"], 4)

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
                run_one["depths"][2]["pending_new_at_depth"],
                run_two["depths"][2]["pending_new_at_depth"],
                run_three["depths"][2]["pending_new_at_depth"],
            ],
            [5, 2, 0],
        )
        self.assertEqual(
            [run_one["verdict"], run_two["verdict"], run_three["verdict"]],
            ["incomplete", "incomplete", "complete"],
        )


if __name__ == "__main__":
    unittest.main()
