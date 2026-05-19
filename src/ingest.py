"""
Orchestrazione della pipeline di ingest.

Questo script coordina:
1. discovery degli URL
2. conversione HTML in Markdown
3. estrazione PDF in Markdown
4. marcatura dei documenti duplicati nel manifest processed
5. generazione di un report compatto in data/processed/stats.json e nello
   storico run, con discovery del run, copertura per depth, estrazione e stato
   cumulativo del corpus processed

La fase di ingest non crea ancora chunk o embedding: prepara un corpus
Markdown pulito, misurabile e pronto per la fase di indexing.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from uuid import uuid4

from discover import run_discovery
from discovery_io import load_config, project_path, validate_config
from extract_pdf import run_extract_pdf_async
from ingest_stats import (
    blocked_pdf_category,
    build_depth_coverage,
    build_discovery_stats,
    build_extraction_stats,
    build_pdf_coverage,
    build_processed_stats,
    checkpoint_summary,
    closed_without_visit_by_depth_from_step_stats,
    compact_depth_rows,
    compact_seed_rows,
    config_snapshot,
    count_by,
    discovery_failure_kind,
    discovery_failures_by_kind,
    domain_from_record,
    frontier_by_depth_from_state,
    frontier_by_seed_and_depth_from_state,
    html_records_by_depth,
    http_failure_kind,
    is_main_opportunity_record,
    markdown_char_stats,
    pdf_failure_kind,
    pdf_failures_by_kind,
    pdf_filename_keywords,
    pdf_keyword_text,
    pdf_source_section,
    print_stats_summary,
    records_by_depth,
    records_by_seed_and_depth,
    reexpansion_by_depth_from_state,
    run_stats_path,
    suspicious_allowed_attachment_hints,
    write_stats,
)
from pipeline_io import latest_record_indexes, load_jsonl, write_jsonl_atomic
from pipeline_types import ProcessedRecord
from scrape import run_scrape


def make_run_id() -> str:
    """Identificativo breve del run, utile nei log e negli stats."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"ingest-{timestamp}-{uuid4().hex[:8]}"


def is_duplicate_candidate(record: ProcessedRecord) -> bool:
    """True se il record può essere confrontato tramite content_hash."""
    return (
        record.get("status") == "ok"
        and bool(record.get("index_markdown_path"))
        and record.get("text_extracted") is not False
        and bool(record.get("content_hash"))
    )


def mark_duplicate_documents(
    records: list[ProcessedRecord],
) -> tuple[list[ProcessedRecord], list[ProcessedRecord], dict]:
    """
    Marca i duplicati esatti sullo stato corrente del manifest.

    Il manifest è append-only: uno stesso URL può avere prima un failed e poi
    un ok. Le stats devono descrivere lo stato attuale, quindi i duplicati si
    calcolano solo sull'ultimo record disponibile per ogni source+url. A parità
    di content_hash, il record canonico preferisce HTML rispetto a PDF.
    """
    first_by_content_hash: dict[str, ProcessedRecord] = {}
    updated_records = [dict(record) for record in records]
    current_indexes = latest_record_indexes(updated_records)
    duplicate_count = 0
    unique_content_hashes = 0

    for record in updated_records:
        for field in ("duplicate_of", "duplicate_of_url", "duplicate_reason", "is_duplicate"):
            record.pop(field, None)

    from datetime import datetime as _dt

    def duplicate_priority(index: int) -> tuple[int, _dt, int]:
        record = updated_records[index]
        source_priority = {"html": 0, "pdf": 1}.get(str(record.get("source")), 2)
        crawled_str = str(record.get("last_crawled", ""))
        try:
            crawled_dt = _dt.fromisoformat(crawled_str.replace("Z", "+00:00"))
            if crawled_dt.tzinfo is None:
                # Le run piu vecchie possono contenere timestamp naive: li
                # trattiamo come UTC per confrontarli con quelli timezone-aware.
                crawled_dt = crawled_dt.replace(tzinfo=UTC)
        except ValueError:
            # Un timestamp corrotto non deve impedire di rigenerare gli stats.
            # Va in fondo all'ordinamento, ma il record resta osservabile.
            crawled_dt = _dt.min.replace(tzinfo=UTC)
        return source_priority, crawled_dt, index

    for index in sorted(current_indexes, key=duplicate_priority):
        record = updated_records[index]
        if not is_duplicate_candidate(record):
            continue

        content_hash = record["content_hash"]
        first_record = first_by_content_hash.get(content_hash)
        if first_record is None:
            first_by_content_hash[content_hash] = record
            record["is_duplicate"] = False
            unique_content_hashes += 1
        else:
            record["is_duplicate"] = True
            record["duplicate_of"] = first_record.get("hash")
            record["duplicate_of_url"] = first_record.get("url")
            record["duplicate_reason"] = "same_content_hash"
            duplicate_count += 1

    current_records = [updated_records[index] for index in current_indexes]
    return updated_records, current_records, {
        "duplicates": duplicate_count,
        "unique_content_hashes": unique_content_hashes,
        "history_records": len(updated_records),
    }


async def run_pipeline_steps(config: dict) -> dict[str, dict]:
    """Esegue gli step pesanti della pipeline e raccoglie le statistiche."""
    step_stats: dict[str, dict] = {}

    print("\n[1/5] Discovery")
    step_stats["discover"] = await run_discovery(config)

    print("\n[2/5] Scrape HTML")
    step_stats["scrape"] = await run_scrape()

    print("\n[3/5] Estrazione PDF")
    step_stats["extract_pdf"] = await run_extract_pdf_async(config)
    pdf_stats = step_stats["extract_pdf"]
    print(
        "PDF: "
        f"scoperti_in_attesa={pdf_stats['pdf_pending_discovered']}, "
        f"saltati_recenti={pdf_stats['skipped_recent']}, "
        f"da_processare={pdf_stats['pdf_candidates']}, "
        f"estratti_ok={pdf_stats['extracted_ok']}, "
        f"falliti={pdf_stats['failed']}, "
        f"troppo_grandi={pdf_stats['too_large']}"
    )

    return step_stats


async def run_ingest(stats_only: bool = False) -> dict:
    """Esegue pipeline, marca documenti duplicati e genera stats."""
    config = load_config()
    validate_config(config)

    run_id = make_run_id()
    manifest_path = project_path(config["paths"].get("processed_manifest_file", "data/processed/manifest.jsonl"))

    print(f"Ingest avviato: {run_id}")

    if stats_only:
        print("Pipeline saltata: genero solo marcatura duplicati e stats dai file esistenti.")
        step_stats: dict[str, dict] = {}
    else:
        step_stats = await run_pipeline_steps(config)

    duplicate_label = "[1/2]" if stats_only else "[4/5]"
    stats_label = "[2/2]" if stats_only else "[5/5]"

    print(f"\n{duplicate_label} Marcatura documenti duplicati")
    manifest_records = load_jsonl(manifest_path)
    manifest_records, current_manifest_records, duplicate_stats = mark_duplicate_documents(manifest_records)
    write_jsonl_atomic(manifest_path, manifest_records)
    print(f"Duplicati marcati: {duplicate_stats['duplicates']}")

    print(f"\n{stats_label} Stats")
    stats = write_stats(config, run_id, step_stats, current_manifest_records, duplicate_stats)
    print_stats_summary(stats)
    return stats


def parse_args() -> argparse.Namespace:
    """CLI minimale dell'orchestratore."""
    parser = argparse.ArgumentParser(description="Orchestra discovery, scraping, PDF e stats.")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help=(
            "Non rilancia la pipeline: marca i duplicati nel manifest esistente "
            "e rigenera le statistiche correnti e storiche."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    asyncio.run(run_ingest(stats_only=args.stats_only))


if __name__ == "__main__":
    main()
