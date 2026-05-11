"""
Orchestrazione della pipeline di ingest.

Questo script coordina:
1. discovery degli URL
2. conversione HTML in Markdown
3. estrazione PDF in Markdown
4. marcatura dei documenti duplicati nel manifest processed
5. generazione di data/processed/stats.json

La fase di ingest non crea ancora chunk o embedding: prepara un corpus
Markdown pulito, misurabile e pronto per la fase di indexing.
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from datetime import UTC, datetime
from statistics import mean, median
from urllib.parse import urlparse
from uuid import uuid4

from discover import run_discovery
from discovery_io import load_config, project_path, validate_config
from extract_pdf import run_extract_pdf
from pipeline_io import BASE_DIR, load_jsonl, now_iso, write_json, write_jsonl_atomic
from scrape import run_scrape


def make_run_id() -> str:
    """Identificativo breve del run, utile nei log e negli stats."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"ingest-{timestamp}-{uuid4().hex[:8]}"


def domain_from_record(record: dict) -> str:
    """Estrae il dominio da document_url o url."""
    url = record.get("document_url") or record.get("url") or ""
    return urlparse(url).netloc or "unknown"


def is_duplicate_candidate(record: dict) -> bool:
    """True se il record può essere confrontato tramite content_hash."""
    if record.get("status") != "ok":
        return False
    if not record.get("markdown_path"):
        return False
    if record.get("text_extracted") is False:
        return False
    return bool(record.get("content_hash"))


def mark_duplicate_documents(records: list[dict]) -> tuple[list[dict], dict]:
    """
    Marca i duplicati esatti basandosi su content_hash.

    Lo status resta invariato: aggiungiamo duplicate_of/is_duplicate senza
    cambiare status="ok", così gli skip incrementali di scrape.py continuano
    a funzionare anche nei run successivi.
    """
    first_by_content_hash: dict[str, dict] = {}
    updated_records: list[dict] = []
    duplicate_count = 0
    unique_content_hashes = 0

    for original in records:
        record = dict(original)
        record.pop("duplicate_of", None)
        record.pop("duplicate_of_url", None)
        record.pop("duplicate_reason", None)
        record.pop("is_duplicate", None)

        if not is_duplicate_candidate(record):
            updated_records.append(record)
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

        updated_records.append(record)

    return updated_records, {
        "duplicates": duplicate_count,
        "unique_content_hashes": unique_content_hashes,
    }


def count_by(records: list[dict], key: str) -> dict[str, int]:
    """Conta i valori di una chiave JSON."""
    counter = Counter(str(record.get(key, "missing")) for record in records)
    return dict(counter)


def markdown_char_stats(records: list[dict]) -> dict:
    """Statistiche di lunghezza Markdown sui record OK non duplicati."""
    values = [
        int(record.get("markdown_chars", 0))
        for record in records
        if record.get("status") == "ok" and not record.get("is_duplicate", False)
    ]

    if not values:
        return {"min": 0, "max": 0, "avg": 0, "median": 0}

    return {
        "min": min(values),
        "max": max(values),
        "avg": round(mean(values), 2),
        "median": median(values),
    }


def build_discovery_stats(config: dict) -> dict:
    """Aggrega statistiche leggere da discovered_urls.jsonl."""
    path = project_path(config["paths"]["discovered_urls_file"])
    records = load_jsonl(path)

    return {
        "records": len(records),
        "by_status": count_by(records, "status"),
        "by_type": count_by(records, "type"),
        "by_domain": dict(Counter(domain_from_record(record) for record in records)),
        "by_depth": dict(Counter(str(record.get("depth", "missing")) for record in records)),
    }


def build_processed_stats(
    records: list[dict],
    duplicate_stats: dict,
) -> dict:
    """Aggrega statistiche del manifest processed."""
    indexable_records = [
        record
        for record in records
        if record.get("status") == "ok"
        and record.get("markdown_path")
        and not record.get("is_duplicate", False)
        and record.get("text_extracted") is not False
    ]
    empty_or_short_records = [
        record
        for record in records
        if record.get("status") == "ok" and int(record.get("markdown_chars", 0)) < 100
    ]

    return {
        "records": len(records),
        "indexable_records": len(indexable_records),
        "duplicates": duplicate_stats["duplicates"],
        "unique_content_hashes": duplicate_stats["unique_content_hashes"],
        "empty_or_short_records": len(empty_or_short_records),
        "by_source": count_by(records, "source"),
        "by_status": count_by(records, "status"),
        "by_domain": dict(Counter(domain_from_record(record) for record in records)),
        "markdown_chars": markdown_char_stats(records),
    }


def config_snapshot(config: dict) -> dict:
    """Salva negli stats solo i parametri utili a interpretare il run."""
    crawler = config["crawler"]
    return {
        "max_total_urls": crawler["max_total_urls"],
        "max_depth": crawler["max_depth"],
        "max_concurrent_requests": crawler["max_concurrent_requests"],
        "rate_limit_per_domain_rps": crawler["rate_limit_per_domain_rps"],
        "use_sitemap": crawler.get("use_sitemap", True),
        "respect_robots_txt": crawler.get("respect_robots_txt", True),
        "per_domain_limits": crawler["per_domain_limits"],
    }


def write_stats(
    config: dict,
    run_id: str,
    step_stats: dict,
    manifest_records: list[dict],
    duplicate_stats: dict,
) -> dict:
    """Genera e salva data/processed/stats.json."""
    stats_path = project_path(config["paths"].get("processed_stats_file", "data/processed/stats.json"))
    stats = {
        "crawl_run_id": run_id,
        "generated_at": now_iso(),
        "stats_file": str(stats_path.relative_to(BASE_DIR)),
        "config": config_snapshot(config),
        "steps": step_stats,
        "discovery": build_discovery_stats(config),
        "processed": build_processed_stats(manifest_records, duplicate_stats),
    }

    write_json(stats_path, stats)
    return stats


def print_stats_summary(stats: dict) -> None:
    """Mostra un riepilogo breve a fine ingest."""
    processed = stats["processed"]
    discovery = stats["discovery"]

    print("Ingest completato")
    print(f"- crawl_run_id: {stats['crawl_run_id']}")
    print(f"- discovered_records: {discovery['records']}")
    print(f"- processed_records: {processed['records']}")
    print(f"- indexable_records: {processed['indexable_records']}")
    print(f"- duplicates: {processed['duplicates']}")
    print(f"- empty_or_short_records: {processed['empty_or_short_records']}")
    print(f"- stats_file: {stats['stats_file']}")


async def run_ingest(stats_only: bool = False) -> dict:
    """Esegue pipeline, marca documenti duplicati e genera stats."""
    config = load_config()
    validate_config(config)

    run_id = make_run_id()
    manifest_path = project_path(config["paths"].get("processed_manifest_file", "data/processed/manifest.jsonl"))
    step_stats: dict[str, dict] = {}

    print(f"Ingest avviato: {run_id}")

    if stats_only:
        print("Pipeline saltata: genero solo marcatura duplicati e stats dai file esistenti.")
    else:
        print("\n[1/5] Discovery")
        step_stats["discover"] = await run_discovery(config)

        print("\n[2/5] Scrape HTML")
        step_stats["scrape"] = await run_scrape()

        print("\n[3/5] Estrazione PDF")
        step_stats["extract_pdf"] = run_extract_pdf()

    duplicate_label = "[1/2]" if stats_only else "[4/5]"
    stats_label = "[2/2]" if stats_only else "[5/5]"

    print(f"\n{duplicate_label} Marcatura documenti duplicati")
    manifest_records = load_jsonl(manifest_path)
    manifest_records, duplicate_stats = mark_duplicate_documents(manifest_records)
    write_jsonl_atomic(manifest_path, manifest_records)
    print(f"Duplicati marcati: {duplicate_stats['duplicates']}")

    print(f"\n{stats_label} Stats")
    stats = write_stats(config, run_id, step_stats, manifest_records, duplicate_stats)
    print_stats_summary(stats)
    return stats


def parse_args() -> argparse.Namespace:
    """CLI minimale dell'orchestratore."""
    parser = argparse.ArgumentParser(description="Orchestra discovery, scraping, PDF e stats.")
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Non rilancia la pipeline: marca i duplicati nel manifest esistente e rigenera stats.json.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    asyncio.run(run_ingest(stats_only=args.stats_only))


if __name__ == "__main__":
    main()
