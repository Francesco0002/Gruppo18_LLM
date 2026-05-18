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
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from urllib.parse import urlparse
from uuid import uuid4

from discover import run_discovery
from discovery_io import load_config, load_discovery_state, project_path, validate_config
from extract_pdf import run_extract_pdf
from pipeline_io import (
    latest_record_indexes,
    load_jsonl,
    now_iso,
    relative_path,
    write_json,
    write_jsonl_atomic,
)
from pipeline_types import ProcessedRecord
from scrape import run_scrape


def make_run_id() -> str:
    """Identificativo breve del run, utile nei log e negli stats."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"ingest-{timestamp}-{uuid4().hex[:8]}"


def domain_from_record(record: dict) -> str:
    """Estrae il dominio da document_url o url."""
    url = record.get("document_url") or record.get("url") or ""
    return urlparse(url).netloc or "unknown"


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
        except ValueError:
            crawled_dt = _dt.min
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


def count_by(records: list[dict], key: str) -> dict[str, int]:
    """Conta i valori di una chiave JSON."""
    counter = Counter(str(record.get(key, "missing")) for record in records)
    return dict(counter)


def discovery_failure_kind(record: dict) -> str:
    """Classifica i fallimenti di discovery in categorie leggibili nei report."""
    error = str(record.get("error", "")).lower()
    if "404 not found" in error:
        return "http_404"
    if "timeout" in error or "timed out" in error:
        return "timeout"
    if any(f"{status} " in error for status in range(500, 600)):
        return "http_5xx"
    if "client error" in error or "server error" in error:
        return "other_http"
    return "other"


def discovery_failures_by_kind(records: list[dict]) -> dict[str, int]:
    """Aggrega i record failed distinguendo link rotti, timeout e server error."""
    counts = Counter(
        discovery_failure_kind(record)
        for record in records
        if record.get("status") == "failed"
    )
    return {kind: counts[kind] for kind in sorted(counts)}


# Keyword mostrate nei report PDF: servono a spiegare cosa viene perso o ammesso.
REPORTABLE_PDF_KEYWORDS = (
    "bando",
    "graduatoria",
    "regolamento",
    "linee-guida",
    "guida",
    "manifesto",
    "piano-di-studi",
    "calendario",
    "schedule",
    "ofa",
    "requirements",
    "accordi",
    "erasmus",
    "sua-cds",
    "schede-sua",
    "almalaurea",
    "decreto",
)
STABLE_DOCUMENT_PDF_KEYWORDS = {
    "regolamento",
    "linee-guida",
    "guida",
    "manifesto",
    "piano-di-studi",
}
# Stesse famiglie semantiche usate nella discovery, riprese qui solo per report.
MAIN_OPPORTUNITY_PDF_KEYWORDS = {"bando", "call", "premio", "borsa", "concorso"}
TEACHING_OPERATIONS_PDF_KEYWORDS = {"calendario", "schedule", "ofa", "requirements"}
INTERNATIONAL_PROGRAM_PDF_KEYWORDS = {"accordo", "accordi", "erasmus"}
COURSE_EVIDENCE_PDF_KEYWORDS = {"sua-cds", "schede-sua", "almalaurea"}
OPPORTUNITY_ATTACHMENT_HINTS = {
    "graduatoria",
    "approvazione-atti",
    "decreto",
    "allegato",
    "domanda",
    "modello",
    "locandina",
    "manifesto-elettorale",
    "scrutino",
    "elettorato",
    "faq",
    "modulo",
    "moduloadesione",
    "presentazione",
    "comunicato",
    "avviso-proroga",
    "proroga",
}
def pdf_source_section(record: dict) -> str:
    """Raggruppa il parent di un PDF in sezioni leggibili del sito."""
    if record.get("pdf_source_section"):
        return str(record["pdf_source_section"])
    source = str(record.get("discovered_from", ""))
    path = urlparse(source).path.lower().rstrip("/")
    if path.startswith("/home/bandi"):
        return "home_bandi"
    if path.startswith("/home/news"):
        return "home_news"
    if path.startswith("/home/eventi"):
        return "home_eventi"
    if path.startswith("/didattica") or "/didattica/" in path:
        return "didattica"
    if path.startswith("/ricerca") or "/ricerca/" in path:
        return "ricerca"
    if path.startswith("/dipartimento") or "/dipartimento/" in path:
        return "dipartimento"
    if path.startswith("/international") or "/international/" in path:
        return "international"
    if path.startswith("/terza-missione"):
        return "terza_missione"
    return "other"


def pdf_filename_keywords(record: dict) -> set[str]:
    """Keyword indicative estratte dal filename del PDF."""
    if record.get("pdf_match_keywords"):
        return {str(keyword) for keyword in record["pdf_match_keywords"]}
    filename = Path(urlparse(str(record.get("url", ""))).path).name.lower()
    return {keyword for keyword in REPORTABLE_PDF_KEYWORDS if keyword in filename}


def pdf_keyword_text(record: dict) -> str:
    """Testo normalizzato usato per riconoscere opportunità e allegati."""
    filename = Path(urlparse(str(record.get("url", ""))).path).name.lower()
    link_text = str(record.get("link_text", "")).lower()
    return "-".join(" ".join([filename, link_text]).replace("_", " ").split())


def is_main_opportunity_record(record: dict) -> bool:
    """True per PDF principali di opportunità, non per allegati o risultati."""
    text = pdf_keyword_text(record)
    return (
        any(keyword in text for keyword in MAIN_OPPORTUNITY_PDF_KEYWORDS)
        and not any(hint in text for hint in OPPORTUNITY_ATTACHMENT_HINTS)
    )


def blocked_pdf_category(record: dict) -> str:
    """Classifica i PDF negati tra candidati da rivedere ed esclusioni volute."""
    section = pdf_source_section(record)
    keywords = pdf_filename_keywords(record)
    has_stable_document_keyword = bool(keywords & STABLE_DOCUMENT_PDF_KEYWORDS)
    has_teaching_operations_keyword = bool(keywords & TEACHING_OPERATIONS_PDF_KEYWORDS)
    has_international_program_keyword = bool(keywords & INTERNATIONAL_PROGRAM_PDF_KEYWORDS)
    has_course_evidence_keyword = bool(keywords & COURSE_EVIDENCE_PDF_KEYWORDS)

    if section == "home_bandi" and not is_main_opportunity_record(record):
        return "intentionally_excluded"
    if (
        has_stable_document_keyword
        or has_teaching_operations_keyword
        or has_international_program_keyword
        or has_course_evidence_keyword
        or is_main_opportunity_record(record)
    ):
        return "review_candidate"
    return "other_blocked"


def build_pdf_coverage(records: list[dict]) -> dict:
    """Sintesi leggibile dei PDF trovati, bloccati e ammessi dalla whitelist."""
    pdf_records = [record for record in records if record.get("type") == "pdf"]
    robots_denied = [
        record
        for record in pdf_records
        if record.get("status") == "robots_denied"
    ]
    unique_denied_by_url = {
        str(record.get("url")): record
        for record in robots_denied
        if record.get("url")
    }
    unique_denied = list(unique_denied_by_url.values())
    keyword_counts = Counter(
        keyword
        for record in unique_denied
        for keyword in pdf_filename_keywords(record)
    )
    allowed_by_policy = [
        record
        for record in pdf_records
        if str(record.get("pdf_download_decision", "")).startswith("allowed_")
        and record.get("pdf_download_decision") != "allowed_by_robots"
    ]
    unique_allowed_by_url = {
        str(record.get("url")): record
        for record in allowed_by_policy
        if record.get("url")
    }
    unique_allowed = list(unique_allowed_by_url.values())
    allowed_keyword_counts = Counter(
        keyword
        for record in unique_allowed
        for keyword in pdf_filename_keywords(record)
    )
    blocked_by_category = {
        "review_candidate": [
            record
            for record in unique_denied
            if blocked_pdf_category(record) == "review_candidate"
        ],
        "intentionally_excluded": [
            record
            for record in unique_denied
            if blocked_pdf_category(record) == "intentionally_excluded"
        ],
        "other_blocked": [
            record
            for record in unique_denied
            if blocked_pdf_category(record) == "other_blocked"
        ],
    }

    return {
        "pdfs_found": len(pdf_records),
        "pdfs_blocked_by_robots": len(robots_denied),
        "unique_pdfs_blocked_by_robots": len(unique_denied),
        "blocked_by_robots_by_section": dict(
            Counter(pdf_source_section(record) for record in unique_denied)
        ),
        "blocked_by_robots_by_keyword": {
            keyword: keyword_counts[keyword]
            for keyword in sorted(keyword_counts)
        },
        "pdfs_allowed_by_policy": len(unique_allowed),
        "pdfs_allowed_by_policy_by_section": dict(
            Counter(pdf_source_section(record) for record in unique_allowed)
        ),
        "pdfs_allowed_by_policy_by_keyword": {
            keyword: allowed_keyword_counts[keyword]
            for keyword in sorted(allowed_keyword_counts)
        },
        "blocked_review_candidates": {
            "count": len(blocked_by_category["review_candidate"]),
            "by_section": dict(
                Counter(pdf_source_section(record) for record in blocked_by_category["review_candidate"])
            ),
            "by_keyword": dict(
                Counter(
                    keyword
                    for record in blocked_by_category["review_candidate"]
                    for keyword in pdf_filename_keywords(record)
                )
            ),
        },
        "blocked_intentionally_excluded": {
            "count": len(blocked_by_category["intentionally_excluded"]),
            "by_section": dict(
                Counter(pdf_source_section(record) for record in blocked_by_category["intentionally_excluded"])
            ),
            "by_keyword": dict(
                Counter(
                    keyword
                    for record in blocked_by_category["intentionally_excluded"]
                    for keyword in pdf_filename_keywords(record)
                )
            ),
        },
        "blocked_other": {
            "count": len(blocked_by_category["other_blocked"]),
            "by_section": dict(
                Counter(pdf_source_section(record) for record in blocked_by_category["other_blocked"])
            ),
            "by_keyword": dict(
                Counter(
                    keyword
                    for record in blocked_by_category["other_blocked"]
                    for keyword in pdf_filename_keywords(record)
                )
            ),
        },
    }


def markdown_char_stats(records: list[ProcessedRecord]) -> dict:
    """Statistiche di lunghezza Markdown sui record OK non duplicati."""
    values = [
        int(record.get("markdown_chars", 0))
        for record in records
        if record.get("status") == "ok" and not record.get("is_duplicate", False)
    ]

    return (
        {
            "min": min(values),
            "max": max(values),
            "avg": round(mean(values), 2),
            "median": median(values),
        }
        if values
        else {"min": 0, "max": 0, "avg": 0, "median": 0}
    )


def html_records_by_depth(records: list[dict]) -> dict[str, int]:
    """Conta gli HTML della run corrente: sono i nodi che possono espandere la BFS."""
    counts = Counter(
        str(record["depth"])
        for record in records
        if record.get("type") == "html" and isinstance(record.get("depth"), int)
    )
    return {depth: counts[depth] for depth in sorted(counts, key=int)}


def compact_depth_rows(
    coverage_depths: list[dict],
    found_by_depth: dict[str, int],
    visited_html_by_depth: dict[str, int],
) -> list[dict]:
    """Unisce copertura, URL trovati e HTML visitati in una tabella per depth."""
    rows: list[dict] = []
    for depth in coverage_depths:
        depth_key = str(depth["depth"])
        row = {
            "depth": depth["depth"],
            "found_in_run": found_by_depth.get(depth_key, 0),
            "visited_html_in_run": visited_html_by_depth.get(depth_key, 0),
            "pending_new": depth["pending_new_at_depth"],
            "pending_reexpansion": depth["pending_reexpansion_at_depth"],
            "pending_from_lower_depths": depth["pending_below_depth"],
            "complete": depth["complete"],
            "visited_complete": depth["visited_complete"],
        }
        if depth["closed_without_visit"]:
            row["closed_without_visit"] = depth["closed_without_visit"]
        rows.append(row)
    return rows


def frontier_by_depth_from_state(config: dict) -> dict[str, int]:
    """Conta solo gli URL nuovi residui, escludendo le riespansioni."""
    state = load_discovery_state(config)
    counts = Counter(item.depth for item in state.frontier if not item.force_revisit)
    return {str(depth): counts[depth] for depth in sorted(counts)}


def reexpansion_by_depth_from_state(config: dict) -> dict[str, int]:
    """Conta riespansioni già in frontier più backlog attivabile alla depth corrente."""
    state = load_discovery_state(config)
    max_depth = config["crawler"]["max_depth"]
    pending_items: dict[str, int] = {
        item.url: item.depth
        for item in state.frontier
        if item.force_revisit
    }
    for item in state.expansion_backlog:
        if item.depth < max_depth:
            pending_items.setdefault(item.url, item.depth)
    counts = Counter(pending_items.values())
    return {str(depth): counts[depth] for depth in sorted(counts)}


def checkpoint_summary(config: dict) -> tuple[str | None, str | None]:
    """Legge lo stato finale della discovery necessario al verdetto di copertura."""
    path = project_path(config["paths"]["checkpoint_file"])
    if not path.exists():
        return None, None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, None

    return data.get("status"), data.get("stop_reason")


def build_depth_coverage(
    max_depth: int,
    frontier_by_depth: dict[str, int],
    checkpoint_status: str | None,
    stop_reason: str | None,
    reexpansion_by_depth: dict[str, int] | None = None,
    closed_without_visit_by_depth: dict[str, int] | None = None,
) -> dict:
    """
    Calcola la chiusura cumulativa della BFS per ogni profondità HTML.

    Una depth può dirsi completa solo se non ha URL pendenti propri e se tutte
    le depth inferiori sono già sigillate, perché solo allora non possono più
    emergere nuovi figli a quel livello nelle run successive.
    """
    numeric_frontier = {
        int(depth): int(count)
        for depth, count in frontier_by_depth.items()
    }
    numeric_reexpansion = {
        int(depth): int(count)
        for depth, count in (reexpansion_by_depth or {}).items()
    }
    numeric_closed_without_visit = {
        int(depth): int(count)
        for depth, count in (closed_without_visit_by_depth or {}).items()
    }

    depths: list[dict] = []
    pending_below = 0
    closed_without_visit_below = 0
    for depth in range(max_depth + 1):
        pending_new_at_depth = numeric_frontier.get(depth, 0)
        pending_reexpansion_at_depth = numeric_reexpansion.get(depth, 0)
        pending_total_at_depth = pending_new_at_depth + pending_reexpansion_at_depth
        closed_without_visit = numeric_closed_without_visit.get(depth, 0)
        sealed = pending_below == 0
        complete = sealed and pending_new_at_depth == 0
        visited_complete = (
            complete
            and closed_without_visit == 0
            and closed_without_visit_below == 0
        )
        depths.append(
            {
                "depth": depth,
                "pending_new_at_depth": pending_new_at_depth,
                "pending_reexpansion_at_depth": pending_reexpansion_at_depth,
                "pending_total_at_depth": pending_total_at_depth,
                "pending_below_depth": pending_below,
                "closed_without_visit": closed_without_visit,
                "sealed": sealed,
                "complete": complete,
                "visited_complete": visited_complete,
            }
        )
        pending_below += pending_total_at_depth
        closed_without_visit_below += closed_without_visit

    frontier_remaining = sum(numeric_frontier.values())
    reexpansion_remaining = sum(numeric_reexpansion.values())
    closed_without_visit = sum(numeric_closed_without_visit.values())
    all_depths_complete = all(depth["complete"] for depth in depths)
    all_depths_visited_complete = all(depth["visited_complete"] for depth in depths)
    blocking_reasons: list[str] = []

    if checkpoint_status != "completed":
        verdict = "unknown"
        blocking_reasons.append("checkpoint_not_completed")
    elif stop_reason != "queue_exhausted":
        verdict = "incomplete"
        blocking_reasons.append(f"stop_reason:{stop_reason or 'missing'}")
    elif frontier_remaining > 0:
        verdict = "incomplete"
        blocking_reasons.append("frontier_not_drained")
    elif reexpansion_remaining > 0:
        verdict = "incomplete"
        blocking_reasons.append("reexpansion_not_drained")
    elif not all_depths_complete:
        verdict = "incomplete"
        blocking_reasons.append("depths_not_complete")
    elif not all_depths_visited_complete:
        verdict = "incomplete"
        blocking_reasons.append("closed_without_visit")
    else:
        verdict = "complete"

    return {
        "verdict": verdict,
        "configured_max_depth": max_depth,
        "checkpoint_status": checkpoint_status,
        "stop_reason": stop_reason,
        "frontier_remaining": frontier_remaining,
        "frontier_by_depth": frontier_by_depth,
        "reexpansion_remaining": reexpansion_remaining,
        "reexpansion_by_depth": reexpansion_by_depth or {},
        "closed_without_visit": closed_without_visit,
        "closed_without_visit_by_depth": closed_without_visit_by_depth or {},
        "depths": depths,
        "blocking_reasons": blocking_reasons,
    }


def closed_without_visit_by_depth_from_step_stats(discover_stats: dict | None) -> dict[str, int]:
    """Conta gli URL rimossi senza fetch perché esclusi definitivamente dal run."""
    by_reason = (discover_stats or {}).get("skipped_by_reason_by_depth", {})
    # `recently_known` è già stato visitato in una run precedente; `out_of_scope`
    # non appartiene alla copertura richiesta; `domain_limit` resta in frontier.
    terminal_unvisited_reasons: set[str] = set()
    counts: Counter[int] = Counter()
    for reason in terminal_unvisited_reasons:
        for depth, count in by_reason.get(reason, {}).items():
            counts[int(depth)] += int(count)
    return {str(depth): counts[depth] for depth in sorted(counts)}


def build_discovery_stats(config: dict, discover_stats: dict | None = None) -> dict:
    """Costruisce un report discovery compatto per il run corrente."""
    path = project_path(config["paths"]["discovered_urls_file"])
    records = load_jsonl(path)
    frontier_by_depth = frontier_by_depth_from_state(config)
    reexpansion_by_depth = reexpansion_by_depth_from_state(config)
    checkpoint_status, stop_reason = checkpoint_summary(config)
    found_by_depth = dict(Counter(str(record.get("depth", "missing")) for record in records))
    visited_html_by_depth = html_records_by_depth(records)
    coverage = build_depth_coverage(
        config["crawler"]["max_depth"],
        frontier_by_depth,
        checkpoint_status,
        stop_reason,
        reexpansion_by_depth,
        closed_without_visit_by_depth_from_step_stats(discover_stats),
    )
    run_stats = discover_stats or {}

    report = {
        "found": {
            "records": len(records),
            "by_status": count_by(records, "status"),
            "by_type": count_by(records, "type"),
            "by_domain": dict(Counter(domain_from_record(record) for record in records)),
        },
        "failed_by_kind": discovery_failures_by_kind(records),
        "pdfs": build_pdf_coverage(records),
        "remaining_work": {
            "new_urls": coverage["frontier_remaining"],
            "new_urls_by_depth": coverage["frontier_by_depth"],
            "reexpansions": coverage["reexpansion_remaining"],
            "reexpansions_by_depth": coverage["reexpansion_by_depth"],
            "future_expansion_backlog_by_depth": run_stats.get(
                "expansion_backlog_after_by_depth",
                {},
            ),
        },
        "coverage": {
            "verdict": coverage["verdict"],
            "configured_max_depth": coverage["configured_max_depth"],
            "blocking_reasons": coverage["blocking_reasons"],
            "depths": compact_depth_rows(
                coverage["depths"],
                found_by_depth,
                visited_html_by_depth,
            ),
        },
    }
    if discover_stats is not None:
        report["run"] = {
            "seed_urls": run_stats.get("seed_urls", 0),
            "visited_urls": run_stats.get("visited_urls", 0),
            "documents_seen": run_stats.get("documents_seen", 0),
            "visited_urls_by_domain": run_stats.get("by_domain", {}),
            "skipped_by_reason": run_stats.get("skipped_by_reason", {}),
            "stop_reason": run_stats.get("stop_reason", stop_reason),
        }
        report["run_progress"] = {
            "new_urls_before_by_depth": run_stats.get("frontier_before_by_depth", {}),
            "new_urls_after_by_depth": run_stats.get("frontier_after_by_depth", frontier_by_depth),
        }
        report["run"] = {
            key: value
            for key, value in report["run"].items()
            if value not in ({}, [])
        }
        report["run_progress"] = {
            key: value
            for key, value in report["run_progress"].items()
            if value not in ({}, [])
        }
    report["remaining_work"] = {
        key: value
        for key, value in report["remaining_work"].items()
        if value not in ({}, [])
    }
    if not any(report["remaining_work"].values()):
        report.pop("remaining_work")
    if not report["failed_by_kind"]:
        report.pop("failed_by_kind")
    return report


def build_processed_stats(
    records: list[ProcessedRecord],
    duplicate_stats: dict,
) -> dict:
    """Aggrega statistiche sullo stato corrente del manifest processed."""
    indexable_records = [
        record
        for record in records
        if record.get("status") == "ok"
        and record.get("index_markdown_path")
        and not record.get("is_duplicate", False)
        and record.get("text_extracted") is not False
        and record.get("indexable", True) is not False
    ]
    empty_or_short_records = [
        record
        for record in records
        if record.get("status") == "ok" and int(record.get("markdown_chars", 0)) < 100
    ]

    return {
        "records": len(records),
        "history_records": duplicate_stats.get("history_records", len(records)),
        "indexable_records": len(indexable_records),
        "duplicates": duplicate_stats["duplicates"],
        "unique_content_hashes": duplicate_stats["unique_content_hashes"],
        "empty_or_short_records": len(empty_or_short_records),
        "by_source": count_by(records, "source"),
        "by_status": count_by(records, "status"),
        "by_clean_status": count_by(records, "clean_status"),
        "by_domain": dict(Counter(domain_from_record(record) for record in records)),
        "markdown_chars": markdown_char_stats(records),
    }


def build_extraction_stats(step_stats: dict) -> dict:
    """Rende leggibili gli esiti di estrazione del solo run corrente."""
    scrape = step_stats.get("scrape", {})
    pdf = step_stats.get("extract_pdf", {})
    report: dict[str, dict] = {}

    if scrape:
        report["html"] = {
            "candidates": scrape.get("html_candidates", 0),
            "processed_ok": scrape.get("processed_ok", 0),
            "failed": scrape.get("failed", 0),
        }

    if pdf:
        report["pdf"] = {
            "pending_found": pdf.get("pdf_pending_discovered", 0),
            "skipped_recent": pdf.get("skipped_recent", 0),
            "selected_for_processing": pdf.get("pdf_candidates", 0),
            "downloaded": pdf.get("downloaded", 0),
            "extracted_ok": pdf.get("extracted_ok", 0),
            "failed": pdf.get("failed", 0),
            "too_large": pdf.get("too_large", 0),
        }

    return report


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


def run_stats_path(config: dict, run_id: str) -> tuple[str, Path]:
    """Path storico delle statistiche per uno specifico run."""
    current_stats_path = project_path(config["paths"].get("processed_stats_file", "data/processed/stats.json"))
    runs_dir = current_stats_path.parent / "runs" / run_id
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_path = runs_dir / "stats.json"
    return relative_path(run_path), run_path


def write_stats(
    config: dict,
    run_id: str,
    step_stats: dict,
    current_manifest_records: list[ProcessedRecord],
    duplicate_stats: dict,
) -> dict:
    """Genera le statistiche e le salva come ultimo report e nello storico run."""
    stats_path = project_path(config["paths"].get("processed_stats_file", "data/processed/stats.json"))
    run_stats_relative_path, run_stats_full_path = run_stats_path(config, run_id)
    stats = {
        "crawl_run_id": run_id,
        "generated_at": now_iso(),
        "stats_file": relative_path(stats_path),
        "run_stats_file": run_stats_relative_path,
        "config": config_snapshot(config),
        "discovery": build_discovery_stats(config, step_stats.get("discover")),
        "processed": build_processed_stats(current_manifest_records, duplicate_stats),
    }
    extraction = build_extraction_stats(step_stats)
    if extraction:
        stats["extraction"] = extraction

    write_json(stats_path, stats)
    write_json(run_stats_full_path, stats)
    return stats


def print_stats_summary(stats: dict) -> None:
    """Mostra un riepilogo breve a fine ingest."""
    processed = stats["processed"]
    discovery = stats["discovery"]

    print("Ingest completato")
    print(f"- crawl_run_id: {stats['crawl_run_id']}")
    print(f"- discovered_records: {discovery['found']['records']}")
    print(f"- processed_records: {processed['records']}")
    print(f"- manifest_history_records: {processed['history_records']}")
    print(f"- indexable_records: {processed['indexable_records']}")
    print(f"- duplicates: {processed['duplicates']}")
    print(f"- empty_or_short_records: {processed['empty_or_short_records']}")
    print(f"- stats_file: {stats['stats_file']}")
    print(f"- run_stats_file: {stats['run_stats_file']}")


async def run_pipeline_steps(config: dict) -> dict[str, dict]:
    """Esegue gli step pesanti della pipeline e raccoglie le statistiche."""
    step_stats: dict[str, dict] = {}

    print("\n[1/5] Discovery")
    step_stats["discover"] = await run_discovery(config)

    print("\n[2/5] Scrape HTML")
    step_stats["scrape"] = await run_scrape()

    print("\n[3/5] Estrazione PDF")
    step_stats["extract_pdf"] = run_extract_pdf()
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
