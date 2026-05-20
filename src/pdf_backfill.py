"""
Backfill dei PDF linkati da HTML gia presenti nel manifest.

La discovery registra i PDF mentre visita una pagina. Se pero una pagina e'
stata visitata prima di una correzione dei filtri, i link PDF gia salvati nel
raw HTML possono restare invisibili alle run successive finche la pagina non
scade dal refresh. Questo modulo rilegge gli HTML correnti e aggiunge i PDF
mancanti a discovered_urls.jsonl prima dell'estrazione.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from discovery_io import append_jsonl, make_record, project_path
from discovery_models import CrawlItem
from html_utils import extract_link_entries_from_soup
from pdf_policy import matched_pdf_keywords, pdf_source_section_from_url
from pipeline_io import latest_records_by_url, load_jsonl
from pipeline_types import DiscoveryRecord, ProcessedRecord
from url_filters import can_traverse_url, is_pdf_url


def existing_pdf_urls(config: dict, manifest_records: list[ProcessedRecord]) -> set[str]:
    """URL PDF gia noti nella discovery o nel manifest processed."""
    discovered_path = project_path(config["paths"]["discovered_urls_file"])
    discovered_records = load_jsonl(discovered_path)
    urls = {
        str(record.get("url"))
        for record in discovered_records
        if record.get("type") == "pdf" and record.get("url")
    }
    urls.update(
        str(record.get("url"))
        for record in manifest_records
        if record.get("source") == "pdf" and record.get("url")
    )
    return urls


def is_course_html_record(record: ProcessedRecord, config: dict) -> bool:
    """True per HTML di corsi.unisa.it gia indicizzabili."""
    course_domain = str(config["scope"].get("course_domain", "corsi.unisa.it")).lower()
    return (
        record.get("source") == "html"
        and record.get("status") == "ok"
        and bool(record.get("raw_html_path"))
        and urlparse(str(record.get("url", ""))).netloc.lower() == course_domain
    )


def linked_pdf_records_from_html_record(
    record: ProcessedRecord,
    config: dict,
    known_pdf_urls: set[str],
) -> list[DiscoveryRecord]:
    """Estrae record PDF mancanti da un singolo raw HTML gia salvato."""
    raw_html_path = project_path(str(record["raw_html_path"]))
    if not raw_html_path.exists():
        return []

    base_url = str(record.get("url") or record.get("document_url"))
    html = raw_html_path.read_text(encoding="utf-8", errors="ignore")
    links = extract_link_entries_from_soup(BeautifulSoup(html, "lxml"), base_url, config)

    records: list[DiscoveryRecord] = []
    for link in links:
        pdf_url = link["url"]
        if not is_pdf_url(pdf_url) or pdf_url in known_pdf_urls:
            continue
        ok, _ = can_traverse_url(pdf_url, config, {"discovered_from": base_url})
        if not ok:
            continue

        link_text = link.get("text", "")
        item = CrawlItem(
            pdf_url,
            int(record.get("depth") or 0) + 1,
            base_url,
            origin_seed=str(record.get("origin_seed") or base_url),
        )
        records.append(
            make_record(
                item,
                "pdf",
                "pending_download",
                final_url=pdf_url,
                document_url=pdf_url,
                discovery_method="processed_html_link_backfill",
                link_text=link_text,
                pdf_source_section=pdf_source_section_from_url(base_url),
                pdf_download_decision="allowed_processed_html_backfill",
                pdf_match_keywords=matched_pdf_keywords(pdf_url, link_text),
                robots_txt_denied=None,
            )
        )
        known_pdf_urls.add(pdf_url)

    return records


def backfill_linked_pdfs_from_processed_html(config: dict) -> dict[str, int]:
    """Aggiunge a discovered_urls.jsonl i PDF mancanti linkati dagli HTML correnti."""
    manifest_path = project_path(
        config["paths"].get("processed_manifest_file", "data/processed/manifest.jsonl")
    )
    discovered_path = project_path(config["paths"]["discovered_urls_file"])
    manifest_records = latest_records_by_url(load_jsonl(manifest_path))
    known_pdf_urls = existing_pdf_urls(config, manifest_records)

    added = 0
    scanned_html = 0
    for record in manifest_records:
        if not is_course_html_record(record, config):
            continue
        scanned_html += 1
        for pdf_record in linked_pdf_records_from_html_record(record, config, known_pdf_urls):
            append_jsonl(discovered_path, pdf_record)
            added += 1

    return {
        "scanned_html": scanned_html,
        "added_pdf_records": added,
    }
