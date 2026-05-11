"""
Download ed estrazione Markdown dei PDF scoperti dalla discovery.

Input:
- data/discovered_urls.jsonl, record con type=pdf

Output:
- data/raw_pdf/<sh>/<hash>.pdf
- data/processed/markdown/<sh>/<hash>.md
- data/processed/manifest.jsonl
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import httpx
import pymupdf4llm

from discovery_io import load_config, project_path, validate_config
from pipeline_io import (
    append_jsonl_batch,
    content_hash,
    load_jsonl,
    now_iso,
    recent_successful_urls,
    relative_path,
    write_text,
)


MAX_PDF_BYTES = 100 * 1024 * 1024
PDF_WORKERS = 4
SKIP_RECENT_DAYS = 7


def raw_pdf_path(url_hash: str, config: dict) -> Path:
    """Path del PDF raw."""
    base = project_path(config["paths"].get("raw_pdf_dir", "data/raw_pdf"))
    return base / url_hash[:2] / f"{url_hash}.pdf"


def markdown_path(url_hash: str, config: dict) -> Path:
    """Path del Markdown estratto."""
    base = project_path(config["paths"].get("processed_markdown_dir", "data/processed/markdown"))
    return base / url_hash[:2] / f"{url_hash}.md"


def pdf_records(config: dict, recent_urls: set[str]) -> list[dict]:
    """Record PDF scoperti e non processati di recente."""
    discovered_path = project_path(config["paths"]["discovered_urls_file"])
    return [
        record
        for record in load_jsonl(discovered_path)
        if record.get("type") == "pdf" and record.get("status") == "pending_download"
        and record.get("url") not in recent_urls
    ]


def download_pdf(record: dict, config: dict, client: httpx.Client) -> dict:
    """Scarica un PDF rispettando MAX_PDF_BYTES."""
    url = record["url"]
    url_hash = record["hash"]
    output_path = raw_pdf_path(url_hash, config)

    with client.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()
        content_length = response.headers.get("Content-Length")
        if content_length and content_length.isdigit() and int(content_length) > MAX_PDF_BYTES:
            return {
                "record": record,
                "status": "too_large",
                "content_length": int(content_length),
                "raw_pdf_path": None,
            }

        downloaded = 0
        too_large = False
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with output_path.open("wb") as file:
                for chunk in response.iter_bytes():
                    downloaded += len(chunk)
                    if downloaded > MAX_PDF_BYTES:
                        too_large = True
                        break
                    file.write(chunk)
        except Exception:
            output_path.unlink(missing_ok=True)
            raise

        if too_large:
            output_path.unlink(missing_ok=True)
            return {
                "record": record,
                "status": "too_large",
                "content_length": downloaded,
                "raw_pdf_path": None,
            }

    return {
        "record": record,
        "status": "downloaded",
        "content_length": output_path.stat().st_size,
        "raw_pdf_path": relative_path(output_path),
    }


def extract_pdf_markdown(pdf_path: str) -> tuple[str, str | None]:
    """Worker process: converte PDF in Markdown."""
    try:
        markdown = pymupdf4llm.to_markdown(str(project_path(pdf_path)))
        return markdown or "", None
    except Exception as error:
        return "", str(error)


def base_manifest_record(record: dict, status: str, crawled_at: str) -> dict:
    """Campi comuni dei record manifest PDF."""
    return {
        "source": "pdf",
        "status": status,
        "url": record["url"],
        "document_url": record.get("document_url", record["url"]),
        "hash": record["hash"],
        "last_crawled": crawled_at,
    }


def build_manifest_record(
    download: dict,
    markdown: str = "",
    error: str | None = None,
    config: dict | None = None,
) -> dict:
    """Crea un record processed per un PDF."""
    record = download["record"]
    crawled_at = now_iso()

    if download["status"] == "too_large":
        manifest_record = base_manifest_record(record, "too_large", crawled_at)
        manifest_record.update(
            content_hash=None,
            markdown_path=None,
            raw_pdf_path=download.get("raw_pdf_path"),
            content_length=download.get("content_length"),
            text_extracted=False,
        )
        return manifest_record

    if error:
        manifest_record = base_manifest_record(record, "failed", crawled_at)
        manifest_record.update(
            content_hash=None,
            markdown_path=None,
            raw_pdf_path=download.get("raw_pdf_path"),
            text_extracted=False,
            error=error,
        )
        return manifest_record

    if config is None:
        raise ValueError("config è richiesto per salvare il Markdown PDF.")

    output_path = markdown_path(record["hash"], config)
    markdown_output = write_text(output_path, markdown)
    text_extracted = len(markdown.strip()) >= 100

    manifest_record = base_manifest_record(record, "ok", crawled_at)
    manifest_record.update(
        content_hash=content_hash(markdown),
        markdown_path=markdown_output,
        raw_pdf_path=download.get("raw_pdf_path"),
        text_extracted=text_extracted,
        markdown_chars=len(markdown),
    )
    return manifest_record


def run_extract_pdf() -> dict:
    """Esegue download PDF e conversione in Markdown."""
    config = load_config()
    validate_config(config)
    manifest_path = project_path(config["paths"].get("processed_manifest_file", "data/processed/manifest.jsonl"))
    records = pdf_records(
        config,
        recent_successful_urls(manifest_path, "pdf", SKIP_RECENT_DAYS),
    )

    headers = {"User-Agent": config["crawler"]["user_agent"]}
    timeout = httpx.Timeout(config["crawler"]["timeout"])

    downloads: list[dict] = []
    manifest_records: list[dict] = []

    with httpx.Client(headers=headers, timeout=timeout) as client:
        for record in records:
            try:
                download = download_pdf(record, config, client)
            except httpx.HTTPError as error:
                download = {
                    "record": record,
                    "status": "failed",
                    "raw_pdf_path": None,
                    "error": str(error),
                }

            if download["status"] == "downloaded":
                downloads.append(download)
            elif download["status"] == "too_large":
                manifest_records.append(build_manifest_record(download))
            else:
                manifest_records.append(
                    build_manifest_record(
                        download,
                        error=download.get("error", "download failed"),
                    )
                )

    if downloads:
        with ProcessPoolExecutor(max_workers=PDF_WORKERS) as executor:
            futures = {
                executor.submit(extract_pdf_markdown, download["raw_pdf_path"]): download
                for download in downloads
            }

            for future in as_completed(futures):
                download = futures[future]
                markdown, error = future.result()
                manifest_records.append(
                    build_manifest_record(download, markdown=markdown, error=error, config=config)
                )

    append_jsonl_batch(manifest_path, manifest_records)

    return {
        "pdf_candidates": len(records),
        "downloaded": len(downloads),
        "manifest_records": len(manifest_records),
        "manifest": relative_path(manifest_path),
    }


def main() -> None:
    """Entry point."""
    stats = run_extract_pdf()
    print("Estrazione PDF completata")
    for key, value in stats.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
