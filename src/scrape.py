"""
Conversione degli HTML scoperti in Markdown raw e Markdown pulito.

Input:
- data/discovered_urls.jsonl
- data/raw_html/<sh>/<hash>.html

Output:
- data/processed/markdown_raw/<sh>/<hash>.md
- data/processed/markdown/<sh>/<hash>.md
- data/processed/manifest.jsonl
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from markdown_cleaner import MIN_INDEXABLE_CHARS, clean_markdown
from pipeline_io import (
    BASE_DIR,
    append_jsonl_batch,
    content_hash,
    load_jsonl,
    now_iso,
    recent_successful_urls,
    relative_path,
    write_text,
)
from pipeline_types import DiscoveryRecord, ProcessedRecord

# Crawl4AI usa questa variabile con il nome CRAWL4_AI_BASE_DIRECTORY.
# La impostiamo prima dell'import per tenere cache e DB dentro il progetto.
load_dotenv(BASE_DIR / ".env")
if not os.environ.get("CRAWL4_AI_BASE_DIRECTORY"):
    os.environ["CRAWL4_AI_BASE_DIRECTORY"] = str(BASE_DIR)

from bs4 import BeautifulSoup
from crawl4ai import (  # noqa: E402
    AsyncWebCrawler,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
    PruningContentFilter,
)

from discovery_io import load_config, project_path, validate_config  # noqa: E402


BATCH_SIZE = 100
SKIP_RECENT_DAYS = 7


def raw_markdown_path(url_hash: str, config: dict) -> Path:
    """Path del Markdown raw generato dallo scraper."""
    base = project_path(config["paths"].get("processed_raw_markdown_dir", "data/processed/markdown_raw"))
    return base / url_hash[:2] / f"{url_hash}.md"


def processed_markdown_path(url_hash: str, config: dict) -> Path:
    """Path del Markdown pulito e indicizzabile."""
    base = project_path(config["paths"].get("processed_markdown_dir", "data/processed/markdown"))
    return base / url_hash[:2] / f"{url_hash}.md"


def html_records(config: dict, recent_urls: set[str]) -> list[DiscoveryRecord]:
    """Record HTML scoperti e non processati di recente."""
    discovered_path = project_path(config["paths"]["discovered_urls_file"])
    records: list[DiscoveryRecord] = []

    for record in load_jsonl(discovered_path):
        if record.get("type") != "html":
            continue
        if record.get("status") != "ok" or not record.get("indexable", False):
            continue
        if not record.get("raw_path"):
            continue
        if record.get("url") in recent_urls:
            continue
        records.append(record)

    return records


def extract_title_and_breadcrumb(html: str) -> tuple[str | None, list[str]]:
    """Estrae titolo e breadcrumb dall'HTML originale."""
    soup = BeautifulSoup(html, "lxml")

    title = None
    if soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(" ", strip=True)
    elif soup.find("h1"):
        title = soup.find("h1").get_text(" ", strip=True)

    selectors = [
        ".breadcrumb",
        ".breadcrumbs",
        "nav[aria-label*=breadcrumb]",
        "nav[aria-label*=Breadcrumb]",
    ]
    breadcrumb: list[str] = []
    for selector in selectors:
        node = soup.select_one(selector)
        if not node:
            continue
        breadcrumb = [
            item.get_text(" ", strip=True)
            for item in node.find_all(["a", "li", "span"])
            if item.get_text(" ", strip=True)
        ]
        if breadcrumb:
            break

    return title, list(dict.fromkeys(breadcrumb))


def markdown_from_crawl4ai_direct(html: str, base_url: str) -> str:
    """Fallback locale: usa il generator crawl4ai senza avviare browser."""
    generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.48, threshold_type="fixed"),
    )
    result = generator.generate_markdown(html, base_url=base_url, citations=False)
    fit_markdown = result.fit_markdown or result.raw_markdown or ""
    return fit_markdown.strip()


async def markdown_from_crawl4ai_raw(
    crawler: AsyncWebCrawler | None,
    run_config: CrawlerRunConfig,
    html: str,
    base_url: str,
) -> str:
    """Converte HTML raw in Markdown con crawl4ai, senza scaricare di nuovo."""
    if crawler is None:
        return markdown_from_crawl4ai_direct(html, base_url)

    try:
        run_config.base_url = base_url
        result = await crawler.arun(url=f"raw://{html}", config=run_config)
        markdown = result.markdown
        fit_markdown = markdown.fit_markdown or markdown.raw_markdown or ""
        return fit_markdown.strip()
    except Exception as error:
        print(f"crawl4ai raw:// non disponibile per {base_url}: {error}")
        return markdown_from_crawl4ai_direct(html, base_url)


async def open_crawler() -> AsyncWebCrawler | None:
    """Prova ad aprire AsyncWebCrawler; se fallisce si userà il fallback locale."""
    try:
        crawler = AsyncWebCrawler(verbose=False, base_directory=str(BASE_DIR))
        await crawler.start()
        return crawler
    except Exception as error:
        print(f"AsyncWebCrawler non avviato, uso fallback locale: {error.__class__.__name__}")
        return None


async def close_crawler(crawler: AsyncWebCrawler | None) -> None:
    """Chiude il crawler se è stato avviato."""
    if crawler is not None:
        await crawler.close()


async def process_html_record(
    record: DiscoveryRecord,
    crawler: AsyncWebCrawler | None,
    run_config: CrawlerRunConfig,
    config: dict,
) -> ProcessedRecord:
    """Processa un HTML raw e produce un record del manifest processed."""
    crawled_at = now_iso()
    raw_path = project_path(record["raw_path"])
    url = record["url"]
    url_hash = record["hash"]

    try:
        html = raw_path.read_text(encoding="utf-8")
        title, breadcrumb = extract_title_and_breadcrumb(html)
        raw_markdown = await markdown_from_crawl4ai_raw(
            crawler,
            run_config,
            html,
            url,
        )
        raw_output_path = write_text(raw_markdown_path(url_hash, config), raw_markdown)
        metadata = {
            "url": url,
            "document_url": record.get("document_url", url),
            "source": "html",
            "hash": url_hash,
            "title": title,
            "breadcrumb": breadcrumb,
            "last_crawled": crawled_at,
        }
        clean_body, index_markdown, quality = clean_markdown(
            raw_markdown,
            source="html",
            metadata=metadata,
        )
        index_output_path = write_text(processed_markdown_path(url_hash, config), index_markdown)
        text_extracted = len(clean_body.strip()) >= MIN_INDEXABLE_CHARS

        return {
            "source": "html",
            "status": "ok",
            "url": url,
            "document_url": record.get("document_url", url),
            "hash": url_hash,
            "content_hash": content_hash(clean_body),
            "raw_content_hash": content_hash(raw_markdown),
            "raw_markdown_path": raw_output_path,
            "index_markdown_path": index_output_path,
            "raw_html_path": record["raw_path"],
            "title": title,
            "breadcrumb": breadcrumb,
            "last_crawled": crawled_at,
            "text_extracted": text_extracted,
            "markdown_chars": len(clean_body),
            "indexable": text_extracted,
            **quality,
        }
    except Exception as error:
        return {
            "source": "html",
            "status": "failed",
            "url": url,
            "hash": url_hash,
            "raw_html_path": record.get("raw_path"),
            "last_crawled": crawled_at,
            "error": str(error),
        }


def chunked(items: list[DiscoveryRecord], size: int) -> Iterable[list[DiscoveryRecord]]:
    """Divide una lista in batch."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


async def run_scrape() -> dict:
    """Esegue la conversione HTML -> Markdown."""
    config = load_config()
    validate_config(config)
    manifest_path = project_path(config["paths"].get("processed_manifest_file", "data/processed/manifest.jsonl"))
    records = html_records(
        config,
        recent_successful_urls(manifest_path, "html", SKIP_RECENT_DAYS),
    )

    run_config = CrawlerRunConfig(
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.48, threshold_type="fixed"),
        ),
        verbose=False,
        process_in_browser=False,
    )

    crawler = await open_crawler()
    processed_count = 0
    failed_count = 0

    try:
        for batch_index, batch in enumerate(chunked(records, BATCH_SIZE), start=0):
            output_records: list[ProcessedRecord] = []
            for record in batch:
                processed = await process_html_record(record, crawler, run_config, config)
                output_records.append(processed)
                if processed["status"] == "ok":
                    processed_count += 1
                else:
                    failed_count += 1

            append_jsonl_batch(manifest_path, output_records)
            print(f"Batch salvato: {len(output_records)} record")
    finally:
        await close_crawler(crawler)

    return {
        "html_candidates": len(records),
        "processed_ok": processed_count,
        "failed": failed_count,
        "manifest": relative_path(manifest_path),
    }


async def main() -> None:
    """Entry point."""
    stats = await run_scrape()
    print("Scrape completato")
    for key, value in stats.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
