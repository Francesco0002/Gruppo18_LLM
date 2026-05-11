"""
Discovery degli URL per il chatbot RAG sul DIEM.

Questo script orchestra il primo step della pipeline:
1. legge gli URL iniziali da data/urls.txt
2. aggiunge URL trovati nelle sitemap dei domini configurati
3. visita le pagine HTML con una BFS
4. salva l'HTML grezzo in data/raw_html/<sh>/<hash>.html
5. scrive gli URL scoperti in data/discovered_urls.jsonl
6. aggiorna data/checkpoint.json per poter riprendere un crawl interrotto

La logica di processing del singolo URL vive in discovery_processor.py.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter, deque
from pathlib import Path
from urllib.parse import urlparse

import httpx
from tqdm import tqdm

from discovery_fetch import DomainRateLimiter, RobotsCache, fetch_sitemap_urls
from discovery_io import (
    load_checkpoint,
    load_config,
    project_path,
    read_seed_urls,
    save_checkpoint,
    save_html,
    validate_config,
    write_record,
)
from discovery_models import CrawlItem, CrawlState, ProcessedDiscoveryItem
from discovery_processor import filter_context, process_item
from url_filters import can_traverse_url, is_pdf_url


def create_initial_state(seed_urls: list[str], sitemap_urls: list[str]) -> CrawlState:
    """Crea la coda iniziale della BFS da seed + sitemap."""
    initial_urls = list(dict.fromkeys(seed_urls + sitemap_urls))
    queue = deque(CrawlItem(url, 0, "seed") for url in initial_urls)
    return CrawlState(
        queue=queue,
        queued=set(initial_urls),
        visited=set(),
        seen_documents=set(),
        domain_counts=Counter(),
    )


def can_visit(item: CrawlItem, state: CrawlState, config: dict) -> bool:
    """True se l'URL può essere inserito nel prossimo batch."""
    if item.url in state.visited:
        return False

    ok, _ = can_traverse_url(item.url, config, filter_context(item.discovered_from))
    if not ok:
        return False

    domain = urlparse(item.url).netloc
    domain_limit = config["crawler"]["per_domain_limits"].get(domain, 0)
    if state.domain_counts[domain] >= domain_limit:
        return False

    return len(state.visited) < config["crawler"]["max_total_urls"]


def mark_visited(state: CrawlState, url: str) -> bool:
    """Marca un URL visitato e aggiorna il contatore del suo dominio."""
    if url in state.visited:
        return False

    state.visited.add(url)
    domain = urlparse(url).netloc
    if domain:
        state.domain_counts[domain] += 1
    return True


def take_batch(state: CrawlState, config: dict) -> list[CrawlItem]:
    """Prende dalla coda un batch di URL validi da scaricare."""
    batch: list[CrawlItem] = []
    batch_size = config["crawler"]["max_concurrent_requests"]

    max_total_urls = config["crawler"]["max_total_urls"]

    while state.queue and len(batch) < batch_size and len(state.visited) < max_total_urls:
        item = state.queue.popleft()
        # queued rappresenta solo gli URL ancora presenti nella coda BFS.
        state.queued.discard(item.url)

        if not can_visit(item, state, config):
            continue

        mark_visited(state, item.url)
        batch.append(item)

    return batch


def load_recorded_pdf_urls(output_file: Path) -> set[str]:
    """URL PDF già scritti in discovered_urls.jsonl, utile nei resume."""
    if not output_file.exists():
        return set()

    urls: set[str] = set()
    with output_file.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            url = record.get("url")
            if record.get("type") == "pdf" or (url and is_pdf_url(url)):
                urls.add(url)

    return urls


def enqueue_links(
    parent: CrawlItem,
    links: list[str],
    state: CrawlState,
    config: dict,
) -> None:
    """Aggiunge alla coda i link trovati, se la profondità lo consente."""
    if parent.depth >= config["crawler"]["max_depth"]:
        return

    for link in links:
        if link in state.visited or link in state.queued:
            continue
        state.queue.append(CrawlItem(link, parent.depth + 1, parent.url))
        state.queued.add(link)


def mark_same_batch_redirect_duplicates(processed_items: list[ProcessedDiscoveryItem]) -> None:
    """Marca duplicati sullo stesso final_url emersi nello stesso batch."""
    seen_final_urls: set[str] = set()

    for processed in processed_items:
        record = processed.record
        if record.get("type") != "html" or not record.get("indexable", False):
            continue

        final_url = record.get("final_url")
        if not final_url:
            continue
        if final_url in seen_final_urls:
            record["status"] = "duplicate_redirect"
            record["indexable"] = False
            record["duplicate_of"] = final_url
            record["raw_path"] = None
            continue

        seen_final_urls.add(final_url)


async def run_discovery(config: dict) -> dict:
    """Esegue la discovery e ritorna statistiche finali."""
    seed_urls = read_seed_urls(config)
    output_file = project_path(config["paths"]["discovered_urls_file"])

    headers = {"User-Agent": config["crawler"]["user_agent"]}
    timeout = httpx.Timeout(config["crawler"]["timeout"])
    limiter = DomainRateLimiter(config["crawler"]["rate_limit_per_domain_rps"])
    robots = None
    if config["crawler"].get("respect_robots_txt", True):
        robots = RobotsCache(
            config["crawler"]["user_agent"],
            config["crawler"]["timeout"],
        )

    status_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()

    async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
        state = load_checkpoint(config)

        if state is None:
            output_file.unlink(missing_ok=True)
            recorded_pdf_urls: set[str] = set()
            sitemap_urls = await fetch_sitemap_urls(client, config, limiter)
            state = create_initial_state(seed_urls, sitemap_urls)
        else:
            print("Checkpoint trovato: riprendo il crawl precedente.")
            recorded_pdf_urls = load_recorded_pdf_urls(output_file)

        max_total_urls = config["crawler"]["max_total_urls"]
        with tqdm(
            total=max_total_urls,
            initial=min(len(state.visited), max_total_urls),
            desc="Discovery",
            unit="url",
            dynamic_ncols=True,
        ) as progress:
            while state.queue and len(state.visited) < max_total_urls:
                batch = take_batch(state, config)
                if not batch:
                    break

                tasks = [
                    process_item(item, client, limiter, robots, config, state.seen_documents)
                    for item in batch
                ]
                results = await asyncio.gather(*tasks)
                mark_same_batch_redirect_duplicates(results)

                for item, processed in zip(batch, results, strict=True):
                    record = processed.record

                    for extra in processed.additional_visited:
                        mark_visited(state, extra)

                    expand_links = record["status"] not in {
                        "duplicate_canonical",
                        "duplicate_redirect",
                    }
                    if record["type"] == "html" and record["indexable"]:
                        document_url = record["document_url"]
                        if document_url in state.seen_documents:
                            record["status"] = "duplicate_canonical"
                            record["indexable"] = False
                            record["duplicate_of"] = document_url
                            record["raw_path"] = None
                            expand_links = False
                        else:
                            state.seen_documents.add(document_url)
                            record["raw_path"] = save_html(document_url, processed.html or "", config)

                    write_record(output_file, record, status_counts, type_counts)
                    if record["type"] == "pdf":
                        recorded_pdf_urls.add(record["url"])

                    if expand_links:
                        for pdf_record in processed.linked_pdf_records:
                            if pdf_record["url"] in recorded_pdf_urls:
                                continue
                            write_record(output_file, pdf_record, status_counts, type_counts)
                            recorded_pdf_urls.add(pdf_record["url"])

                        enqueue_links(item, processed.traversal_links, state, config)

                save_checkpoint(state, config, completed=False)
                current_progress = min(len(state.visited), max_total_urls)
                if current_progress > progress.n:
                    progress.update(current_progress - progress.n)
                progress.set_postfix(
                    coda=len(state.queue),
                    documenti=len(state.seen_documents),
                    refresh=False,
                )

    save_checkpoint(state, config, completed=True)

    return {
        "seed_urls": len(seed_urls),
        "visited_urls": len(state.visited),
        "remaining_queue": len(state.queue),
        "documents_seen": len(state.seen_documents),
        "by_domain": dict(state.domain_counts),
        "by_status": dict(status_counts),
        "by_type": dict(type_counts),
        "output_file": config["paths"]["discovered_urls_file"],
        "raw_html_dir": config["paths"]["raw_html_dir"],
        "checkpoint_file": config["paths"]["checkpoint_file"],
    }


def print_config_summary(config: dict) -> None:
    """Mostra i parametri principali prima di iniziare."""
    crawler = config["crawler"]
    paths = config["paths"]

    print("Configurazione discovery")
    print(f"- max_total_urls: {crawler['max_total_urls']}")
    print(f"- max_depth: {crawler['max_depth']}")
    print(f"- max_concurrent_requests: {crawler['max_concurrent_requests']}")
    print(f"- raw_html_dir: {paths['raw_html_dir']}")
    print(f"- discovered_urls_file: {paths['discovered_urls_file']}")
    print()


async def main() -> None:
    """Entry point."""
    config = load_config()
    validate_config(config)
    print_config_summary(config)

    stats = await run_discovery(config)

    print("Discovery completata")
    for key, value in stats.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
