"""
Discovery degli URL per il chatbot RAG sul DIEM.

Questo script orchestra il primo step della pipeline:
1. legge gli URL iniziali da data/urls.txt
2. aggiunge URL trovati nelle sitemap dei domini configurati
3. visita le pagine HTML con una BFS
4. salva l'HTML grezzo in data/raw_html/<sh>/<hash>.html
5. scrive gli URL scoperti in data/discovered_urls.jsonl
6. aggiorna data/checkpoint.json per poter riprendere un crawl interrotto
7. aggiorna data/discovery_state.json per continuare la frontier tra run

La logica di processing del singolo URL vive in discovery_processor.py.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter, deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse

import httpx
from tqdm import tqdm

from discovery_fetch import DomainRateLimiter, RobotsCache, fetch_sitemap_urls
from discovery_io import (
    load_checkpoint,
    load_config,
    load_discovery_state,
    project_path,
    read_seed_urls,
    save_checkpoint,
    save_discovery_state,
    save_html,
    validate_config,
    write_record,
)
from discovery_models import CrawlItem, CrawlState, PersistentDiscoveryState, ProcessedDiscoveryItem
from discovery_processor import filter_context, process_item
from pipeline_io import now_iso
from url_filters import can_traverse_url, is_pdf_url


def create_initial_state(
    seed_urls: list[str],
    sitemap_urls: list[str],
    persistent_state: PersistentDiscoveryState | None = None,
) -> CrawlState:
    """Crea la BFS da frontier persistita, seed e sitemap."""
    persistent_state = persistent_state or PersistentDiscoveryState(deque(), {}, {})
    initial_items = list(persistent_state.frontier)
    initial_items.extend(CrawlItem(url, 0, "seed") for url in seed_urls)
    initial_items.extend(CrawlItem(url, 0, "sitemap") for url in sitemap_urls)

    deduped_items: list[CrawlItem] = []
    seen_urls: set[str] = set()
    for item in initial_items:
        if item.url in seen_urls:
            continue
        deduped_items.append(item)
        seen_urls.add(item.url)

    queue = deque(deduped_items)
    return CrawlState(
        queue=queue,
        queued={item.url for item in queue},
        visited=set(),
        seen_documents=set(),
        domain_counts=Counter(),
        known_urls=dict(persistent_state.known_urls),
        known_documents=dict(persistent_state.known_documents),
    )


def is_due_for_refresh(last_seen: str | None, config: dict, reference_time: datetime | None = None) -> bool:
    """True se un URL noto può essere ricontrollato."""
    if not last_seen:
        return True

    try:
        seen_at = datetime.fromisoformat(last_seen)
    except ValueError:
        return True

    refresh_after_days = int(config["crawler"].get("refresh_after_days", 7))
    now = reference_time or datetime.now(UTC)
    return seen_at <= now - timedelta(days=refresh_after_days)


def can_visit(
    item: CrawlItem,
    state: CrawlState,
    config: dict,
    reference_time: datetime | None = None,
) -> bool:
    """True se l'URL può essere inserito nel prossimo batch."""
    if item.url in state.visited:
        return False

    if item.url in state.known_urls and not is_due_for_refresh(
        state.known_urls[item.url],
        config,
        reference_time,
    ):
        return False

    ok, _ = can_traverse_url(item.url, config, filter_context(item.discovered_from))
    if not ok:
        return False

    domain = urlparse(item.url).netloc
    domain_limit = config["crawler"]["per_domain_limits"].get(domain, 0)
    if state.domain_counts[domain] >= domain_limit:
        return False

    return len(state.visited) < config["crawler"]["max_total_urls"]


def mark_visited(state: CrawlState, url: str, visited_at: str | None = None) -> bool:
    """Marca un URL visitato e aggiorna il contatore del suo dominio."""
    if url in state.visited:
        return False

    state.visited.add(url)
    state.known_urls[url] = visited_at or now_iso()
    domain = urlparse(url).netloc
    if domain:
        state.domain_counts[domain] += 1
    return True


def take_batch(
    state: CrawlState,
    config: dict,
    reference_time: datetime | None = None,
    visited_at: str | None = None,
) -> list[CrawlItem]:
    """Prende dalla coda un batch di URL validi da scaricare."""
    batch: list[CrawlItem] = []
    batch_size = config["crawler"]["max_concurrent_requests"]

    max_total_urls = config["crawler"]["max_total_urls"]

    while state.queue and len(batch) < batch_size and len(state.visited) < max_total_urls:
        item = state.queue.popleft()
        # queued rappresenta solo gli URL ancora presenti nella coda BFS.
        state.queued.discard(item.url)

        if not can_visit(item, state, config, reference_time):
            continue

        mark_visited(state, item.url, visited_at)
        batch.append(item)

    return batch


def discovery_stop_reason(state: CrawlState, config: dict, exhausted_without_batch: bool = False) -> str:
    """Classifica il motivo di arresto della discovery."""
    if len(state.visited) >= config["crawler"]["max_total_urls"]:
        return "max_total_urls"
    if not state.queue:
        return "queue_exhausted"
    if exhausted_without_batch:
        return "no_eligible_urls"
    return "unknown"


def load_recorded_pdf_urls(output_file: Path) -> set[str]:
    """URL PDF già scritti in discovered_urls.jsonl, utile nei resume."""
    if not output_file.exists():
        return set()

    urls: set[str] = set()
    with output_file.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping malformed line in discovered_urls.jsonl: {exc}")
                continue
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
        if link in state.known_urls and not is_due_for_refresh(state.known_urls[link], config):
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
    run_started_at = now_iso()
    reference_time = datetime.fromisoformat(run_started_at)
    persistent_state = load_discovery_state(config)
    frontier_loaded = len(persistent_state.frontier)

    async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
        state = load_checkpoint(config, persistent_state)

        if state is None:
            output_file.unlink(missing_ok=True)
            recorded_pdf_urls: set[str] = set()
            sitemap_urls = await fetch_sitemap_urls(client, config, limiter)
            state = create_initial_state(seed_urls, sitemap_urls, persistent_state)
        else:
            print("Checkpoint trovato: riprendo il crawl precedente.")
            recorded_pdf_urls = load_recorded_pdf_urls(output_file)

        max_total_urls = config["crawler"]["max_total_urls"]
        exhausted_without_batch = False
        with tqdm(
            total=max_total_urls,
            initial=min(len(state.visited), max_total_urls),
            desc="Discovery",
            unit="url",
            dynamic_ncols=True,
        ) as progress:
            while state.queue and len(state.visited) < max_total_urls:
                batch = take_batch(state, config, reference_time, run_started_at)
                if not batch:
                    exhausted_without_batch = True
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
                            state.known_documents[document_url] = run_started_at
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

                save_checkpoint(
                    state,
                    config,
                    status="in_progress",
                    persistent_state=persistent_state,
                )
                current_progress = min(len(state.visited), max_total_urls)
                if current_progress > progress.n:
                    progress.update(current_progress - progress.n)
                progress.set_postfix(
                    coda=len(state.queue),
                    documenti=len(state.seen_documents),
                    refresh=False,
                )

    stop_reason = discovery_stop_reason(state, config, exhausted_without_batch)
    save_checkpoint(
        state,
        config,
        status="completed",
        stop_reason=stop_reason,
        persistent_state=persistent_state,
    )
    save_discovery_state(state, config)

    return {
        "seed_urls": len(seed_urls),
        "visited_urls": len(state.visited),
        "remaining_queue": len(state.queue),
        "documents_seen": len(state.seen_documents),
        "by_domain": dict(state.domain_counts),
        "by_status": dict(status_counts),
        "by_type": dict(type_counts),
        "frontier_loaded": frontier_loaded,
        "frontier_remaining": len(state.queue),
        "known_urls": len(state.known_urls),
        "known_documents": len(state.known_documents),
        "stop_reason": stop_reason,
        "output_file": config["paths"]["discovered_urls_file"],
        "raw_html_dir": config["paths"]["raw_html_dir"],
        "checkpoint_file": config["paths"]["checkpoint_file"],
        "discovery_state_file": config["paths"]["discovery_state_file"],
    }


def print_config_summary(config: dict) -> None:
    """Mostra i parametri principali prima di iniziare."""
    crawler = config["crawler"]
    paths = config["paths"]

    print("Configurazione discovery")
    print(f"- max_total_urls: {crawler['max_total_urls']}")
    print(f"- max_depth: {crawler['max_depth']}")
    print(f"- max_concurrent_requests: {crawler['max_concurrent_requests']}")
    print(f"- refresh_after_days: {crawler.get('refresh_after_days', 7)}")
    print(f"- raw_html_dir: {paths['raw_html_dir']}")
    print(f"- discovered_urls_file: {paths['discovered_urls_file']}")
    print(f"- discovery_state_file: {paths['discovery_state_file']}")
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
