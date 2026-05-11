"""
Discovery degli URL per il chatbot RAG sul DIEM.

Questo script orchestra il primo step della pipeline:
1. legge gli URL iniziali da data/urls.txt
2. aggiunge URL trovati nelle sitemap dei domini configurati
3. visita le pagine HTML con una BFS
4. salva l'HTML grezzo in data/raw_html/<sh>/<hash>.html
5. scrive gli URL scoperti in data/discovered_urls.jsonl
6. aggiorna data/checkpoint.json per poter riprendere un crawl interrotto
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter, deque
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from tqdm import tqdm

from discovery_fetch import (
    DomainRateLimiter,
    RobotsCache,
    fetch_discovery_candidate,
    fetch_sitemap_urls,
)
from discovery_io import (
    load_checkpoint,
    load_config,
    make_record,
    project_path,
    read_seed_urls,
    save_checkpoint,
    save_html,
    validate_config,
    write_record,
)
from discovery_models import CrawlItem, CrawlState, ProcessedDiscoveryItem
from html_utils import extract_canonical_from_soup, extract_links_from_soup
from url_filters import can_index_url, can_traverse_url, is_pdf_url


def filter_context(discovered_from: str) -> dict[str, str]:
    """Crea il contesto da passare ai filtri che dipendono dalla sorgente."""
    return {"discovered_from": discovered_from}


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

        state.visited.add(item.url)
        state.domain_counts[urlparse(item.url).netloc] += 1
        batch.append(item)

    return batch


def pick_document_url(
    canonical_url: str,
    final_url: str,
    config: dict,
    context: dict[str, str],
) -> tuple[str, str | None]:
    """Usa canonical_url come chiave solo se resta nello scope della discovery."""
    ok, reason = can_traverse_url(canonical_url, config, context)
    if ok:
        return canonical_url, None

    canonical_domain = urlparse(canonical_url).netloc.lower()
    final_domain = urlparse(final_url).netloc.lower()
    if canonical_domain and canonical_domain != final_domain:
        return final_url, "canonical_out_of_scope"

    return final_url, reason


def split_document_links(links: list[str]) -> tuple[list[str], list[str]]:
    """Separa i link HTML da attraversare dai PDF da registrare subito."""
    traversal_links: list[str] = []
    pdf_links: list[str] = []

    for link in links:
        if is_pdf_url(link):
            pdf_links.append(link)
        else:
            traversal_links.append(link)

    return traversal_links, pdf_links


async def make_linked_pdf_records(
    pdf_links: list[str],
    parent_url: str,
    parent_depth: int,
    robots: RobotsCache | None,
) -> list[dict]:
    """Crea record PDF appena il link viene trovato in una pagina HTML."""
    records: list[dict] = []

    for pdf_url in pdf_links:
        item = CrawlItem(pdf_url, parent_depth + 1, parent_url)
        status = "pending_download"
        if robots and not await robots.can_fetch(pdf_url):
            status = "robots_denied"

        records.append(
            make_record(
                item,
                "pdf",
                status,
                final_url=pdf_url,
                document_url=pdf_url,
                raw_path=None,
                discovery_method="html_link",
            )
        )

    return records


def processed_item(
    record: dict,
    traversal_links: list[str] | None = None,
    html: str | None = None,
    additional_visited: list[str] | None = None,
    linked_pdf_records: list[dict] | None = None,
) -> ProcessedDiscoveryItem:
    """Costruisce un risultato di discovery con campi espliciti."""
    return ProcessedDiscoveryItem(
        record=record,
        traversal_links=traversal_links or [],
        html=html,
        additional_visited=additional_visited or [],
        linked_pdf_records=linked_pdf_records or [],
    )


async def process_item(
    item: CrawlItem,
    client: httpx.AsyncClient,
    limiter: DomainRateLimiter,
    robots: RobotsCache | None,
    config: dict,
    seen_documents: set[str],
) -> ProcessedDiscoveryItem:
    """Scarica un URL e restituisce un risultato di discovery.

    Flusso dei filtri:
    - seed, sitemap e link estratti passano da can_traverse_url;
    - solo il document_url HTML finale passa da can_index_url;
    - i PDF restano nel manifest con status pending_download.

    I PDF linkati da una pagina HTML vengono registrati subito, senza attendere
    che vengano pescati dalla coda BFS.
    """
    if is_pdf_url(item.url):
        if robots and not await robots.can_fetch(item.url):
            return processed_item(make_record(item, "pdf", "robots_denied"))

        record = make_record(
            item,
            "pdf",
            "pending_download",
            final_url=item.url,
            document_url=item.url,
            raw_path=None,
        )
        return processed_item(record)

    if robots and not await robots.can_fetch(item.url):
        return processed_item(make_record(item, "unknown", "robots_denied"))

    max_bytes = int(config["crawler"].get("max_html_bytes", 5_000_000))
    try:
        candidate = await fetch_discovery_candidate(client, item.url, limiter, max_bytes)
    except httpx.HTTPError as error:
        return processed_item(make_record(item, "html", "failed", error=str(error), raw_path=None))

    final_url = candidate.final_url
    context = filter_context(item.url)
    additional_visited = [final_url] if final_url != item.url else []

    if final_url in seen_documents:
        record = make_record(
            item,
            "html",
            "duplicate_redirect",
            final_url=final_url,
            document_url=final_url,
            content_type=candidate.content_type,
            mime=candidate.mime,
            duplicate_of=final_url,
            raw_path=None,
        )
        return processed_item(record, additional_visited=additional_visited)

    ok_final, final_reason = can_traverse_url(final_url, config, context)
    if not ok_final:
        record = make_record(
            item,
            "other",
            "redirected_out_of_scope",
            final_url=final_url,
            document_url=final_url,
            content_type=candidate.content_type,
            mime=candidate.mime,
            skip_reason=final_reason,
            raw_path=None,
        )
        return processed_item(record, additional_visited=additional_visited)

    if is_pdf_url(final_url) or candidate.mime == "application/pdf":
        record = make_record(
            item,
            "pdf",
            "pending_download",
            final_url=final_url,
            document_url=final_url,
            content_type=candidate.content_type,
            mime=candidate.mime,
            raw_path=None,
        )
        return processed_item(record, additional_visited=additional_visited)

    if candidate.html is None:
        status = "too_large" if candidate.too_large else "non_html"
        source_type = "html" if candidate.too_large else "other"
        record = make_record(
            item,
            source_type,
            status,
            final_url=final_url,
            document_url=final_url,
            content_type=candidate.content_type,
            mime=candidate.mime,
            content_length=candidate.content_length,
            raw_path=None,
        )
        return processed_item(record, additional_visited=additional_visited)

    soup = BeautifulSoup(candidate.html, "lxml")
    canonical_url = extract_canonical_from_soup(soup, final_url)
    document_url, canonical_skip_reason = pick_document_url(
        canonical_url,
        final_url,
        config,
        context,
    )
    all_links = extract_links_from_soup(soup, final_url, config)
    links, pdf_links = split_document_links(all_links)
    linked_pdf_records = await make_linked_pdf_records(
        pdf_links,
        final_url,
        item.depth,
        robots,
    )
    indexable, index_reason = can_index_url(document_url, config, context)
    status = "ok" if indexable else "not_indexable"

    if document_url not in (item.url, final_url):
        additional_visited.append(document_url)

    record = make_record(
        item,
        "html",
        status,
        final_url=final_url,
        canonical_url=canonical_url,
        document_url=document_url,
        indexable=indexable,
        raw_path=None,
        content_type=candidate.content_type,
        mime=candidate.mime,
        links_found=len(all_links),
        pdf_links_found=len(pdf_links),
        canonical_skip_reason=canonical_skip_reason,
        index_skip_reason=None if indexable else index_reason,
    )
    return processed_item(
        record,
        traversal_links=links,
        html=candidate.html if indexable else None,
        additional_visited=additional_visited,
        linked_pdf_records=linked_pdf_records,
    )


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


async def run_discovery(config: dict) -> dict:
    """Esegue la discovery e ritorna statistiche finali."""
    seed_urls = read_seed_urls(config)
    output_file = project_path(config["paths"]["discovered_urls_file"])

    headers = {"User-Agent": config["crawler"]["user_agent"]}
    timeout = httpx.Timeout(config["crawler"]["timeout"])
    limiter = DomainRateLimiter(config["crawler"]["rate_limit_per_domain_rps"])
    robots = None
    if config["crawler"].get("respect_robots_txt", True):
        robots = RobotsCache(config["crawler"]["user_agent"])

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

                for item, processed in zip(batch, results, strict=True):
                    record = processed.record

                    for extra in processed.additional_visited:
                        state.visited.add(extra)

                    if record["type"] == "html" and record["indexable"]:
                        document_url = record["document_url"]
                        if document_url in state.seen_documents:
                            record["status"] = "duplicate_canonical"
                            record["indexable"] = False
                            record["duplicate_of"] = document_url
                            record["raw_path"] = None
                        else:
                            state.seen_documents.add(document_url)
                            record["raw_path"] = save_html(document_url, processed.html or "", config)

                    write_record(output_file, record, status_counts, type_counts)
                    if record["type"] == "pdf":
                        recorded_pdf_urls.add(record["url"])

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
