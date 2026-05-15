"""
Processing di un singolo URL durante la discovery.

Questo modulo contiene la logica che trasforma un CrawlItem in un
ProcessedDiscoveryItem:
- controllo robots.txt;
- fetch HTTP leggero;
- gestione redirect e canonical;
- distinzione HTML/PDF;
- estrazione link da HTML;
- registrazione immediata dei PDF linkati.

`discover.py` resta così focalizzato sull'orchestrazione BFS, checkpoint e
progress bar.
"""

from __future__ import annotations

from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from discovery_fetch import DomainRateLimiter, RobotsCache, fetch_discovery_candidate
from discovery_io import make_record
from discovery_models import CrawlItem, FetchResult, ProcessedDiscoveryItem
from html_utils import extract_canonical_from_soup, extract_links_from_soup
from pipeline_types import DiscoveryRecord
from url_filters import can_index_url, can_traverse_url, is_pdf_url


def filter_context(discovered_from: str) -> dict[str, str]:
    """Crea il contesto da passare ai filtri che dipendono dalla sorgente."""
    return {"discovered_from": discovered_from}


def pick_document_url(
    canonical_url: str,
    final_url: str,
    config: dict,
    context: dict[str, str],
) -> tuple[str, str | None]:
    """Usa canonical_url come chiave solo se resta nello scope della discovery.

    Se il canonical dichiarato esce dallo scope, deduplichiamo sul final_url:
    è una scelta conservativa per non far collassare pagine in-scope diverse
    su una chiave fuori perimetro.
    """
    if not canonical_url or not canonical_url.startswith(("http://", "https://")):
        return final_url, "no_canonical"

    ok, reason = can_traverse_url(canonical_url, config, context)
    if ok:
        return canonical_url, None

    canonical_domain = urlparse(canonical_url).netloc.lower()
    final_domain = urlparse(final_url).netloc.lower()
    if canonical_domain and canonical_domain != final_domain:
        return final_url, "canonical_out_of_scope"

    return final_url, reason


def split_document_links(links: list[str]) -> tuple[list[str], list[str]]:
    """Separa link HTML da attraversare e PDF da registrare subito."""
    traversal_links: list[str] = []
    pdf_links: list[str] = []

    for link in links:
        if is_pdf_url(link):
            pdf_links.append(link)
        else:
            traversal_links.append(link)

    return traversal_links, pdf_links


def processed_item(
    record: DiscoveryRecord,
    traversal_links: list[str] | None = None,
    html: str | None = None,
    additional_visited: list[str] | None = None,
    linked_pdf_records: list[DiscoveryRecord] | None = None,
) -> ProcessedDiscoveryItem:
    """Costruisce un risultato di discovery con campi espliciti."""
    return ProcessedDiscoveryItem(
        record=record,
        traversal_links=traversal_links or [],
        html=html,
        additional_visited=additional_visited or [],
        linked_pdf_records=linked_pdf_records or [],
    )


def make_pdf_record(item: CrawlItem, status: str, **extra: object) -> DiscoveryRecord:
    """Crea un record PDF con campi comuni coerenti."""
    final_url = str(extra.pop("final_url", item.url))
    return make_record(
        item,
        "pdf",
        status,
        final_url=final_url,
        document_url=final_url,
        raw_path=None,
        **extra,
    )


def make_fetch_record(
    item: CrawlItem,
    source_type: str,
    status: str,
    final_url: str,
    candidate: FetchResult,
    **extra: object,
) -> DiscoveryRecord:
    """Crea un record per esiti legati a una risposta HTTP già classificata."""
    return make_record(
        item,
        source_type,
        status,
        final_url=final_url,
        document_url=final_url,
        content_type=candidate.content_type,
        mime=candidate.mime,
        raw_path=None,
        **extra,
    )


async def make_linked_pdf_records(
    pdf_links: list[str],
    parent_url: str,
    parent_depth: int,
    robots: RobotsCache | None,
) -> list[DiscoveryRecord]:
    """Crea record PDF appena il link viene trovato in una pagina HTML."""
    records: list[DiscoveryRecord] = []

    for pdf_url in pdf_links:
        item = CrawlItem(pdf_url, parent_depth + 1, parent_url)
        status = "pending_download"
        if robots and not await robots.can_fetch(pdf_url):
            status = "robots_denied"

        records.append(
            make_pdf_record(
                item,
                status,
                final_url=pdf_url,
                discovery_method="html_link",
            )
        )

    return records


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
        ok, reason = can_traverse_url(item.url, config, context)
        if not ok:
            return processed_item(
                make_pdf_record(item, "out_of_scope", skip_reason=reason)
            )
        if robots and not await robots.can_fetch(item.url):
            return processed_item(make_pdf_record(item, "robots_denied"))

        return processed_item(make_pdf_record(item, "pending_download"))

    if robots and not await robots.can_fetch(item.url):
        return processed_item(make_record(item, "unknown", "robots_denied"))

    max_bytes = int(config["crawler"].get("max_html_bytes", 5_000_000))
    try:
        candidate = await fetch_discovery_candidate(client, item.url, limiter, max_bytes)
    except httpx.HTTPError as error:
        return processed_item(make_record(item, "html", "failed", error=str(error), raw_path=None))

    final_url = candidate.final_url
    context = filter_context(item.discovered_from)
    additional_visited = [final_url] if final_url != item.url else []

    ok_final, final_reason = can_traverse_url(final_url, config, context)
    if not ok_final:
        record = make_fetch_record(
            item,
            "other",
            "redirected_out_of_scope",
            final_url=final_url,
            candidate=candidate,
            skip_reason=final_reason,
        )
        return processed_item(record, additional_visited=additional_visited)

    if final_url in seen_documents:
        record = make_fetch_record(
            item,
            "html",
            "duplicate_redirect",
            final_url=final_url,
            candidate=candidate,
            duplicate_of=final_url,
        )
        return processed_item(record, additional_visited=additional_visited)

    if is_pdf_url(final_url) or candidate.mime == "application/pdf":
        record = make_pdf_record(
            item,
            "pending_download",
            final_url=final_url,
            content_type=candidate.content_type,
            mime=candidate.mime,
        )
        return processed_item(record, additional_visited=additional_visited)

    if candidate.html is None:
        status = "too_large" if candidate.too_large else "non_html"
        source_type = "html" if candidate.too_large else "other"
        record = make_fetch_record(
            item,
            source_type,
            status,
            final_url=final_url,
            candidate=candidate,
            content_length=candidate.content_length,
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
    if document_url not in (item.url, final_url):
        additional_visited.append(document_url)

    if document_url in seen_documents:
        record = make_record(
            item,
            "html",
            "duplicate_canonical",
            final_url=final_url,
            canonical_url=canonical_url,
            document_url=document_url,
            indexable=False,
            duplicate_of=document_url,
            raw_path=None,
            content_type=candidate.content_type,
            mime=candidate.mime,
            canonical_skip_reason=canonical_skip_reason,
        )
        return processed_item(record, additional_visited=additional_visited)

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
