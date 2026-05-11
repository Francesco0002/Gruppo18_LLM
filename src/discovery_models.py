"""
Dataclass condivise dalla fase di discovery.

I modelli tengono separati i contratti dati dalla logica:
- CrawlItem rappresenta un URL nella coda BFS;
- CrawlState rappresenta lo stato persistito nel checkpoint;
- FetchResult rappresenta una risposta HTTP già classificata;
- ProcessedDiscoveryItem rappresenta l'esito del processing di un URL.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass


@dataclass
class CrawlItem:
    """Singolo URL nella coda BFS."""

    url: str
    depth: int
    discovered_from: str


@dataclass
class CrawlState:
    """Stato salvato nel checkpoint."""

    queue: deque[CrawlItem]
    queued: set[str]
    visited: set[str]
    seen_documents: set[str]
    domain_counts: Counter[str]


@dataclass
class FetchResult:
    """Risultato leggero di una richiesta HTTP nella discovery."""

    final_url: str
    content_type: str
    mime: str
    content_length: int | None
    too_large: bool
    html: str | None


@dataclass
class ProcessedDiscoveryItem:
    """Risultato del processing di un singolo URL in discovery."""

    record: dict
    traversal_links: list[str]
    html: str | None
    additional_visited: list[str]
    linked_pdf_records: list[dict]
