"""
Dataclass condivise dalla fase di discovery.

I modelli tengono separati i contratti dati dalla logica:
- CrawlItem rappresenta un URL nella coda BFS;
- CrawlState rappresenta lo stato persistito nel checkpoint;
- PersistentDiscoveryState conserva anche le pagine di bordo da riespandere
  quando una run successiva aumenta max_depth e la whitelist dei docenti DIEM;
- FetchResult rappresenta una risposta HTTP già classificata;
- ProcessedDiscoveryItem rappresenta l'esito del processing di un URL.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

from pipeline_types import DiscoveryRecord


@dataclass
class CrawlItem:
    """Singolo URL nella coda BFS."""

    url: str
    depth: int
    discovered_from: str
    force_revisit: bool = False
    # Seed/bootstrap root da cui discende il ramo; resta opzionale per leggere
    # checkpoint e frontier prodotti dalle versioni precedenti della pipeline.
    origin_seed: str | None = None


@dataclass
class CrawlState:
    """Stato salvato nel checkpoint.

    `allowed_teacher_profiles` contiene solo profili docenti autorizzati dal
    personale DIEM. `expansion_backlog` conserva pagine gia viste al bordo della
    depth corrente, da riaprire se una run successiva aumenta `max_depth`.
    """

    queue: deque[CrawlItem]
    queued: set[str]
    visited: set[str]
    seen_documents: set[str]
    domain_counts: Counter[str]
    known_urls: dict[str, str] = field(default_factory=dict)
    known_documents: dict[str, str] = field(default_factory=dict)
    allowed_teacher_profiles: set[str] = field(default_factory=set)
    expansion_backlog: dict[str, CrawlItem] = field(default_factory=dict)


@dataclass
class PersistentDiscoveryState:
    """Memoria cumulativa condivisa tra run completati.

    `frontier` è lavoro ancora da visitare; `expansion_backlog` è lavoro già
    visitato ma utile solo per aprire depth future.
    """

    frontier: deque[CrawlItem]
    known_urls: dict[str, str]
    known_documents: dict[str, str]
    allowed_teacher_profiles: set[str] = field(default_factory=set)
    expansion_backlog: deque[CrawlItem] = field(default_factory=deque)


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

    record: DiscoveryRecord
    traversal_links: list[str]
    html: str | None
    additional_visited: list[str]
    linked_pdf_records: list[DiscoveryRecord]
