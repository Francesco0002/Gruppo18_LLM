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
8. espone metriche per depth utili a capire se la frontiera si sta esaurendo
   tra run successive con budget limitato
9. conserva le pagine HTML di bordo da riespandere quando aumenta max_depth
10. mantiene una whitelist persistente dei profili docente DIEM autorizzati
11. rinvia gli URL bloccati dal limite di dominio invece di perderli

La logica di processing del singolo URL vive in discovery_processor.py.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter, deque
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode, urlparse

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
from pdf_policy import DIEM_BANDI_STRUCTURED_MODULES, DIEM_BANDI_UNSTRUCTURED_MODULES
from url_filters import can_traverse_url, is_pdf_url
from url_filters import (
    DIEM_BANDI_STRUCTURE_ID,
    directory_person_matricola,
    is_diem_personnel_url,
    is_directory_person_url,
    normalize_url,
    teacher_profile_key,
)


DIEM_BANDI_ARCHIVE_START_YEAR = 2013


def count_items_by_depth(
    items: list[CrawlItem] | deque[CrawlItem],
    *,
    force_revisit: bool | None = None,
) -> dict[str, int]:
    """Conta item BFS per depth, con filtro opzionale sulle riespansioni."""
    counts = Counter(
        item.depth
        for item in items
        if force_revisit is None or item.force_revisit is force_revisit
    )
    return {str(depth): counts[depth] for depth in sorted(counts)}


def item_origin_seed(item: CrawlItem) -> str:
    """Chiave di fairness del ramo, compatibile anche con stati legacy."""
    if item.origin_seed:
        return item.origin_seed
    if item.depth == 0:
        return item.url
    # Nei vecchi checkpoint non esisteva origin_seed: il parent immediato è la
    # migliore informazione disponibile senza ricostruire l'intero grafo.
    return item.discovered_from


def count_items_by_seed_and_depth(
    items: list[CrawlItem] | deque[CrawlItem],
    *,
    force_revisit: bool | None = None,
) -> dict[str, dict[str, int]]:
    """Conta il lavoro residuo per origine e depth, utile a vedere rami sbilanciati."""
    counts: dict[str, Counter[int]] = {}
    for item in items:
        if force_revisit is not None and item.force_revisit != force_revisit:
            continue
        seed_counts = counts.setdefault(item_origin_seed(item), Counter())
        seed_counts[item.depth] += 1
    return {
        seed: {str(depth): seed_counts[depth] for depth in sorted(seed_counts)}
        for seed, seed_counts in sorted(counts.items())
    }


def fair_bfs_order(items: list[CrawlItem]) -> list[CrawlItem]:
    """Ordina prima per depth e poi alterna i rami dei seed nello stesso livello."""
    by_depth: dict[int, list[CrawlItem]] = {}
    for item in items:
        by_depth.setdefault(item.depth, []).append(item)

    ordered: list[CrawlItem] = []
    for depth in sorted(by_depth):
        per_seed: dict[str, deque[CrawlItem]] = {}
        for item in by_depth[depth]:
            per_seed.setdefault(item_origin_seed(item), deque()).append(item)

        # Un giro prende al massimo un URL per seed, poi riparte: un ramo molto
        # prolifico non può occupare tutta la capacità del livello da solo.
        seed_order = list(per_seed)
        while seed_order:
            next_round: list[str] = []
            for seed in seed_order:
                ordered.append(per_seed[seed].popleft())
                if per_seed[seed]:
                    next_round.append(seed)
            seed_order = next_round
    return ordered


def rebalance_queue_for_fair_bfs(state: CrawlState) -> None:
    """Rende esplicita la BFS equa prima di salvare o scegliere il batch seguente."""
    if len(state.queue) < 2:
        return
    state.queue = deque(fair_bfs_order(list(state.queue)))


def count_recorded_html_by_depth(output_file: Path) -> dict[str, int]:
    """Conta gli HTML prodotti nella run corrente, separandoli dagli URL PDF terminali."""
    if not output_file.exists():
        return {}

    counts: Counter[int] = Counter()
    with output_file.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping malformed line in discovered_urls.jsonl: {exc}")
                continue
            if record.get("type") != "html":
                continue
            depth = record.get("depth")
            if isinstance(depth, int):
                counts[depth] += 1

    return {str(depth): counts[depth] for depth in sorted(counts)}


def eligible_reexpansion_items(
    persistent_state: PersistentDiscoveryState,
    max_depth: int,
) -> list[CrawlItem]:
    """Pagine di bordo già viste che possono aprire il livello successivo."""
    return [
        CrawlItem(
            item.url,
            item.depth,
            item.discovered_from,
            force_revisit=True,
            origin_seed=item.origin_seed,
        )
        for item in persistent_state.expansion_backlog
        if item.depth < max_depth
    ]


def due_bootstrap_items(
    urls: list[str],
    source: str,
    persistent_state: PersistentDiscoveryState,
    config: dict | None,
    reference_time: datetime | None,
) -> list[CrawlItem]:
    """Seed e sitemap da accodare solo se nuovi o dovuti al refresh."""
    items: list[CrawlItem] = []
    for url in urls:
        last_seen = persistent_state.known_urls.get(url)
        if (
            last_seen
            and url in persistent_state.known_documents
            and config is not None
            and not is_due_for_refresh(
                last_seen,
                config,
                reference_time,
            )
        ):
            continue
        items.append(CrawlItem(url, 0, source, origin_seed=url))
    return items


def missing_teacher_profile_items(
    persistent_state: PersistentDiscoveryState,
    config: dict,
) -> tuple[list[CrawlItem], set[str]]:
    """Accoda profili docente numerici mancanti partendo da rubrica gia' validata.

    Se una run precedente ha indicizzato la scheda rubrica DIEM ma ha saltato il
    profilo `docenti.unisa.it/<matricola>/home`, la frontier puo' essere vuota:
    questo backfill ricostruisce solo quei profili partendo da documenti rubrica
    gia' entrati nello scope.
    """
    teacher_domain = str(
        config.get("scope", {}).get("teacher_domain", "docenti.unisa.it")
    ).lower()
    items: list[CrawlItem] = []
    allowed_profiles: set[str] = set()

    for directory_url in sorted(persistent_state.known_documents):
        matricola = directory_person_matricola(directory_url, config)
        if not matricola:
            continue

        profile_url = normalize_url(f"https://{teacher_domain}/{matricola}/home")
        if profile_url in persistent_state.known_documents:
            continue

        allowed_profiles.add(matricola.lower())
        items.append(
            CrawlItem(
                profile_url,
                0,
                directory_url,
                origin_seed=directory_url,
            )
        )

    return items, allowed_profiles


def diem_bandi_archive_urls(
    config: dict,
    reference_time: datetime | None = None,
) -> list[str]:
    """URL espliciti per archiviare tutte le sezioni bandi DIEM per anno."""
    crawler_config = config.get("crawler", {})
    if not crawler_config.get("include_diem_bandi_archives", True):
        return []

    start_year = int(
        crawler_config.get(
            "diem_bandi_archive_start_year",
            DIEM_BANDI_ARCHIVE_START_YEAR,
        )
    )
    end_year = int(
        crawler_config.get(
            "diem_bandi_archive_end_year",
            (reference_time or datetime.now(UTC)).year,
        )
    )
    if start_year > end_year:
        return []

    diem_domain = str(
        config.get("scope", {}).get("diem_domain", "www.diem.unisa.it")
    ).lower()
    structure_id = str(
        config.get("scope", {}).get("diem_bandi_structure_id", DIEM_BANDI_STRUCTURE_ID)
    )
    base_url = f"https://{diem_domain}/home/bandi"
    modules: list[tuple[str, str | None]] = [
        *sorted(DIEM_BANDI_STRUCTURED_MODULES.items()),
        *((module, None) for module in sorted(DIEM_BANDI_UNSTRUCTURED_MODULES)),
    ]

    urls: list[str] = []
    for module, structure_param in modules:
        base_params = {"modulo": module}
        if structure_param is not None:
            query_param = (
                "cdsStruttura"
                if structure_param == "cdsstruttura"
                else structure_param
            )
            base_params[query_param] = structure_id
        urls.append(normalize_url(f"{base_url}?{urlencode(base_params)}"))

        for year in range(end_year, start_year - 1, -1):
            params = {"anno": str(year), **base_params}
            urls.append(normalize_url(f"{base_url}?{urlencode(params)}"))

    return urls


def missing_diem_bandi_archive_items(
    persistent_state: PersistentDiscoveryState,
    config: dict,
    reference_time: datetime | None = None,
) -> list[CrawlItem]:
    """Backfill dei bandi annuali, indipendente dai link oggi visibili."""
    return due_bootstrap_items(
        diem_bandi_archive_urls(config, reference_time),
        "diem_bandi_archive",
        persistent_state,
        config,
        reference_time,
    )


def has_diem_bandi_seed(seed_urls: list[str], config: dict) -> bool:
    """True se i seed dichiarano che la sezione bandi DIEM fa parte del crawl."""
    diem_domain = str(
        config.get("scope", {}).get("diem_domain", "www.diem.unisa.it")
    ).lower()
    return any(
        urlparse(url).netloc.lower() == diem_domain
        and urlparse(url).path.rstrip("/").lower() == "/home/bandi"
        for url in seed_urls
    )


def create_initial_state(
    seed_urls: list[str],
    sitemap_urls: list[str],
    persistent_state: PersistentDiscoveryState | None = None,
    max_depth: int | None = None,
    config: dict | None = None,
    reference_time: datetime | None = None,
) -> CrawlState:
    """Crea la BFS da lavoro persistito più bootstrap nuovi o da refresh."""
    persistent_state = persistent_state or PersistentDiscoveryState(deque(), {}, {})
    initial_items = list(persistent_state.frontier)
    if max_depth is not None:
        initial_items.extend(eligible_reexpansion_items(persistent_state, max_depth))
    initial_items.extend(
        due_bootstrap_items(seed_urls, "seed", persistent_state, config, reference_time)
    )
    initial_items.extend(
        due_bootstrap_items(sitemap_urls, "sitemap", persistent_state, config, reference_time)
    )
    backfill_teacher_profiles: set[str] = set()
    if config is not None:
        backfill_items, backfill_teacher_profiles = missing_teacher_profile_items(
            persistent_state,
            config,
        )
        initial_items.extend(backfill_items)
        if has_diem_bandi_seed(seed_urls, config):
            initial_items.extend(
                missing_diem_bandi_archive_items(
                    persistent_state,
                    config,
                    reference_time,
                )
            )

    deduped_items: list[CrawlItem] = []
    seen_urls: set[str] = set()
    for item in initial_items:
        if item.url in seen_urls:
            continue
        deduped_items.append(item)
        seen_urls.add(item.url)

    queue = deque(fair_bfs_order(deduped_items))
    return CrawlState(
        queue=queue,
        queued={item.url for item in queue},
        visited=set(),
        seen_documents=set(),
        domain_counts=Counter(),
        known_urls=dict(persistent_state.known_urls),
        known_documents=dict(persistent_state.known_documents),
        allowed_teacher_profiles={
            *persistent_state.allowed_teacher_profiles,
            *backfill_teacher_profiles,
        },
        expansion_backlog={item.url: item for item in persistent_state.expansion_backlog},
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
    return visit_skip_reason(item, state, config, reference_time) is None


def visit_skip_reason(
    item: CrawlItem,
    state: CrawlState,
    config: dict,
    reference_time: datetime | None = None,
) -> str | None:
    """Motivo per cui un item non può entrare nel batch corrente, se esiste."""
    if item.url in state.visited:
        return "already_visited"

    has_known_document = item.url in state.known_documents
    if (
        not item.force_revisit
        and item.url in state.known_urls
        and has_known_document
        and not is_due_for_refresh(
            state.known_urls[item.url],
            config,
            reference_time,
        )
    ):
        return "recently_known"

    ok, reason = can_traverse_url(
        item.url,
        config,
        filter_context(item.discovered_from, state.allowed_teacher_profiles),
    )
    if not ok:
        return reason

    domain = urlparse(item.url).netloc
    domain_limit = config["crawler"]["per_domain_limits"].get(domain, 0)
    if state.domain_counts[domain] >= domain_limit:
        return "domain_limit"

    if len(state.visited) >= config["crawler"]["max_total_urls"]:
        return "max_total_urls"

    return None


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
    skip_counts: Counter[str] | None = None,
    skip_counts_by_depth: Counter[tuple[str, int]] | None = None,
) -> list[CrawlItem]:
    """Prende dalla coda un batch di URL validi senza perdere lavoro rinviabile."""
    batch: list[CrawlItem] = []
    batch_size = config["crawler"]["max_concurrent_requests"]
    max_total_urls = config["crawler"]["max_total_urls"]
    items_to_scan = len(state.queue)
    deferred_items: list[CrawlItem] = []

    while (
        state.queue
        and items_to_scan > 0
        and len(batch) < batch_size
        and len(state.visited) < max_total_urls
    ):
        item = state.queue.popleft()
        items_to_scan -= 1
        # queued rappresenta solo gli URL ancora presenti nella coda BFS.
        state.queued.discard(item.url)

        skip_reason = visit_skip_reason(item, state, config, reference_time)
        if skip_reason is not None:
            if skip_counts is not None:
                skip_counts[skip_reason] += 1
            if skip_counts_by_depth is not None:
                skip_counts_by_depth[(skip_reason, item.depth)] += 1
            if skip_reason == "domain_limit":
                # Il limite di dominio è un blocco del run corrente, non prova
                # che l'URL sia stato visitato: va riprovato in run future.
                deferred_items.append(item)
                state.queued.add(item.url)
            elif item.force_revisit:
                # Una riespansione forzata che oggi non passa piu' i filtri non
                # deve restare nel backlog persistente e bloccare ogni run.
                state.expansion_backlog.pop(item.url, None)
            continue

        mark_visited(state, item.url, visited_at)
        batch.append(item)

    state.queue.extend(deferred_items)
    return batch


def discovery_stop_reason(state: CrawlState, config: dict, exhausted_without_batch: bool = False) -> str:
    """Classifica il motivo di arresto della discovery."""
    if len(state.visited) >= config["crawler"]["max_total_urls"]:
        return "max_total_urls"
    if exhausted_without_batch:
        return "no_eligible_urls"
    if not state.queue:
        return "queue_exhausted"
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
) -> bool:
    """Aggiunge i link trovati e indica se il parent è stato espanso."""
    if parent.depth >= config["crawler"]["max_depth"]:
        return False

    for link in links:
        profile = teacher_profile_key(link, config)
        if profile and (
            is_diem_personnel_url(parent.url, config)
            or (
                is_directory_person_url(parent.url, config)
                and is_diem_personnel_url(parent.discovered_from, config)
            )
        ):
            state.allowed_teacher_profiles.add(profile)

        if link in state.visited or link in state.queued:
            continue
        if (
            link in state.known_urls
            and link in state.known_documents
            and not is_due_for_refresh(state.known_urls[link], config)
        ):
            continue
        state.queue.append(
            CrawlItem(
                link,
                parent.depth + 1,
                parent.url,
                origin_seed=parent.origin_seed,
            )
        )
        state.queued.add(link)

    return True


def can_track_expansion(record: dict, expand_links: bool) -> bool:
    """True quando la pagina HTML è stata letta e può generare figli."""
    return (
        expand_links
        and record.get("type") == "html"
        and record.get("status") in {"ok", "not_indexable"}
    )


def update_expansion_backlog(
    item: CrawlItem,
    record: dict,
    expand_links: bool,
    expanded: bool,
    state: CrawlState,
) -> None:
    """Aggiorna le pagine di bordo da riespandere in run con depth maggiore."""
    if not expand_links:
        state.expansion_backlog.pop(item.url, None)
        return

    if not can_track_expansion(record, expand_links):
        return

    if expanded:
        state.expansion_backlog.pop(item.url, None)
        return

    state.expansion_backlog[item.url] = CrawlItem(
        item.url,
        item.depth,
        item.discovered_from,
        origin_seed=item.origin_seed,
    )


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
    skip_counts: Counter[str] = Counter()
    skip_counts_by_depth: Counter[tuple[str, int]] = Counter()
    run_started_at = now_iso()
    reference_time = datetime.fromisoformat(run_started_at)
    persistent_state = load_discovery_state(config)
    frontier_loaded = len(persistent_state.frontier)
    frontier_before_by_depth = count_items_by_depth(
        persistent_state.frontier,
        force_revisit=False,
    )
    expansion_backlog_before_by_depth = count_items_by_depth(persistent_state.expansion_backlog)
    reexpansion_loaded = len(
        eligible_reexpansion_items(persistent_state, config["crawler"]["max_depth"])
    )

    async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
        state = load_checkpoint(config, persistent_state)

        if state is None:
            output_file.unlink(missing_ok=True)
            recorded_pdf_urls: set[str] = set()
            sitemap_urls = await fetch_sitemap_urls(client, config, limiter)
            state = create_initial_state(
                seed_urls,
                sitemap_urls,
                persistent_state,
                config["crawler"]["max_depth"],
                config,
                reference_time,
            )
        else:
            print("Checkpoint trovato: riprendo il crawl precedente.")
            recorded_pdf_urls = load_recorded_pdf_urls(output_file)
            rebalance_queue_for_fair_bfs(state)

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
                batch = take_batch(
                    state,
                    config,
                    reference_time,
                    run_started_at,
                    skip_counts,
                    skip_counts_by_depth,
                )
                if not batch:
                    exhausted_without_batch = True
                    break

                tasks = [
                    process_item(
                        item,
                        client,
                        limiter,
                        robots,
                        config,
                        state.seen_documents,
                        state.allowed_teacher_profiles,
                    )
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

                    expanded = False
                    if expand_links:
                        for pdf_record in processed.linked_pdf_records:
                            if pdf_record["url"] in recorded_pdf_urls:
                                continue
                            write_record(output_file, pdf_record, status_counts, type_counts)
                            recorded_pdf_urls.add(pdf_record["url"])

                        expanded = enqueue_links(item, processed.traversal_links, state, config)

                    update_expansion_backlog(
                        item,
                        record,
                        expand_links,
                        expanded,
                        state,
                    )

                # Dopo aver accodato tutti i figli del batch, ripristiniamo
                # l'alternanza per seed prima di scegliere il batch successivo.
                rebalance_queue_for_fair_bfs(state)

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

    if stop_reason == "no_eligible_urls":
        print(
            "Discovery senza nuovi URL eleggibili: "
            f"{dict(skip_counts)}"
        )

    return {
        "seed_urls": len(seed_urls),
        "visited_urls": len(state.visited),
        "remaining_queue": len(state.queue),
        "documents_seen": len(state.seen_documents),
        "by_domain": dict(state.domain_counts),
        "by_status": dict(status_counts),
        "by_type": dict(type_counts),
        "skipped_by_reason": dict(skip_counts),
        "skipped_by_reason_by_depth": {
            reason: {
                str(depth): count
                for (skip_reason, depth), count in sorted(skip_counts_by_depth.items())
                if skip_reason == reason
            }
            for reason in sorted({reason for reason, _ in skip_counts_by_depth})
        },
        "frontier_loaded": frontier_loaded,
        "frontier_remaining": len(state.queue),
        "frontier_before_by_depth": frontier_before_by_depth,
        "frontier_before_by_seed_and_depth": count_items_by_seed_and_depth(
            persistent_state.frontier,
            force_revisit=False,
        ),
        "frontier_after_by_depth": count_items_by_depth(
            state.queue,
            force_revisit=False,
        ),
        "frontier_after_by_seed_and_depth": count_items_by_seed_and_depth(
            state.queue,
            force_revisit=False,
        ),
        "reexpansion_frontier_after_by_depth": count_items_by_depth(
            state.queue,
            force_revisit=True,
        ),
        "reexpansion_loaded": reexpansion_loaded,
        "expansion_backlog_before_by_depth": expansion_backlog_before_by_depth,
        "expansion_backlog_after_by_depth": count_items_by_depth(
            list(state.expansion_backlog.values())
        ),
        "processed_html_by_depth": count_recorded_html_by_depth(output_file),
        "known_urls": len(state.known_urls),
        "known_documents": len(state.known_documents),
        "allowed_teacher_profiles": len(state.allowed_teacher_profiles),
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
