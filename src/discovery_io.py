"""
Input/output su filesystem per la fase di discovery.

Responsabilità:
- leggere e validare config.yaml;
- risolvere path relativi alla root progetto;
- leggere seed URL;
- creare record per discovered_urls.jsonl;
- salvare HTML grezzo, checkpoint BFS e backlog delle pagine di bordo da
  riespandere quando aumenta max_depth;
- persistere la whitelist dei profili docente DIEM tra run e resume.

Questo modulo non effettua richieste HTTP e non processa HTML.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv

from discovery_models import CrawlItem, CrawlState, PersistentDiscoveryState
from pipeline_io import BASE_DIR, relative_path, write_json, write_text
from pipeline_types import DiscoveryRecord
from url_filters import can_traverse_url, normalize_url


CONFIG_FILE = BASE_DIR / "config.yaml"


def project_path(path: str | Path) -> Path:
    """Converte path relativi al progetto in path assoluti."""
    path = Path(path)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def load_config() -> dict:
    """Legge config.yaml."""
    load_dotenv(BASE_DIR / ".env")
    with CONFIG_FILE.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def validate_config(config: dict) -> None:
    """Controlla le chiavi minime richieste dalla discovery."""
    required_crawler = [
        "user_agent",
        "allowed_domains",
        "pdf_allowed_domains",
        "max_total_urls",
        "max_depth",
        "per_domain_limits",
        "timeout",
        "max_concurrent_requests",
        "rate_limit_per_domain_rps",
        "max_html_bytes",
    ]
    required_paths = [
        "urls_seed",
        "raw_html_dir",
        "discovered_urls_file",
        "checkpoint_file",
        "discovery_state_file",
    ]
    required_scope = [
        "diem_domain",
        "teacher_domain",
        "course_domain",
        "allowed_course_paths",
        "allowed_course_codes",
    ]

    if "crawler" not in config or "paths" not in config:
        raise ValueError("config.yaml deve contenere le sezioni 'crawler' e 'paths'.")

    for key in required_crawler:
        if key not in config["crawler"]:
            raise ValueError(f"Chiave mancante in crawler: {key}")

    for key in required_paths:
        if key not in config["paths"]:
            raise ValueError(f"Chiave mancante in paths: {key}")

    if "scope" not in config:
        raise ValueError("config.yaml deve contenere la sezione 'scope'.")

    for key in required_scope:
        if key not in config["scope"]:
            raise ValueError(f"Chiave mancante in scope: {key}")

    if config["crawler"]["max_concurrent_requests"] <= 0:
        raise ValueError("crawler.max_concurrent_requests deve essere > 0.")

    if config["crawler"]["rate_limit_per_domain_rps"] <= 0:
        raise ValueError("crawler.rate_limit_per_domain_rps deve essere > 0.")

    if config["crawler"]["max_total_urls"] <= 0:
        raise ValueError("crawler.max_total_urls deve essere > 0.")

    if config["crawler"]["max_depth"] < 0:
        raise ValueError("crawler.max_depth deve essere >= 0.")

    if config["crawler"]["timeout"] <= 0:
        raise ValueError("crawler.timeout deve essere > 0.")

    if config["crawler"]["max_html_bytes"] <= 0:
        raise ValueError("crawler.max_html_bytes deve essere > 0.")

    if config["crawler"].get("refresh_after_days", 0) < 0:
        raise ValueError("crawler.refresh_after_days deve essere >= 0.")

    allowed_domains = {
        str(domain).lower()
        for domain in config["crawler"]["allowed_domains"]
    }
    limited_domains = {
        str(domain).lower()
        for domain in config["crawler"]["per_domain_limits"]
    }
    missing_limits = sorted(allowed_domains - limited_domains)
    if missing_limits:
        raise ValueError(
            "crawler.per_domain_limits manca per: "
            + ", ".join(missing_limits)
        )


def ensure_parent_dir(path: Path) -> None:
    """Crea la cartella padre del file, se non esiste."""
    path.parent.mkdir(parents=True, exist_ok=True)


def short_hash(text: str, length: int = 16) -> str:
    """Hash breve e stabile, usato per i nomi file."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:length]


def html_output_path(document_url: str, config: dict) -> Path:
    """Path di salvataggio dell'HTML grezzo per un documento."""
    url_id = short_hash(document_url)
    raw_html_dir = project_path(config["paths"]["raw_html_dir"])
    return raw_html_dir / url_id[:2] / f"{url_id}.html"


def read_seed_urls(config: dict) -> list[str]:
    """Legge data/urls.txt, normalizza e filtra gli URL iniziali."""
    seed_file = project_path(config["paths"]["urls_seed"])
    urls: list[str] = []

    with seed_file.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            url = normalize_url(line)
            ok, reason = can_traverse_url(url, config)
            if ok:
                urls.append(url)
            else:
                print(f"Seed ignorato ({reason}): {url}")

    return list(dict.fromkeys(urls))


def save_html(document_url: str, html: str, config: dict) -> str:
    """Salva HTML grezzo e ritorna il path relativo."""
    path = html_output_path(document_url, config)
    return write_text(path, html)


def append_jsonl(path: Path, record: dict) -> None:
    """Aggiunge una riga JSON al file JSONL."""
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def make_record(
    item: CrawlItem,
    source_type: str,
    status: str,
    *,
    final_url: str | None = None,
    canonical_url: str | None = None,
    document_url: str | None = None,
    indexable: bool = False,
    **extra: object,
) -> DiscoveryRecord:
    """Crea un record per data/discovered_urls.jsonl."""
    requested_url = item.url
    final_url = normalize_url(final_url or requested_url)
    document_url = normalize_url(document_url or canonical_url or final_url)
    record: DiscoveryRecord = {
        "hash": short_hash(document_url),
        "requested_hash": short_hash(requested_url),
        "url": document_url,
        "requested_url": requested_url,
        "final_url": final_url,
        "canonical_url": canonical_url,
        "document_url": document_url,
        "domain": urlparse(document_url).netloc,
        "depth": item.depth,
        "discovered_from": item.discovered_from,
        "type": source_type,
        "status": status,
        "indexable": indexable,
    }
    record.update(extra)
    return record


def write_record(
    output_file: Path,
    record: DiscoveryRecord,
    status_counts: Counter[str],
    type_counts: Counter[str],
) -> None:
    """Scrive un record utile allo scraping e aggiorna le statistiche."""
    status_counts[record["status"]] += 1
    type_counts[record["type"]] += 1

    append_jsonl(output_file, record)


def save_checkpoint(
    state: CrawlState,
    config: dict,
    status: str,
    stop_reason: str | None = None,
    persistent_state: PersistentDiscoveryState | None = None,
) -> None:
    """Salva lo stato del crawl per poter riprendere in caso di interruzione."""
    persistent_state = persistent_state or empty_persistent_state()
    # Il checkpoint salva solo il delta rispetto allo stato persistito: così non
    # duplica ad ogni run l'intera memoria cumulativa del crawler.
    new_known_urls = {
        url: timestamp
        for url, timestamp in state.known_urls.items()
        if persistent_state.known_urls.get(url) != timestamp
    }
    # Documenti canonici già incontrati nella run corrente; servono a non
    # riscrivere come nuovi URL che puntano allo stesso contenuto noto.
    new_known_documents = {
        url: timestamp
        for url, timestamp in state.known_documents.items()
        if persistent_state.known_documents.get(url) != timestamp
    }
    new_allowed_teacher_profiles = sorted(
        state.allowed_teacher_profiles - persistent_state.allowed_teacher_profiles
    )
    checkpoint = {
        "status": status,
        "queue": [asdict(item) for item in state.queue],
        "queued": sorted(state.queued),
        "visited": sorted(state.visited),
        "seen_documents": sorted(state.seen_documents),
        "domain_counts": dict(state.domain_counts),
        "new_known_urls": new_known_urls,
        "new_known_documents": new_known_documents,
        "new_allowed_teacher_profiles": new_allowed_teacher_profiles,
        "expansion_backlog": [
            asdict(item)
            for item in sorted(
                state.expansion_backlog.values(),
                key=lambda item: (item.depth, item.url),
            )
        ],
    }
    if status == "completed" and stop_reason is not None:
        checkpoint["stop_reason"] = stop_reason
    write_json(project_path(config["paths"]["checkpoint_file"]), checkpoint)


def load_checkpoint(
    config: dict,
    persistent_state: PersistentDiscoveryState | None = None,
) -> CrawlState | None:
    """Carica un checkpoint incompleto, se esiste."""
    path = project_path(config["paths"]["checkpoint_file"])
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        print(f"Checkpoint file corrupted or unreadable: {exc}")
        return None

    status = data.get("status")
    if status is None:
        status = "completed" if data.get("completed", False) else "in_progress"

    if status == "completed":
        return None

    try:
        queue = deque(CrawlItem(**item) for item in data.get("queue", []))
    except (TypeError, KeyError) as exc:
        print(f"Checkpoint queue item invalid: {exc}")
        return None

    queued = {item.url for item in queue}

    persistent_state = persistent_state or empty_persistent_state()
    if (
        "new_known_urls" in data
        or "new_known_documents" in data
        or "new_allowed_teacher_profiles" in data
    ):
        known_urls = {**persistent_state.known_urls, **dict(data.get("new_known_urls", {}))}
        known_documents = {
            **persistent_state.known_documents,
            **dict(data.get("new_known_documents", {})),
        }
        allowed_teacher_profiles = {
            *persistent_state.allowed_teacher_profiles,
            *data.get("new_allowed_teacher_profiles", []),
        }
    else:
        # Compatibilità con i checkpoint precedenti che salvavano la copia completa.
        known_urls = {**persistent_state.known_urls, **dict(data.get("known_urls", {}))}
        known_documents = {
            **persistent_state.known_documents,
            **dict(data.get("known_documents", {})),
        }
        allowed_teacher_profiles = {
            *persistent_state.allowed_teacher_profiles,
            *data.get("allowed_teacher_profiles", []),
        }

    raw_backlog = data.get("expansion_backlog")
    if raw_backlog is None:
        backlog_items = list(persistent_state.expansion_backlog)
    else:
        try:
            backlog_items = [CrawlItem(**item) for item in raw_backlog]
        except (TypeError, KeyError) as exc:
            print(f"Checkpoint expansion backlog invalid: {exc}")
            backlog_items = list(persistent_state.expansion_backlog)

    return CrawlState(
        queue=queue,
        queued=queued,
        visited=set(data.get("visited", [])),
        seen_documents=set(data.get("seen_documents", [])),
        domain_counts=Counter(data.get("domain_counts", {})),
        known_urls=known_urls,
        known_documents=known_documents,
        allowed_teacher_profiles=allowed_teacher_profiles,
        expansion_backlog={item.url: item for item in backlog_items},
    )


def empty_persistent_state() -> PersistentDiscoveryState:
    """Ritorna uno stato persistente iniziale vuoto."""
    return PersistentDiscoveryState(
        frontier=deque(),
        known_urls={},
        known_documents={},
        allowed_teacher_profiles=set(),
        expansion_backlog=deque(),
    )


def load_discovery_state(config: dict) -> PersistentDiscoveryState:
    """Carica la memoria della discovery tra run completati."""
    path = project_path(config["paths"]["discovery_state_file"])
    if not path.exists():
        return empty_persistent_state()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        frontier = deque(CrawlItem(**item) for item in data.get("frontier", []))
        expansion_backlog = deque(
            CrawlItem(**item) for item in data.get("expansion_backlog", [])
        )
    except (json.JSONDecodeError, OSError, TypeError, KeyError) as exc:
        print(f"Discovery state corrupted or unreadable: {exc}")
        return empty_persistent_state()

    return PersistentDiscoveryState(
        frontier=frontier,
        known_urls=dict(data.get("known_urls", {})),
        known_documents=dict(data.get("known_documents", {})),
        allowed_teacher_profiles=set(data.get("allowed_teacher_profiles", [])),
        expansion_backlog=expansion_backlog,
    )


def save_discovery_state(state: CrawlState, config: dict) -> None:
    """Salva frontier, memoria cumulativa e pagine di bordo da riespandere."""
    discovery_state = {
        "frontier": [asdict(item) for item in state.queue],
        "known_urls": state.known_urls,
        "known_documents": state.known_documents,
        "allowed_teacher_profiles": sorted(state.allowed_teacher_profiles),
        "expansion_backlog": [
            asdict(item)
            for item in sorted(
                state.expansion_backlog.values(),
                key=lambda item: (item.depth, item.url),
            )
        ],
    }
    write_json(project_path(config["paths"]["discovery_state_file"]), discovery_state)
