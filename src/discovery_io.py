"""Input/output su filesystem per la fase di discovery."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path
from urllib.parse import urlparse

import yaml

from discovery_models import CrawlItem, CrawlState
from pipeline_io import write_json
from url_filters import can_traverse_url, normalize_url


BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "config.yaml"


def project_path(path: str | Path) -> Path:
    """Converte path relativi al progetto in path assoluti."""
    path = Path(path)
    if path.is_absolute():
        return path
    return BASE_DIR / path


def load_config() -> dict:
    """Legge config.yaml."""
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
    ]
    required_scope = [
        "diem_domain",
        "teacher_domain",
        "course_domain",
        "allowed_course_slugs",
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


def relative_path(path: Path) -> str:
    """Path relativo alla root del progetto, più leggibile nel JSONL."""
    return str(path.relative_to(BASE_DIR))


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
    ensure_parent_dir(path)
    path.write_text(html, encoding="utf-8")
    return relative_path(path)


def append_jsonl(path: Path, record: dict) -> None:
    """Aggiunge una riga JSON al file JSONL."""
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


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
) -> dict:
    """Crea un record per data/discovered_urls.jsonl."""
    requested_url = item.url
    final_url = normalize_url(final_url or requested_url)
    document_url = normalize_url(document_url or canonical_url or final_url)
    record = {
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
    record: dict,
    status_counts: Counter[str],
    type_counts: Counter[str],
) -> None:
    """Scrive un record utile allo scraping e aggiorna le statistiche."""
    status_counts[record["status"]] += 1
    type_counts[record["type"]] += 1

    if record["status"] == "redirected_out_of_scope":
        return

    append_jsonl(output_file, record)


def save_checkpoint(state: CrawlState, config: dict, completed: bool) -> None:
    """Salva lo stato del crawl per poter riprendere in caso di interruzione."""
    checkpoint = {
        "completed": completed,
        "queue": [asdict(item) for item in state.queue],
        "queued": sorted(state.queued),
        "visited": sorted(state.visited),
        "seen_documents": sorted(state.seen_documents),
        "domain_counts": dict(state.domain_counts),
    }
    write_json(project_path(config["paths"]["checkpoint_file"]), checkpoint)


def load_checkpoint(config: dict) -> CrawlState | None:
    """Carica un checkpoint incompleto, se esiste."""
    path = project_path(config["paths"]["checkpoint_file"])
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("completed", False):
        return None

    queue = deque(CrawlItem(**item) for item in data.get("queue", []))
    queued = set(data.get("queued", [])) or {item.url for item in queue}

    return CrawlState(
        queue=queue,
        queued=queued,
        visited=set(data.get("visited", [])),
        seen_documents=set(data.get("seen_documents", [])),
        domain_counts=Counter(data.get("domain_counts", {})),
    )
