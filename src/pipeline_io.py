"""
Utility condivise dagli step della pipeline di processing.

Contiene operazioni generiche e prive di logica di dominio:
- timestamp UTC;
- hash contenuto;
- lettura/scrittura JSONL;
- scrittura JSON;
- scrittura testo con path relativo.
- vista corrente dei manifest append-only.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pipeline_types import ProcessedRecord


BASE_DIR = Path(__file__).resolve().parent.parent


def now_iso() -> str:
    """Timestamp UTC leggibile e stabile."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def content_hash(text: str) -> str:
    """Hash SHA-256 del contenuto testuale."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def relative_path(path: Path) -> str:
    """Path relativo alla root progetto, assoluto se fuori repo."""
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> list[dict]:
    """Legge un JSONL, restituendo [] se il file non esiste."""
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def manifest_record_key(record: ProcessedRecord) -> tuple[str, str]:
    """Chiave stabile per identificare un documento nel manifest processed.

    Il manifest è append-only: lo stesso URL può avere più tentativi nel tempo.
    La coppia source+url permette di tenere l'ultimo stato di ciascun documento.
    """
    source = str(record.get("source", "unknown"))
    url = str(record.get("url") or record.get("document_url") or record.get("hash") or "")
    return source, url


def latest_record_indexes(records: list[ProcessedRecord]) -> list[int]:
    """Indici dell'ultimo record disponibile per ogni documento."""
    latest_by_key: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        latest_by_key[manifest_record_key(record)] = index
    return sorted(latest_by_key.values())


def latest_records_by_url(records: list[ProcessedRecord]) -> list[ProcessedRecord]:
    """Vista corrente di un manifest append-only.

    Restituisce solo l'ultimo record per ogni source+url, preservando l'ordine
    con cui quei record compaiono nel manifest.
    """
    return [records[index] for index in latest_record_indexes(records)]


def recent_successful_urls(manifest_path: Path, source: str, days: int) -> set[str]:
    """URL processati con status ok negli ultimi `days` giorni."""
    cutoff = datetime.now(UTC) - timedelta(days=days)
    recent: set[str] = set()

    for record in load_jsonl(manifest_path):
        if record.get("source") != source or record.get("status") != "ok":
            continue

        last_crawled = record.get("last_crawled")
        if not last_crawled:
            continue

        try:
            crawled_at = datetime.fromisoformat(last_crawled)
        except ValueError:
            continue

        if crawled_at >= cutoff:
            recent.add(record.get("url", ""))

    recent.discard("")
    return recent


def append_jsonl_batch(path: Path, records: list[dict]) -> None:
    """Aggiunge un batch JSONL con fsync per resilienza."""
    if not records:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def write_text_atomic(path: Path, text: str) -> None:
    """Scrive testo in modo atomico tramite file temporaneo e replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        file.write(text)
        file.flush()
        os.fsync(file.fileno())

    temp_path.replace(path)


def write_json(path: Path, data: dict) -> None:
    """Scrive un JSON leggibile su disco in modo atomico."""
    write_text_atomic(path, json.dumps(data, indent=2, ensure_ascii=False))


def write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    """Riscrive un JSONL in modo atomico."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())

    temp_path.replace(path)


def write_text(path: Path, text: str) -> str:
    """Scrive testo UTF-8 e ritorna un path relativo alla root progetto."""
    write_text_atomic(path, text)
    return relative_path(path)
