"""Utility condivise dagli step della pipeline di processing."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


def now_iso() -> str:
    """Timestamp UTC leggibile e stabile."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def content_hash(text: str) -> str:
    """Hash SHA-256 del contenuto testuale."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def write_json(path: Path, data: dict) -> None:
    """Scrive un JSON leggibile su disco."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return str(path.relative_to(BASE_DIR))
