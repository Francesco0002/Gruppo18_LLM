from __future__ import annotations

"""
Audit di copertura per manifest e chunk.

Serve come controllo di qualità prima di ricostruire il vector store o fare una
demo: verifica che ogni macro-topic DIEM abbia almeno una copertura indicizzata
e rende visibili duplicati, fallimenti di estrazione e peso delle rescue upload.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

from chunk_metadata import chunk_metadata_value
from pipeline_io import BASE_DIR, latest_records_by_url, load_jsonl


MANIFEST_PATH = BASE_DIR / "data" / "processed" / "manifest.jsonl"
CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks" / "chunks.jsonl"

CORE_TOPIC_FAMILIES = {
    "dipartimento",
    "didattica",
    "docenti",
    "ricerca",
    "terza_missione",
    "international",
    "dottorati",
    "laboratori",
    "bandi",
    "qualita_statistiche",
}


def normalize(text: object) -> str:
    return str(text or "").lower()


def source_family(url: str) -> str:
    # Classificazione per dominio. È volutamente stabile e indipendente dal
    # contenuto, utile per capire se stiamo indicizzando le fonti giuste.
    domain = urlparse(url).netloc.lower()
    if "docenti.unisa.it" in domain:
        return "teacher_profile"
    if "coursecatalogue" in domain:
        return "course_catalogue"
    if "corsi.unisa.it" in domain:
        return "course"
    if "diem.unisa.it" in domain:
        return "diem"
    if "rubrica.unisa.it" in domain:
        return "directory"
    return "other"


def topic_family(record_or_text: dict) -> str:
    # Classificazione topica conservativa: se il chunking ha già scritto
    # topic_family, questa funzione è solo fallback/reporting.
    url = normalize(record_or_text.get("url") or record_or_text.get("document_url") or record_or_text.get("source_url"))
    title = normalize(record_or_text.get("title"))
    breadcrumb = normalize(record_or_text.get("breadcrumb") or record_or_text.get("breadcrumb_text"))
    link_text = normalize(record_or_text.get("link_text"))
    chunk_kind = normalize(record_or_text.get("chunk_kind"))
    probe = " ".join([url, title, breadcrumb, link_text, chunk_kind])

    if "docenti.unisa.it" in url or chunk_kind in {"office_hours", "publication_summary", "teacher_projects"}:
        return "docenti"
    if "coursecatalogue" in url or "offerta-formativa" in url or "piano-di-studi" in url or "piano studi" in probe:
        return "didattica"
    if "erasmus" in probe or "international" in url:
        return "international"
    if "dottorato" in probe or "phd" in probe:
        return "dottorati"
    if "laboratori" in url or "strutture" in url or chunk_kind == "lab_equipment":
        return "laboratori"
    if "terza-missione" in url or "spin-off" in url or "brevetti" in probe or "public-engagement" in url:
        return "terza_missione"
    if "/ricerca/" in url or "progetti-finanziati" in url or "aree-di-ricerca" in url:
        return "ricerca"
    if "bando" in probe or "graduatoria" in probe or "avviso" in probe or "/home/bandi" in url:
        return "bandi"
    if "almalaurea" in probe or "sua-cds" in probe or "statistiche" in url or "qualita" in probe or "qualità" in probe:
        return "qualita_statistiche"
    if "/dipartimento" in url or "dipartimento" in breadcrumb:
        return "dipartimento"
    return "generale"


def manifest_summary() -> dict:
    # Usiamo solo lo stato corrente del manifest append-only: altrimenti vecchi
    # record falliti/duplicati gonfierebbero il report operativo.
    history = load_jsonl(MANIFEST_PATH)
    current = latest_records_by_url(history)
    rows = list(current)

    by_status = Counter(str(row.get("status") or "unknown") for row in rows)
    by_source = Counter(source_family(str(row.get("url") or row.get("document_url") or "")) for row in rows)
    by_topic = Counter(topic_family(row) for row in rows)
    failed = [
        row
        for row in rows
        if row.get("status") != "ok" or row.get("text_extracted") is not True
    ]
    duplicates = [row for row in rows if row.get("is_duplicate") is True or row.get("duplicate") is True]
    rescue_urls = [
        row
        for row in rows
        if "/uploads/rescue/" in str(row.get("url") or row.get("document_url") or "")
    ]

    return {
        "history_records": len(history),
        "current_records": len(rows),
        "by_status": dict(by_status.most_common()),
        "by_source_family": dict(by_source.most_common()),
        "by_topic_family": dict(by_topic.most_common()),
        "failed_or_unextracted": len(failed),
        "duplicates": len(duplicates),
        "rescue_upload_records": len(rescue_urls),
    }


def chunk_summary() -> dict:
    chunks = load_jsonl(CHUNKS_FILE) if CHUNKS_FILE.exists() else []
    # Il report sui chunk misura la conoscenza effettivamente disponibile al
    # retrieval, non solo le pagine visitate dal crawler.
    by_kind = Counter(str(chunk_metadata_value(chunk, "chunk_kind") or "unknown") for chunk in chunks)
    by_topic = Counter(
        str(chunk_metadata_value(chunk, "topic_family") or topic_family({
            "source_url": chunk_metadata_value(chunk, "source_url"),
            "document_url": chunk_metadata_value(chunk, "document_url"),
            "title": chunk_metadata_value(chunk, "title"),
            "breadcrumb_text": chunk_metadata_value(chunk, "breadcrumb_text"),
            "chunk_kind": chunk_metadata_value(chunk, "chunk_kind"),
        }))
        for chunk in chunks
    )
    by_source_topic: dict[str, Counter] = defaultdict(Counter)

    for chunk in chunks:
        url = str(chunk_metadata_value(chunk, "source_url") or chunk_metadata_value(chunk, "document_url") or "")
        family = source_family(url)
        topic = str(chunk_metadata_value(chunk, "topic_family") or topic_family({"source_url": url}))
        by_source_topic[family][topic] += 1

    return {
        "chunks": len(chunks),
        "by_chunk_kind": dict(by_kind.most_common()),
        "by_topic_family": dict(by_topic.most_common()),
        "by_source_topic": {
            family: dict(counter.most_common())
            for family, counter in sorted(by_source_topic.items())
        },
    }


def build_report() -> dict:
    manifest = manifest_summary()
    chunks = chunk_summary()
    chunk_topics = set(chunks.get("by_topic_family", {}))
    # Questo è il gate principale: se manca un topic core, la pipeline può
    # sembrare sana sui test noti ma fallire su una classe intera di domande.
    missing_core_topics = sorted(CORE_TOPIC_FAMILIES - chunk_topics)

    return {
        "manifest": manifest,
        "chunks": chunks,
        "missing_core_topics": missing_core_topics,
        "ok": not missing_core_topics,
    }


def write_markdown(report: dict, path: Path) -> None:
    lines = ["# DIEM Coverage Audit", ""]
    lines.append(f"- Manifest corrente: `{report['manifest']['current_records']}` record")
    lines.append(f"- Chunk: `{report['chunks']['chunks']}`")
    lines.append(f"- Topic core mancanti: `{', '.join(report['missing_core_topics']) or 'nessuno'}`")
    lines.append("")
    lines.append("## Topic Chunk")
    lines.append("")
    for topic, count in report["chunks"]["by_topic_family"].items():
        lines.append(f"- `{topic}`: {count}")
    lines.append("")
    lines.append("## Source Family")
    lines.append("")
    for family, count in report["manifest"]["by_source_family"].items():
        lines.append(f"- `{family}`: {count}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit copertura manifest/chunk DIEM.")
    parser.add_argument("--json", action="store_true", help="Stampa report JSON.")
    parser.add_argument(
        "--out-md",
        type=Path,
        default=BASE_DIR / "eval" / "results" / "coverage_audit.md",
        help="Path markdown del report.",
    )
    parser.add_argument(
        "--fail-on-gaps",
        action="store_true",
        help="Esce con codice 1 se mancano topic core.",
    )
    args = parser.parse_args()

    report = build_report()
    write_markdown(report, args.out_md)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Coverage audit scritto in: {args.out_md.relative_to(BASE_DIR)}")
        if report["missing_core_topics"]:
            print("Topic core mancanti:", ", ".join(report["missing_core_topics"]))
        else:
            print("Topic core coperti.")

    if args.fail_on_gaps and not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
