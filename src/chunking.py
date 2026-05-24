from __future__ import annotations

import re
from pathlib import Path
from statistics import mean, median
from urllib.parse import urlparse

from pdf_policy import diem_rescue_upload_scope_reason
from pipeline_io import (
    BASE_DIR,
    content_hash,
    latest_records_by_url,
    load_jsonl,
    write_json,
    write_jsonl_atomic,
)


MANIFEST_PATH = BASE_DIR / "data" / "processed" / "manifest.jsonl"

CHUNKS_DIR = BASE_DIR / "data" / "processed" / "chunks"
CHUNKS_FILE = CHUNKS_DIR / "chunks.jsonl"
STATS_FILE = CHUNKS_DIR / "stats.json"

# Parametri principali
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
MIN_CHUNK_CHARS = 150

# Parametri per documenti molto grandi
LARGE_DOC_THRESHOLD = 50_000
LARGE_DOC_CHUNK_SIZE = 1500
LARGE_DOC_OVERLAP = 220


def project_path(path_value: str) -> Path:
    """
    Converte un path salvato nel manifest in Path assoluto.
    Gestisce sia path Windows con \\ sia path Unix con /.
    """
    normalized = str(path_value).replace("\\", "/")
    return BASE_DIR / normalized


def is_valid_record(record: dict) -> bool:
    """
    Tiene solo i documenti utili per l'indicizzazione.
    """
    if diem_rescue_upload_scope_reason(
        str(record.get("url") or record.get("document_url") or ""),
        str(record.get("discovered_from") or ""),
    ):
        return False

    return (
        record.get("status") == "ok"
        and record.get("indexable") is True
        and record.get("text_extracted") is True
        and record.get("is_duplicate") is not True
        and record.get("duplicate") is not True
        and bool(record.get("index_markdown_path"))
    )


def read_markdown(record: dict) -> str:
    """
    Legge il Markdown pulito del documento.
    """
    markdown_path = project_path(record["index_markdown_path"])

    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown non trovato: {markdown_path}")

    return markdown_path.read_text(encoding="utf-8")


def remove_front_matter(text: str) -> str:
    """
    Rimuove il front matter YAML iniziale:

    ---
    url: ...
    title: ...
    ---

    I metadata li prendiamo già dal manifest, quindi non devono entrare negli embedding.
    """
    pattern = r"^\s*---\s*\n.*?\n---\s*\n?"
    return re.sub(pattern, "", text, count=1, flags=re.DOTALL).strip()

def remove_existing_context_header(text: str) -> str:
    """
    Rimuove eventuali header contestuali già presenti nel Markdown.
    Serve per evitare doppio [CONTESTO DOCUMENTO] nei chunk finali.
    """
    pattern = (
        r"^\s*\[CONTESTO DOCUMENTO\]\s*\n"
        r"(?:Titolo:.*\n)?"
        r"(?:Percorso:.*\n)?"
        r"(?:Fonte:.*\n)?"
        r"\s*(?:\[CONTENUTO\]\s*\n)?"
    )

    previous = None

    while previous != text:
        previous = text
        text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)

    return text.strip()


def normalize_text(text: str) -> str:
    """
    Normalizza il testo senza distruggere la struttura Markdown.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("¿", "'")
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()

def breadcrumb_to_text(breadcrumb: object) -> str:
    """
    Converte il breadcrumb in stringa leggibile.
    """
    if isinstance(breadcrumb, list) and breadcrumb:
        return " > ".join(str(item) for item in breadcrumb)

    return "N/D"


def build_context_header(record: dict) -> str:
    """
    Header contestuale da aggiungere a ogni chunk.

    Serve perché molti chunk possono contenere frasi generiche come:
    - Contatti
    - Didattica
    - Calendario
    - Insegnamenti

    Con l'header, il retriever capisce da quale documento arriva quel testo.
    """
    title = record.get("title") or "Documento senza titolo"
    url = record.get("url") or record.get("document_url") or ""
    breadcrumb_text = breadcrumb_to_text(record.get("breadcrumb"))

    return (
        "[CONTESTO DOCUMENTO]\n"
        f"Titolo: {title}\n"
        f"Percorso: {breadcrumb_text}\n"
        f"Fonte: {url}\n\n"
        "[CONTENUTO]\n"
    )


def split_by_markdown_sections(text: str) -> list[str]:
    """
    Divide il documento in sezioni Markdown.
    Ogni nuova sezione parte da un heading:

    # Titolo
    ## Sezione
    ### Sottosezione
    """
    lines = text.splitlines()
    sections: list[str] = []
    current: list[str] = []

    heading_pattern = re.compile(r"^#{1,6}\s+")

    for line in lines:
        if heading_pattern.match(line) and current:
            section = "\n".join(current).strip()
            if section:
                sections.append(section)
            current = [line]
        else:
            current.append(line)

    if current:
        section = "\n".join(current).strip()
        if section:
            sections.append(section)

    return sections


def find_best_cut(text: str, start: int, target_end: int) -> int:
    """
    Cerca un punto di taglio naturale prima di target_end.
    Preferisce paragrafi, righe, frasi e spazi.
    """
    min_end = start + int((target_end - start) * 0.55)
    window = text[start:target_end]

    separators = ["\n\n", "\n", ". ", "; ", ", ", " "]

    best_cut = -1

    for sep in separators:
        pos = window.rfind(sep)

        if pos != -1:
            candidate = start + pos + len(sep)

            if candidate >= min_end:
                best_cut = candidate
                break

    if best_cut == -1 or best_cut <= start:
        best_cut = target_end

    return best_cut


def split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Spezza una sezione troppo lunga in chunk.
    Usa tagli naturali quando possibile e mantiene overlap.
    """
    text = text.strip()

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        target_end = min(start + chunk_size, text_len)

        if target_end >= text_len:
            cut = text_len
        else:
            cut = find_best_cut(text, start, target_end)

        chunk = text[start:cut].strip()

        if chunk:
            chunks.append(chunk)

        if cut >= text_len:
            break

        next_start = max(cut - overlap, 0)

        if next_start <= start:
            next_start = cut

        start = next_start

    return chunks


def is_protected_short_section(chunk: str) -> bool:
    """
    Alcune sezioni sono corte ma importanti e non devono essere eliminate.
    Esempio: orari di ricevimento con una sola riga.
    """
    chunk_lower = chunk.lower()

    protected_markers = [
        "orario di ricevimento",
        "ricevimento",
    ]

    return any(marker in chunk_lower for marker in protected_markers)


def merge_small_chunks(chunks: list[str], min_chars: int, max_chars: int) -> list[str]:
    """
    Unisce chunk troppo piccoli quando possibile.
    Evita chunk inutili da poche parole, ma non elimina sezioni corte importanti
    come gli orari di ricevimento.
    """
    if not chunks:
        return []

    merged: list[str] = []
    buffer = ""

    for chunk in chunks:
        chunk = chunk.strip()

        if not chunk:
            continue

        if not buffer:
            buffer = chunk
            continue

        candidate = buffer + "\n\n" + chunk

        # Se il buffer è piccolo oppure il nuovo chunk è piccolo,
        # proviamo a unirli invece di rischiare di perdere il chunk corto.
        if (len(buffer) < min_chars or len(chunk) < min_chars) and len(candidate) <= max_chars:
            buffer = candidate
        else:
            merged.append(buffer.strip())
            buffer = chunk

    if buffer:
        merged.append(buffer.strip())

    good_chunks = [
        chunk
        for chunk in merged
        if len(chunk) >= min_chars or is_protected_short_section(chunk)
    ]

    if good_chunks:
        return good_chunks

    return [max(merged, key=len)] if merged else []


def make_chunk_id(document_hash: str, chunk_index: int, text: str) -> str:
    """
    Crea un ID stabile per il chunk.
    """
    chunk_hash = content_hash(text)[:16]
    return f"{document_hash}_{chunk_index:04d}_{chunk_hash}"


def chunk_document(record: dict) -> list[dict]:
    """
    Produce i chunk di un singolo documento.
    """
    raw_markdown = read_markdown(record)

    text = remove_front_matter(raw_markdown)
    text = remove_existing_context_header(text)
    text = normalize_text(text)

    if not text:
        return []

    markdown_chars = int(record.get("markdown_chars") or len(text))

    if markdown_chars >= LARGE_DOC_THRESHOLD:
        chunk_size = LARGE_DOC_CHUNK_SIZE
        overlap = LARGE_DOC_OVERLAP
    else:
        chunk_size = CHUNK_SIZE
        overlap = CHUNK_OVERLAP

    sections = split_by_markdown_sections(text)

    preliminary_chunks: list[str] = []

    for section in sections:
        if len(section) <= chunk_size:
            preliminary_chunks.append(section)
        else:
            preliminary_chunks.extend(
                split_long_text(
                    text=section,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )

    preliminary_chunks = merge_small_chunks(
        chunks=preliminary_chunks,
        min_chars=MIN_CHUNK_CHARS,
        max_chars=chunk_size,
    )

    context_header = build_context_header(record)

    document_hash = str(record.get("hash") or content_hash(text)[:16])
    chunks: list[dict] = []

    for index, chunk_body in enumerate(preliminary_chunks):
        final_text = context_header + chunk_body.strip()
        text_hash = content_hash(final_text)

        chunk = {
            "chunk_id": make_chunk_id(document_hash, index, final_text),
            "document_hash": document_hash,
            "document_content_hash": record.get("content_hash"),
            "chunk_index": index,
            "chunk_count": None,
            "source": record.get("source"),
            "source_url": record.get("url"),
            "document_url": record.get("document_url"),
            "title": record.get("title"),
            "breadcrumb": record.get("breadcrumb", []),
            "breadcrumb_text": breadcrumb_to_text(record.get("breadcrumb")),
            "index_markdown_path": record.get("index_markdown_path"),
            "last_crawled": record.get("last_crawled"),
            "clean_status": record.get("clean_status"),
            "clean_warnings": record.get("clean_warnings", []),
            "text": final_text,
            "text_hash": text_hash,
            "chars": len(final_text),
        }

        chunks.append(chunk)

    for chunk in chunks:
        chunk["chunk_count"] = len(chunks)

    return chunks


def load_valid_records() -> list[dict]:
    """
    Legge il manifest append-only e restituisce solo lo stato corrente dei documenti validi.
    """
    history_records = load_jsonl(MANIFEST_PATH)
    current_records = latest_records_by_url(history_records)

    valid_records = [record for record in current_records if is_valid_record(record)]

    print(f"Record storici nel manifest: {len(history_records)}")
    print(f"Record correnti: {len(current_records)}")
    print(f"Documenti validi per chunking: {len(valid_records)}")

    return valid_records


def domain_from_url(url: str | None) -> str:
    """
    Estrae il dominio da un URL.
    """
    if not url:
        return "unknown"

    parsed = urlparse(url)
    return parsed.netloc or "unknown"


def build_stats(
    valid_records: list[dict],
    chunks: list[dict],
    failed_documents: list[dict],
) -> dict:
    """
    Crea statistiche del chunking.
    """
    chunk_lengths = [chunk["chars"] for chunk in chunks]

    by_source: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    chunks_by_document: dict[str, int] = {}

    for chunk in chunks:
        source = chunk.get("source") or "unknown"
        by_source[source] = by_source.get(source, 0) + 1

        domain = domain_from_url(chunk.get("source_url"))
        by_domain[domain] = by_domain.get(domain, 0) + 1

        document_hash = chunk.get("document_hash") or "unknown"
        chunks_by_document[document_hash] = chunks_by_document.get(document_hash, 0) + 1

    top_documents_by_chunks = sorted(
        [
            {
                "document_hash": document_hash,
                "chunks": count,
            }
            for document_hash, count in chunks_by_document.items()
        ],
        key=lambda item: item["chunks"],
        reverse=True,
    )[:10]

    return {
        "valid_documents": len(valid_records),
        "total_chunks": len(chunks),
        "failed_documents_count": len(failed_documents),
        "failed_documents": failed_documents,
        "chunk_chars": {
            "min": min(chunk_lengths) if chunk_lengths else 0,
            "max": max(chunk_lengths) if chunk_lengths else 0,
            "avg": round(mean(chunk_lengths), 2) if chunk_lengths else 0,
            "median": median(chunk_lengths) if chunk_lengths else 0,
        },
        "by_source": by_source,
        "by_domain": by_domain,
        "top_documents_by_chunks": top_documents_by_chunks,
        "config": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "min_chunk_chars": MIN_CHUNK_CHARS,
            "large_doc_threshold": LARGE_DOC_THRESHOLD,
            "large_doc_chunk_size": LARGE_DOC_CHUNK_SIZE,
            "large_doc_overlap": LARGE_DOC_OVERLAP,
        },
        "output_file": str(CHUNKS_FILE.relative_to(BASE_DIR)),
    }


def build_all_chunks() -> None:
    """
    Esegue tutto lo step di chunking.
    """
    valid_records = load_valid_records()

    all_chunks: list[dict] = []
    failed_documents: list[dict] = []

    for record in valid_records:
        try:
            document_chunks = chunk_document(record)
            all_chunks.extend(document_chunks)
        except Exception as error:
            failed_documents.append(
                {
                    "url": record.get("url"),
                    "title": record.get("title"),
                    "hash": record.get("hash"),
                    "error": str(error),
                }
            )

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    write_jsonl_atomic(CHUNKS_FILE, all_chunks)

    stats = build_stats(
        valid_records=valid_records,
        chunks=all_chunks,
        failed_documents=failed_documents,
    )

    write_json(STATS_FILE, stats)

    print()
    print("Chunking completato.")
    print(f"Chunk creati: {len(all_chunks)}")
    print(f"File chunks: {CHUNKS_FILE.relative_to(BASE_DIR)}")
    print(f"File stats: {STATS_FILE.relative_to(BASE_DIR)}")

    if failed_documents:
        print(f"Documenti falliti: {len(failed_documents)}")


def main() -> None:
    build_all_chunks()


if __name__ == "__main__":
    main()
