from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import warnings
from functools import lru_cache
from pathlib import Path
from statistics import mean, median
from urllib.parse import urlparse

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

from pipeline_io import BASE_DIR, load_jsonl, write_json


load_dotenv(BASE_DIR / ".env", override=True)

CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks" / "chunks.jsonl"

VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore" / "chroma"
VECTORSTORE_STATS_FILE = BASE_DIR / "data" / "vectorstore" / "stats.json"

COLLECTION_NAME = "diem_knowledge"

# Modello multilingua moderno, adatto a italiano + inglese e query lunghe.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
EMBEDDING_FALLBACK_MODEL_NAME = os.getenv("EMBEDDING_FALLBACK_MODEL", "BAAI/bge-m3")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))
EMBEDDING_TRUNCATE_DIM = os.getenv("EMBEDDING_TRUNCATE_DIM", "1024").strip()
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto").strip().lower()

VECTORSTORE_BATCH_SIZE = int(os.getenv("VECTORSTORE_BATCH_SIZE", "512"))
DENSE_INDEX_PROFILE = os.getenv("DENSE_INDEX_PROFILE", "core").strip().lower()
DENSE_PDF_MIN_YEAR = int(os.getenv("DENSE_PDF_MIN_YEAR", "2024"))
DENSE_MAX_CHUNKS_PER_PDF = int(os.getenv("DENSE_MAX_CHUNKS_PER_PDF", "16"))


def truthy_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DENSE_INCLUDE_RECENT_PDFS = truthy_env("DENSE_INCLUDE_RECENT_PDFS", default=True)
DENSE_INCLUDE_REGULATION_PDFS = truthy_env("DENSE_INCLUDE_REGULATION_PDFS", default=True)
DENSE_INCLUDE_BANDO_PDFS = truthy_env("DENSE_INCLUDE_BANDO_PDFS", default=True)


def expected_vectorstore_config() -> dict[str, object]:
    """
    Configurazione che deve corrispondere allo stats.json del vector store.

    Se cambia uno di questi valori, Chroma va ricreato: altrimenti il retrieval
    usa un indice costruito con una policy diversa da quella dichiarata nel .env.
    """
    return {
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_truncate_dim": int(EMBEDDING_TRUNCATE_DIM) if EMBEDDING_TRUNCATE_DIM else None,
        "dense_index_profile": DENSE_INDEX_PROFILE,
        "dense_pdf_min_year": DENSE_PDF_MIN_YEAR,
        "dense_max_chunks_per_pdf": DENSE_MAX_CHUNKS_PER_PDF,
        "dense_include_recent_pdfs": DENSE_INCLUDE_RECENT_PDFS,
        "dense_include_regulation_pdfs": DENSE_INCLUDE_REGULATION_PDFS,
        "dense_include_bando_pdfs": DENSE_INCLUDE_BANDO_PDFS,
    }


def vectorstore_config_mismatches(stats: dict) -> list[str]:
    """
    Restituisce le differenze tra config corrente e vector store persistito.
    """
    mismatches: list[str] = []

    for key, expected in expected_vectorstore_config().items():
        actual = stats.get(key)
        if actual != expected:
            mismatches.append(f"{key}={actual!r} (atteso {expected!r})")

    return mismatches


def warn_if_vectorstore_config_stale() -> None:
    """
    Avvisa se lo stats.json indica un vector store creato con vecchia config.
    """
    if not VECTORSTORE_STATS_FILE.exists():
        warnings.warn(
            f"Stats vector store non trovate: {VECTORSTORE_STATS_FILE}. "
            "Se la directory Chroma esiste, ricreala con: python src/vector_store.py --reset",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    try:
        stats = json.loads(VECTORSTORE_STATS_FILE.read_text(encoding="utf-8"))
    except Exception as error:
        warnings.warn(
            f"Impossibile leggere {VECTORSTORE_STATS_FILE}: {error}. "
            "Ricrea il vector store con: python src/vector_store.py --reset",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    mismatches = vectorstore_config_mismatches(stats)
    if mismatches:
        warnings.warn(
            "Vector store non allineato alla configurazione corrente: "
            + "; ".join(mismatches)
            + ". Ricrealo con: python src/vector_store.py --reset",
            RuntimeWarning,
            stacklevel=2,
        )


def embedding_model_kwargs(device: str) -> dict:
    kwargs: dict[str, object] = {"device": device}

    # Qwen3 Embedding usa codice custom lato sentence-transformers; lo rendiamo
    # configurabile per ambienti più restrittivi.
    if truthy_env("EMBEDDING_TRUST_REMOTE_CODE", default=True):
        kwargs["trust_remote_code"] = True

    # Usiamo 1024 dimensioni per contenere memoria/dimensione dell'indice.
    # Cambiare questo valore richiede di ricreare Chroma con --reset.
    if EMBEDDING_TRUNCATE_DIM:
        kwargs["truncate_dim"] = int(EMBEDDING_TRUNCATE_DIM)

    return kwargs


def pick_embedding_device() -> str:
    """
    Sceglie il device per gli embedding.

    Su Mac M1 con 8 GB, Qwen3 su MPS può saturare la memoria condivisa durante
    l'indicizzazione. In quel caso impostare EMBEDDING_DEVICE=cpu nel .env.
    """
    if EMBEDDING_DEVICE in {"cpu", "mps", "cuda"}:
        return EMBEDDING_DEVICE

    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Carica il modello di embedding HuggingFace.
    La prima esecuzione può richiedere tempo perché scarica il modello.
    Usa automaticamente MPS (Apple Silicon), CUDA (NVIDIA) o CPU in base all'hardware.
    """
    device = pick_embedding_device()
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print(f"Embedding device: {device}")

    try:
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs=embedding_model_kwargs(device),
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": EMBEDDING_BATCH_SIZE,
            },
        )
    except Exception as error:
        # Di default falliamo in modo esplicito: usare un fallback diverso senza
        # reindicizzare produrrebbe query embedding incompatibili con Chroma.
        if not truthy_env("EMBEDDING_ALLOW_FALLBACK", default=False):
            raise RuntimeError(
                f"Impossibile caricare il modello embedding {EMBEDDING_MODEL_NAME}. "
                "Se i pesi sono già in cache, puoi usare HF_HUB_OFFLINE=1. "
                "Per usare il fallback BGE imposta EMBEDDING_ALLOW_FALLBACK=true."
            ) from error

        print(
            "Embedding fallback attivo: "
            f"{EMBEDDING_MODEL_NAME} non disponibile, uso {EMBEDDING_FALLBACK_MODEL_NAME}."
        )
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_FALLBACK_MODEL_NAME,
            model_kwargs=embedding_model_kwargs(device),
            encode_kwargs={
                "normalize_embeddings": True,
                "batch_size": EMBEDDING_BATCH_SIZE,
            },
        )


def is_valid_chunk(chunk: dict) -> bool:
    """
    Tiene solo chunk con id e testo utilizzabile.
    """
    return (
        bool(chunk.get("chunk_id"))
        and bool(chunk.get("text"))
        and isinstance(chunk.get("text"), str)
        and len(chunk.get("text", "").strip()) > 0
    )


def chunk_url(chunk: dict) -> str:
    return str(chunk.get("source_url") or chunk.get("document_url") or "")


def metadata_text_for_pdf_policy(chunk: dict) -> str:
    return " ".join(
        [
            chunk_url(chunk),
            str(chunk.get("title") or ""),
            str(chunk.get("breadcrumb_text") or ""),
        ]
    ).lower()


def text_for_pdf_policy(chunk: dict) -> str:
    """
    Testo compatto per decidere se un PDF entra nel dense index.

    Usiamo URL, titolo e inizio chunk: abbastanza per riconoscere regolamenti,
    bandi e anni recenti senza fare classificazione pesante.
    """
    return " ".join(
        [
            metadata_text_for_pdf_policy(chunk),
            str(chunk.get("text") or "")[:1200],
        ]
    ).lower()


def years_in_text(text: str) -> list[int]:
    return [
        int(match)
        for match in re.findall(r"\b(?:19|20)\d{2}\b", text)
    ]


def is_pdf_chunk(chunk: dict) -> bool:
    url = chunk_url(chunk).lower()
    source = str(chunk.get("source") or "").lower()
    return source == "pdf" or url.endswith(".pdf") or "/uploads/" in url


def is_regulation_pdf(chunk: dict) -> bool:
    policy_text = text_for_pdf_policy(chunk)
    return any(
        marker in policy_text
        for marker in [
            "__regolamenti-cds",
            "regolamento",
            "regolamenti",
            "manifesto degli studi",
        ]
    )


def is_bando_pdf(chunk: dict) -> bool:
    policy_text = text_for_pdf_policy(chunk)
    return any(
        marker in policy_text
        for marker in [
            "bando",
            "bandi",
            "graduatoria",
            "avviso",
            "selezione",
            "concorso",
            "decreto",
        ]
    )


def is_recent_pdf(chunk: dict) -> bool:
    years = years_in_text(metadata_text_for_pdf_policy(chunk))
    return bool(years) and max(years) >= DENSE_PDF_MIN_YEAR


def pdf_dense_priority(chunk: dict) -> tuple[int, int]:
    """
    Priorità dei chunk PDF da tenere nel dense index core.

    I PDF lunghi restano completi in BM25. Nel dense index teniamo solo chunk
    di apertura e chunk con segnali utili per domande tipiche.
    """
    text = str(chunk.get("text") or "").lower()
    chunk_index = int(chunk.get("chunk_index") or 0)
    score = 0

    if chunk_index <= 2:
        score += 25

    high_value_terms = [
        "regolamento",
        "manifesto degli studi",
        "bando",
        "graduatoria",
        "decreto",
        "avviso",
        "selezione",
        "scadenza",
        "requisiti",
        "modalità di accesso",
        "modalita di accesso",
        "immatricolazione",
        "erasmus",
        "università partner",
        "universita partner",
        "ricevimento",
    ]

    score += 10 * sum(1 for term in high_value_terms if term in text)

    return score, -chunk_index


def should_index_dense(chunk: dict) -> bool:
    """
    Decide cosa entra nel vector store dense.

    - all: indicizza tutto, utile solo su hardware generoso o benchmark completi.
    - no_pdf: indicizza solo HTML/catalogo, lasciando tutti i PDF a BM25.
    - core: default; HTML/catalogo sempre, PDF solo se recenti o ad alto valore.
    """
    if DENSE_INDEX_PROFILE == "all":
        return True

    if not is_pdf_chunk(chunk):
        return True

    if DENSE_INDEX_PROFILE == "no_pdf":
        return False

    if DENSE_INDEX_PROFILE != "core":
        raise ValueError(
            "DENSE_INDEX_PROFILE deve essere uno tra: core, all, no_pdf."
        )

    if DENSE_INCLUDE_RECENT_PDFS and is_recent_pdf(chunk):
        return True

    if DENSE_INCLUDE_REGULATION_PDFS and is_regulation_pdf(chunk):
        return True

    if DENSE_INCLUDE_BANDO_PDFS and is_bando_pdf(chunk) and is_recent_pdf(chunk):
        return True

    return False


def dense_document_key(chunk: dict) -> str:
    return str(
        chunk.get("document_url")
        or chunk.get("source_url")
        or chunk.get("document_hash")
        or chunk.get("chunk_id")
    )


def select_dense_chunks(valid_chunks: list[dict]) -> list[dict]:
    """
    Applica il profilo dense e, nel profilo core, limita i chunk per PDF.

    Questo è il punto che rende il vector store adatto a hardware limitato:
    Chroma contiene la conoscenza più utile per il retrieval semantico, mentre
    BM25 continua a coprire l'archivio completo.
    """
    candidates = [chunk for chunk in valid_chunks if should_index_dense(chunk)]

    if DENSE_INDEX_PROFILE != "core" or DENSE_MAX_CHUNKS_PER_PDF <= 0:
        return candidates

    selected: list[dict] = []
    pdf_groups: dict[str, list[dict]] = {}

    for chunk in candidates:
        if not is_pdf_chunk(chunk):
            selected.append(chunk)
            continue

        pdf_groups.setdefault(dense_document_key(chunk), []).append(chunk)

    for group in pdf_groups.values():
        sorted_group = sorted(
            group,
            key=pdf_dense_priority,
            reverse=True,
        )
        kept = sorted_group[:DENSE_MAX_CHUNKS_PER_PDF]
        selected.extend(sorted(kept, key=lambda chunk: int(chunk.get("chunk_index") or 0)))

    return selected


def domain_from_url(url: str | None) -> str:
    """
    Estrae il dominio da un URL.
    """
    if not url:
        return "unknown"

    parsed = urlparse(url)
    return parsed.netloc or "unknown"


def clean_metadata_value(value):
    """
    Chroma accetta metadata scalari: str, int, float, bool.
    Liste e dizionari vanno convertiti in stringa.
    """
    if value is None:
        return ""

    if isinstance(value, (str, int, float, bool)):
        return value

    return json.dumps(value, ensure_ascii=False)


def chunk_to_document(chunk: dict) -> Document:
    """
    Converte un record chunk JSON in Document LangChain.
    """
    source_url = chunk.get("source_url") or chunk.get("document_url") or ""

    metadata = {
        "chunk_id": chunk.get("chunk_id"),
        "document_hash": chunk.get("document_hash"),
        "document_content_hash": chunk.get("document_content_hash"),
        "chunk_index": chunk.get("chunk_index"),
        "chunk_count": chunk.get("chunk_count"),
        "source": chunk.get("source"),
        "source_url": source_url,
        "document_url": chunk.get("document_url"),
        "domain": domain_from_url(source_url),
        "title": chunk.get("title"),
        "breadcrumb": chunk.get("breadcrumb"),
        "breadcrumb_text": chunk.get("breadcrumb_text"),
        "index_markdown_path": chunk.get("index_markdown_path"),
        "last_crawled": chunk.get("last_crawled"),
        "text_hash": chunk.get("text_hash"),
        "chars": chunk.get("chars"),
    }

    clean_metadata = {
        key: clean_metadata_value(value)
        for key, value in metadata.items()
    }

    return Document(
        page_content=chunk["text"],
        metadata=clean_metadata,
    )


def count_by_source(chunks: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        source = str(chunk.get("source") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def load_documents() -> tuple[list[Document], list[str], dict]:
    """
    Legge chunks.jsonl e prepara Document + ids per Chroma.
    """
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"File chunk non trovato: {CHUNKS_FILE}. "
            "Esegui prima: python src/chunking.py"
        )

    chunks = load_jsonl(CHUNKS_FILE)
    valid_chunks = [chunk for chunk in chunks if is_valid_chunk(chunk)]
    dense_chunks = select_dense_chunks(valid_chunks)
    dense_chunk_ids = {chunk.get("chunk_id") for chunk in dense_chunks}
    skipped_chunks = len(valid_chunks) - len(dense_chunks)

    documents = [chunk_to_document(chunk) for chunk in dense_chunks]
    ids = [chunk["chunk_id"] for chunk in dense_chunks]

    filter_stats = {
        "dense_index_profile": DENSE_INDEX_PROFILE,
        "chunks_read": len(chunks),
        "valid_chunks": len(valid_chunks),
        "dense_chunks": len(dense_chunks),
        "skipped_chunks": skipped_chunks,
        "dense_by_source": count_by_source(dense_chunks),
        "skipped_by_source": count_by_source(
            [
                chunk
                for chunk in valid_chunks
                if chunk.get("chunk_id") not in dense_chunk_ids
            ]
        ),
    }

    print(f"Chunk letti: {len(chunks)}")
    print(f"Chunk validi: {len(valid_chunks)}")
    print(f"Profilo dense index: {DENSE_INDEX_PROFILE}")
    print(f"Chunk indicizzati in dense: {len(dense_chunks)}")
    print(f"Chunk esclusi dal dense index: {skipped_chunks}")
    print(f"Documenti LangChain creati: {len(documents)}")

    return documents, ids, filter_stats


def batched(items: list, batch_size: int):
    """
    Genera batch di dimensione batch_size.
    """
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def build_vector_store(reset: bool = False) -> None:
    """
    Crea il vector store Chroma a partire dai chunk.
    """
    if VECTORSTORE_DIR.exists() and reset:
        print(f"Rimuovo vector store precedente: {VECTORSTORE_DIR}")
        shutil.rmtree(VECTORSTORE_DIR)

    if VECTORSTORE_DIR.exists() and any(VECTORSTORE_DIR.iterdir()) and not reset:
        raise RuntimeError(
            f"Vector store già presente in {VECTORSTORE_DIR}. "
            "Usa --reset per ricrearlo da zero."
        )

    documents, ids, filter_stats = load_documents()

    if not documents:
        raise RuntimeError("Nessun documento valido da indicizzare.")

    embeddings = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

    total = len(documents)

    for batch_index, (doc_batch, id_batch) in enumerate(
        zip(
            batched(documents, VECTORSTORE_BATCH_SIZE),
            batched(ids, VECTORSTORE_BATCH_SIZE),
        ),
        start=1,
    ):
        print(f"Indicizzo batch {batch_index} ({len(doc_batch)} documenti)...")
        vector_store.add_documents(documents=doc_batch, ids=id_batch)

    if hasattr(vector_store, "persist"):
        vector_store.persist()

    stats = build_stats(documents, filter_stats)
    write_json(VECTORSTORE_STATS_FILE, stats)

    print()
    print("Vector store creato correttamente.")
    print(f"Documenti indicizzati: {total}")
    print(f"Directory Chroma: {VECTORSTORE_DIR.relative_to(BASE_DIR)}")
    print(f"Stats: {VECTORSTORE_STATS_FILE.relative_to(BASE_DIR)}")


def build_stats(documents: list[Document], filter_stats: dict | None = None) -> dict:
    """
    Crea statistiche semplici sui documenti indicizzati.
    """
    lengths = [len(doc.page_content) for doc in documents]

    by_source: dict[str, int] = {}
    by_domain: dict[str, int] = {}

    for doc in documents:
        source = doc.metadata.get("source") or "unknown"
        domain = doc.metadata.get("domain") or "unknown"

        by_source[source] = by_source.get(source, 0) + 1
        by_domain[domain] = by_domain.get(domain, 0) + 1

    return {
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_fallback_model": EMBEDDING_FALLBACK_MODEL_NAME,
        "embedding_truncate_dim": int(EMBEDDING_TRUNCATE_DIM) if EMBEDDING_TRUNCATE_DIM else None,
        "embedding_device": EMBEDDING_DEVICE,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "vectorstore_batch_size": VECTORSTORE_BATCH_SIZE,
        "dense_index_profile": DENSE_INDEX_PROFILE,
        "dense_pdf_min_year": DENSE_PDF_MIN_YEAR,
        "dense_max_chunks_per_pdf": DENSE_MAX_CHUNKS_PER_PDF,
        "dense_include_recent_pdfs": DENSE_INCLUDE_RECENT_PDFS,
        "dense_include_regulation_pdfs": DENSE_INCLUDE_REGULATION_PDFS,
        "dense_include_bando_pdfs": DENSE_INCLUDE_BANDO_PDFS,
        "vectorstore_dir": str(VECTORSTORE_DIR.relative_to(BASE_DIR)),
        "documents_indexed": len(documents),
        "filter": filter_stats or {},
        "content_chars": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "avg": round(mean(lengths), 2) if lengths else 0,
            "median": median(lengths) if lengths else 0,
        },
        "by_source": by_source,
        "by_domain": by_domain,
    }


@lru_cache(maxsize=1)
def load_vector_store() -> Chroma:
    """
    Carica un vector store Chroma già creato.
    """
    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"Vector store non trovato: {VECTORSTORE_DIR}. "
            "Esegui prima: python src/vector_store.py --reset"
        )

    warn_if_vectorstore_config_stale()

    embeddings = get_embedding_model()

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def preview_text(text: str, max_chars: int = 700) -> str:
    """
    Accorcia il testo per stamparlo a terminale.
    """
    text = text.replace("\n", " ").strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def query_vector_store(query: str, k: int = 5) -> None:
    """
    Esegue una query di test sul vector store.
    """
    vector_store = load_vector_store()

    results = vector_store.similarity_search_with_score(query, k=k)

    print()
    print(f"Query: {query}")
    print(f"Risultati trovati: {len(results)}")
    print("-" * 80)

    for index, (doc, score) in enumerate(results, start=1):
        print(f"Risultato {index}")
        print(f"Score distanza: {score}")
        print(f"Titolo: {doc.metadata.get('title')}")
        print(f"URL: {doc.metadata.get('source_url')}")
        print(f"Chunk ID: {doc.metadata.get('chunk_id')}")
        print()
        print(preview_text(doc.page_content))
        print("-" * 80)
        
def dense_retrieve(query: str, k: int = 20):
    """
    Restituisce i risultati del dense retrieval da Chroma.
    Usata da retrieval.py per costruire il retrieval ibrido.
    """
    vector_store = load_vector_store()
    return vector_store.similarity_search_with_score(query, k=k)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crea o interroga il vector store Chroma del chatbot DIEM."
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Ricrea il vector store da zero eliminando quello precedente.",
    )

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Esegue una query di test sul vector store già creato.",
    )

    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Numero di risultati da recuperare nella query.",
    )

    args = parser.parse_args()

    if args.query:
        query_vector_store(args.query, k=args.k)
    else:
        build_vector_store(reset=args.reset)


if __name__ == "__main__":
    main()
