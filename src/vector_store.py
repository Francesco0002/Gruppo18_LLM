from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from statistics import mean, median
from urllib.parse import urlparse

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from pipeline_io import BASE_DIR, load_jsonl, write_json


CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks" / "chunks.jsonl"

VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore" / "chroma"
VECTORSTORE_STATS_FILE = BASE_DIR / "data" / "vectorstore" / "stats.json"

COLLECTION_NAME = "diem_knowledge"

# Modello leggero e multilingua, adatto a italiano + inglese
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

BATCH_SIZE = 128


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Carica il modello di embedding HuggingFace.
    La prima esecuzione può richiedere tempo perché scarica il modello.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 32,
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


def load_documents() -> tuple[list[Document], list[str]]:
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

    documents = [chunk_to_document(chunk) for chunk in valid_chunks]
    ids = [chunk["chunk_id"] for chunk in valid_chunks]

    print(f"Chunk letti: {len(chunks)}")
    print(f"Chunk validi: {len(valid_chunks)}")
    print(f"Documenti LangChain creati: {len(documents)}")

    return documents, ids


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

    documents, ids = load_documents()

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
        zip(batched(documents, BATCH_SIZE), batched(ids, BATCH_SIZE)),
        start=1,
    ):
        print(f"Indicizzo batch {batch_index} ({len(doc_batch)} documenti)...")
        vector_store.add_documents(documents=doc_batch, ids=id_batch)

    if hasattr(vector_store, "persist"):
        vector_store.persist()

    stats = build_stats(documents)
    write_json(VECTORSTORE_STATS_FILE, stats)

    print()
    print("Vector store creato correttamente.")
    print(f"Documenti indicizzati: {total}")
    print(f"Directory Chroma: {VECTORSTORE_DIR.relative_to(BASE_DIR)}")
    print(f"Stats: {VECTORSTORE_STATS_FILE.relative_to(BASE_DIR)}")


def build_stats(documents: list[Document]) -> dict:
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
        "vectorstore_dir": str(VECTORSTORE_DIR.relative_to(BASE_DIR)),
        "documents_indexed": len(documents),
        "content_chars": {
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "avg": round(mean(lengths), 2) if lengths else 0,
            "median": median(lengths) if lengths else 0,
        },
        "by_source": by_source,
        "by_domain": by_domain,
    }


def load_vector_store() -> Chroma:
    """
    Carica un vector store Chroma già creato.
    """
    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"Vector store non trovato: {VECTORSTORE_DIR}. "
            "Esegui prima: python src/vector_store.py --reset"
        )

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