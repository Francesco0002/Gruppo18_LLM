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

from chunk_metadata import (
    CHUNK_METADATA_SCHEMA_VERSION,
    chunk_metadata_value,
    flatten_chunk_metadata,
)
from pipeline_io import BASE_DIR, load_jsonl, write_json


load_dotenv(BASE_DIR / ".env", override=True)

CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks" / "chunks.jsonl"

VECTORSTORE_DIR = BASE_DIR / "data" / "vectorstore" / "chroma"
VECTORSTORE_STATS_FILE = BASE_DIR / "data" / "vectorstore" / "stats.json"

# Collection primaria: contiene embedding del solo body_text, senza context
# header, per non diluire il segnale semantico.
COLLECTION_NAME = "diem_knowledge"

# Collection secondaria: contiene locator_text, cioè metadata compatti fielded.
# Aiuta a trovare il documento giusto per corso/docente/anno/curriculum senza
# sporcare l'embedding del contenuto.
LOCATOR_COLLECTION_NAME = "diem_knowledge_locator"
CHUNK_CONTEXT_SCHEMA_VERSION = 5

# Modello multilingua moderno, adatto a italiano + inglese e query lunghe.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "e5-small-v2")
EMBEDDING_FALLBACK_MODEL_NAME = os.getenv("EMBEDDING_FALLBACK_MODEL", "BAAI/bge-m3")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))
EMBEDDING_TRUNCATE_DIM = os.getenv("EMBEDDING_TRUNCATE_DIM", "1024").strip()
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "auto").strip().lower()
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "torch").strip().lower()
EMBEDDING_ONNX_MODEL_PATH = os.getenv("EMBEDDING_ONNX_MODEL_PATH", "").strip()
EMBEDDING_ONNX_PROVIDER = os.getenv("EMBEDDING_ONNX_PROVIDER", "CPUExecutionProvider").strip()
EMBEDDING_ONNX_FILE_NAME = os.getenv("EMBEDDING_ONNX_FILE_NAME", "").strip()

VECTORSTORE_BATCH_SIZE = int(os.getenv("VECTORSTORE_BATCH_SIZE", "128"))
DENSE_INDEX_PROFILE = os.getenv("DENSE_INDEX_PROFILE", "core").strip().lower()
DENSE_PDF_MIN_YEAR = int(os.getenv("DENSE_PDF_MIN_YEAR", "2023"))
DENSE_MAX_CHUNKS_PER_PDF = int(os.getenv("DENSE_MAX_CHUNKS_PER_PDF", "16"))


def truthy_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


DENSE_INCLUDE_RECENT_PDFS = truthy_env("DENSE_INCLUDE_RECENT_PDFS", default=True)
DENSE_INCLUDE_REGULATION_PDFS = truthy_env("DENSE_INCLUDE_REGULATION_PDFS", default=True)
DENSE_INCLUDE_BANDO_PDFS = truthy_env("DENSE_INCLUDE_BANDO_PDFS", default=True)
VECTORSTORE_ALLOW_STALE = truthy_env("VECTORSTORE_ALLOW_STALE", default=False)


def expected_vectorstore_config() -> dict[str, object]:
    """
    Configurazione che deve corrispondere allo stats.json del vector store.

    Se cambia uno di questi valori, Chroma va ricreato: altrimenti il retrieval
    usa un indice costruito con una policy diversa da quella dichiarata nel .env.
    """
    return {
        "collection_name": COLLECTION_NAME,
        # La presenza del locator store fa parte della compatibilità dell'indice:
        # se manca, dense body può funzionare ma la nuova pipeline è incompleta.
        "locator_collection_name": LOCATOR_COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_backend": EMBEDDING_BACKEND,
        "embedding_onnx_model_path": EMBEDDING_ONNX_MODEL_PATH if EMBEDDING_BACKEND == "onnx" else "",
        "embedding_onnx_provider": EMBEDDING_ONNX_PROVIDER if EMBEDDING_BACKEND == "onnx" else "",
        "embedding_onnx_file_name": EMBEDDING_ONNX_FILE_NAME if EMBEDDING_BACKEND == "onnx" else "",
        "embedding_truncate_dim": int(EMBEDDING_TRUNCATE_DIM) if EMBEDDING_TRUNCATE_DIM else None,
        "dense_index_profile": DENSE_INDEX_PROFILE,
        "dense_pdf_min_year": DENSE_PDF_MIN_YEAR,
        "dense_max_chunks_per_pdf": DENSE_MAX_CHUNKS_PER_PDF,
        "dense_include_recent_pdfs": DENSE_INCLUDE_RECENT_PDFS,
        "dense_include_regulation_pdfs": DENSE_INCLUDE_REGULATION_PDFS,
        "dense_include_bando_pdfs": DENSE_INCLUDE_BANDO_PDFS,
        "chunk_context_schema_version": CHUNK_CONTEXT_SCHEMA_VERSION,
        "chunk_metadata_schema_version": CHUNK_METADATA_SCHEMA_VERSION,
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


def load_vectorstore_stats() -> dict:
    if not VECTORSTORE_STATS_FILE.exists():
        raise FileNotFoundError(
            f"Stats vector store non trovate: {VECTORSTORE_STATS_FILE}. "
            "Ricrea il vector store con: python src/vector_store.py --reset"
        )

    try:
        return json.loads(VECTORSTORE_STATS_FILE.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(
            f"Impossibile leggere {VECTORSTORE_STATS_FILE}: {error}. "
            "Ricrea il vector store con: python src/vector_store.py --reset"
        ) from error


def ensure_vectorstore_config_current() -> None:
    """
    Impedisce di usare Chroma quando l'indice non corrisponde ai chunk/config.

    Il fallback BM25 in retrieval mantiene il chatbot usabile durante sviluppo,
    ma evita risposte contaminate da metadata vecchi.
    """
    stats = load_vectorstore_stats()
    mismatches = vectorstore_config_mismatches(stats)
    if not mismatches:
        return

    message = (
        "Vector store non allineato alla configurazione corrente: "
        + "; ".join(mismatches)
        + ". Ricrealo con: python src/vector_store.py --reset"
    )

    if VECTORSTORE_ALLOW_STALE:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
        return

    raise RuntimeError(message)


def warn_if_vectorstore_config_stale() -> None:
    """
    Avvisa se lo stats.json indica un vector store creato con vecchia config.
    """
    try:
        stats = load_vectorstore_stats()
    except Exception as error:
        warnings.warn(
            str(error),
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


def resolve_project_path(path_value: str) -> str:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path
    return str(path)


def runtime_embedding_model_name() -> str:
    """
    Restituisce il nome/path da passare a SentenceTransformer.

    Con backend torch usiamo il modello HuggingFace dichiarato in EMBEDDING_MODEL.
    Con backend onnx, se EMBEDDING_ONNX_MODEL_PATH è valorizzato, carichiamo il
    modello esportato localmente: è il percorso consigliato per rebuild veloci.
    """
    if EMBEDDING_BACKEND == "onnx" and EMBEDDING_ONNX_MODEL_PATH:
        return resolve_project_path(EMBEDDING_ONNX_MODEL_PATH)
    return EMBEDDING_MODEL_NAME


def embedding_model_kwargs(device: str, backend: str | None = None) -> dict:
    selected_backend = backend or EMBEDDING_BACKEND
    kwargs: dict[str, object] = {}

    if selected_backend == "onnx":
        # ONNX Runtime non usa MPS. Su Mac M1 il provider CPU e la quantizzazione
        # arm64 sono il percorso più prevedibile per rebuild sotto deadline.
        kwargs["backend"] = "onnx"
        kwargs["model_kwargs"] = {
            "provider": EMBEDDING_ONNX_PROVIDER or "CPUExecutionProvider",
        }
        if EMBEDDING_ONNX_FILE_NAME:
            kwargs["model_kwargs"]["file_name"] = EMBEDDING_ONNX_FILE_NAME
    else:
        kwargs["device"] = device

    # Qwen3 Embedding usa codice custom lato sentence-transformers; lo rendiamo
    # configurabile per ambienti più restrittivi.
    if truthy_env("EMBEDDING_TRUST_REMOTE_CODE", default=True):
        kwargs["trust_remote_code"] = True

    # Evita tokenizzazione non corretta sui tokenizer affetti dal bug regex
    # noto (warning `fix_mistral_regex=True`).
    if truthy_env("EMBEDDING_FIX_MISTRAL_REGEX", default=True):
        kwargs["processor_kwargs"] = {"fix_mistral_regex": True}

    # Usiamo 1024 dimensioni per contenere memoria/dimensione dell'indice.
    # Cambiare questo valore richiede di ricreare Chroma con --reset.
    if EMBEDDING_TRUNCATE_DIM:
        kwargs["truncate_dim"] = int(EMBEDDING_TRUNCATE_DIM)

    return kwargs


def embedding_encode_kwargs() -> dict:
    return {
        "normalize_embeddings": True,
        "batch_size": EMBEDDING_BATCH_SIZE,
    }


def pick_embedding_device() -> str:
    """
    Sceglie il device per gli embedding.

    Su Mac M1 con 8 GB, Qwen3 su MPS può saturare la memoria condivisa durante
    l'indicizzazione. In quel caso impostare EMBEDDING_DEVICE=cpu nel .env.
    """
    if EMBEDDING_BACKEND == "onnx":
        return "cpu"

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
    model_name = runtime_embedding_model_name()
    print(f"Embedding model: {model_name}")
    print(f"Embedding backend: {EMBEDDING_BACKEND}")
    print(f"Embedding device: {device}")
    if EMBEDDING_BACKEND == "onnx":
        print(f"Embedding ONNX provider: {EMBEDDING_ONNX_PROVIDER or 'CPUExecutionProvider'}")

    try:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=embedding_model_kwargs(device),
            encode_kwargs=embedding_encode_kwargs(),
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
            model_kwargs=embedding_model_kwargs(device, backend="torch"),
            encode_kwargs=embedding_encode_kwargs(),
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
    return str(
        chunk_metadata_value(chunk, "source_url")
        or chunk_metadata_value(chunk, "document_url")
        or ""
    )


def metadata_text_for_pdf_policy(chunk: dict) -> str:
    return " ".join(
        [
            chunk_url(chunk),
            str(chunk_metadata_value(chunk, "title") or ""),
            str(chunk_metadata_value(chunk, "breadcrumb_text") or ""),
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
    source = str(chunk_metadata_value(chunk, "source") or "").lower()
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


def is_study_plan_pdf(chunk: dict) -> bool:
    policy_text = text_for_pdf_policy(chunk)
    return any(
        marker in policy_text
        for marker in [
            "__piano-studi-cds",
            "piano di studi",
            "piano degli studi",
            "manifesto degli studi",
            "1° anno",
            "2° anno",
            "3° anno",
        ]
    )


def is_international_pdf(chunk: dict) -> bool:
    policy_text = text_for_pdf_policy(chunk)
    return any(
        marker in policy_text
        for marker in [
            "international",
            "erasmus",
            "mobilità",
            "mobilita",
            "accordi",
            "learning agreement",
        ]
    )


def is_high_value_chunk_kind(chunk: dict) -> bool:
    return str(chunk_metadata_value(chunk, "chunk_kind") or "").lower() in {
        "course_syllabus",
        "study_plan",
        "course_statistic",
        "official_document_summary",
        "publication_summary",
        "office_hours",
        "lab_equipment",
    }


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
        "piano di studi",
        "piano degli studi",
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

    chunk_kind = str(chunk_metadata_value(chunk, "chunk_kind") or "").lower()
    if chunk_kind == "study_plan":
        score += 55
    elif chunk_kind == "course_statistic":
        score += 45
    elif chunk_kind in {"official_document_summary", "lab_equipment", "office_hours"}:
        score += 35

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

    if is_high_value_chunk_kind(chunk):
        return True

    if is_study_plan_pdf(chunk):
        return True

    if DENSE_INCLUDE_RECENT_PDFS and is_recent_pdf(chunk):
        return True

    if DENSE_INCLUDE_REGULATION_PDFS and is_regulation_pdf(chunk):
        return True

    if DENSE_INCLUDE_BANDO_PDFS and is_bando_pdf(chunk) and is_recent_pdf(chunk):
        return True

    if is_international_pdf(chunk) and is_recent_pdf(chunk):
        return True

    return False


def dense_document_key(chunk: dict) -> str:
    return str(
        chunk_metadata_value(chunk, "document_url")
        or chunk_metadata_value(chunk, "source_url")
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


def chunk_to_document(chunk: dict, representation: str = "body") -> Document:
    """
    Converte un record chunk JSON in Document LangChain.

    Usa text_for_embedding (senza context header) per creare l'embedding,
    così la similarità semantica si basa solo sul contenuto effettivo.
    Salva text_for_display nei metadata per retrieval e reranking.
    """
    metadata = flatten_chunk_metadata(chunk)
    source_url = metadata.get("source_url") or metadata.get("document_url") or ""
    metadata["source_url"] = source_url
    metadata["domain"] = domain_from_url(str(source_url))

    # Salva il testo completo (con context header) per retrieval/reranking
    text_for_display = chunk.get("text_for_display") or chunk.get("text", "")
    metadata["text_for_display"] = text_for_display
    metadata["vector_representation"] = representation

    clean_metadata = {
        key: clean_metadata_value(value)
        for key, value in metadata.items()
    }

    # Usa body_text/text_for_embedding (senza header) per l'embedding primario.
    # Usa locator_text per la seconda collection, con soli metadati compatti.
    if representation == "locator":
        page_content = chunk.get("locator_text") or ""
    else:
        page_content = (
            chunk.get("body_text")
            or chunk.get("text_for_embedding")
            or chunk.get("text", "")
        )

    return Document(
        page_content=str(page_content),
        metadata=clean_metadata,
    )


def count_by_source(chunks: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for chunk in chunks:
        source = str(chunk_metadata_value(chunk, "source") or "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def load_documents() -> tuple[list[Document], list[str], list[Document], list[str], dict]:
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

    documents = [chunk_to_document(chunk, representation="body") for chunk in dense_chunks]
    ids = [chunk["chunk_id"] for chunk in dense_chunks]
    # Ogni chunk dense può produrre due documenti Chroma con lo stesso id:
    # uno nella collection body e uno nella collection locator. Gli id uguali
    # rendono semplice fondere i risultati nel retrieval.
    locator_chunks = [
        chunk
        for chunk in dense_chunks
        if str(chunk.get("locator_text") or "").strip()
    ]
    locator_documents = [
        chunk_to_document(chunk, representation="locator")
        for chunk in locator_chunks
    ]
    locator_ids = [chunk["chunk_id"] for chunk in locator_chunks]

    filter_stats = {
        "dense_index_profile": DENSE_INDEX_PROFILE,
        "chunks_read": len(chunks),
        "valid_chunks": len(valid_chunks),
        "dense_chunks": len(dense_chunks),
        "locator_chunks": len(locator_chunks),
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
    print(f"Chunk indicizzati in locator dense: {len(locator_documents)}")
    print(f"Chunk esclusi dal dense index: {skipped_chunks}")
    print(f"Documenti LangChain creati: {len(documents)}")

    return documents, ids, locator_documents, locator_ids, filter_stats


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

    documents, ids, locator_documents, locator_ids, filter_stats = load_documents()

    if not documents:
        raise RuntimeError("Nessun documento valido da indicizzare.")

    embeddings = get_embedding_model()

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )
    # Le due collection condividono directory e modello embedding, ma restano
    # interrogabili separatamente per mantenere trasparente il contributo di
    # body dense e locator dense nel trace.
    locator_vector_store = Chroma(
        collection_name=LOCATOR_COLLECTION_NAME,
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

    for batch_index, (doc_batch, id_batch) in enumerate(
        zip(
            batched(locator_documents, VECTORSTORE_BATCH_SIZE),
            batched(locator_ids, VECTORSTORE_BATCH_SIZE),
        ),
        start=1,
    ):
        print(f"Indicizzo locator batch {batch_index} ({len(doc_batch)} documenti)...")
        locator_vector_store.add_documents(documents=doc_batch, ids=id_batch)

    if hasattr(vector_store, "persist"):
        vector_store.persist()
    if hasattr(locator_vector_store, "persist"):
        locator_vector_store.persist()

    stats = build_stats(documents, filter_stats)
    write_json(VECTORSTORE_STATS_FILE, stats)

    print()
    print("Vector store creato correttamente.")
    print(f"Documenti indicizzati: {total}")
    print(f"Locator indicizzati: {len(locator_documents)}")
    print(f"Directory Chroma: {VECTORSTORE_DIR.relative_to(BASE_DIR)}")
    print(f"Stats: {VECTORSTORE_STATS_FILE.relative_to(BASE_DIR)}")


def build_stats(
    documents: list[Document],
    filter_stats: dict | None = None,
) -> dict:
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
        "locator_collection_name": LOCATOR_COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_fallback_model": EMBEDDING_FALLBACK_MODEL_NAME,
        "embedding_backend": EMBEDDING_BACKEND,
        "embedding_onnx_model_path": EMBEDDING_ONNX_MODEL_PATH if EMBEDDING_BACKEND == "onnx" else "",
        "embedding_onnx_provider": EMBEDDING_ONNX_PROVIDER if EMBEDDING_BACKEND == "onnx" else "",
        "embedding_onnx_file_name": EMBEDDING_ONNX_FILE_NAME if EMBEDDING_BACKEND == "onnx" else "",
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
        "chunk_context_schema_version": CHUNK_CONTEXT_SCHEMA_VERSION,
        "chunk_metadata_schema_version": CHUNK_METADATA_SCHEMA_VERSION,
        "vectorstore_dir": str(VECTORSTORE_DIR.relative_to(BASE_DIR)),
        "documents_indexed": len(documents),
        "locator_documents_indexed": int((filter_stats or {}).get("locator_chunks") or 0),
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


@lru_cache(maxsize=4)
def load_vector_store(collection_name: str = COLLECTION_NAME) -> Chroma:
    """
    Carica un vector store Chroma già creato.
    """
    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"Vector store non trovato: {VECTORSTORE_DIR}. "
            "Esegui prima: python src/vector_store.py --reset"
        )

    ensure_vectorstore_config_current()

    embeddings = get_embedding_model()

    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(VECTORSTORE_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )


def load_locator_vector_store() -> Chroma:
    return load_vector_store(LOCATOR_COLLECTION_NAME)


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
        
def dense_retrieve(query: str, k: int = 20, representation: str = "body"):
    """
    Restituisce i risultati del dense retrieval da Chroma.
    Usata da retrieval.py per costruire il retrieval ibrido.
    """
    # Punto unico di accesso al dense retrieval: retrieval.py decide quale
    # representation interrogare e poi fonde i ranking.
    if representation == "locator":
        vector_store = load_locator_vector_store()
    else:
        vector_store = load_vector_store()
    return vector_store.similarity_search_with_score(query, k=k)


def dense_locator_retrieve(query: str, k: int = 20):
    """
    Restituisce risultati dal vector store locator, separato dal body embedding.
    """
    return dense_retrieve(query, k=k, representation="locator")


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
