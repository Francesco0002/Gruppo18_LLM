from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

DEFAULT_RERANKER_MODEL = "mixedbread-ai/mxbai-rerank-base-v2"
FALLBACK_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Il reranker è volutamente opzionale: migliora la precisione del top-k,
# ma su CPU/Mac piccoli può diventare il collo di bottiglia della chat.
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
RERANKER_FALLBACK_MODEL_NAME = os.getenv("RERANKER_FALLBACK_MODEL", FALLBACK_RERANKER_MODEL)
RERANKER_BACKEND = os.getenv("RERANKER_BACKEND", "auto").strip().lower()
RERANKER_DEVICE = os.getenv("RERANKER_DEVICE", "auto").strip().lower()
RERANKER_WEIGHT = float(os.getenv("RERANKER_WEIGHT", "0.75"))
HYBRID_WEIGHT = float(os.getenv("HYBRID_WEIGHT", "0.25"))
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "4"))


def truthy_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def reranker_enabled() -> bool:
    return truthy_env("RERANKER_ENABLED", default=True)


def normalize(values: list[float]) -> list[float]:
    """Porta score ibridi eterogenei in scala 0..1 prima della combinazione."""
    if not values:
        return []

    min_value = min(values)
    max_value = max(values)

    if math.isclose(min_value, max_value):
        return [1.0 for _ in values]

    return [
        (value - min_value) / (max_value - min_value)
        for value in values
    ]


def sigmoid(value: float) -> float:
    """Converte score cross-encoder non limitati in un valore comparabile 0..1."""
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)

    z = math.exp(value)
    return z / (1 + z)


def cross_encoder_relevance_score(value: float) -> float:
    """
    Normalizza output CrossEncoder eterogenei.

    Alcuni reranker Sentence Transformers restituiscono già probabilità 0..1,
    altri logit non limitati. Preserviamo i primi e applichiamo sigmoid ai secondi.
    """
    if math.isnan(value) or math.isinf(value):
        return 0.0

    if 0.0 <= value <= 1.0:
        return value

    return sigmoid(value)


def selected_reranker_backend(model_name: str | None = None) -> str:
    if RERANKER_BACKEND and RERANKER_BACKEND != "auto":
        return RERANKER_BACKEND

    model = (model_name or RERANKER_MODEL_NAME).lower()
    if "jina-reranker-v3" in model:
        return "jina"

    return "cross_encoder"


def resolve_reranker_device(backend: str | None = None) -> str | None:
    if RERANKER_DEVICE in {"", "none"}:
        return None

    if RERANKER_DEVICE != "auto":
        return RERANKER_DEVICE

    try:
        import torch
    except Exception:
        return None

    resolved_backend = backend or selected_reranker_backend()
    if (
        resolved_backend == "jina"
        and getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_available()
        and not truthy_env("RERANKER_JINA_ALLOW_MPS", default=False)
    ):
        return "cpu"

    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


@lru_cache(maxsize=1)
def get_reranker():
    """
    Carica il reranker una sola volta per processo.

    Il primo caricamento può essere lento perché inizializza i pesi; le query
    successive riusano lo stesso oggetto, cosa importante per Chainlit.
    """
    backend = selected_reranker_backend()

    if backend == "jina":
        from transformers import AutoModel

        load_kwargs = {
            "trust_remote_code": truthy_env("RERANKER_TRUST_REMOTE_CODE", default=True),
        }

        try:
            model = AutoModel.from_pretrained(
                RERANKER_MODEL_NAME,
                dtype="auto",
                **load_kwargs,
            )
        except TypeError:
            model = AutoModel.from_pretrained(
                RERANKER_MODEL_NAME,
                torch_dtype="auto",
                **load_kwargs,
            )

        device = resolve_reranker_device(backend)
        if device:
            model = model.to(device)

        model.eval()
        return model

    from sentence_transformers import CrossEncoder

    device = resolve_reranker_device(backend)
    cross_encoder_kwargs = {
        "trust_remote_code": truthy_env("RERANKER_TRUST_REMOTE_CODE", default=True),
    }
    if device:
        cross_encoder_kwargs["device"] = device

    try:
        return CrossEncoder(
            RERANKER_MODEL_NAME,
            **cross_encoder_kwargs,
        )
    except Exception as error:
        if not truthy_env("RERANKER_ALLOW_FALLBACK", default=False):
            raise RuntimeError(
                f"Impossibile caricare il reranker {RERANKER_MODEL_NAME}. "
                "Scarica il modello o imposta RERANKER_ENABLED=false. "
                "Per usare il fallback veloce imposta RERANKER_ALLOW_FALLBACK=true."
            ) from error

        return CrossEncoder(
            RERANKER_FALLBACK_MODEL_NAME,
            **cross_encoder_kwargs,
        )


def coerce_relevance_score(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0

    return max(0.0, min(1.0, value))


def cross_encoder_rerank_scores(query: str, candidates: list[Any]) -> list[float]:
    model = get_reranker()
    pairs = [(query, reranker_passage(result)) for result in candidates]
    raw_scores = model.predict(
        pairs,
        batch_size=RERANKER_BATCH_SIZE,
        show_progress_bar=False,
    )

    return [cross_encoder_relevance_score(float(score)) for score in raw_scores]


def metadata_value(metadata: dict[str, Any], key: str) -> str:
    value = metadata.get(key)

    if value is None:
        return ""

    if isinstance(value, list):
        return " > ".join(str(item) for item in value if str(item).strip())

    return str(value).strip()


def reranker_passage(result: Any) -> str:
    """
    Costruisce il passaggio visto dal cross-encoder.

    Il reranker deve poter usare anche segnali di entità e provenienza:
    molti chunk hanno corpo generico ("### Strumentazione"), mentre titolo
    contenuto, sezione, URL o testo del link chiariscono a quale corso,
    laboratorio o documento appartengono.
    """
    metadata = getattr(result, "metadata", {}) or {}
    text = str(getattr(result, "text", "") or "")

    fields = [
        ("Titolo", metadata_value(metadata, "title")),
        ("Titolo contenuto", metadata_value(metadata, "content_title")),
        ("Sezione", metadata_value(metadata, "section_heading")),
        ("Percorso", metadata_value(metadata, "breadcrumb") or metadata_value(metadata, "breadcrumb_text")),
        ("URL", metadata_value(metadata, "source_url") or metadata_value(metadata, "document_url")),
        ("Fonte originaria", metadata_value(metadata, "discovered_from")),
        ("Testo link sorgente", metadata_value(metadata, "link_text")),
        ("Tipo chunk", metadata_value(metadata, "chunk_kind")),
        ("Tipo entità", metadata_value(metadata, "entity_type")),
        ("Nome entità", metadata_value(metadata, "entity_name")),
        ("Famiglia fonte", metadata_value(metadata, "source_family")),
        ("ID docente", metadata_value(metadata, "teacher_id")),
        ("ID corso", metadata_value(metadata, "course_id")),
        ("ID laboratorio", metadata_value(metadata, "lab_id")),
        ("Tipo documento", metadata_value(metadata, "document_type")),
        ("Anni documento", metadata_value(metadata, "document_years")),
        ("Titolo pubblicazione", metadata_value(metadata, "publication_title")),
        ("Anno pubblicazione", metadata_value(metadata, "publication_year")),
        ("Tipologia pubblicazione", metadata_value(metadata, "publication_type")),
        ("Sede pubblicazione", metadata_value(metadata, "publication_venue")),
        ("Autori pubblicazione", metadata_value(metadata, "publication_authors")),
        ("DOI pubblicazione", metadata_value(metadata, "publication_doi")),
        ("IRIS pubblicazione", metadata_value(metadata, "publication_iris_url")),
    ]

    context_lines = [
        f"{label}: {value}"
        for label, value in fields
        if value
    ]

    if context_lines:
        return "\n".join(context_lines) + "\n\nContenuto:\n" + text

    return text


def neural_rerank(
    query: str,
    results: list[Any],
    top_k: int,
) -> list[Any]:
    """
    Reranking neurale dei candidati già recuperati.

    Se il modello non è disponibile e RERANKER_STRICT non è attivo, ritorna
    l'ordine ibrido: la chat resta utilizzabile anche su macchine senza pesi
    locali o senza rete.
    """
    if not reranker_enabled() or not results:
        return results[:top_k]

    candidates = results[:top_k]

    try:
        backend = selected_reranker_backend()
        if backend == "jina":
            reranker_scores = jina_rerank_scores(query, candidates)
        else:
            reranker_scores = cross_encoder_rerank_scores(query, candidates)
    except Exception:
        if truthy_env("RERANKER_STRICT", default=False):
            raise
        return results[:top_k]

    hybrid_scores = normalize([float(result.score) for result in candidates])

    for result, reranker_score, hybrid_score in zip(
        candidates,
        reranker_scores,
        hybrid_scores,
        strict=True,
    ):
        # Conserviamo gli score intermedi nei metadata per debug/evaluation.
        result.metadata["reranker_score"] = reranker_score
        result.metadata["pre_rerank_score"] = result.score
        result.metadata["reranker_backend"] = backend
        result.score = (RERANKER_WEIGHT * reranker_score) + (HYBRID_WEIGHT * hybrid_score)

    reranked = sorted(candidates, key=lambda result: result.score, reverse=True)
    reranked.extend(results[top_k:])

    for rank, result in enumerate(reranked, start=1):
        result.rank = rank

    return reranked
