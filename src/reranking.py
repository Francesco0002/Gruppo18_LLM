from __future__ import annotations

import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env", override=True)

DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
FAST_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

# Il reranker è volutamente opzionale: migliora la precisione del top-k,
# ma su CPU/Mac piccoli può diventare il collo di bottiglia della chat.
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL", DEFAULT_RERANKER_MODEL)
RERANKER_FALLBACK_MODEL_NAME = os.getenv("RERANKER_FALLBACK_MODEL", FAST_RERANKER_MODEL)
RERANKER_WEIGHT = float(os.getenv("RERANKER_WEIGHT", "0.75"))
HYBRID_WEIGHT = float(os.getenv("HYBRID_WEIGHT", "0.25"))
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "16"))


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


@lru_cache(maxsize=1)
def get_reranker():
    """
    Carica il CrossEncoder una sola volta per processo.

    Il primo caricamento può essere lento perché inizializza i pesi; le query
    successive riusano lo stesso oggetto, cosa importante per Chainlit.
    """
    from sentence_transformers import CrossEncoder

    try:
        return CrossEncoder(
            RERANKER_MODEL_NAME,
            trust_remote_code=truthy_env("RERANKER_TRUST_REMOTE_CODE", default=True),
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
            trust_remote_code=truthy_env("RERANKER_TRUST_REMOTE_CODE", default=True),
        )


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
        model = get_reranker()
        pairs = [(query, result.text) for result in candidates]
        raw_scores = model.predict(
            pairs,
            batch_size=RERANKER_BATCH_SIZE,
            show_progress_bar=False,
        )
    except Exception:
        if truthy_env("RERANKER_STRICT", default=False):
            raise
        return results[:top_k]

    reranker_scores = [sigmoid(float(score)) for score in raw_scores]
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
        result.score = (RERANKER_WEIGHT * reranker_score) + (HYBRID_WEIGHT * hybrid_score)

    reranked = sorted(candidates, key=lambda result: result.score, reverse=True)
    reranked.extend(results[top_k:])

    for rank, result in enumerate(reranked, start=1):
        result.rank = rank

    return reranked
