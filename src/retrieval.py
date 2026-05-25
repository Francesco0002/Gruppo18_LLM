"""
Retrieval ibrido BM25 + Dense con reranking neurale.

Pipeline semplificata:
1. BM25 retrieval (ricerca lessicale)
2. Dense retrieval da Chroma (ricerca semantica)
3. Reciprocal Rank Fusion per combinare i ranking
4. Neural reranking con cross-encoder
5. Deduplica per chunk_id
"""

from __future__ import annotations

import argparse
import os
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from chunk_metadata import flatten_chunk_metadata
from pipeline_io import load_jsonl
from query_planner import QueryPlan, build_retrieval_query, plan_query
from reranking import neural_rerank
from structured_evidence import structured_retrieve
from vector_store import CHUNKS_FILE, dense_locator_retrieve, dense_retrieve, preview_text


load_dotenv(CHUNKS_FILE.parent.parent.parent.parent / ".env", override=False)


DEFAULT_BM25_K = int(os.getenv("RETRIEVAL_BM25_K", "80"))
DEFAULT_DENSE_K = int(os.getenv("RETRIEVAL_DENSE_K", "80"))
DEFAULT_RERANK_K = int(os.getenv("RETRIEVAL_RERANK_K", "20"))

_DENSE_RETRIEVAL_WARNING_EMITTED = False
_LOCATOR_RETRIEVAL_WARNING_EMITTED = False

# Ultimo trace prodotto da hybrid_retrieve. Chainlit/RAG possono leggerlo per
# debug senza cambiare l'interfaccia pubblica del retrieval.
_LAST_RETRIEVAL_TRACE: dict[str, Any] = {}


ITALIAN_STOPWORDS = {
    "a", "ad", "al", "allo", "alla", "alle", "agli", "ai",
    "con", "da", "dal", "dalla", "dalle", "dei", "del", "dell",
    "della", "delle", "di", "e", "è", "il", "lo", "la", "i",
    "gli", "le", "in", "nel", "nella", "nelle", "nei", "per",
    "su", "sul", "sulla", "sulle", "sono", "un", "una", "uno",
    "che", "chi", "cosa", "come", "quando", "dove", "quale",
    "quali", "quanto", "quanti", "mi", "puoi", "sapere",
}


@dataclass
class RetrievalResult:
    """Risultato del retrieval con metadata e score."""
    
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    source: str
    rank: int
    score: float


def tokenize(text: str) -> list[str]:
    """
    Tokenizza testo italiano rimuovendo stopwords e punteggiatura.
    """
    import re
    tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9_]+", text.lower())
    return [
        token
        for token in tokens
        if token not in ITALIAN_STOPWORDS and len(token) > 1
    ]


@lru_cache(maxsize=1)
def load_valid_chunks() -> tuple[dict[str, Any], ...]:
    """
    Carica tutti i chunk validi da chunks.jsonl.
    Usa cache per evitare reload multipli.
    """
    chunks = load_jsonl(CHUNKS_FILE)
    return tuple(
        chunk
        for chunk in chunks
        if chunk.get("chunk_id") and chunk.get("text")
    )


def metadata_for_bm25(chunk: dict[str, Any]) -> str:
    # BM25 lavora su body + campi importanti, perché la ricerca lessicale trae
    # beneficio da corso/anno/docente anche se questi non entrano nel body embedding.
    metadata = flatten_chunk_metadata(chunk)
    fields = [
        metadata.get("title"),
        metadata.get("content_title"),
        metadata.get("section_heading"),
        metadata.get("breadcrumb_text"),
        metadata.get("chunk_kind"),
        metadata.get("topic_family"),
        metadata.get("source_family"),
        metadata.get("entity_name"),
        metadata.get("course_name"),
        metadata.get("course_level"),
        metadata.get("curriculum"),
        metadata.get("course_year"),
        metadata.get("academic_year"),
        metadata.get("cohort"),
        metadata.get("document_type"),
        metadata.get("publication_title"),
        metadata.get("publication_authors"),
        chunk.get("locator_text"),
    ]
    return " ".join(str(field or "") for field in fields)


def bm25_text_for_chunk(chunk: dict[str, Any]) -> str:
    body = (
        chunk.get("body_text")
        or chunk.get("text_for_embedding")
        or chunk.get("text")
        or ""
    )
    return f"{metadata_for_bm25(chunk)}\n{body}"


@lru_cache(maxsize=1)
def load_tokenized_corpus() -> tuple[tuple[str, ...], ...]:
    """
    Carica e tokenizza tutto il corpus per BM25.
    Usa cache per evitare rielaborazioni.
    """
    valid_chunks = load_valid_chunks()
    return tuple(
        tuple(tokenize(bm25_text_for_chunk(chunk)))
        for chunk in valid_chunks
    )


@lru_cache(maxsize=1)
def get_bm25_index() -> BM25Okapi:
    """
    Costruisce l'indice BM25 una sola volta.
    """
    tokenized_corpus = load_tokenized_corpus()
    return BM25Okapi(tokenized_corpus)


def bm25_retrieve(query: str, k: int = 20) -> list[RetrievalResult]:
    """
    Recupera chunk tramite BM25 (ricerca lessicale).
    """
    valid_chunks = load_valid_chunks()
    bm25 = get_bm25_index()
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    ranked_indices = [
        i for i in sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )
        if scores[i] > 0
    ][:k]

    results = []
    for rank, index in enumerate(ranked_indices, start=1):
        chunk = valid_chunks[index]
        # Usa text_for_display (con header) per reranking e generazione
        text_for_display = chunk.get("text_for_display") or chunk.get("text", "")
        results.append(
            RetrievalResult(
                chunk_id=chunk["chunk_id"],
                text=text_for_display,
                metadata=flatten_chunk_metadata(chunk),
                source="bm25",
                rank=rank,
                score=float(scores[index]),
            )
        )

    return results


def dense_retrieve_wrapped(query: str, k: int = 20) -> list[RetrievalResult]:
    """
    Recupera chunk tramite dense retrieval da Chroma (ricerca semantica).
    """
    global _DENSE_RETRIEVAL_WARNING_EMITTED

    if k <= 0:
        return []

    try:
        docs_and_scores = dense_retrieve(query, k=k)
    except Exception as error:
        if not _DENSE_RETRIEVAL_WARNING_EMITTED:
            warnings.warn(
                "Dense retrieval non disponibile; uso solo BM25. "
                f"Dettaglio: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            _DENSE_RETRIEVAL_WARNING_EMITTED = True
        return []

    results = []
    for rank, (doc, score) in enumerate(docs_and_scores, start=1):
        metadata = dict(doc.metadata or {})
        chunk_id = str(
            metadata.get("chunk_id")
            or metadata.get("text_hash")
            or f"dense_rank_{rank}"
        )
        # Usa text_for_display (con header) per reranking e generazione
        text_for_display = metadata.get("text_for_display", doc.page_content)
        results.append(
            RetrievalResult(
                chunk_id=chunk_id,
                text=text_for_display,
                metadata=metadata,
                source="dense",
                rank=rank,
                score=float(score),
            )
        )

    return results


def dense_locator_retrieve_wrapped(query: str, k: int = 20) -> list[RetrievalResult]:
    """
    Recupera chunk tramite dense retrieval sulla collection locator.

    Questa collection indicizza solo metadati compatti fielded e non contamina
    l'embedding semantico del contenuto.
    """
    global _LOCATOR_RETRIEVAL_WARNING_EMITTED

    if k <= 0:
        return []

    try:
        docs_and_scores = dense_locator_retrieve(query, k=k)
    except Exception as error:
        if not _LOCATOR_RETRIEVAL_WARNING_EMITTED:
            warnings.warn(
                "Locator dense retrieval non disponibile; continuo senza locator. "
                f"Dettaglio: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            _LOCATOR_RETRIEVAL_WARNING_EMITTED = True
        return []

    results = []
    for rank, (doc, score) in enumerate(docs_and_scores, start=1):
        metadata = dict(doc.metadata or {})
        chunk_id = str(
            metadata.get("chunk_id")
            or metadata.get("text_hash")
            or f"locator_rank_{rank}"
        )
        text_for_display = metadata.get("text_for_display", doc.page_content)
        results.append(
            RetrievalResult(
                chunk_id=chunk_id,
                text=text_for_display,
                metadata=metadata,
                source="locator_dense",
                rank=rank,
                score=float(score),
            )
        )

    return results


def structured_retrieve_wrapped(
    query: str,
    plan: QueryPlan,
    k: int = 30,
) -> list[RetrievalResult]:
    # Converte le evidenze strutturate nello stesso tipo usato dagli altri
    # retriever, così RRF e reranker restano agnostici rispetto alla sorgente.
    evidence = structured_retrieve(
        query=query,
        plan=plan,
        chunks=load_valid_chunks(),
        limit=k,
    )
    results: list[RetrievalResult] = []

    for rank, item in enumerate(evidence, start=1):
        chunk = item.chunk
        metadata = flatten_chunk_metadata(chunk)
        metadata["structured_reason"] = item.reason
        text_for_display = chunk.get("text_for_display") or chunk.get("text") or ""
        results.append(
            RetrievalResult(
                chunk_id=str(chunk["chunk_id"]),
                text=str(text_for_display),
                metadata=metadata,
                source="structured",
                rank=rank,
                score=float(item.score),
            )
        )

    return results


def reciprocal_rank_fusion(
    result_lists: dict[str, list[RetrievalResult]],
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    """
    Combina BM25 e dense retrieval usando Reciprocal Rank Fusion.

    Non confronta direttamente gli score (scale diverse), ma usa le posizioni
    nei ranking. Formula: score = sum(1 / (rrf_k + rank)) per ogni lista.
    """
    fused_results: dict[str, RetrievalResult] = {}
    fused_scores: dict[str, float] = {}
    fused_ranks: dict[str, dict[str, int]] = {}

    for source_name, results in result_lists.items():
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk_id

            if chunk_id not in fused_results:
                fused_results[chunk_id] = RetrievalResult(
                    chunk_id=result.chunk_id,
                    text=result.text,
                    metadata=result.metadata,
                    source="hybrid",
                    rank=0,
                    score=0.0,
                )

            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
                1.0 / (rrf_k + rank)
            )
            fused_ranks.setdefault(chunk_id, {})[source_name] = rank

    final_results = []
    for chunk_id, result in fused_results.items():
        result.score = fused_scores[chunk_id]
        result.metadata["retrieval_source_ranks"] = fused_ranks.get(chunk_id, {})
        result.metadata["retrieval_sources"] = sorted(fused_ranks.get(chunk_id, {}))
        final_results.append(result)

    def tie_breaker(result: RetrievalResult) -> tuple[float, int, int]:
        # A pari RRF preferiamo evidenze structured/locator e rank migliore.
        # Non è un boost moltiplicativo: rompe solo i pareggi tra liste diverse.
        ranks = result.metadata.get("retrieval_source_ranks") or {}
        structured_priority = 1 if "structured" in ranks else 0
        locator_priority = 1 if "locator_dense" in ranks else 0
        best_rank = min((int(rank) for rank in ranks.values()), default=999_999)
        return result.score, structured_priority + locator_priority, -best_rank

    final_results.sort(key=tie_breaker, reverse=True)

    for rank, result in enumerate(final_results, start=1):
        result.rank = rank

    return final_results


def pin_structured_evidence(
    results: list[RetrievalResult],
    structured_results: list[RetrievalResult],
    plan: QueryPlan,
    final_k: int,
) -> list[RetrievalResult]:
    """
    Porta in testa evidenze strutturate quando la query lo richiede chiaramente.

    Non genera risposte predeterminate e non altera gli score: assicura solo che
    tabelle/listati ufficiali recuperati non vengano sepolti da rumore BM25 a
    pari RRF, soprattutto nelle domande multi-sezione.
    """
    # Pinning solo ad alta confidenza: serve alle domande con lista completa
    # (es. piani di studio multi-anno) dove il rumore BM25 tende a intercalarsi.
    if not structured_results or plan.confidence < 0.7:
        return results

    if plan.task_type == "study_plan" and plan.requires_complete_answer:
        pin_count = min(final_k, 8, len(structured_results))
    elif plan.needs_structured_data:
        pin_count = min(max(2, final_k // 3), len(structured_results))
    else:
        return results

    pinned = structured_results[:pin_count]
    pinned_ids = {result.chunk_id for result in pinned}
    combined = pinned + [
        result
        for result in results
        if result.chunk_id not in pinned_ids
    ]

    for rank, result in enumerate(combined, start=1):
        result.rank = rank

    return combined


def hybrid_retrieve(
    query: str,
    bm25_k: int = DEFAULT_BM25_K,
    dense_k: int = DEFAULT_DENSE_K,
    final_k: int = 5,
    rerank_k: int = DEFAULT_RERANK_K,
    max_per_url: int = 2,
) -> list[RetrievalResult]:
    """
    Retrieval ibrido semplificato:
    1. BM25 retrieval (ricerca lessicale)
    2. Dense retrieval (ricerca semantica)
    3. Reciprocal Rank Fusion per combinare i ranking
    4. Neural reranking con cross-encoder
    5. Deduplica URL (max 2 chunk per pagina)
    6. Return top-k risultati
    """
    global _LAST_RETRIEVAL_TRACE

    # Il planner orienta la ricerca ma non blocca mai BM25/dense. In caso di
    # bassa confidenza il risultato resta una fusione ampia.
    query_plan = plan_query(query)
    retrieval_query = build_retrieval_query(query, query_plan)
    if max_per_url == 2 and (
        query_plan.requires_complete_answer
        or query_plan.task_type in {"study_plan", "course_catalog", "teacher_publications"}
    ):
        max_per_url = 8 if query_plan.task_type != "teacher_publications" else 12

    # Candidate generation multi-representation.
    result_lists = {
        "bm25": bm25_retrieve(retrieval_query, k=bm25_k),
        "dense": dense_retrieve_wrapped(retrieval_query, k=dense_k),
        "locator_dense": dense_locator_retrieve_wrapped(retrieval_query, k=max(20, min(dense_k, 60))),
        "structured": structured_retrieve_wrapped(retrieval_query, query_plan, k=max(20, final_k * 4)),
    }
    result_lists = {
        name: results
        for name, results in result_lists.items()
        if results
    }

    # Fusione dei ranking con RRF
    # RRF mantiene scale di score diverse separate: BM25, Chroma body, Chroma
    # locator e structured evidence contribuiscono come ranking indipendenti.
    hybrid_results = reciprocal_rank_fusion(result_lists)

    # Neural reranking per migliorare la precisione
    neural_pool_k = max(rerank_k, final_k * 4)
    hybrid_results = neural_rerank(
        query=query,
        results=hybrid_results,
        top_k=neural_pool_k,
    )
    hybrid_results = pin_structured_evidence(
        results=hybrid_results,
        structured_results=result_lists.get("structured", []),
        plan=query_plan,
        final_k=final_k,
    )

    # Deduplica URL: max 2 chunk per pagina per diversificare i risultati
    if max_per_url > 0:
        url_counts: dict[str, int] = {}
        deduped_results: list[RetrievalResult] = []
        
        for result in hybrid_results:
            url = result.metadata.get("source_url") or result.metadata.get("document_url") or result.chunk_id
            count = url_counts.get(url, 0)
            
            if count < max_per_url:
                deduped_results.append(result)
                url_counts[url] = count + 1
        
        hybrid_results = deduped_results

    final_results = hybrid_results[:final_k]
    # Trace compatto per capire perché una risposta non trova fonti o perché
    # una certa evidenza è arrivata nel top-k.
    _LAST_RETRIEVAL_TRACE = {
        "query": query,
        "retrieval_query": retrieval_query,
        "plan": {
            "task_type": query_plan.task_type,
            "confidence": query_plan.confidence,
            "entities": query_plan.entities,
            "filters": query_plan.filters,
            "source_families": list(query_plan.source_families),
            "needs_structured_data": query_plan.needs_structured_data,
            "requires_complete_answer": query_plan.requires_complete_answer,
            "reason": query_plan.reason,
        },
        "candidate_counts": {
            name: len(results)
            for name, results in result_lists.items()
        },
        "max_per_url": max_per_url,
        "returned": [
            {
                "rank": result.rank,
                "chunk_id": result.chunk_id,
                "score": result.score,
                "title": result.metadata.get("title"),
                "url": result.metadata.get("source_url") or result.metadata.get("document_url"),
                "chunk_kind": result.metadata.get("chunk_kind"),
                "sources": result.metadata.get("retrieval_sources", []),
            }
            for result in final_results
        ],
    }

    for result in final_results:
        result.metadata["query_plan_task"] = query_plan.task_type
        result.metadata["query_plan_confidence"] = query_plan.confidence

    return final_results


def get_last_retrieval_trace() -> dict[str, Any]:
    return dict(_LAST_RETRIEVAL_TRACE)


def print_results(query: str, results: list[RetrievalResult]) -> None:
    """Stampa risultati in formato leggibile."""
    print()
    print(f"Query: {query}")
    print(f"Risultati trovati: {len(results)}")
    print("-" * 80)

    for result in results:
        title = result.metadata.get("title")
        url = result.metadata.get("source_url") or result.metadata.get("document_url")
        breadcrumb = result.metadata.get("breadcrumb") or result.metadata.get("breadcrumb_text")

        print(f"Risultato {result.rank}")
        print(f"Score: {result.score:.6f}")
        print(f"Source: {result.source}")
        print(f"Titolo: {title}")
        print(f"URL: {url}")
        print(f"Breadcrumb: {breadcrumb}")
        print(f"Chunk ID: {result.chunk_id}")
        print()
        print(preview_text(result.text))
        print("-" * 80)


def main() -> None:
    """CLI per testare il retrieval ibrido."""
    parser = argparse.ArgumentParser(
        description="Retrieval ibrido BM25 + Dense per il chatbot DIEM."
    )

    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Domanda da cercare nel corpus DIEM.",
    )

    parser.add_argument(
        "--bm25-k",
        type=int,
        default=80,
        help="Numero di risultati BM25 da recuperare.",
    )

    parser.add_argument(
        "--dense-k",
        type=int,
        default=80,
        help="Numero di risultati dense da recuperare.",
    )

    parser.add_argument(
        "--final-k",
        type=int,
        default=5,
        help="Numero finale di risultati hybrid da mostrare.",
    )

    parser.add_argument(
        "--rerank-k",
        type=int,
        default=DEFAULT_RERANK_K,
        help="Numero di candidati da passare al reranker neurale.",
    )

    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Mostra testo completo dei chunk.",
    )

    args = parser.parse_args()

    results = hybrid_retrieve(
        query=args.query,
        bm25_k=args.bm25_k,
        dense_k=args.dense_k,
        final_k=args.final_k,
        rerank_k=args.rerank_k,
    )

    print_results(args.query, results)


if __name__ == "__main__":
    main()
