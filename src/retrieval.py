from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from typing import Any

from rank_bm25 import BM25Okapi

from pipeline_io import load_jsonl
from vector_store import CHUNKS_FILE, dense_retrieve, preview_text


ITALIAN_STOPWORDS = {
    "a", "ad", "al", "allo", "alla", "alle", "agli", "ai",
    "con", "da", "dal", "dalla", "dalle", "dei", "del", "dell",
    "della", "delle", "di", "e", "è", "il", "lo", "la", "i",
    "gli", "le", "in", "nel", "nella", "nelle", "nei", "per",
    "su", "sul", "sulla", "sulle", "sono", "un", "una", "uno",
    "che", "chi", "cosa", "come", "quando", "dove", "quale",
    "quali", "quanto", "quanti", "mi", "puoi", "sapere"
}


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    source: str
    rank: int
    score: float


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9_]+", text.lower())
    return [
        token
        for token in tokens
        if token not in ITALIAN_STOPWORDS and len(token) > 1
    ]


def bm25_retrieve(query: str, k: int = 20) -> list[RetrievalResult]:
    """
    Recupera chunk tramite BM25, quindi ricerca lessicale basata su parole chiave.
    """
    chunks = load_jsonl(CHUNKS_FILE)

    valid_chunks = [
        chunk
        for chunk in chunks
        if chunk.get("chunk_id") and chunk.get("text")
    ]

    texts = [chunk["text"] for chunk in valid_chunks]
    tokenized_corpus = [tokenize(text) for text in texts]

    bm25 = BM25Okapi(tokenized_corpus)
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

        results.append(
            RetrievalResult(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                metadata=chunk,
                source="bm25",
                rank=rank,
                score=float(scores[index]),
            )
        )

    return results


def dense_retrieve_wrapped(query: str, k: int = 20) -> list[RetrievalResult]:
    """
    Recupera chunk tramite dense retrieval usando Chroma.
    """
    docs_and_scores = dense_retrieve(query, k=k)

    results = []

    for rank, (doc, score) in enumerate(docs_and_scores, start=1):
        metadata = dict(doc.metadata or {})

        chunk_id = str(
            metadata.get("chunk_id")
            or metadata.get("text_hash")
            or f"dense_rank_{rank}"
        )

        results.append(
            RetrievalResult(
                chunk_id=chunk_id,
                text=doc.page_content,
                metadata=metadata,
                source="dense",
                rank=rank,
                score=float(score),
            )
        )

    return results


def reciprocal_rank_fusion(
    result_lists: dict[str, list[RetrievalResult]],
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    """
    Combina BM25 e dense retrieval usando Reciprocal Rank Fusion.

    Non confronta direttamente gli score, ma usa le posizioni nei ranking.
    Questo è utile perché BM25 e Chroma producono score con scale diverse.
    """
    fused_results: dict[str, RetrievalResult] = {}
    fused_scores: dict[str, float] = {}

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

    final_results = []

    for chunk_id, result in fused_results.items():
        result.score = fused_scores[chunk_id]
        final_results.append(result)

    final_results.sort(key=lambda result: result.score, reverse=True)

    for rank, result in enumerate(final_results, start=1):
        result.rank = rank

    return final_results

def deduplicate_by_url(
    results: list[RetrievalResult],
    max_per_url: int = 1,
) -> list[RetrievalResult]:
    counts: dict[str, int] = {}
    deduped: list[RetrievalResult] = []

    for result in results:
        url = (
            result.metadata.get("source_url")
            or result.metadata.get("document_url")
            or result.metadata.get("url")
            or "unknown"
        )

        current_count = counts.get(url, 0)

        if current_count >= max_per_url:
            continue

        counts[url] = current_count + 1
        deduped.append(result)

    for rank, result in enumerate(deduped, start=1):
        result.rank = rank

    return deduped


def rerank_with_metadata_signals(
    results: list[RetrievalResult],
    query: str,
) -> list[RetrievalResult]:
    """
    Applica piccoli aggiustamenti al ranking hybrid usando segnali sui metadati.

    Obiettivi:
    - penalizzare pagine inglesi quando la query è in italiano;
    - penalizzare pagine di anni accademici molto vecchi se la query non specifica un anno;
    - favorire pagine specifiche sull'offerta formativa quando la query riguarda corsi/lauree;
    - favorire pagine docente solo quando la query contiene un possibile nome/cognome.
    """
    query_lower = query.lower()

    query_tokens = set(tokenize(query_lower))

    generic_teacher_words = {
        "professor",
        "professore",
        "professori",
        "ricevimenti",
        "professoressa",
        "prof",
        "prof.",
        "docente",
        "docenti",
        "ricevimento",
        "orario",
        "orari",
        "ore",
        "quando",
    }

    teacher_query = any(
        keyword in query_lower
        for keyword in [
            "professor",
            "professore",
            "professoressa",
            "prof.",
            "prof",
            "docente",
            "ricevimento",
        ]
    )

    teacher_name_tokens = query_tokens - generic_teacher_words

    asks_office_hours = any(
        keyword in query_lower
        for keyword in ["ricevimento", "orario di ricevimento", "orari di ricevimento"]
    )
    
    wants_degree_info = any(
        keyword in query_lower
        for keyword in [
            "corsi di laurea",
            "corso di laurea",
            "lauree",
            "laurea",
            "laurea magistrale",
            "offerta formativa",
            "corsi di studio",
            "programmi di studio",
        ]
    )

    wants_teaching_info = any(
        keyword in query_lower
        for keyword in [
            "corso",
            "corsi",
            "didattica",
            "insegnamento",
            "insegnamenti",
            "lezioni",
            "esami",
            "programma",
            "programmi",
        ]
    )

    query_mentions_year = bool(re.search(r"20\d{2}", query_lower))

    italian_query = not any(
        word in query_lower
        for word in ["what", "which", "where", "who", "degree", "course", "teaching"]
    )

    for result in results:
        url = (
            result.metadata.get("source_url")
            or result.metadata.get("document_url")
            or result.metadata.get("url")
            or ""
        )

        title = str(result.metadata.get("title") or "").lower()
        breadcrumb = str(
            result.metadata.get("breadcrumb")
            or result.metadata.get("breadcrumb_text")
            or ""
        ).lower()

        adjusted_score = result.score

        # Query su docenti:
        # favorisce pagine docente che contengono nome/cognome presenti nella query.
        if teacher_query and teacher_name_tokens:
            combined_metadata_text = f"{title} {breadcrumb} {url}".lower()
            metadata_tokens = set(tokenize(combined_metadata_text))

            matching_name_tokens = teacher_name_tokens.intersection(metadata_tokens)

            is_docenti_page = "docenti.unisa.it" in url
            is_requested_teacher_page = len(matching_name_tokens) >= 1

            text_lower = result.text.lower()
            contains_office_hours = (
                "orario di ricevimento" in text_lower
                or "ricevimento" in text_lower
            )

            if asks_office_hours:
                if is_docenti_page and is_requested_teacher_page and contains_office_hours:
                    adjusted_score *= 1.60

                elif is_docenti_page and not is_requested_teacher_page:
                    adjusted_score *= 0.35

                elif not is_docenti_page:
                    adjusted_score *= 0.60

            else:
                if is_docenti_page and is_requested_teacher_page:
                    adjusted_score *= 1.35

                elif is_docenti_page and not is_requested_teacher_page:
                    adjusted_score *= 0.60

        # Penalizza pagina inglese se la query è italiana
        if italian_query and "/en" in url:
            adjusted_score *= 0.75

        # Penalizza anni vecchi se l'utente non ha chiesto uno specifico anno
        if not query_mentions_year:
            year_match = re.search(r"anno=(20\d{2})", url)
            if year_match:
                year = int(year_match.group(1))
                if year < 2025:
                    adjusted_score *= 0.70

        # Query specifica su corsi di laurea/offerta formativa
        if wants_degree_info:
            if "offerta-formativa" in url:
                adjusted_score *= 1.30

            if "presentazione" in url and "dipartimento" in url:
                adjusted_score *= 1.10

            if "didattica" in title or "didattica" in breadcrumb:
                adjusted_score *= 1.05

        # Query didattica generica, ma non specifica sui corsi di laurea
        elif wants_teaching_info:
            if "didattica" in title or "didattica" in breadcrumb:
                adjusted_score *= 1.10

            if "offerta-formativa" in url:
                adjusted_score *= 1.10

        result.score = adjusted_score

    results.sort(key=lambda result: result.score, reverse=True)

    for rank, result in enumerate(results, start=1):
        result.rank = rank

    return results

def hybrid_retrieve(
    query: str,
    bm25_k: int = 20,
    dense_k: int = 20,
    final_k: int = 5,
) -> list[RetrievalResult]:
    """
    Retrieval ibrido:
    - BM25 per parole chiave;
    - Chroma per similarità semantica;
    - RRF per fusione dei ranking.
    """
    bm25_results = bm25_retrieve(query, k=bm25_k)
    dense_results = dense_retrieve_wrapped(query, k=dense_k)

    hybrid_results = reciprocal_rank_fusion(
        {
            "bm25": bm25_results,
            "dense": dense_results,
        }
    )

    hybrid_results = deduplicate_by_url(hybrid_results, max_per_url=1)
    hybrid_results = rerank_with_metadata_signals(hybrid_results, query)

    return hybrid_results[:final_k]


def print_results(query: str, results: list[RetrievalResult]) -> None:
    print()
    print(f"Query: {query}")
    print(f"Risultati hybrid trovati: {len(results)}")
    print("-" * 80)

    for result in results:
        title = result.metadata.get("title")
        url = result.metadata.get("source_url") or result.metadata.get("document_url")
        breadcrumb = result.metadata.get("breadcrumb") or result.metadata.get("breadcrumb_text")

        print(f"Risultato {result.rank}")
        print(f"Score hybrid: {result.score:.6f}")
        print(f"Titolo: {title}")
        print(f"URL: {url}")
        print(f"Breadcrumb: {breadcrumb}")
        print(f"Chunk ID: {result.chunk_id}")
        print()
        print(preview_text(result.text))
        print("-" * 80)


def main() -> None:
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
        default=20,
        help="Numero di risultati BM25 da recuperare.",
    )

    parser.add_argument(
        "--dense-k",
        type=int,
        default=20,
        help="Numero di risultati dense da recuperare.",
    )

    parser.add_argument(
        "--final-k",
        type=int,
        default=5,
        help="Numero finale di risultati hybrid da mostrare.",
    )

    args = parser.parse_args()

    results = hybrid_retrieve(
        query=args.query,
        bm25_k=args.bm25_k,
        dense_k=args.dense_k,
        final_k=args.final_k,
    )

    print_results(args.query, results)


if __name__ == "__main__":
    main()