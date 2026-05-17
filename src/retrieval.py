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
    


def expand_query_for_retrieval(query: str) -> str:
    """
    Espande query brevi o colloquiali con termini più vicini
    al linguaggio usato nelle pagine DIEM.
    
    """
    query_lower = query.lower()

    expanded_terms: list[str] = []

    wants_people_info = is_aggregate_query(query) and any(
        keyword in query_lower
        for keyword in [
            "prof",
            "professori",
            "professore",
            "professor",
            "docenti",
            "docente",
            "personale docente",
            "personale",
            "insegnanti",
            "ricercatori",
            "ricercatore",
        ]
    )

    if wants_people_info:
        expanded_terms.extend(
            [
                "docenti",
                "personale",
                "docenti e personale",
                "personale docente",
                "professori",
                "ricercatori",
                "dipartimento",
                "DIEM",
                "Università di Salerno",
            ]
        )

    wants_location_info = any(
        keyword in query_lower
        for keyword in [
            "dove si trova",
            "dove è",
            "dove sta",
            "indirizzo",
            "sede",
            "ubicazione",
            "contatti",
            "raggiungere",
        ]
    )

    if wants_location_info:
        expanded_terms.extend(
            [
                "contatti",
                "indirizzo",
                "sede",
                "ubicazione",
                "campus",
                "edificio",
                "Fisciano",
            ]
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

    if wants_degree_info:
        expanded_terms.extend(
            [
                "offerta formativa",
                "corsi di laurea",
                "laurea magistrale",
                "didattica",
                "corsi di studio",
            ]
        )

    wants_structures_info = any(
        keyword in query_lower
        for keyword in [
            "laboratori",
            "laboratorio",
            "strutture",
            "aule",
            "biblioteche",
            "centri",
        ]
    )

    if wants_structures_info:
        expanded_terms.extend(
            [
                "strutture",
                "laboratori",
                "aule",
                "centri",
                "dipartimento",
            ]
        )

    wants_international_info = any(
        keyword in query_lower
        for keyword in [
            "erasmus",
            "internazionale",
            "international",
            "mobilità",
            "mobilita",
            "accordi",
            "estero",
        ]
    )

    if wants_international_info:
        expanded_terms.extend(
            [
                "international",
                "erasmus",
                "mobilità",
                "accordi erasmus plus",
                "traineeship",
                "studio all'estero",
            ]
        )

    wants_admission_info = any(
        keyword in query_lower
        for keyword in [
            "requisiti",
            "accesso",
            "modalità di accesso",
            "modalita di accesso",
            "ammissione",
            "immatricolazione",
            "immatricolazioni",
            "iscriversi",
            "iscrizione",
            "tolc",
            "ofa",
            "verifica dei requisiti",
        ]
    )

    if wants_admission_info:
        expanded_terms.extend(
            [
                "modalità di accesso",
                "immatricolazioni",
                "requisiti di accesso",
                "verifica dei requisiti",
                "ammissione",
                "OFA",
                "TOLC",
            ]
        )
    
    wants_erasmus_bando_info = any(
        keyword in query_lower
        for keyword in [
            "bando erasmus",
            "bando",
            "informazioni erasmus",
            "informazioni sul bando",
            "mobilità internazionale",
            "mobilita internazionale",
            "mobilità in uscita",
            "mobilita in uscita",
            "candidatura erasmus",
            "scadenze erasmus",
            "call erasmus",
        ]
    ) and "erasmus" in query_lower

    if wants_erasmus_bando_info:
        expanded_terms.extend(
            [
                "informazioni bando erasmus",
                "international mobility",
                "mobilità internazionale",
                "mobilità in uscita",
                "bando erasmus",
                "studio",
                "tirocinio",
                "learning agreement",
                "scadenze",
                "candidatura",
            ]
        )

    wants_erasmus_agreements = any(
        keyword in query_lower
        for keyword in [
            "accordi erasmus",
            "accordo erasmus",
            "accordi",
            "università partner",
            "universita partner",
            "partner",
            "traineeship",
            "studio",
            "docenza",
        ]
    ) and "erasmus" in query_lower

    if wants_erasmus_agreements:
        expanded_terms.extend(
            [
                "accordi erasmus plus",
                "mobilità per studio",
                "mobilità per traineeship",
                "mobilità per docenza",
                "università partner",
                "paese",
                "data scadenza",
            ]
        )
    
    if not expanded_terms:
        return query

    return query + " " + " ".join(expanded_terms)


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


def is_single_profile_page(url: str) -> bool:
    """
    Riconosce pagine profilo singole.
    Per query aggregative sono meno adatte rispetto a pagine indice/lista.
    """
    normalized_url = url.rstrip("/")
    return "docenti.unisa.it" in normalized_url and normalized_url.endswith("/home")


def is_detail_or_news_page(url: str) -> bool:
    """
    Riconosce pagine di dettaglio, news o focus.

    Per query aggregative sono spesso meno centrali rispetto alle pagine sezione/indice.
    """
    normalized_url = url.lower()

    detail_patterns = [
        "unisa-rescue-page/dettaglio",
        "/didattica/focus?id=",
        "/news",
        "row/",
        "module/",
    ]

    return any(pattern in normalized_url for pattern in detail_patterns)


def is_parameterized_detail_page(url: str) -> bool:
    """
    Riconosce pagine dettaglio con parametri.
    Per query aggregative sono spesso meno centrali rispetto alle pagine indice.
    """
    normalized_url = url.lower()

    detail_markers = [
        "?id=",
        "&id=",
        "dettaglio=",
        "row/",
        "module/",
    ]

    return any(marker in normalized_url for marker in detail_markers)


def metadata_relevance_multiplier(
    result: RetrievalResult,
    query: str,
) -> float:
    """
    Boost generale basato sulla coerenza tra query espansa e metadati.

    Per query aggregative, titolo/breadcrumb/URL contano di più perché
    indicano pagine centrali o pagine indice, non semplici pagine che
    citano le parole nel contenuto.
    """
    generic_metadata_tokens = {
        "diem",
        "unisa",
        "università",
        "universita",
        "salerno",
        "dipartimento",
        "www",
        "https",
        "http",
        "it",
        "home",
    }

    expanded_query = expand_query_for_retrieval(query)
    query_tokens = set(tokenize(expanded_query)) - generic_metadata_tokens

    url = str(
        result.metadata.get("source_url")
        or result.metadata.get("document_url")
        or result.metadata.get("url")
        or ""
    )

    title = str(result.metadata.get("title") or "")
    breadcrumb = str(
        result.metadata.get("breadcrumb")
        or result.metadata.get("breadcrumb_text")
        or ""
    )

    metadata_text = f"{title} {breadcrumb} {url}".lower()
    metadata_tokens = set(tokenize(metadata_text)) - generic_metadata_tokens

    overlap = len(query_tokens.intersection(metadata_tokens))

    if is_aggregate_query(query):
        if overlap >= 4:
            return 3.20

        if overlap == 3:
            return 2.80

        if overlap == 2:
            return 2.50

        if overlap == 1:
            return 1.35

        return 1.0

    if overlap >= 4:
        return 2.20

    if overlap == 3:
        return 1.90

    if overlap == 2:
        return 1.60

    if overlap == 1:
        return 1.25

    return 1.0


def is_aggregate_query(query: str) -> bool:
    """
    Riconosce domande che chiedono elenchi, panoramiche o più elementi.
    Non è legata solo ai professori.
    """
    query_lower = query.lower()

    aggregate_intent = any(
        keyword in query_lower
        for keyword in [
            "elenca",
            "elencami",
            "lista",
            "quali sono",
            "chi sono",
            "mostrami",
            "dimmi quali",
            "che cosa offre",
            "cosa offre",
            "quali",
        ]
    )

    aggregate_objects = any(
        keyword in query_lower
        for keyword in [
            "professori",
            "docenti",
            "personale",
            "ricercatori",
            "corsi",
            "corsi di laurea",
            "lauree",
            "laboratori",
            "strutture",
            "servizi",
            "opportunità",
            "opportunita",
            "accordi",
            "erasmus",
            "aule",
        ]
    )

    return aggregate_intent and aggregate_objects


def is_erasmus_query(query: str) -> bool:
    query_lower = query.lower()
    return "erasmus" in query_lower


def deduplicate_for_query(
    results: list[RetrievalResult],
    query: str,
) -> list[RetrievalResult]:
    """
    Deduplica adattata al tipo di domanda.

    Query puntuali:
    - massimo 1 chunk per URL.

    Query aggregative:
    - più chunk per URL, perché liste e panoramiche possono essere distribuite
      su più sezioni della stessa pagina.

    Query Erasmus:
    - massimo 2 chunk per URL, per evitare che una sola modalità
      monopolizzi il contesto.
    """
    if is_erasmus_query(query):
        return deduplicate_by_url(results, max_per_url=2)

    if is_aggregate_query(query):
        return deduplicate_by_url(results, max_per_url=4)

    return deduplicate_by_url(results, max_per_url=1)


def teacher_metadata_matches(
    teacher_name_tokens: set[str],
    metadata_tokens: set[str],
) -> bool:
    """
    Match generale per pagine docente.

    Se nella query ci sono almeno due token utili, richiede almeno due match.
    Evita casi come "Luca Greco" -> "Antonio Greco".
    """
    matching_tokens = teacher_name_tokens.intersection(metadata_tokens)

    if len(teacher_name_tokens) >= 2:
        return len(matching_tokens) >= 2

    return len(matching_tokens) >= 1


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
    
    wants_admission_info = any(
        keyword in query_lower
        for keyword in [
            "requisiti",
            "accesso",
            "modalità di accesso",
            "modalita di accesso",
            "ammissione",
            "immatricolazione",
            "immatricolazioni",
            "iscriversi",
            "iscrizione",
            "tolc",
            "ofa",
            "verifica dei requisiti",
        ]
    )

    wants_master_degree = any(
        keyword in query_lower
        for keyword in [
            "magistrale",
            "laurea magistrale",
        ]
    )
    
    wants_erasmus_bando_info = any(
        keyword in query_lower
        for keyword in [
            "bando erasmus",
            "bando",
            "informazioni erasmus",
            "informazioni sul bando",
            "mobilità internazionale",
            "mobilita internazionale",
            "mobilità in uscita",
            "mobilita in uscita",
            "candidatura erasmus",
            "scadenze erasmus",
            "call erasmus",
        ]
    ) and "erasmus" in query_lower

    wants_erasmus_agreements = any(
        keyword in query_lower
        for keyword in [
            "accordi erasmus",
            "accordo erasmus",
            "accordi",
            "università partner",
            "universita partner",
            "partner",
            "traineeship",
            "studio",
            "docenza",
        ]
    ) and "erasmus" in query_lower
    
    wants_aggregate_info = is_aggregate_query(query)

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
        
        # Boost generale: se titolo, breadcrumb o URL sono coerenti con la query,
        # il risultato è probabilmente più centrale.
        adjusted_score *= metadata_relevance_multiplier(result, query)

        # Se la domanda chiede un elenco/panoramica, le pagine profilo singole
        # e le pagine di dettaglio/news sono meno adatte delle pagine indice/lista.
        if wants_aggregate_info:
            if is_single_profile_page(str(url)):
                adjusted_score *= 0.65

            if is_detail_or_news_page(str(url)):
                adjusted_score *= 0.70
            
            if is_parameterized_detail_page(str(url)):
                adjusted_score *= 0.55

        # Query su docenti:
        # favorisce pagine docente che contengono nome/cognome presenti nella query.
        if teacher_query and teacher_name_tokens and not wants_aggregate_info:
            combined_metadata_text = f"{title} {breadcrumb} {url}".lower()
            metadata_tokens = set(tokenize(combined_metadata_text))

            is_docenti_page = "docenti.unisa.it" in url
            is_requested_teacher_page = teacher_metadata_matches(
                teacher_name_tokens=teacher_name_tokens,
                metadata_tokens=metadata_tokens,
            )

            text_lower = result.text.lower()
            contains_office_hours = (
                "orario di ricevimento" in text_lower
                or "ricevimento" in text_lower
            )

            if asks_office_hours:
                is_home_page = str(url).rstrip("/").endswith("/home")

                if (
                    is_docenti_page
                    and is_requested_teacher_page
                    and contains_office_hours
                    and is_home_page
                ):
                    adjusted_score *= 2.20

                elif is_docenti_page and is_requested_teacher_page and contains_office_hours:
                    adjusted_score *= 1.70

                elif is_docenti_page and is_requested_teacher_page and not contains_office_hours:
                    adjusted_score *= 0.60

                elif is_docenti_page and not is_requested_teacher_page:
                    adjusted_score *= 0.30

                elif not is_docenti_page:
                    adjusted_score *= 0.55

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

        # Query su requisiti di accesso / immatricolazioni / ammissione
        if wants_admission_info:
            if "immatricolazioni" in str(url):
                adjusted_score *= 1.80

            if "modalità di accesso" in title or "modalità di accesso" in breadcrumb:
                adjusted_score *= 1.70

            if "modalita di accesso" in title or "modalita di accesso" in breadcrumb:
                adjusted_score *= 1.70

            if "verifica dei requisiti" in result.text.lower():
                adjusted_score *= 1.50

            if wants_master_degree and "magistrale" in str(url).lower():
                adjusted_score *= 1.50
        
        # Query su Erasmus: distingue informazioni sul bando da accordi/partner
        if wants_erasmus_bando_info:
            text_lower = result.text.lower()

            if "international-mobility" in str(url).lower():
                adjusted_score *= 2.20

            if "informazioni bando erasmus" in title or "informazioni bando erasmus" in breadcrumb:
                adjusted_score *= 2.00

            if "bando erasmus" in text_lower:
                adjusted_score *= 1.70

            if "learning agreement" in text_lower:
                adjusted_score *= 1.30

            if "accordi-erasmus-plus" in str(url).lower():
                adjusted_score *= 0.70

        if wants_erasmus_agreements:
            if "accordi-erasmus-plus" in str(url).lower():
                adjusted_score *= 1.70

            if "mobilità per studio" in title or "mobilità per studio" in breadcrumb:
                adjusted_score *= 1.30

            if "mobilità per traineeship" in title or "mobilità per traineeship" in breadcrumb:
                adjusted_score *= 1.30

            if "mobilità per docenza" in title or "mobilità per docenza" in breadcrumb:
                adjusted_score *= 1.20
        
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
    bm25_k: int = 30,
    dense_k: int = 30,
    final_k: int = 5,
) -> list[RetrievalResult]:
    """
    Retrieval ibrido:
    - BM25 per parole chiave;
    - Chroma per similarità semantica;
    - RRF per fusione dei ranking.
    """
    retrieval_query = expand_query_for_retrieval(query)

    bm25_results = bm25_retrieve(retrieval_query, k=bm25_k)
    dense_results = dense_retrieve_wrapped(retrieval_query, k=dense_k)

    hybrid_results = reciprocal_rank_fusion(
        {
            "bm25": bm25_results,
            "dense": dense_results,
        }
    )

    hybrid_results = rerank_with_metadata_signals(hybrid_results, query)

    hybrid_results = deduplicate_for_query(
        results=hybrid_results,
        query=query,
    )

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
        default=30,
        help="Numero di risultati BM25 da recuperare.",
    )

    parser.add_argument(
        "--dense-k",
        type=int,
        default=30,
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