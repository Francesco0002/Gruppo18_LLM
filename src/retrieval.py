from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi

from chunk_metadata import flatten_chunk_metadata
from pipeline_io import load_jsonl
from reranking import neural_rerank
from vector_store import CHUNKS_FILE, dense_retrieve, preview_text


DEFAULT_BM25_K = int(os.getenv("RETRIEVAL_BM25_K", "80"))
DEFAULT_DENSE_K = int(os.getenv("RETRIEVAL_DENSE_K", "80"))
DEFAULT_RERANK_K = int(os.getenv("RETRIEVAL_RERANK_K", "20"))


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


@dataclass(frozen=True)
class SiteIntentSpec:
    name: str
    query_patterns: tuple[str, ...]
    expansion_terms: tuple[str, ...] = ()
    preferred_chunk_kinds: tuple[str, ...] = ()
    positive_url_patterns: tuple[str, ...] = ()
    negative_url_patterns: tuple[str, ...] = ()
    negative_query_patterns: tuple[str, ...] = ()
    max_per_url: int = 1
    dedup_metadata_keys: tuple[str, ...] = ()

    def matches(self, query: str) -> bool:
        if any(re.search(pattern, query, re.IGNORECASE) for pattern in self.negative_query_patterns):
            return False

        return any(re.search(pattern, query, re.IGNORECASE) for pattern in self.query_patterns)


SITE_INTENT_SPECS: tuple[SiteIntentSpec, ...] = (
    SiteIntentSpec(
        name="teacher_office_hours",
        query_patterns=(r"\bricevimento\b", r"\borari?\s+di\s+ricevimento\b", r"\briceve\b"),
        expansion_terms=("orario di ricevimento", "ricevimento docente", "pagina personale docente"),
        preferred_chunk_kinds=("office_hours",),
        positive_url_patterns=(r"docenti\.unisa\.it/.+/home",),
        negative_url_patterns=(r"/dipartimento/personale",),
        max_per_url=2,
    ),
    SiteIntentSpec(
        name="teacher_publications",
        query_patterns=(r"\bpubblicazioni?\b", r"\barticoli?\b", r"\bpaper\b", r"\blavori\s+scientifici\b"),
        expansion_terms=("pubblicazioni docente", "publication_summary", "anno pubblicazione", "doi iris"),
        preferred_chunk_kinds=("publication_summary", "teacher_publications_page"),
        positive_url_patterns=(r"docenti\.unisa\.it/.+/ricerca/pubblicazioni",),
        max_per_url=8,
        dedup_metadata_keys=("publication_id",),
    ),
    SiteIntentSpec(
        name="teacher_projects",
        query_patterns=(r"\bprogetti?\b", r"\bricerca\b", r"\bresponsabile\s+scientifico\b"),
        expansion_terms=("progetti di ricerca", "progetti finanziati", "responsabile scientifico"),
        preferred_chunk_kinds=("teacher_projects",),
        positive_url_patterns=(r"docenti\.unisa\.it/.+/ricerca/progetti", r"diem\.unisa\.it/.+progetti"),
        max_per_url=3,
    ),
    SiteIntentSpec(
        name="lab_equipment",
        query_patterns=(r"\bstrumentazione\b", r"\battrezzature\b", r"\bdotazione\b", r"\blaborator[io]\b", r"\blabrob\b", r"\bmivia\b"),
        expansion_terms=("strumentazione", "dotazione", "attrezzature", "laboratori", "strutture"),
        preferred_chunk_kinds=("lab_equipment",),
        positive_url_patterns=(r"/dipartimento/strutture", r"/ricerca/laboratori"),
        max_per_url=3,
    ),
    SiteIntentSpec(
        name="course_statistics",
        query_patterns=(r"\balmalaurea\b", r"\bstatistiche\b", r"\bvalutazion[ei]\b", r"\bsoddisfazione\b", r"\bcondizione\s+occupazionale\b"),
        expansion_terms=("AlmaLaurea", "statistiche", "profilo dei laureati", "condizione occupazionale", "valutazione della didattica"),
        preferred_chunk_kinds=("course_statistic",),
        positive_url_patterns=(r"statistiche", r"__almalaurea"),
        max_per_url=3,
    ),
    SiteIntentSpec(
        name="course_info",
        query_patterns=(r"\bcorsi?\s+di\s+laurea\b", r"\blaure[ae]\b", r"\bofferta\s+formativa\b", r"\binsegnamenti?\b", r"\bpiano\s+di\s+studi\b"),
        expansion_terms=("offerta formativa", "corsi di laurea", "insegnamenti", "piano di studi", "didattica"),
        preferred_chunk_kinds=("course_info",),
        positive_url_patterns=(r"offerta-formativa", r"piano-di-studi", r"coursecatalogue"),
        negative_query_patterns=(r"\bsedut[ae]\s+di\s+laurea\b", r"\bprova\s+finale\b", r"\besame\s+finale\b", r"\bdomanda\s+di\s+laurea\b", r"\bconseguimento\s+(?:del\s+)?titolo\b"),
        max_per_url=3,
    ),
    SiteIntentSpec(
        name="final_exam",
        query_patterns=(r"\bsedut[ae]\s+di\s+laurea\b", r"\bprova\s+finale\b", r"\besame\s+finale\b", r"\bdomanda\s+di\s+laurea\b", r"\bconseguimento\s+(?:del\s+)?titolo\b", r"\bappell[oi]\s+di\s+laurea\b", r"\blaurearsi\b"),
        expansion_terms=("esame finale", "prova finale", "sedute di laurea", "domanda conseguimento titolo", "appello di laurea", "calendario sedute di laurea"),
        preferred_chunk_kinds=("course_info", "official_document_summary", "official_document"),
        positive_url_patterns=(r"didattica/esame-finale", r"sedut[ae]-di-laurea", r"prova-finale", r"conseguimento"),
        max_per_url=3,
    ),
    SiteIntentSpec(
        name="admission",
        query_patterns=(r"\baccesso\b", r"\bammissione\b", r"\bimmatricolazion[ei]\b", r"\btolc\b", r"\bofa\b", r"\brequisiti\b"),
        expansion_terms=("modalità di accesso", "requisiti di accesso", "immatricolazioni", "OFA", "TOLC"),
        preferred_chunk_kinds=("course_info", "official_document_summary"),
        positive_url_patterns=(r"immatricolazioni", r"modalit", r"requisiti", r"ofa", r"tolc"),
        negative_query_patterns=(r"\bsedut[ae]\s+di\s+laurea\b", r"\bprova\s+finale\b", r"\besame\s+finale\b", r"\bdomanda\s+di\s+laurea\b", r"\bconseguimento\s+(?:del\s+)?titolo\b"),
        max_per_url=3,
    ),
    SiteIntentSpec(
        name="erasmus",
        query_patterns=(r"\berasmus\b", r"\bmobilit[aà]\b", r"\binternational\b", r"\blearning\s+agreement\b"),
        expansion_terms=("Erasmus", "mobilità internazionale", "accordi Erasmus", "traineeship", "learning agreement"),
        preferred_chunk_kinds=("erasmus", "official_document_summary"),
        positive_url_patterns=(r"erasmus", r"international", r"accordi-erasmus-plus"),
        max_per_url=3,
    ),
    SiteIntentSpec(
        name="phd",
        query_patterns=(r"\bdottorat[oi]\b", r"\bphd\b", r"\bdoctoral\b"),
        expansion_terms=("dottorato di ricerca", "phd", "doctoral", "collegio dei docenti"),
        preferred_chunk_kinds=("phd", "official_document_summary"),
        positive_url_patterns=(r"dottorat", r"phd", r"doctoral", r"DOT"),
        max_per_url=3,
    ),
    SiteIntentSpec(
        name="official_document",
        query_patterns=(r"\bbando\b", r"\bregolament[oi]\b", r"\bdecreto\b", r"\bgraduatoria\b", r"\bavviso\b", r"\bpdf\b"),
        expansion_terms=("documento ufficiale", "bando", "regolamento", "decreto", "graduatoria", "avviso"),
        preferred_chunk_kinds=("official_document_summary", "official_document"),
        positive_url_patterns=(r"uploads", r"bando", r"regolament", r"graduatoria", r"avviso"),
        max_per_url=3,
    ),
    SiteIntentSpec(
        name="contacts",
        query_patterns=(r"\bcontatti?\b", r"\bsede\b", r"\bindirizzo\b", r"\bdove\s+si\s+trova\b", r"\bubicazione\b"),
        expansion_terms=("contatti", "sede", "indirizzo", "ubicazione", "campus", "edificio"),
        preferred_chunk_kinds=("office_hours", "text"),
        positive_url_patterns=(r"contatti", r"dipartimento", r"docenti\.unisa\.it"),
        max_per_url=2,
    ),
    SiteIntentSpec(
        name="news",
        query_patterns=(r"\bnews\b", r"\bavvisi?\b", r"\bnotizie\b", r"\beventi?\b"),
        expansion_terms=("news", "avvisi", "eventi", "notizie"),
        positive_url_patterns=(r"news", r"avvisi", r"dettaglio"),
        max_per_url=3,
    ),
)


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9_]+", text.lower())
    return [
        token
        for token in tokens
        if token not in ITALIAN_STOPWORDS and len(token) > 1
    ]
    

@lru_cache(maxsize=1)
def load_valid_chunks() -> tuple[dict[str, Any], ...]:
    chunks = load_jsonl(CHUNKS_FILE)
    return tuple(
        chunk
        for chunk in chunks
        if chunk.get("chunk_id") and chunk.get("text")
    )


@lru_cache(maxsize=1)
def load_tokenized_corpus() -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(tokenize(str(chunk["text"])))
        for chunk in load_valid_chunks()
    )


@lru_cache(maxsize=1)
def get_bm25_index() -> BM25Okapi:
    return BM25Okapi([list(tokens) for tokens in load_tokenized_corpus()])


def get_result_url(result: RetrievalResult) -> str:
    return str(
        result.metadata.get("source_url")
        or result.metadata.get("document_url")
        or result.metadata.get("url")
        or "unknown"
    )


def query_mentions_explicit_year(query: str) -> bool:
    """
    Riconosce se l'utente chiede esplicitamente un anno.
    In quel caso non dobbiamo penalizzare o nascondere pagine storiche.
    """
    return bool(re.search(r"\b(19|20)\d{2}\b", query.lower()))


def query_wants_pdf_evidence(query: str) -> bool:
    query_lower = query.lower()
    return any(
        keyword in query_lower
        for keyword in [
            "bando",
            "bandi",
            "regolamento",
            "regolamenti",
            "decreto",
            "verbale",
            "graduatoria",
            "pdf",
            "allegato",
            "avviso",
            "concorso",
            "selezione",
        ]
    )


def query_wants_final_exam_info(query: str) -> bool:
    query_lower = query.lower()
    return bool(
        re.search(
            r"\b(?:sedut[ae]\s+di\s+laurea|prova\s+finale|esame\s+finale|domanda\s+di\s+laurea|conseguimento\s+(?:del\s+)?titolo|appell[oi]\s+di\s+laurea|laurearsi)\b",
            query_lower,
        )
    )


def active_site_intent_specs(query: str) -> list[SiteIntentSpec]:
    return [spec for spec in SITE_INTENT_SPECS if spec.matches(query)]


def query_intents(query: str) -> set[str]:
    """
    Riconosce famiglie di bisogno informativo, non singole domande.

    Gli intenti servono per aggiungere segnali di ranking mirati senza
    trasformare il retriever in una lista fragile di casi speciali.
    """
    query_lower = query.lower()

    intents: set[str] = {spec.name for spec in active_site_intent_specs(query)}

    equipment_terms = [
        "strumentazione",
        "strumenti",
        "dotazione",
        "attrezzature",
        "apparecchiature",
        "possiede",
        "dispone",
    ]
    structure_terms = [
        "laboratorio",
        "laboratori",
        "struttura",
        "strutture",
        "lab",
        "robotica",
        "mivia",
        "labrob",
    ]

    if any(term in query_lower for term in equipment_terms) and any(
        term in query_lower for term in structure_terms
    ):
        intents.add("lab_equipment")

    statistics_terms = [
        "almalaurea",
        "statistiche",
        "laureati",
        "laureandi",
        "profilo dei laureati",
        "condizione occupazionale",
        "soddisfazione",
        "valutazione della didattica",
        "valutazione dei laureati",
    ]

    if any(term in query_lower for term in statistics_terms):
        intents.add("course_statistics")

    if query_wants_pdf_evidence(query) or "pdf" in query_lower:
        intents.add("official_document")

    return intents


def extract_years_from_text(text: str) -> list[int]:
    return [
        int(match)
        for match in re.findall(r"\b(?:19|20)\d{2}\b", text)
    ]


def normalize_url_for_dedup(url: str) -> str:
    """
    Normalizza URL per deduplicare versioni canoniche e versioni con parametri.

    Esempio:
    /offerta-formativa
    /offerta-formativa?anno=2021

    diventano la stessa chiave logica.
    """
    if not url or url == "unknown":
        return "unknown"

    parts = urlsplit(url)
    path = parts.path.rstrip("/") or "/"
    query_params = {
        name.lower(): value
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
    }
    query = ""

    # Le pagine dettaglio delle strutture DIEM condividono lo stesso path ma
    # descrivono laboratori diversi tramite ?id=. Se togliamo sempre la query,
    # una strumentazione può eliminare l'altra in deduplica.
    if (
        path.lower().endswith("/dipartimento/strutture")
        or path.lower().endswith("/ricerca/laboratori")
    ) and query_params.get("id"):
        query = f"id={query_params['id']}"

    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            path,
            query,
            "",
        )
    )


def query_param_value(url: str, name: str) -> str:
    try:
        params = parse_qsl(urlsplit(url).query, keep_blank_values=True)
    except ValueError:
        return ""

    for param_name, value in params:
        if param_name.lower() == name.lower():
            return value

    return ""


def expand_query_for_retrieval(query: str) -> str:
    """
    Espande query brevi o colloquiali con termini più vicini
    al linguaggio usato nelle pagine DIEM.
    
    """
    query_lower = query.lower()

    expanded_terms: list[str] = []

    for spec in active_site_intent_specs(query):
        expanded_terms.extend(spec.expansion_terms)

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
            "corsi del dipartimento",
            "corsi del diem",
            "corsi offerti",
            "corsi disponibili",
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

    intents = query_intents(query)

    if "lab_equipment" in intents:
        expanded_terms.extend(
            [
                "strumentazione",
                "dotazione",
                "attrezzature",
                "apparecchiature",
                "sezione strumentazione",
                "dipartimento strutture",
            ]
        )

        if "robotica" in query_lower or "labrob" in query_lower:
            expanded_terms.extend(
                [
                    "Laboratorio di Robotica",
                    "LabROB",
                    "Robotica",
                    "strutture id 2",
                ]
            )

    if "course_statistics" in intents:
        expanded_terms.extend(
            [
                "statistiche",
                "AlmaLaurea",
                "livello di soddisfazione dei laureandi",
                "profilo dei laureati",
                "condizione occupazionale dei laureati",
                "valutazione della didattica",
            ]
        )

    if "final_exam" in intents:
        expanded_terms.extend(
            [
                "esame finale",
                "prova finale",
                "sedute di laurea",
                "domanda di laurea",
                "domanda conseguimento titolo",
                "appello di laurea",
                "calendario sedute di laurea",
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
                "mobilità per studio",
                "erasmus studio",
                "traineeship",
                "studio all'estero",
            ]
        )

    wants_final_exam_info = query_wants_final_exam_info(query)

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
    ) and not wants_final_exam_info

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
        
    wants_phd_info = any(
        keyword in query_lower
        for keyword in [
            "dottorato",
            "dottorati",
            "dottorato di ricerca",
            "dottorati di ricerca",
            "phd",
            "ph.d",
            "ph.d.",
        ]
    )

    if wants_phd_info:
        expanded_terms.extend(
            [
                "dottorato di ricerca",
                "dottorati di ricerca",
                "phd",
                "doctoral research program",
                "doctoral education",
                "information engineering",
                "photovoltaics",
                "collegio di dottorato",
                "coordinatore del dottorato",
            ]
        )
    
    if not expanded_terms:
        return query

    return query + " " + " ".join(expanded_terms)


def bm25_retrieve(query: str, k: int = 20) -> list[RetrievalResult]:
    """
    Recupera chunk tramite BM25, quindi ricerca lessicale basata su parole chiave.
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

        results.append(
            RetrievalResult(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                metadata=flatten_chunk_metadata(chunk),
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
    query: str = "",
) -> list[RetrievalResult]:
    """
    Deduplica i risultati per URL.

    Se la query non cita un anno, usa una chiave canonica senza parametri:
    questo evita di mostrare insieme pagina corrente e pagina storica.

    Se la query cita un anno, mantiene gli URL completi:
    questo permette di recuperare pagine storiche come ?anno=2021.
    """
    counts: dict[str, int] = {}
    deduped: list[RetrievalResult] = []

    query_has_year = query_mentions_explicit_year(query)

    for result in results:
        url = get_result_url(result)

        if query_has_year:
            dedup_key = url
        else:
            dedup_key = normalize_url_for_dedup(url)

        current_count = counts.get(dedup_key, 0)

        if current_count >= max_per_url:
            continue

        counts[dedup_key] = current_count + 1
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
    content_title = str(result.metadata.get("content_title") or "")
    section_heading = str(result.metadata.get("section_heading") or "")
    link_text = str(result.metadata.get("link_text") or "")
    discovered_from = str(result.metadata.get("discovered_from") or "")
    document_type = str(result.metadata.get("document_type") or "")
    chunk_kind = str(result.metadata.get("chunk_kind") or "")
    entity_type = str(result.metadata.get("entity_type") or "")
    entity_name = str(result.metadata.get("entity_name") or "")

    metadata_text = (
        f"{title} {content_title} {section_heading} {breadcrumb} "
        f"{url} {discovered_from} {link_text} {document_type} "
        f"{chunk_kind} {entity_type} {entity_name}"
    ).lower()
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
            "pubblicazioni",
            "pubblicazione",
            "articoli",
            "paper",
            "progetti",
            "servizi",
            "opportunità",
            "opportunita",
            "accordi",
            "erasmus",
            "aule",
            "dottorato",
            "dottorati",
            "phd",
        ]
    )

    return aggregate_intent and aggregate_objects


def is_erasmus_query(query: str) -> bool:
    query_lower = query.lower()
    return "erasmus" in query_lower


def requested_erasmus_mobility(query: str) -> str | None:
    query_lower = query.lower()

    if any(term in query_lower for term in [
        "per studio",
        "studio",
        "studenti",
        "student mobility",
        "study mobility",
    ]):
        return "studio"

    if any(term in query_lower for term in [
        "traineeship",
        "tirocinio",
        "tirocini",
        "placement",
    ]):
        return "traineeship"

    if any(term in query_lower for term in [
        "docenza",
        "teaching",
        "docenti",
    ]):
        return "teaching"

    return None


def deduplicate_for_query(
    results: list[RetrievalResult],
    query: str,
) -> list[RetrievalResult]:
    """
    Deduplica adattata al tipo di domanda.

    Query puntuali:
    - massimo 1 chunk per URL logico.

    Query aggregative:
    - più chunk per URL logico, perché liste e panoramiche possono essere distribuite
      su più sezioni della stessa pagina.

    Query Erasmus:
    - massimo 2 chunk per URL logico, per evitare che una sola modalità
      monopolizzi il contesto.

    Se la query contiene un anno esplicito, gli URL con parametri restano distinguibili.
    """
    active_specs = active_site_intent_specs(query)
    metadata_dedup_specs = [
        spec for spec in active_specs if spec.dedup_metadata_keys
    ]

    if metadata_dedup_specs:
        return deduplicate_by_intent_metadata(
            results=results,
            query=query,
            specs=metadata_dedup_specs,
        )

    if active_specs:
        return deduplicate_by_url(
            results,
            max_per_url=max(spec.max_per_url for spec in active_specs),
            query=query,
        )

    if is_aggregate_query(query):
        return deduplicate_by_url(results, max_per_url=4, query=query)

    if is_erasmus_query(query):
        return deduplicate_by_url(results, max_per_url=2, query=query)

    return deduplicate_by_url(results, max_per_url=1, query=query)


def deduplicate_by_intent_metadata(
    results: list[RetrievalResult],
    query: str,
    specs: list[SiteIntentSpec],
) -> list[RetrievalResult]:
    """
    Deduplica entità ripetitive dentro la stessa pagina.

    Le pagine pubblicazioni docente, per esempio, vivono tutte sotto lo stesso
    URL. La deduplica per URL eliminerebbe quasi tutti gli articoli e lascerebbe
    spesso solo l'header pagina; qui usiamo invece chiavi come publication_id.
    """
    query_has_year = query_mentions_explicit_year(query)
    max_per_url = max(spec.max_per_url for spec in specs)
    counts_by_url: dict[str, int] = {}
    seen_metadata_keys: set[str] = set()
    deduped: list[RetrievalResult] = []

    for result in results:
        metadata_key = ""
        for spec in specs:
            for key in spec.dedup_metadata_keys:
                value = result.metadata.get(key)
                if value not in {None, ""}:
                    metadata_key = f"{spec.name}:{key}:{value}"
                    break
            if metadata_key:
                break

        if metadata_key:
            if metadata_key in seen_metadata_keys:
                continue
            seen_metadata_keys.add(metadata_key)

        url = get_result_url(result)
        dedup_key = url if query_has_year else normalize_url_for_dedup(url)
        current_count = counts_by_url.get(dedup_key, 0)

        if current_count >= max_per_url:
            continue

        counts_by_url[dedup_key] = current_count + 1
        deduped.append(result)

    for rank, result in enumerate(deduped, start=1):
        result.rank = rank

    return deduped

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


def combined_result_text(result: RetrievalResult) -> str:
    metadata = result.metadata or {}
    metadata_text = " ".join(
        str(metadata.get(key) or "")
        for key in [
            "title",
            "content_title",
            "section_heading",
            "breadcrumb",
            "breadcrumb_text",
            "source_url",
            "document_url",
            "discovered_from",
            "link_text",
            "chunk_kind",
            "entity_type",
            "entity_name",
            "source_family",
            "teacher_id",
            "course_id",
            "lab_id",
            "year",
            "document_type",
            "document_years",
            "publication_id",
            "publication_title",
            "publication_year",
            "publication_type",
            "publication_venue",
            "publication_authors",
            "publication_doi",
            "publication_iris_url",
        ]
    )
    return f"{metadata_text} {result.text}".lower()


def site_intent_relevance_multiplier(
    result: RetrievalResult,
    query: str,
) -> float:
    specs = active_site_intent_specs(query)
    if not specs:
        return 1.0

    url = get_result_url(result).lower()
    chunk_kind = str(result.metadata.get("chunk_kind") or "").lower()
    metadata_probe = combined_result_text(result)
    multiplier = 1.0

    for spec in specs:
        if chunk_kind and chunk_kind in spec.preferred_chunk_kinds:
            multiplier *= 2.30

        if any(re.search(pattern, url, re.IGNORECASE) for pattern in spec.positive_url_patterns):
            multiplier *= 1.45

        if any(re.search(pattern, url, re.IGNORECASE) for pattern in spec.negative_url_patterns):
            multiplier *= 0.55

        expansion_tokens = set(tokenize(" ".join(spec.expansion_terms)))
        if expansion_tokens:
            overlap = len(expansion_tokens.intersection(set(tokenize(metadata_probe))))
            if overlap >= 3:
                multiplier *= 1.35
            elif overlap == 2:
                multiplier *= 1.20
            elif overlap == 1:
                multiplier *= 1.08

    return multiplier


def query_wants_recent_items(query: str) -> bool:
    return bool(
        re.search(
            r"\b(?:recent[ei]|ultim[ei]|pi[uù]\s+recent[ei]|nuov[ei])\b",
            query.lower(),
        )
    )


def safe_int(value: object) -> int | None:
    try:
        if value in {None, ""}:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def route_relevance_multiplier(
    result: RetrievalResult,
    query: str,
) -> float:
    """
    Boost/penalità per intenti ad alta ambiguità.

    Le regole sono organizzate per famiglia di query: servono a vincolare
    entità e tipo documento quando BM25/dense confondono pagine simili.
    """
    intents = query_intents(query)
    if not intents:
        return 1.0

    query_lower = query.lower()
    url = get_result_url(result).lower()
    url_id = query_param_value(url, "id")
    source = str(result.metadata.get("source") or "").lower()
    haystack = combined_result_text(result)

    multiplier = site_intent_relevance_multiplier(result, query)

    if "teacher_publications" in intents:
        publication_year = safe_int(
            result.metadata.get("publication_year") or result.metadata.get("year")
        )
        chunk_kind = str(result.metadata.get("chunk_kind") or "").lower()

        if "docenti.unisa.it" in url and "/ricerca/pubblicazioni" in url:
            multiplier *= 2.00

        if chunk_kind == "publication_summary":
            multiplier *= 3.20

        if query_wants_recent_items(query) and publication_year:
            if publication_year >= 2026:
                multiplier *= 2.60
            elif publication_year >= 2025:
                multiplier *= 2.25
            elif publication_year >= 2024:
                multiplier *= 1.90
            elif publication_year < 2020:
                multiplier *= 0.45

    if "final_exam" in intents:
        if "didattica/esame-finale" in url:
            multiplier *= 3.20

        if "esame finale" in haystack:
            multiplier *= 2.20

        if "seduta di laurea" in haystack or "sedute di laurea" in haystack:
            multiplier *= 2.10

        if "domanda conseguimento titolo" in haystack or "domanda di laurea" in haystack:
            multiplier *= 1.90

        if "prova finale" in haystack:
            multiplier *= 1.60

        if "immatricolazioni" in url or "modalità di accesso" in haystack or "modalita di accesso" in haystack:
            multiplier *= 0.18

        if "tolc" in haystack or "ofa" in haystack:
            multiplier *= 0.45

    if "lab_equipment" in intents:
        labrob_requested = (
            "labrob" in query_lower
            or "laboratorio di robotica" in query_lower
            or "laboratorio robotica" in query_lower
        )
        labrob_detail_url = (
            url_id == "2"
            and (
                "dipartimento/strutture" in url
                or "/ricerca/laboratori" in url
            )
        )

        if "strumentazione" in haystack or "dotazione" in haystack:
            multiplier *= 1.45

        if "www.diem.unisa.it/dipartimento/strutture?id=" in url:
            multiplier *= 1.45

        if "docenti.unisa.it" in url and "/ricerca/laboratori?id=" in url:
            multiplier *= 0.82

        if "robotica" in query_lower or "labrob" in query_lower:
            if (
                labrob_detail_url
                or "labrob" in haystack
                or "dipartimento | robotica" in haystack
                or "laboratorio di robotica" in haystack
            ):
                multiplier *= 3.50

            if labrob_requested and labrob_detail_url and (
                "strumentazione" in haystack or "dotazione" in haystack
            ):
                multiplier *= 2.20

            if (
                (
                    url_id == "23"
                    and (
                        "dipartimento/strutture" in url
                        or "/ricerca/laboratori" in url
                    )
                )
                or "telecomunicazioni e teoria" in haystack
            ):
                multiplier *= 0.35

            if (
                labrob_requested
                and (
                    "dipartimento/strutture?id=" in url
                    or "/ricerca/laboratori?id=" in url
                )
                and not (
                    labrob_detail_url
                    or "labrob" in haystack
                    or "laboratorio di robotica" in haystack
                )
            ):
                multiplier *= 0.25

    if "course_statistics" in intents:
        years = extract_years_from_text(query_lower)

        if "statistiche" in url or " statistiche" in haystack:
            multiplier *= 2.10

        if "almalaurea" in haystack or "__almalaurea" in url:
            multiplier *= 2.10

        if "valutazione della didattica" in haystack:
            multiplier *= 1.35

        if "digital medicine" in query_lower or "medicina digitale" in query_lower:
            if (
                "information-engineering-for-digital-medicine" in url
                or "information-engineering-for-digital-medicine" in haystack
                or "medicina digitale" in haystack
            ):
                multiplier *= 1.80

        for year in years:
            if (
                f"__almalaurea/{year}" in url
                or f"almalaurea {year}" in haystack
                or f"anni documento: {year}" in haystack
            ):
                multiplier *= 2.40
                break

        if source == "pdf" and ("almalaurea" in haystack or "__almalaurea" in url):
            multiplier *= 1.25

    return multiplier


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
    intents = query_intents(query)

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

    wants_final_exam_info = query_wants_final_exam_info(query)
    
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
            "corsi del dipartimento",
            "corsi del diem",
            "corsi offerti",
            "corsi disponibili",
        ]
    ) and not wants_final_exam_info

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
    ) and not wants_final_exam_info

    wants_master_degree = any(
        keyword in query_lower
        for keyword in [
            "magistrale",
            "laurea magistrale",
        ]
    )
    
    wants_phd_info = any(
        keyword in query_lower
        for keyword in [
            "dottorato",
            "dottorati",
            "dottorato di ricerca",
            "dottorati di ricerca",
            "phd",
            "ph.d",
            "ph.d.",
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
    
    erasmus_mobility = requested_erasmus_mobility(query)
    
    wants_aggregate_info = is_aggregate_query(query)

    query_mentions_year = query_mentions_explicit_year(query)

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
        adjusted_score *= route_relevance_multiplier(result, query)

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

        # Se l'utente non chiede un anno specifico, preferiamo URL canonici
        # rispetto a versioni parametrizzate o storiche della stessa pagina.
        if not query_mentions_year and "teacher_publications" not in intents:
            parsed_url = urlsplit(str(url))

            if parsed_url.query:
                adjusted_score *= 0.85

            if re.search(r"(?:^|[?&])(?:anno|year|aa)=(?:19|20)\d{2}", str(url).lower()):
                adjusted_score *= 0.75

        # Query specifica su corsi di laurea/offerta formativa
        if wants_degree_info:
            if "offerta-formativa" in url:
                adjusted_score *= 1.30

            if "presentazione" in url and "dipartimento" in url:
                adjusted_score *= 1.10

            if "didattica" in title or "didattica" in breadcrumb:
                adjusted_score *= 1.05

        # Query su esame finale / sedute di laurea:
        # "accesso alla seduta" non è accesso al corso.
        if wants_final_exam_info:
            text_lower = result.text.lower()
            url_lower = str(url).lower()

            if "didattica/esame-finale" in url_lower:
                adjusted_score *= 2.60

            if "esame finale" in title or "esame finale" in breadcrumb:
                adjusted_score *= 2.10

            if "seduta di laurea" in text_lower or "sedute di laurea" in text_lower:
                adjusted_score *= 1.90

            if "domanda conseguimento titolo" in text_lower or "domanda di laurea" in text_lower:
                adjusted_score *= 1.80

            if "prova finale" in text_lower:
                adjusted_score *= 1.55

            if "immatricolazioni" in url_lower:
                adjusted_score *= 0.22

            if "modalità di accesso" in title or "modalità di accesso" in breadcrumb:
                adjusted_score *= 0.25

            if "modalita di accesso" in title or "modalita di accesso" in breadcrumb:
                adjusted_score *= 0.25

            if "tolc" in text_lower or "ofa" in text_lower:
                adjusted_score *= 0.45

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
            
            if erasmus_mobility == "studio":
                if "accordi-erasmus-plus/studio" in str(url).lower():
                    adjusted_score *= 2.60

                if "mobilità per studio" in title or "mobilità per studio" in breadcrumb:
                    adjusted_score *= 2.00

                if "traineeship" in str(url).lower() or "docenza" in str(url).lower() or "teaching" in str(url).lower():
                    adjusted_score *= 0.45

            elif erasmus_mobility == "traineeship":
                if "accordi-erasmus-plus/traineeship" in str(url).lower():
                    adjusted_score *= 2.60

                if "mobilità per traineeship" in title or "mobilità per traineeship" in breadcrumb:
                    adjusted_score *= 2.00

                if "studio" in str(url).lower() or "docenza" in str(url).lower() or "teaching" in str(url).lower():
                    adjusted_score *= 0.45

            elif erasmus_mobility == "teaching":
                if "accordi-erasmus-plus/teaching" in str(url).lower():
                    adjusted_score *= 2.60

                if "mobilità per docenza" in title or "mobilità per docenza" in breadcrumb:
                    adjusted_score *= 2.00

                if "studio" in str(url).lower() or "traineeship" in str(url).lower():
                    adjusted_score *= 0.45
        
        # Query didattica generica, ma non specifica sui corsi di laurea
        elif wants_teaching_info and not wants_degree_info:
            if "didattica" in title or "didattica" in breadcrumb:
                adjusted_score *= 1.10

            if "offerta-formativa" in url:
                adjusted_score *= 1.10
                
        if wants_phd_info:
            text_lower = result.text.lower()
            url_lower = str(url).lower()

            if "diem.unisa.it" in url_lower and (
                "department" in url_lower or "dipartimento" in url_lower
            ):
                adjusted_score *= 1.60

            if "commissions-and-delegates" in url_lower:
                adjusted_score *= 1.60

            if "doctoral" in text_lower or "dottorato" in text_lower or "dottorati" in text_lower:
                adjusted_score *= 1.50

            if (
                "phd program in information engineering" in text_lower
                or "dottorato di ricerca in ingegneria dell" in text_lower
                or "dottorato in ingegneria dell" in text_lower
            ):
                adjusted_score *= 1.80

            if (
                "nationally significant phd program in photovoltaics" in text_lower
                or "dottorato di interesse nazionale in photovoltaics" in text_lower
                or "dottorato di ricerca in photovoltaics" in text_lower
            ):
                adjusted_score *= 1.80

            if "docenti.unisa.it" in url_lower and "curriculum" in url_lower:
                adjusted_score *= 0.65

        result.score = adjusted_score

    results.sort(key=lambda result: result.score, reverse=True)

    for rank, result in enumerate(results, start=1):
        result.rank = rank

    return results


def rerank_with_source_freshness(
    results: list[RetrievalResult],
    query: str,
) -> list[RetrievalResult]:
    """
    Riduce il rumore dei PDF storici quando la domanda non chiede documenti,
    bandi, regolamenti o anni specifici.
    """
    if (
        query_mentions_explicit_year(query)
        or query_wants_pdf_evidence(query)
        or "course_statistics" in query_intents(query)
        or "teacher_publications" in query_intents(query)
    ):
        return results

    for result in results:
        url = get_result_url(result).lower()
        source = str(result.metadata.get("source") or "").lower()
        text_probe = f"{url} {result.metadata.get('title') or ''}"
        years = extract_years_from_text(text_probe)

        is_pdf = source == "pdf" or url.endswith(".pdf") or "/uploads/" in url

        # Il corpus contiene molti PDF storici. Sono utili per bandi/regolamenti,
        # ma rumorosi per domande correnti su corsi, docenti o servizi.
        if is_pdf:
            result.score *= 0.45

        # Le pagine ?anno=YYYY sono mantenute per domande storiche; per default
        # preferiamo la pagina canonica corrente.
        if re.search(r"(?:^|[?&])(?:anno|year|aa)=(?:19|20)\d{2}", url):
            result.score *= 0.35

        if years:
            newest_year = max(years)
            if newest_year < 2020:
                result.score *= 0.30
            elif newest_year < 2024:
                result.score *= 0.65

        if source == "course_catalogue":
            result.score *= 1.12

        if source == "html" and "www.diem.unisa.it" in url:
            result.score *= 1.08

    results.sort(key=lambda result: result.score, reverse=True)

    for rank, result in enumerate(results, start=1):
        result.rank = rank

    return results


def hybrid_retrieve(
    query: str,
    bm25_k: int = DEFAULT_BM25_K,
    dense_k: int = DEFAULT_DENSE_K,
    final_k: int = 5,
    rerank_k: int = DEFAULT_RERANK_K,
) -> list[RetrievalResult]:
    """
    Retrieval ibrido:
    - BM25 per parole chiave;
    - Chroma per similarità semantica;
    - RRF per fusione dei ranking.
    """
    retrieval_query = expand_query_for_retrieval(query)

    # Candidate generation larga: query originale per precisione sui nomi/codici,
    # query espansa per richiamo sui domini DIEM più frequenti.
    result_lists = {
        "bm25_original": bm25_retrieve(query, k=bm25_k),
        "dense_original": dense_retrieve_wrapped(query, k=dense_k),
    }

    if retrieval_query != query:
        result_lists["bm25_expanded"] = bm25_retrieve(retrieval_query, k=bm25_k)
        result_lists["dense_expanded"] = dense_retrieve_wrapped(retrieval_query, k=dense_k)

    hybrid_results = reciprocal_rank_fusion(result_lists)

    # Prima applichiamo segnali deterministici e deduplica; poi il reranker
    # neurale lavora su un pool più pulito e molto più piccolo.
    hybrid_results = rerank_with_metadata_signals(hybrid_results, query)
    hybrid_results = rerank_with_source_freshness(hybrid_results, query)

    hybrid_results = deduplicate_for_query(
        results=hybrid_results,
        query=query,
    )

    hybrid_results = neural_rerank(
        query=query,
        results=hybrid_results,
        top_k=rerank_k,
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
    parser.add_argument(
        "--rerank-k",
        type=int,
        default=DEFAULT_RERANK_K,
        help="Numero di candidati da passare al reranker neurale.",
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
