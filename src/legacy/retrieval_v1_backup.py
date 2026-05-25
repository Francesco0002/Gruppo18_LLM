from __future__ import annotations

import argparse
import os
import re
import warnings
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

_DENSE_RETRIEVAL_WARNING_EMITTED = False

HIGH_CONFIDENCE_ROUTE_INTENTS = {
    "course_syllabus",
    "study_plan",
    "teacher_office_hours",
    "teacher_publications",
    "lab_equipment",
    "course_statistics",
    "final_exam",
    "admission",
    "official_document",
}
ROUTE_MULTIPLIER_MIN = 0.05
ROUTE_MULTIPLIER_MAX = 8.0
HIGH_CONFIDENCE_ROUTE_MULTIPLIER_MIN = 0.03
HIGH_CONFIDENCE_ROUTE_MULTIPLIER_MAX = 18.0
TOTAL_SCORE_FACTOR_MIN = 0.04
TOTAL_SCORE_FACTOR_MAX = 20.0

COURSE_TITLE_GENERIC_TOKENS = {
    "corso",
    "insegnamento",
    "materia",
    "magistrale",
    "triennale",
    "ingegneria",
    "informatica",
    "argomento",
    "argomenti",
    "programma",
    "contenuto",
    "contenuti",
    "obiettivo",
    "obiettivi",
    "formativi",
    "tratta",
    "scheda",
}


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


GENERIC_TEACHER_WORDS = {
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

PEOPLE_INFO_KEYWORDS = (
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
)

LOCATION_INFO_KEYWORDS = (
    "dove si trova",
    "dove è",
    "dove sta",
    "indirizzo",
    "sede",
    "ubicazione",
    "contatti",
    "raggiungere",
)

BACHELOR_DEGREE_KEYWORDS = (
    "triennale",
    "triennali",
    "laurea triennale",
    "lauree triennali",
    "corsi triennali",
    "corsi triennale",
    "l-8",
    "classe l-8",
    "ie127l-8",
    "ie128l-8",
)

MASTER_DEGREE_KEYWORDS = (
    "magistrale",
    "magistrali",
    "laurea magistrale",
    "lauree magistrali",
    "lm-32",
    "lm-28",
    "ie227lm-32",
    "ie232lm-32",
    "ie233lm-28",
)

DEGREE_OVERVIEW_KEYWORDS = (
    "corsi di laurea",
    "corso di laurea",
    "lauree",
    "laurea",
    "offerta formativa",
    "corsi di studio",
    "programmi di studio",
    "corsi del dipartimento",
    "corsi del diem",
    "corsi offerti",
    "corsi disponibili",
)

TEACHING_INFO_KEYWORDS = (
    "corso",
    "corsi",
    "didattica",
    "insegnamento",
    "insegnamenti",
    "lezioni",
    "esami",
    "programma",
    "programmi",
)

SYLLABUS_DETAIL_KEYWORDS = (
    "argomento",
    "argomenti",
    "tratta",
    "programma",
    "contenuto",
    "contenuti",
    "obiettivo",
    "obiettivi",
    "obiettivi formativi",
    "syllabus",
    "testi",
    "modalità esame",
    "modalita esame",
    "come si svolge l'esame",
)

STUDY_PLAN_KEYWORDS = (
    "piano di studi",
    "piano degli studi",
    "manifesto degli studi",
    "primo anno",
    "secondo anno",
    "terzo anno",
    "1° anno",
    "2° anno",
    "3° anno",
    "1 anno",
    "2 anno",
    "3 anno",
)

ADMISSION_KEYWORDS = (
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
)

PHD_KEYWORDS = (
    "dottorato",
    "dottorati",
    "dottorato di ricerca",
    "dottorati di ricerca",
    "phd",
    "ph.d",
    "ph.d.",
)

ERASMUS_BANDO_KEYWORDS = (
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
)

ERASMUS_AGREEMENT_KEYWORDS = (
    "accordi erasmus",
    "accordo erasmus",
    "accordi",
    "università partner",
    "universita partner",
    "partner",
    "traineeship",
    "studio",
    "docenza",
)

STRUCTURES_KEYWORDS = (
    "laboratori",
    "laboratorio",
    "strutture",
    "aule",
    "biblioteche",
    "centri",
)

GOVERNANCE_KEYWORDS = (
    "organi collegiali",
    "consiglio di dipartimento",
    "giunta",
    "commissioni",
    "delegati",
    "direttore",
    "vicedirettore",
    "referenti",
)

RESEARCH_KEYWORDS = (
    "ricerca",
    "progetti di ricerca",
    "progetti",
    "laboratori di ricerca",
    "pubblicazioni",
    "terza missione",
)

INTERNATIONAL_KEYWORDS = (
    "erasmus",
    "internazionale",
    "international",
    "mobilità",
    "mobilita",
    "accordi",
    "estero",
)


@dataclass(frozen=True)
class QuerySignals:
    lower: str
    tokens: set[str]
    intents: set[str]
    active_specs: tuple[SiteIntentSpec, ...]
    wants_people_info: bool
    wants_location_info: bool
    wants_bachelor_degree: bool
    wants_master_degree: bool
    wants_degree_overview: bool
    wants_specific_teaching: bool
    wants_study_plan_courses: bool
    wants_teaching_info: bool
    wants_admission_info: bool
    wants_phd_info: bool
    wants_erasmus_bando_info: bool
    wants_erasmus_agreements: bool
    wants_structures_info: bool
    wants_international_info: bool
    wants_final_exam_info: bool
    wants_aggregate_info: bool
    query_mentions_year: bool
    italian_query: bool
    teacher_query: bool
    teacher_name_tokens: set[str]
    asks_office_hours: bool
    erasmus_mobility: str | None


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
        name="people_directory",
        query_patterns=(
            r"\bdocenti\s+(?:del|di)\s+diem\b",
            r"\bpersonale\s+(?:docente\s+)?(?:del|di)\s+dipartimento\b",
            r"\belenco\s+(?:dei\s+)?(?:docenti|professori|ricercatori|personale)\b",
            r"\brubrica\b",
            r"\bprofessori\s+(?:ordinari|associati)\b",
            r"\bricercatori\s+diem\b",
        ),
        expansion_terms=(
            "docenti e personale",
            "personale docente",
            "rubrica",
            "professori",
            "ricercatori",
            "dipartimento personale",
        ),
        positive_url_patterns=(r"/dipartimento/personale", r"rubrica\.unisa\.it", r"docenti\.unisa\.it"),
        negative_url_patterns=(r"/ricerca/pubblicazioni", r"/ricerca/progetti"),
        max_per_url=4,
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
        name="department_governance",
        query_patterns=(
            r"\borgani\s+collegiali\b",
            r"\bconsiglio\s+di\s+dipartimento\b",
            r"\bcommissioni\b",
            r"\bdelegati\b",
            r"\bdirettore\b",
            r"\bvicedirettore\b",
            r"\breferenti\b",
        ),
        expansion_terms=(
            "organi collegiali",
            "commissioni e delegati",
            "consiglio di dipartimento",
            "direttore",
            "dipartimento",
        ),
        positive_url_patterns=(r"/dipartimento/organi-collegiali", r"/dipartimento/commissioni", r"commissions-and-delegates"),
        max_per_url=4,
    ),
    SiteIntentSpec(
        name="department_research",
        query_patterns=(
            r"\bricerca\s+diem\b",
            r"\bprogetti\s+di\s+ricerca\b",
            r"\blaboratori\s+di\s+ricerca\b",
            r"\bterza\s+missione\b",
        ),
        expansion_terms=(
            "ricerca",
            "progetti di ricerca",
            "laboratori",
            "terza missione",
            "dipartimento",
        ),
        preferred_chunk_kinds=("lab_equipment", "teacher_projects", "text"),
        positive_url_patterns=(r"/ricerca", r"/terza-missione", r"/dipartimento/strutture"),
        max_per_url=4,
    ),
    SiteIntentSpec(
        name="lab_equipment",
        query_patterns=(r"\bstrumentazione\b", r"\battrezzature\b", r"\bdotazione\b", r"\blaborator[io]\b", r"\blabrob\b", r"\bmivia\b"),
        expansion_terms=("strumentazione", "dotazione", "attrezzature", "laboratori", "strutture"),
        preferred_chunk_kinds=("lab_equipment",),
        positive_url_patterns=(r"/dipartimento/strutture", r"/ricerca/laboratori"),
        max_per_url=5,
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
        name="course_syllabus",
        query_patterns=(
            r"(?=.*\b(?:corso|insegnamento|materia)\b)(?=.*\b(?:argomenti?|programma|contenuti?|obiettivi?(?:\s+formativi)?|tratta|syllabus|testi|modalit[aà]\s+esame)\b)",
            r"\bscheda\s+(?:insegnamento|corso)\b",
        ),
        expansion_terms=(
            "scheda insegnamento",
            "coursecatalogue",
            "insegnamenti",
        ),
        preferred_chunk_kinds=("course_syllabus", "text"),
        positive_url_patterns=(r"coursecatalogue", r"/didattica/insegnamenti"),
        negative_url_patterns=(r"offerta-formativa", r"piano-di-studi"),
        max_per_url=6,
    ),
    SiteIntentSpec(
        name="study_plan",
        query_patterns=(
            r"\bpiano\s+(?:di|degli)\s+studi\b",
            r"\bmanifesto\s+degli\s+studi\b",
            r"(?=.*\b(?:corsi?|insegnamenti?)\b)(?=.*\b(?:primo|secondo|terzo|1\s*[°o]|2\s*[°o]|3\s*[°o])\s+anno\b)",
        ),
        expansion_terms=(
            "piano di studi",
            "piano degli studi",
            "manifesto degli studi",
            "insegnamenti",
            "CFU",
            "1° anno",
            "2° anno",
            "3° anno",
        ),
        preferred_chunk_kinds=("study_plan", "course_info", "official_document"),
        positive_url_patterns=(r"piano-di-studi", r"__piano-studi-cds"),
        max_per_url=20,
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
        query_patterns=(
            r"\baccesso\b",
            r"\bammissione\b",
            r"\bimmatricolazion[ei]\b",
            r"\biscriversi\b",
            r"\biscrizione\b",
            r"\biscrizioni\b",
            r"\btolc\b",
            r"\bofa\b",
            r"\brequisiti\b",
        ),
        expansion_terms=("modalità di accesso", "requisiti di accesso", "immatricolazioni", "OFA", "TOLC"),
        preferred_chunk_kinds=("course_info", "official_document_summary"),
        positive_url_patterns=(r"immatricolazioni", r"modalit", r"requisiti", r"ofa", r"tolc"),
        negative_query_patterns=(r"\bsedut[ae]\s+di\s+laurea\b", r"\bprova\s+finale\b", r"\besame\s+finale\b", r"\bdomanda\s+di\s+laurea\b", r"\bconseguimento\s+(?:del\s+)?titolo\b"),
        max_per_url=3,
    ),
    SiteIntentSpec(
        name="course_regulations",
        query_patterns=(
            r"\bregolament[oi]\b.*\b(?:corso|corsi|laurea|ingegneria|informatica|magistrale|triennale)\b",
            r"\b(?:corso|corsi|laurea|ingegneria|informatica|magistrale|triennale)\b.*\bregolament[oi]\b",
        ),
        expansion_terms=(
            "regolamento corso di studio",
            "regolamenti cds",
            "__regolamenti-cds",
            "didattica regolamenti",
            "corsi regolamenti",
        ),
        preferred_chunk_kinds=("official_document", "official_document_summary", "course_info"),
        positive_url_patterns=(r"__regolamenti-cds", r"/didattica/regolamenti", r"regolament"),
        negative_url_patterns=(r"/home/bandi", r"bando", r"graduatoria"),
        max_per_url=4,
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
        max_per_url=3,
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
    specs = [spec for spec in SITE_INTENT_SPECS if spec.matches(query)]
    names = {spec.name for spec in specs}

    # Le query su una scheda insegnamento o sul piano di studi contengono spesso
    # parole come "corso", "laurea" o "magistrale". Senza questa priorità
    # attivano anche l'intento panoramico "course_info", che porta il retriever
    # verso pagine generiche di offerta formativa.
    if {"course_syllabus", "study_plan"}.intersection(names):
        specs = [spec for spec in specs if spec.name != "course_info"]

    return specs


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


def contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def route_multiplier_bounds(intents: set[str]) -> tuple[float, float]:
    if intents.intersection(HIGH_CONFIDENCE_ROUTE_INTENTS):
        return HIGH_CONFIDENCE_ROUTE_MULTIPLIER_MIN, HIGH_CONFIDENCE_ROUTE_MULTIPLIER_MAX

    return ROUTE_MULTIPLIER_MIN, ROUTE_MULTIPLIER_MAX


def bounded_route_multiplier(multiplier: float, intents: set[str]) -> float:
    minimum, maximum = route_multiplier_bounds(intents)
    return clamp(multiplier, minimum, maximum)


def bounded_total_score(
    base_score: float,
    adjusted_score: float,
    intents: set[str],
) -> tuple[float, float]:
    if base_score <= 0:
        return adjusted_score, 1.0

    _, route_max = route_multiplier_bounds(intents)
    total_max = max(TOTAL_SCORE_FACTOR_MAX, route_max)
    factor = clamp(
        adjusted_score / base_score,
        TOTAL_SCORE_FACTOR_MIN,
        total_max,
    )
    return base_score * factor, factor


def requested_course_title_tokens(signals: QuerySignals) -> set[str]:
    return signals.tokens - COURSE_TITLE_GENERIC_TOKENS


def query_wants_specific_teaching(query_lower: str) -> bool:
    """
    Distingue una scheda di insegnamento da una domanda sui corsi di laurea.

    "corso di Machine Learning che argomenti tratta" non deve attivare la rotta
    generica "offerta formativa", anche se contiene parole come corso o
    magistrale.
    """
    if not contains_any(query_lower, SYLLABUS_DETAIL_KEYWORDS):
        return False

    if re.search(r"\bcorsi?\s+di\s+laurea\b", query_lower):
        return False

    return bool(
        re.search(r"\b(?:corso|insegnamento|materia)\b", query_lower)
        or re.search(r"\bscheda\s+(?:insegnamento|corso)\b", query_lower)
    )


def query_wants_study_plan_courses(query_lower: str) -> bool:
    if contains_any(query_lower, ("piano di studi", "piano degli studi", "manifesto degli studi")):
        return True

    if not contains_any(query_lower, ("corso", "corsi", "insegnamento", "insegnamenti")):
        return False

    year_words = sum(
        1
        for keyword in ("primo", "secondo", "terzo", "1°", "2°", "3°", "1 anno", "2 anno", "3 anno")
        if keyword in query_lower
    )
    return "anno" in query_lower and year_words >= 1


@lru_cache(maxsize=512)
def query_signals(query: str) -> QuerySignals:
    query_lower = query.lower()
    query_tokens = set(tokenize(query_lower))
    intents = query_intents(query)
    active_specs = tuple(active_site_intent_specs(query))

    wants_final_exam_info = query_wants_final_exam_info(query)
    wants_specific_teaching = query_wants_specific_teaching(query_lower)
    wants_study_plan_courses = query_wants_study_plan_courses(query_lower)
    wants_admission_info = contains_any(query_lower, ADMISSION_KEYWORDS) and not wants_final_exam_info
    wants_bachelor_degree = contains_any(query_lower, BACHELOR_DEGREE_KEYWORDS)
    wants_master_degree = contains_any(query_lower, MASTER_DEGREE_KEYWORDS)
    wants_degree_overview = (
        contains_any(query_lower, DEGREE_OVERVIEW_KEYWORDS)
        or wants_bachelor_degree
        or wants_master_degree
    ) and not (
        wants_final_exam_info
        or wants_specific_teaching
        or wants_study_plan_courses
        or wants_admission_info
    )

    teacher_query = contains_any(
        query_lower,
        (
            "professor",
            "professore",
            "professoressa",
            "prof.",
            "prof",
            "docente",
            "ricevimento",
        ),
    )

    italian_query = not contains_any(
        query_lower,
        ("what", "which", "where", "who", "degree", "course", "teaching"),
    )

    return QuerySignals(
        lower=query_lower,
        tokens=query_tokens,
        intents=intents,
        active_specs=active_specs,
        wants_people_info=is_aggregate_query(query) and contains_any(query_lower, PEOPLE_INFO_KEYWORDS),
        wants_location_info=contains_any(query_lower, LOCATION_INFO_KEYWORDS),
        wants_bachelor_degree=wants_bachelor_degree,
        wants_master_degree=wants_master_degree,
        wants_degree_overview=wants_degree_overview,
        wants_specific_teaching=wants_specific_teaching,
        wants_study_plan_courses=wants_study_plan_courses,
        wants_teaching_info=contains_any(query_lower, TEACHING_INFO_KEYWORDS),
        wants_admission_info=wants_admission_info,
        wants_phd_info=contains_any(query_lower, PHD_KEYWORDS),
        wants_erasmus_bando_info=contains_any(query_lower, ERASMUS_BANDO_KEYWORDS) and "erasmus" in query_lower,
        wants_erasmus_agreements=contains_any(query_lower, ERASMUS_AGREEMENT_KEYWORDS) and "erasmus" in query_lower,
        wants_structures_info=contains_any(query_lower, STRUCTURES_KEYWORDS),
        wants_international_info=contains_any(query_lower, INTERNATIONAL_KEYWORDS),
        wants_final_exam_info=wants_final_exam_info,
        wants_aggregate_info=is_aggregate_query(query),
        query_mentions_year=query_mentions_explicit_year(query),
        italian_query=italian_query,
        teacher_query=teacher_query,
        teacher_name_tokens=query_tokens - GENERIC_TEACHER_WORDS,
        asks_office_hours=contains_any(
            query_lower,
            ("ricevimento", "orario di ricevimento", "orari di ricevimento"),
        ),
        erasmus_mobility=requested_erasmus_mobility(query),
    )


def extract_years_from_text(text: str) -> list[int]:
    return [
        int(match)
        for match in re.findall(r"\b(?:19|20)\d{2}\b", text)
    ]


def metadata_year_values(metadata: dict[str, Any]) -> list[int]:
    values: list[int] = []

    for key in ("year", "document_years"):
        value = metadata.get(key)
        if isinstance(value, int):
            values.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, int):
                    values.append(item)
                else:
                    values.extend(extract_years_from_text(str(item)))
        else:
            values.extend(extract_years_from_text(str(value or "")))

    return values


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

    I segnali sono calcolati una sola volta in ``query_signals``: questo evita
    che espansione, deduplica e reranking interpretino la stessa query in modi
    diversi.
    """
    signals = query_signals(query)
    expanded_terms: list[str] = []

    for spec in signals.active_specs:
        expanded_terms.extend(spec.expansion_terms)

    if signals.wants_people_info:
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

    if signals.wants_location_info:
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

    if signals.wants_specific_teaching:
        expanded_terms.extend(["coursecatalogue", "scheda insegnamento", "insegnamenti"])

        if contains_any(signals.lower, ("modalità esame", "modalita esame", "esame", "verifica")):
            expanded_terms.extend(["verifica dell'apprendimento", "modalità esame"])
        elif contains_any(signals.lower, ("testi", "libri", "manuale", "bibliografia")):
            expanded_terms.extend(["testi", "materiale didattico", "bibliografia"])
        elif contains_any(signals.lower, ("metodi didattici", "lezioni", "laboratorio", "esercitazioni")):
            expanded_terms.extend(["metodi didattici", "lezioni", "esercitazioni", "laboratorio"])
        else:
            expanded_terms.extend(["contenuti", "obiettivi formativi", "programma", "argomenti"])

    if signals.wants_study_plan_courses:
        expanded_terms.extend(
            [
                "piano di studi",
                "piano degli studi",
                "manifesto degli studi",
                "__piano-studi-cds",
                "insegnamenti",
                "CFU",
                "1° anno",
                "2° anno",
                "3° anno",
            ]
        )

        if signals.wants_bachelor_degree and not signals.wants_master_degree:
            expanded_terms.extend(
                [
                    "IE127",
                    "IE127L-8",
                    "L-8",
                    "classe L-8",
                    "CORSO DI LAUREA IN INGEGNERIA INFORMATICA",
                ]
            )

        elif signals.wants_master_degree and not signals.wants_bachelor_degree:
            expanded_terms.extend(
                [
                    "IE227LM-32",
                    "LM-32",
                    "CORSO DI LAUREA MAGISTRALE IN INGEGNERIA INFORMATICA",
                ]
            )

    if signals.wants_degree_overview:
        expanded_terms.extend(
            [
                "DIEM",
                "Didattica",
                "Offerta Formativa",
                "corsi di studio",
            ]
        )

        if signals.wants_bachelor_degree and not signals.wants_master_degree:
            expanded_terms.extend(
                [
                    "CORSO DI LAUREA",
                    "L-8",
                    "classe L-8",
                    "IE127L-8",
                    "IE128L-8",
                    "Ingegneria Informatica",
                    "Ingegneria dell'Informazione per la Medicina Digitale",
                ]
            )

        elif signals.wants_master_degree and not signals.wants_bachelor_degree:
            expanded_terms.extend(
                [
                    "CORSO DI LAUREA MAGISTRALE",
                    "LM-32",
                    "LM-28",
                    "IE227LM-32",
                    "IE232LM-32",
                    "IE233LM-28",
                    "Ingegneria Informatica magistrale",
                    "Information Engineering for Digital Medicine",
                    "Electrical Engineering for Digital Energy",
                ]
            )

        else:
            expanded_terms.extend(
                [
                    "corsi di laurea",
                    "corsi di laurea magistrale",
                    "CORSO DI LAUREA",
                    "CORSO DI LAUREA MAGISTRALE",
                    "L-8",
                    "LM-32",
                    "LM-28",
                ]
            )

    if signals.wants_structures_info:
        expanded_terms.extend(
            [
                "strutture",
                "laboratori",
                "aule",
                "centri",
                "dipartimento",
            ]
        )

    if "lab_equipment" in signals.intents:
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

        if "robotica" in signals.lower or "labrob" in signals.lower:
            expanded_terms.extend(
                [
                    "Laboratorio di Robotica",
                    "LabROB",
                    "Robotica",
                    "strutture id 2",
                ]
            )

    if "course_statistics" in signals.intents:
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

    if "final_exam" in signals.intents:
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

    if signals.wants_international_info:
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

    if signals.wants_admission_info:
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

    if signals.wants_erasmus_bando_info:
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

    if signals.wants_erasmus_agreements:
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

    if signals.wants_phd_info:
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

    unique_terms = list(dict.fromkeys(expanded_terms))

    if not unique_terms:
        return query

    return query + " " + " ".join(unique_terms)


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


def intent_anchor_patterns(query: str) -> list[tuple[str, int]]:
    signals = query_signals(query)
    intents = signals.intents
    patterns: list[tuple[str, int]] = []

    if signals.teacher_query and signals.teacher_name_tokens and not signals.wants_aggregate_info:
        for person_id in person_ids_for_query(query)[:1]:
            patterns.extend(
                [
                    (rf"docenti\.unisa\.it/{person_id}/home$", 3),
                    (rf"rubrica\.unisa\.it/persone\?matricola={person_id}", 2),
                ]
            )

    if "contacts" in intents and signals.wants_location_info and contains_any(signals.lower, ("diem", "dipartimento")):
        patterns.append((r"/home/contatti$", 3))

    if signals.wants_degree_overview:
        patterns.append((r"/didattica/offerta-formativa$", 3))

    if "study_plan" in intents:
        if "ingegneria informatica" in signals.lower and signals.wants_bachelor_degree:
            patterns.append((r"__piano-studi-cds/\d{4}/(?:ie127|06127)\.pdf$", 9))
        elif "ingegneria informatica" in signals.lower and signals.wants_master_degree:
            patterns.append((r"__piano-studi-cds/\d{4}/(?:ie227|06227)\.pdf$", 9))

        patterns.extend(
            [
                (r"__piano-studi-cds", 9),
                (r"/didattica/piano-di-studi(?:$|\?)", 4),
            ]
        )

    if "people_directory" in intents:
        patterns.append((r"/dipartimento/personale$", 4))

    if "department_governance" in intents:
        patterns.extend(
            [
                (r"/dipartimento/organi-collegiali", 4),
                (r"/dipartimento/commissioni", 4),
            ]
        )

    if "final_exam" in intents:
        patterns.append((r"/didattica/esame-finale", 3))

    if "course_regulations" in intents:
        patterns.append((r"__regolamenti-cds", 6))

    if "official_document" in intents and contains_any(signals.lower, ("bando", "bandi", "graduatoria")):
        patterns.append((r"/home/bandi", 6))

    if "admission" in intents:
        if "ingegneria informatica" in signals.lower and signals.wants_master_degree:
            patterns.append((r"/ingegneria-informatica-magistrale/immatricolazioni$", 6))
        elif "ingegneria informatica" in signals.lower and signals.wants_bachelor_degree:
            patterns.append((r"/ingegneria-informatica/immatricolazioni$", 6))

        patterns.append((r"/immatricolazioni(?:$|\?)", 4))

    if signals.wants_erasmus_bando_info:
        patterns.append((r"/international/international-mobility", 4))

    if signals.wants_erasmus_agreements:
        mobility = signals.erasmus_mobility
        if mobility == "studio":
            patterns.append((r"/international/accordi-erasmus-plus/studio", 4))
        elif mobility == "traineeship":
            patterns.append((r"/international/accordi-erasmus-plus/traineeship", 4))
        elif mobility == "teaching":
            patterns.append((r"/international/accordi-erasmus-plus/teaching", 4))
        else:
            patterns.append((r"/international/accordi-erasmus-plus", 4))

    if "lab_equipment" in intents and ("robotica" in signals.lower or "labrob" in signals.lower):
        patterns.append((r"/dipartimento/strutture\?id=2$", 6))

    if "phd" in intents:
        if contains_any(signals.lower, ("studenti", "studente", "dottorandi", "dottorando", "iscritti")):
            for year in extract_years_from_text(signals.lower):
                patterns.append((rf"/(?:ingegneria-dell-informazione|DOT18CK8F9)/studenti\?anno={year}$", 5))

            patterns.extend(
                [
                    (r"/(?:ingegneria-dell-informazione|DOT18CK8F9)/studenti(?:$|\?)", 5),
                    (r"/photovoltaics/studenti(?:$|\?)", 3),
                ]
            )

        patterns.extend(
            [
                (r"/dipartimento/presentazione$", 2),
                (r"/didattica/offerta-formativa$", 2),
            ]
        )

    return patterns


def anchor_result_priority(
    *,
    metadata: dict[str, Any],
    text: str,
    query: str,
) -> tuple[int, int]:
    signals = query_signals(query)
    intents = signals.intents
    url = str(
        metadata.get("source_url")
        or metadata.get("document_url")
        or metadata.get("url")
        or ""
    ).lower()
    section = str(metadata.get("section_heading") or "").lower()
    title = str(metadata.get("title") or metadata.get("content_title") or "").lower()
    breadcrumb = str(metadata.get("breadcrumb_text") or metadata.get("breadcrumb") or "").lower()
    chunk_kind = str(metadata.get("chunk_kind") or "").lower()
    lab_id = str(metadata.get("lab_id") or "")
    chunk_index = int(metadata.get("chunk_index") or 0)
    haystack = f"{title} {breadcrumb} {section} {url} {text[:2200]}".lower()

    priority = 0

    if signals.teacher_query and signals.teacher_name_tokens:
        person_ids = person_ids_for_query(query)[:1]
        if any(f"docenti.unisa.it/{person_id}" in url for person_id in person_ids):
            priority += 140
        if any(f"matricola={person_id}" in url for person_id in person_ids):
            priority += 90
        if chunk_kind == "office_hours":
            priority += 120
        if contains_any(haystack, ("orario di ricevimento", "ricevimento")):
            priority += 80

    if "lab_equipment" in intents:
        if chunk_kind == "lab_equipment":
            priority += 120
        if "strumentazione" in section or "strumentazione" in haystack:
            priority += 90
        if "dotazione" in haystack or "attrezzature" in haystack:
            priority += 40
        if ("robotica" in signals.lower or "labrob" in signals.lower) and (
            lab_id == "2" or "strutture?id=2" in url or "labrob" in haystack
        ):
            priority += 90
        if contains_any(haystack, ("ur10", "franka", "comau", "husky", "ros2")):
            priority += 25

    if "contacts" in intents:
        if "contatti generali" in section or "contatti generali" in haystack:
            priority += 120
        if contains_any(haystack, ("indirizzo", "sede", "fisciano", "edificio")):
            priority += 45

    if "admission" in intents:
        if "immatricolazioni" in url:
            priority += 70
        if contains_any(haystack, ("per immatricolarsi", "requisiti", "prova di ammissione")):
            priority += 85
        if "ingegneria informatica" in signals.lower and "ingegneria-informatica-magistrale" in url:
            priority += 70
        if signals.wants_master_degree and "laurea magistrale" in haystack:
            priority += 35

    if "study_plan" in intents:
        is_canonical_plan = is_canonical_study_plan_metadata(metadata, text)
        if is_canonical_plan:
            priority += 150
        elif chunk_kind == "study_plan":
            priority -= 120

        if "__piano-studi-cds" in url:
            priority += 100
        elif "piano-di-studi" in url:
            priority += 50

        if signals.wants_bachelor_degree and not signals.wants_master_degree:
            if contains_any(haystack, ("ie127", "06127", "l-8", "corso di laurea in ingegneria informatica")):
                priority += 80
            if contains_any(haystack, ("lm-32", "lm-28", "magistrale")):
                priority -= 90
        elif signals.wants_master_degree and not signals.wants_bachelor_degree:
            if contains_any(haystack, ("ie227", "06227", "lm-32", "corso di laurea magistrale in ingegneria informatica")):
                priority += 80
            if contains_any(haystack, ("l-8", "corso di laurea in ingegneria informatica")):
                priority -= 70

        requested_year_terms = []
        if contains_any(signals.lower, ("primo anno", "1° anno", "1 anno")):
            requested_year_terms.extend(["1° anno", "primo anno"])
        if contains_any(signals.lower, ("secondo anno", "2° anno", "2 anno")):
            requested_year_terms.extend(["2° anno", "secondo anno"])
        if contains_any(signals.lower, ("terzo anno", "3° anno", "3 anno")):
            requested_year_terms.extend(["3° anno", "terzo anno"])

        if requested_year_terms and contains_any(f"{section} {text[:800].lower()}", tuple(requested_year_terms)):
            priority += 85

        if contains_any(
            haystack,
            (
                "insegnamenti da altri curricula",
                "insegnamenti da altri cds",
                "a scelta dello studente",
            ),
        ):
            priority -= 65

        if not signals.query_mentions_year:
            years = metadata_year_values(metadata)
            if years:
                newest_year = max(years)
                if newest_year >= 2025:
                    priority += 70
                elif newest_year == 2024:
                    priority += 30
                elif newest_year < 2023:
                    priority -= 80

    if "phd" in intents:
        if "/studenti" in url and contains_any(signals.lower, ("studenti", "studente", "dottorandi", "dottorando", "iscritti")):
            priority += 120
        if chunk_kind == "phd":
            priority += 45
        for year in extract_years_from_text(signals.lower):
            if str(year) in url or str(year) in haystack:
                priority += 35
                break
        if "ingegneria dell'informazione" in signals.lower and "ingegneria-dell-informazione" in url:
            priority += 55
        if "photovoltaics" in signals.lower and "photovoltaics" in url:
            priority += 55

    if "course_regulations" in intents:
        if "__regolamenti-cds" in url:
            priority += 90
        for year in extract_years_from_text(signals.lower):
            if f"/{year}/" in url or str(year) in haystack:
                priority += 45
                break

    if "official_document" in intents and chunk_kind == "official_document_summary":
        priority += 35

    return priority, chunk_index


def course_syllabus_anchor_priority(
    *,
    metadata: dict[str, Any],
    text: str,
    query: str,
) -> tuple[int, int]:
    signals = query_signals(query)
    title = str(metadata.get("title") or metadata.get("content_title") or "").lower()
    section = str(metadata.get("section_heading") or "").lower()
    chunk_index = int(metadata.get("chunk_index") or 0)
    title_tokens = set(tokenize(title)) - COURSE_TITLE_GENERIC_TOKENS
    query_title_tokens = requested_course_title_tokens(signals)
    title_overlap = len(query_title_tokens.intersection(title_tokens))

    priority, _ = anchor_result_priority(metadata=metadata, text=text, query=query)

    if query_title_tokens and title_overlap:
        priority += title_overlap * 80
        if query_title_tokens.issubset(title_tokens):
            priority += 180
        if title_tokens == query_title_tokens:
            priority += 120

    if contains_any(signals.lower, ("argomenti", "argomento", "tratta", "contenuti", "contenuto", "programma")):
        if "contenuti" in section:
            priority += 180
        elif "obiettivi" in section:
            priority += 90
        elif contains_any(section, ("metodi didattici", "verifica", "testi", "prerequisiti")):
            priority -= 120

    if "docenti" in section:
        priority -= 60

    if not signals.query_mentions_year:
        years = metadata_year_values(metadata)
        if years:
            newest_year = max(years)
            if newest_year >= 2025:
                priority += 90
            elif newest_year == 2024:
                priority += 45
            elif newest_year < 2023:
                priority -= 45

    return priority, chunk_index


def anchor_retrieve(query: str, k: int = 20) -> list[RetrievalResult]:
    patterns = intent_anchor_patterns(query)
    if k <= 0:
        return []

    results: list[RetrievalResult] = []
    seen_chunk_ids: set[str] = set()
    chunks = load_valid_chunks()
    signals = query_signals(query)

    if signals.wants_specific_teaching:
        query_title_tokens = requested_course_title_tokens(signals)
        matches: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []

        if query_title_tokens:
            for chunk in chunks:
                metadata = flatten_chunk_metadata(chunk)
                if str(metadata.get("chunk_kind") or "").lower() != "course_syllabus":
                    continue

                title = str(metadata.get("title") or metadata.get("content_title") or "").lower()
                title_tokens = set(tokenize(title)) - COURSE_TITLE_GENERIC_TOKENS
                if len(query_title_tokens.intersection(title_tokens)) < min(2, len(query_title_tokens)):
                    continue

                priority, chunk_index = course_syllabus_anchor_priority(
                    metadata=metadata,
                    text=str(chunk["text"]),
                    query=query,
                )
                matches.append((priority, chunk_index, chunk, metadata))

        matches.sort(key=lambda item: (-item[0], item[1]))

        for _, _, chunk, metadata in matches[: min(12, k)]:
            chunk_id = str(chunk["chunk_id"])
            if chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(chunk_id)
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=chunk["text"],
                    metadata=metadata,
                    source="anchor",
                    rank=len(results) + 1,
                    score=1.0 / (len(results) + 1),
                )
            )

    if not patterns:
        return results

    for pattern, max_matches in patterns:
        matches: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []

        for chunk in chunks:
            metadata = flatten_chunk_metadata(chunk)
            url = str(
                metadata.get("source_url")
                or metadata.get("document_url")
                or metadata.get("url")
                or ""
            ).lower()

            if not re.search(pattern, url):
                continue

            chunk_id = str(chunk["chunk_id"])
            if chunk_id in seen_chunk_ids:
                continue

            priority, chunk_index = anchor_result_priority(
                metadata=metadata,
                text=str(chunk["text"]),
                query=query,
            )
            matches.append((priority, chunk_index, chunk, metadata))

        matches.sort(key=lambda item: (-item[0], item[1]))

        for _, _, chunk, metadata in matches[:max_matches]:
            if len(results) >= k:
                break

            chunk_id = str(chunk["chunk_id"])
            if chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(chunk_id)
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    text=chunk["text"],
                    metadata=metadata,
                    source="anchor",
                    rank=len(results) + 1,
                    score=1.0 / (len(results) + 1),
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


@lru_cache(maxsize=512)
def person_ids_for_query(query: str) -> tuple[str, ...]:
    signals = query_signals(query)
    name_tokens = tuple(
        token
        for token in sorted(signals.teacher_name_tokens)
        if len(token) >= 3
    )
    if len(name_tokens) < 2:
        return ()

    phrase_patterns = [r"\b" + r"\s+".join(map(re.escape, name_tokens)) + r"\b"]
    if len(name_tokens) == 2:
        phrase_patterns.append(r"\b" + re.escape(name_tokens[1]) + r"\s+" + re.escape(name_tokens[0]) + r"\b")

    found_ids: list[str] = []
    seen_ids: set[str] = set()

    for chunk in load_valid_chunks():
        metadata = flatten_chunk_metadata(chunk)
        url = str(
            metadata.get("source_url")
            or metadata.get("document_url")
            or metadata.get("url")
            or ""
        )
        text = str(chunk.get("text") or "")
        haystack = re.sub(r"\s+", " ", f"{text} {url}".lower())

        for phrase_pattern in phrase_patterns:
            for match in re.finditer(phrase_pattern, haystack):
                after_window = haystack[match.start(): match.end() + 220]
                ids = re.findall(r"(?:matricola=|docenti\.unisa\.it/)(\d{6})", after_window)
                if ids:
                    ids = ids[:1]

                if not ids:
                    ids = re.findall(r"(?:matricola=|docenti\.unisa\.it/)(\d{6})", url.lower())

                if not ids:
                    window = haystack[max(0, match.start() - 120): match.end() + 120]
                    ids = re.findall(r"(?:matricola=|docenti\.unisa\.it/)(\d{6})", window)
                    if ids:
                        ids = ids[:1]

                for person_id in ids:
                    if person_id not in seen_ids:
                        seen_ids.add(person_id)
                        found_ids.append(person_id)

        if len(found_ids) >= 5:
            break

    return tuple(found_ids)


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


def is_canonical_study_plan_metadata(
    metadata: dict[str, Any],
    text: str = "",
) -> bool:
    metadata_probe = " ".join(
        str(metadata.get(key) or "")
        for key in [
            "source_url",
            "document_url",
            "url",
            "discovered_from",
            "link_text",
            "title",
            "content_title",
            "section_heading",
            "breadcrumb",
            "breadcrumb_text",
        ]
    ).lower()
    body_probe = text[:1800].lower()

    if contains_any(
        metadata_probe,
        (
            "__piano-studi-cds",
            "/didattica/piano-di-studi",
            "piano di studi a.a",
            "piano degli studi a.a",
            "manifesto degli studi",
        ),
    ):
        return True

    has_plan_title = contains_any(
        metadata_probe,
        ("piano di studi", "piano degli studi", "manifesto degli studi"),
    )
    has_curricular_table_signal = bool(
        re.search(r"\b[123]\s*[°o]\s+anno\b", body_probe)
        or re.search(r"\b(?:ssd|cfu|taf|ambito|curriculum)\b", body_probe)
    )
    return has_plan_title and has_curricular_table_signal


def is_canonical_study_plan_result(result: RetrievalResult) -> bool:
    return is_canonical_study_plan_metadata(result.metadata or {}, result.text)


def requested_study_plan_year_labels(signals: QuerySignals) -> list[str]:
    labels: list[str] = []

    if contains_any(signals.lower, ("primo anno", "1° anno", "1 anno")) or (
        "primo" in signals.lower and "anno" in signals.lower
    ):
        labels.append("1")
    if contains_any(signals.lower, ("secondo anno", "2° anno", "2 anno")) or (
        "secondo" in signals.lower and "anno" in signals.lower
    ):
        labels.append("2")
    if contains_any(signals.lower, ("terzo anno", "3° anno", "3 anno")) or (
        "terzo" in signals.lower and "anno" in signals.lower
    ):
        labels.append("3")

    return labels


def study_plan_year_label(result: RetrievalResult) -> str:
    section = str(result.metadata.get("section_heading") or "").lower()
    text_lower = result.text[:1400].lower()
    probe = f"{section}\n{text_lower}"

    if re.search(r"\b1\s*[°o]\s+anno\b|\bprimo\s+anno\b|\banno\s+1\b", probe):
        return "1"
    if re.search(r"\b2\s*[°o]\s+anno\b|\bsecondo\s+anno\b|\banno\s+2\b", probe):
        return "2"
    if re.search(r"\b3\s*[°o]\s+anno\b|\bterzo\s+anno\b|\banno\s+3\b", probe):
        return "3"

    return ""


def study_plan_course_preference(result: RetrievalResult, signals: QuerySignals) -> int:
    haystack = combined_result_text(result)
    score = 0

    if is_canonical_study_plan_result(result):
        score += 8

    years = metadata_year_values(result.metadata)
    if years and not signals.query_mentions_year:
        newest_year = max(years)
        if newest_year >= 2025:
            score += 6
        elif newest_year == 2024:
            score += 2
        elif newest_year < 2023:
            score -= 6

    if signals.wants_bachelor_degree and not signals.wants_master_degree:
        if contains_any(
            haystack,
            ("ie127", "06127", "corso di laurea in ingegneria informatica", "classe l-8"),
        ):
            score += 8
        if contains_any(
            haystack,
            ("ie128", "06128", "medicina digitale", "lm-32", "lm-28", "magistrale"),
        ):
            score -= 10
    elif signals.wants_master_degree and not signals.wants_bachelor_degree:
        if contains_any(
            haystack,
            ("ie227", "06227", "corso di laurea magistrale in ingegneria informatica", "lm-32"),
        ):
            score += 8
        if contains_any(
            haystack,
            ("ie127", "06127", "ie128", "06128", "l-8", "medicina digitale"),
        ):
            score -= 8

    return score


def enforce_study_plan_year_coverage(
    results: list[RetrievalResult],
    signals: QuerySignals,
) -> list[RetrievalResult]:
    requested_labels = requested_study_plan_year_labels(signals)
    if not signals.wants_study_plan_courses or not requested_labels:
        return results

    selected: list[RetrievalResult] = []
    selected_ids: set[str] = set()

    for label in requested_labels:
        candidates = [
            result
            for result in results
            if result.chunk_id not in selected_ids
            and str(result.metadata.get("chunk_kind") or "").lower() == "study_plan"
            and is_canonical_study_plan_result(result)
            and study_plan_year_label(result) == label
        ]
        if not candidates:
            continue

        candidates.sort(
            key=lambda result: (
                study_plan_course_preference(result, signals),
                result.score,
                -int(result.metadata.get("chunk_index") or 0),
            ),
            reverse=True,
        )
        selected.append(candidates[0])
        selected_ids.add(candidates[0].chunk_id)

    if not selected:
        return results

    remaining = [result for result in results if result.chunk_id not in selected_ids]
    return selected + remaining


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

    return bounded_route_multiplier(multiplier, {spec.name for spec in specs})


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
    signals = query_signals(query)
    intents = signals.intents
    if not intents:
        return 1.0

    query_lower = signals.lower
    url = get_result_url(result).lower()
    url_id = query_param_value(url, "id")
    source = str(result.metadata.get("source") or "").lower()
    chunk_kind = str(result.metadata.get("chunk_kind") or "").lower()
    title = str(result.metadata.get("title") or "").lower()
    content_title = str(result.metadata.get("content_title") or "").lower()
    section_heading = str(result.metadata.get("section_heading") or "").lower()
    haystack = combined_result_text(result)

    multiplier = site_intent_relevance_multiplier(result, query)

    if "course_syllabus" in intents:
        title_tokens = set(tokenize(f"{title} {content_title}")) - COURSE_TITLE_GENERIC_TOKENS
        query_title_tokens = requested_course_title_tokens(signals)
        title_overlap = len(query_title_tokens.intersection(title_tokens))

        if "coursecatalogue" in url:
            multiplier *= 3.80

        if "/didattica/insegnamenti" in url:
            multiplier *= 1.50

        if chunk_kind == "course_syllabus":
            multiplier *= 2.80

        if title_overlap >= 2:
            multiplier *= 3.60
        elif title_overlap == 1:
            multiplier *= 1.65

        if any(
            marker in section_heading
            for marker in [
                "contenuti",
                "obiettivi formativi",
                "metodi didattici",
                "verifica dell'apprendimento",
                "testi",
            ]
        ):
            multiplier *= 2.25

        if contains_any(query_lower, ("argomenti", "tratta", "programma", "contenuti")):
            if "contenuti" in section_heading:
                multiplier *= 2.80
            elif "obiettivi formativi" in section_heading:
                multiplier *= 1.65

        if "offerta-formativa" in url:
            multiplier *= 0.08

        if "piano-di-studi" in url or "__piano-studi-cds" in url:
            multiplier *= 0.30

        if (
            ("corso di laurea" in haystack or "offerta formativa" in haystack)
            and title_overlap == 0
            and "coursecatalogue" not in url
        ):
            multiplier *= 0.20

    if "study_plan" in intents:
        asks_first_year = contains_any(query_lower, ("primo anno", "1° anno", "1 anno"))
        asks_second_year = contains_any(query_lower, ("secondo anno", "2° anno", "2 anno"))
        asks_third_year = contains_any(query_lower, ("terzo anno", "3° anno", "3 anno"))
        is_canonical_plan = is_canonical_study_plan_result(result)

        if is_canonical_plan and (
            "__piano-studi-cds" in url
            or "piano degli studi" in haystack
            or "manifesto degli studi" in haystack
        ):
            multiplier *= 3.40

        if is_canonical_plan and "piano-di-studi" in url:
            multiplier *= 2.00

        if chunk_kind == "study_plan":
            multiplier *= 2.80 if is_canonical_plan else 0.35
        elif not is_canonical_plan and contains_any(url, ("tutorato", "orientamento")):
            multiplier *= 0.30

        if is_canonical_plan and asks_first_year and ("1° anno" in haystack or "primo anno" in haystack):
            multiplier *= 1.80

        if is_canonical_plan and asks_second_year and ("2° anno" in haystack or "secondo anno" in haystack):
            multiplier *= 1.80

        if is_canonical_plan and asks_third_year and ("3° anno" in haystack or "terzo anno" in haystack):
            multiplier *= 1.80

        if signals.wants_bachelor_degree and not signals.wants_master_degree:
            if (
                "ie127" in haystack
                or "ie127" in url
                or "l-8" in haystack
                or "corso di laurea in ingegneria informatica" in haystack
            ):
                multiplier *= 3.00

            if "lm-32" in haystack or "lm-28" in haystack or "magistrale" in haystack:
                multiplier *= 0.28

        elif signals.wants_master_degree and not signals.wants_bachelor_degree:
            if "ie227" in haystack or "ie227" in url or "lm-32" in haystack:
                multiplier *= 3.00

            if "l-8" in haystack or "corso di laurea in ingegneria informatica" in haystack:
                multiplier *= 0.35

        if is_canonical_plan and not signals.query_mentions_year:
            years = metadata_year_values(result.metadata)
            if years:
                newest_year = max(years)
                if newest_year >= 2025:
                    multiplier *= 1.60
                elif newest_year == 2024:
                    multiplier *= 1.20
                elif newest_year < 2023:
                    multiplier *= 0.25

        if "offerta-formativa" in url:
            multiplier *= 0.25

        if "presentazione" in url or "immatricolazioni" in url:
            multiplier *= 0.35

    if "people_directory" in intents:
        if "/dipartimento/personale" in url:
            multiplier *= 3.40

        if "docenti e personale" in haystack or "personale docente" in haystack:
            multiplier *= 2.20

        if "rubrica.unisa.it" in url:
            multiplier *= 1.50

        if "docenti.unisa.it" in url and is_single_profile_page(url):
            multiplier *= 0.55

        if "/ricerca/pubblicazioni" in url or "/ricerca/progetti" in url:
            multiplier *= 0.30

    if "department_governance" in intents:
        if "/dipartimento/organi-collegiali" in url:
            multiplier *= 3.20

        if "/dipartimento/commissioni" in url or "commissions-and-delegates" in url:
            multiplier *= 2.40

        if "direttore" in query_lower and "direttore" in haystack:
            multiplier *= 1.80

        if "organi collegiali" in haystack or "consiglio di dipartimento" in haystack:
            multiplier *= 1.70

        if "docenti.unisa.it" in url:
            multiplier *= 0.45

    if "department_research" in intents:
        if "/ricerca" in url or "/terza-missione" in url:
            multiplier *= 2.20

        if "/dipartimento/strutture" in url and contains_any(query_lower, ("laboratorio", "laboratori")):
            multiplier *= 1.70

        if "progetti di ricerca" in haystack or "laboratori" in haystack:
            multiplier *= 1.35

    if "contacts" in intents:
        if "/home/contatti" in url:
            multiplier *= 3.20

        if "/didattica/contatti" in url and contains_any(query_lower, ("didattica", "studenti", "segreteria")):
            multiplier *= 1.90

        if "fisciano" in haystack or "via giovanni paolo ii" in haystack:
            multiplier *= 1.80

        if signals.wants_location_info and contains_any(query_lower, ("diem", "dipartimento")):
            if "/home/contatti" in url:
                multiplier *= 4.50

            if "/dipartimento/presentazione" in url:
                multiplier *= 2.20

            if "/international/" in url or "corsi.unisa.it" in url:
                multiplier *= 0.20

        if "docenti.unisa.it" in url and not signals.teacher_query:
            multiplier *= 0.35

    if "official_document" in intents:
        if "/home/bandi" in url:
            multiplier *= 2.50

        if "/uploads/" in url or url.endswith(".pdf"):
            multiplier *= 1.70

        if chunk_kind in {"official_document", "official_document_summary"}:
            multiplier *= 1.80

        if "regolament" in query_lower:
            if "__regolamenti-cds" in url or "regolamento" in haystack:
                multiplier *= 2.00
            if "bando" in haystack and "regolament" not in haystack:
                multiplier *= 0.45

        if "bando" in query_lower or "bandi" in query_lower:
            if "bando" in haystack or "/home/bandi" in url:
                multiplier *= 2.10
            if "regolamento" in haystack and "bando" not in haystack:
                multiplier *= 0.55

        if "graduatoria" in query_lower and "graduatoria" in haystack:
            multiplier *= 1.80

        if query_wants_recent_items(query):
            document_years = extract_years_from_text(f"{url} {haystack}")
            if document_years:
                newest_year = max(document_years)
                if newest_year >= 2026:
                    multiplier *= 2.40
                elif newest_year >= 2025:
                    multiplier *= 2.00
                elif newest_year >= 2024:
                    multiplier *= 1.40
                elif newest_year < 2023:
                    multiplier *= 0.45

    if "course_regulations" in intents:
        if "__regolamenti-cds" in url:
            multiplier *= 4.00

        if "regolamento" in haystack or "regolamenti" in haystack:
            multiplier *= 2.40

        if "ingegneria-informatica" in query_lower.replace(" ", "-") or "ingegneria informatica" in query_lower:
            if "ingegneria-informatica" in url or "ingegneria informatica" in haystack:
                multiplier *= 2.20

        for year in extract_years_from_text(query_lower):
            if f"/{year}/" in url or str(year) in haystack:
                multiplier *= 1.80
                break

        if "/home/bandi" in url or "bando" in haystack or "graduatoria" in haystack:
            multiplier *= 0.08

    if "news" in intents:
        if "/home/news" in url or "/news" in url:
            multiplier *= 2.20

        if "avviso" in query_lower and ("avviso" in haystack or "avvisi" in haystack):
            multiplier *= 1.70

        if "evento" in query_lower and ("evento" in haystack or "eventi" in haystack):
            multiplier *= 1.70

    if "teacher_publications" in intents:
        publication_year = safe_int(
            result.metadata.get("publication_year") or result.metadata.get("year")
        )

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

    return bounded_route_multiplier(multiplier, intents)


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
    signals = query_signals(query)
    intents = signals.intents

    wants_degree_info = signals.wants_degree_overview
    wants_bachelor_degree = signals.wants_bachelor_degree
    wants_master_degree = signals.wants_master_degree
    wants_teaching_info = signals.wants_teaching_info
    wants_admission_info = signals.wants_admission_info
    wants_final_exam_info = signals.wants_final_exam_info
    wants_phd_info = signals.wants_phd_info
    wants_erasmus_bando_info = signals.wants_erasmus_bando_info
    wants_erasmus_agreements = signals.wants_erasmus_agreements
    erasmus_mobility = signals.erasmus_mobility
    wants_aggregate_info = signals.wants_aggregate_info
    query_mentions_year = signals.query_mentions_year
    italian_query = signals.italian_query
    teacher_query = signals.teacher_query
    teacher_name_tokens = signals.teacher_name_tokens
    asks_office_hours = signals.asks_office_hours

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
            requested_person_ids = person_ids_for_query(query)

            is_docenti_page = "docenti.unisa.it" in url
            is_requested_teacher_page = teacher_metadata_matches(
                teacher_name_tokens=teacher_name_tokens,
                metadata_tokens=metadata_tokens,
            ) or any(
                f"docenti.unisa.it/{person_id}" in str(url).lower()
                or f"matricola={person_id}" in str(url).lower()
                for person_id in requested_person_ids
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
                
        text_lower = result.text.lower()
        url_lower = str(url).lower()

        if wants_degree_info and wants_bachelor_degree and not wants_master_degree:
            if "didattica/offerta-formativa" in url_lower:
                adjusted_score *= 4.00

            if "corso di laurea" in text_lower and "l-8" in text_lower:
                adjusted_score *= 4.00

            if "ie127l-8" in text_lower or "ie128l-8" in text_lower:
                adjusted_score *= 4.00

            if "corso di laurea magistrale" in text_lower:
                adjusted_score *= 0.15

            if "focus della didattica" in title or "focus della didattica" in breadcrumb:
                adjusted_score *= 0.20

            if "percorso di eccellenza" in text_lower or "percorso di eccellenza" in title:
                adjusted_score *= 0.10

        elif wants_degree_info and wants_master_degree and not wants_bachelor_degree:
            if "didattica/offerta-formativa" in url_lower:
                adjusted_score *= 1.80

            if "corso di laurea magistrale" in text_lower:
                adjusted_score *= 2.00

            if "lm-32" in text_lower or "lm-28" in text_lower:
                adjusted_score *= 2.20

            if "percorso di eccellenza" in text_lower or "percorso di eccellenza" in title:
                adjusted_score *= 0.35

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
        elif (
            wants_teaching_info
            and not wants_degree_info
            and "course_syllabus" not in intents
            and "study_plan" not in intents
        ):
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

        result.score, adjustment_factor = bounded_total_score(
            base_score=result.score,
            adjusted_score=adjusted_score,
            intents=intents,
        )
        result.metadata["metadata_adjustment_factor"] = adjustment_factor

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
        or "study_plan" in query_intents(query)
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


def apply_final_exact_match_guards(
    results: list[RetrievalResult],
    query: str,
) -> list[RetrievalResult]:
    """
    Protegge pochi match deterministici ad alta confidenza dopo il reranker.

    Il cross-encoder vede solo testo e metadata del singolo chunk. Se una pagina
    docente ha titolo generico ma URL numerica corretta, il reranker può
    preferire un altro docente con nome esplicito nei metadata. La risoluzione
    nome -> matricola, e alcuni vincoli di sezione, evitano questi errori senza
    disattivare il reranker.
    """
    signals = query_signals(query)
    requested_person_ids = (
        person_ids_for_query(query)[:1]
        if signals.teacher_query and signals.teacher_name_tokens and not signals.wants_aggregate_info
        else ()
    )
    asks_syllabus_contents = contains_any(
        signals.lower,
        ("argomenti", "argomento", "tratta", "contenuti", "contenuto", "programma"),
    )
    asks_phd_students = contains_any(
        signals.lower,
        ("studenti", "studente", "dottorandi", "dottorando", "iscritti"),
    )
    asks_information_engineering_phd = (
        "ingegneria dell'informazione" in signals.lower
        or "ingegneria dell informazione" in signals.lower
    )
    study_plan_year_terms = (
        "1° anno",
        "2° anno",
        "3° anno",
        "primo anno",
        "secondo anno",
        "terzo anno",
        "anno 1",
        "anno 2",
        "anno 3",
    )

    for result in results:
        url = get_result_url(result).lower()
        text_lower = result.text.lower()
        chunk_kind = str(result.metadata.get("chunk_kind") or "").lower()
        section = str(result.metadata.get("section_heading") or "").lower()
        local_context = f"{section} {text_lower[:1200]}"
        is_canonical_plan = is_canonical_study_plan_result(result)

        if requested_person_ids:
            is_requested_person = any(
                f"docenti.unisa.it/{person_id}" in url
                or f"matricola={person_id}" in url
                for person_id in requested_person_ids
            )
            is_office_hours = (
                chunk_kind == "office_hours"
                or "orario di ricevimento" in text_lower
                or "ricevimento" in text_lower
            )

            if is_requested_person:
                result.score *= 5.0 if signals.asks_office_hours and is_office_hours else 2.5
                result.metadata["exact_person_match"] = True
            elif signals.asks_office_hours and "docenti.unisa.it" in url:
                result.score *= 0.35

        if signals.wants_specific_teaching and chunk_kind == "course_syllabus":
            if asks_syllabus_contents:
                if "contenuti" in section:
                    result.score *= 2.6
                elif "obiettivi" in section:
                    result.score *= 1.6
                elif contains_any(section, ("metodi didattici", "verifica", "testi")):
                    result.score *= 0.45

            if not signals.query_mentions_year:
                years = metadata_year_values(result.metadata)
                if years:
                    newest_year = max(years)
                    if newest_year >= 2025:
                        result.score *= 1.35
                    elif newest_year == 2024:
                        result.score *= 1.10
                    elif newest_year < 2023:
                        result.score *= 0.70

        if signals.wants_study_plan_courses:
            has_requested_year_section = contains_any(local_context, study_plan_year_terms)
            is_elective_group = contains_any(
                local_context,
                (
                    "insegnamenti da altri curricula",
                    "insegnamenti da altri cds",
                    "insegnamenti dal cds",
                    "a scelta dello studente",
                ),
            )

            if is_canonical_plan and chunk_kind == "study_plan":
                if has_requested_year_section:
                    result.score *= 3.0
                elif is_elective_group:
                    result.score *= 0.40
                else:
                    result.score *= 1.25

                if not signals.query_mentions_year:
                    years = metadata_year_values(result.metadata)
                    if years:
                        newest_year = max(years)
                        if newest_year >= 2025:
                            result.score *= 1.45
                        elif newest_year == 2024:
                            result.score *= 1.10
                        elif newest_year < 2023:
                            result.score *= 0.20
            elif chunk_kind == "study_plan" or contains_any(url, ("tutorato", "orientamento")):
                result.score *= 0.22
            elif is_elective_group:
                result.score *= 0.40

        if signals.wants_phd_info and asks_phd_students:
            if "/studenti" in url:
                result.score *= 1.8
            if asks_information_engineering_phd and "ingegneria-dell-informazione" in url:
                result.score *= 2.6
            elif asks_information_engineering_phd and "photovoltaics" in url:
                result.score *= 0.30

        result.metadata["final_guard_applied"] = True

    results.sort(key=lambda result: result.score, reverse=True)
    results = enforce_study_plan_year_coverage(results, signals)

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
    anchors = anchor_retrieve(query, k=min(20, max(final_k * 3, 10)))
    if anchors:
        result_lists["anchors"] = anchors

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

    signals = query_signals(query)
    neural_pool_k = rerank_k
    if signals.wants_study_plan_courses:
        neural_pool_k = max(neural_pool_k, final_k * 8, 60)
    elif signals.wants_specific_teaching or signals.intents.intersection(HIGH_CONFIDENCE_ROUTE_INTENTS):
        neural_pool_k = max(neural_pool_k, final_k * 6, 40)

    hybrid_results = neural_rerank(
        query=query,
        results=hybrid_results,
        top_k=neural_pool_k,
    )
    hybrid_results = apply_final_exact_match_guards(hybrid_results, query)

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
