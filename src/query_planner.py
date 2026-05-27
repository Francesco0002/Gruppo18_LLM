from __future__ import annotations

"""
Planner leggero per orientare il retrieval senza renderlo deterministico.

Il modulo non prova a "capire tutto" e non decide mai la risposta finale:
produce solo un QueryPlan con topic, entità e filtri probabili. Quando la
confidenza è bassa, retrieval.py continua comunque a interrogare più retriever.
"""

import re
import unicodedata
from dataclasses import dataclass, field


TASK_TYPES = {
    "study_plan",
    "course_catalog",
    "teacher_profile",
    "teacher_publications",
    "office_hours",
    "lab_equipment",
    "research",
    "third_mission",
    "international",
    "phd",
    "official_docs",
    "general",
}


FINAL_EXAM_GRADE_MARKERS = (
    "voto di laurea",
    "voto laurea",
    "voto finale di laurea",
    "voto finale laurea",
    "voto dell esame finale",
    "voto esame finale",
    "esame finale",
    "valutazione conclusiva",
    "media voto",
    "media voti",
    "media esami",
)


def normalize_query(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    return " ".join(text.lower().split())


def contains_any(text: str, markers: tuple[str, ...]) -> bool:
    return any(marker in text for marker in markers)


def extract_years(text: str) -> list[int]:
    return sorted({int(match) for match in re.findall(r"\b(?:19|20)\d{2}\b", text)})


def extract_course_years(text: str) -> list[int]:
    # Gestisce sia "2 anno" sia formulazioni naturali come
    # "primo, secondo e terzo anno", molto frequenti nelle query sui piani.
    years = {
        int(match)
        for match in re.findall(r"\b([123])\s*(?:°|o)?\s*anno\b", text)
    }
    ordinal_words = {
        "primo": 1,
        "secondo": 2,
        "terzo": 3,
    }
    for word, value in ordinal_words.items():
        if re.search(rf"\b{word}\s+anno\b", text) or (word in text and "anno" in text):
            years.add(value)
    return sorted(years)


def extract_course_hint(text: str) -> str:
    # Lista intenzionalmente corta: serve a riconoscere i corsi DIEM canonici,
    # non a costruire un'ontologia rigida. Se manca il match, gli altri
    # retriever restano attivi.
    course_markers = [
        "ingegneria informatica magistrale",
        "ingegneria informatica",
        "ingegneria dell informazione per la medicina digitale",
        "information engineering for digital medicine",
        "electrical engineering for digital energy",
    ]
    for marker in course_markers:
        if marker in text:
            return marker
    return ""


def extract_course_level(text: str) -> str:
    if contains_any(text, ("magistrale", "laurea magistrale", "lm-")):
        return "magistrale"
    if contains_any(text, ("triennale", "laurea triennale", "classe l-")):
        return "triennale"
    return ""


def extract_teacher_hint(original: str, normalized: str) -> str:
    # Estrazione conservativa del nome docente: rimuove parole funzionali
    # della query e tiene gli ultimi token significativi, evitando regex per
    # ogni possibile cognome.
    if not contains_any(
        normalized,
        (
            "prof ",
            "prof.",
            "professore",
            "professoressa",
            "docente",
            "ricevimento",
            "pubblicazioni",
            "paper",
        ),
    ):
        return ""

    cleaned = re.sub(
        r"\b(?:prof\.?|professore|professoressa|docente|orari?|ricevimento|pubblicazioni|paper|articoli|di|del|della|il|la|lo|quali|sono|recenti)\b",
        " ",
        original,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[^A-Za-zÀ-ÿ\s'-]", " ", cleaned)
    tokens = [token for token in cleaned.split() if len(token) > 2]
    if len(tokens) >= 2:
        return " ".join(tokens[-3:])
    return ""


@dataclass(frozen=True)
class QueryPlan:
    """Contratto piccolo e serializzabile usato da retrieval e tracing."""

    task_type: str = "general"
    confidence: float = 0.35
    entities: dict[str, object] = field(default_factory=dict)
    filters: dict[str, object] = field(default_factory=dict)
    source_families: tuple[str, ...] = ()
    needs_structured_data: bool = False
    needs_vector_search: bool = True
    requires_complete_answer: bool = False
    reason: str = ""

    @property
    def low_confidence(self) -> bool:
        return self.confidence < 0.55


def build_retrieval_query(query: str, plan: QueryPlan) -> str:
    """
    Rafforza la query di retrieval con segnali già emersi dal planner.

    Non cambia l'intento della domanda: aggiunge solo termini utili a BM25,
    dense retrieval e structured retrieval per evitare che soggetti o syllabus
    affini prendano il sopravvento.
    """
    terms: list[str] = []
    teacher = str(plan.entities.get("teacher") or "").strip()
    course = str(plan.entities.get("course") or "").strip()
    normalized_query = normalize_query(query)

    if teacher:
        terms.append(teacher)
        if plan.task_type == "teacher_publications":
            terms.extend([f"{teacher} pubblicazioni", "pubblicazioni docente"])
        elif plan.task_type == "teacher_profile":
            terms.append("profilo docente")

    if course:
        terms.append(course)

    if plan.task_type == "teacher_publications":
        terms.extend(["pubblicazioni", "produzione scientifica", "articoli"])

    if contains_any(normalized_query, FINAL_EXAM_GRADE_MARKERS):
        terms.extend([
            "regolamento esame finale lauree magistrali",
            "attribuzione voto esame finale",
            "voto base media pesata crediti",
            "valutazione conclusiva centodecimi",
        ])

    if plan.task_type == "lab_equipment":
        terms.extend(["dipartimento strutture laboratori", "laboratori DIEM", "strutture DIEM"])

    if plan.task_type in {"study_plan", "course_catalog"} and contains_any(
        normalized_query,
        (
            "programma",
            "esame",
            "sillabo",
            "syllabus",
            "verifica dell apprendimento",
            "modalita esame",
        ),
    ):
        terms.extend([
            "programma esame",
            "verifica dell'apprendimento",
            "modalita esame",
            "contenuti insegnamento",
        ])

    if plan.task_type == "office_hours" and teacher:
        terms.append("orario di ricevimento")

    if not terms:
        return query

    merged_terms: list[str] = []
    seen: set[str] = set()

    for term in [query, *terms]:
        cleaned_term = " ".join(str(term).split())
        normalized_term = normalize_query(cleaned_term)
        if not cleaned_term or normalized_term in seen:
            continue
        seen.add(normalized_term)
        merged_terms.append(cleaned_term)

    return " ".join(merged_terms)


def infer_task_type(text: str) -> tuple[str, float, str]:
    # Heuristiche ad alto segnale: non sono intent esclusivi e non bloccano la
    # pipeline. Il valore di confidence serve solo a capire quanto fidarsi per
    # operazioni come pinning di evidenze strutturate.
    if contains_any(text, FINAL_EXAM_GRADE_MARKERS):
        return "official_docs", 0.82, "query su regolamento dell'esame finale e voto di laurea"
    if contains_any(text, ("piano di studi", "piano degli studi", "esame", "esami", "insegnamenti", "cfu", "curriculum")):
        return "study_plan", 0.78, "query didattica con segnali di piano di studi/insegnamenti"
    if "anno" in text and contains_any(text, ("primo", "secondo", "terzo", "1 anno", "2 anno", "3 anno")) and "ingegneria" in text:
        return "study_plan", 0.76, "query didattica multi-anno su corso di ingegneria"
    if contains_any(text, ("corsi di laurea", "offerta formativa", "lauree", "corso di laurea", "laurea magistrale")):
        return "course_catalog", 0.74, "query su offerta formativa o corsi"
    if contains_any(text, ("ricevimento", "riceve", "orario di ricevimento", "orari di ricevimento")):
        return "office_hours", 0.82, "query su ricevimento docente"
    if contains_any(text, ("pubblicazioni", "pubblicazione", "paper", "articoli", "produzione scientifica")):
        return "teacher_publications", 0.77, "query su pubblicazioni"
    if contains_any(text, ("docente", "professore", "professoressa", "studio", "email", "telefono")):
        return "teacher_profile", 0.66, "query su profilo docente"
    if contains_any(text, ("laboratorio", "laboratori", "strumentazione", "attrezzature", "strutture")):
        return "lab_equipment", 0.72, "query su laboratori o strutture"
    if contains_any(text, ("erasmus", "mobilita", "mobilità", "learning agreement", "international", "accordi")):
        return "international", 0.75, "query international/erasmus"
    if contains_any(text, ("dottorato", "dottorati", "phd", "collegio", "ciclo")):
        return "phd", 0.74, "query su dottorati"
    if contains_any(text, ("terza missione", "spin off", "brevetti", "public engagement", "conto terzi", "trasferimento tecnologico")):
        return "third_mission", 0.73, "query terza missione"
    if contains_any(text, ("ricerca", "progetti finanziati", "aree di ricerca", "gruppo di ricerca")):
        return "research", 0.65, "query ricerca"
    if contains_any(text, ("bando", "bandi", "graduatoria", "decreto", "regolamento", "calendario", "avviso")):
        return "official_docs", 0.68, "query documentale ufficiale"
    return "general", 0.35, "nessun topic specifico ad alta confidenza"


def plan_query(query: str) -> QueryPlan:
    normalized = normalize_query(query)
    task_type, confidence, reason = infer_task_type(normalized)
    course_years = extract_course_years(normalized)
    academic_years = extract_years(normalized)
    course_hint = extract_course_hint(normalized)
    course_level = extract_course_level(normalized)
    teacher_hint = extract_teacher_hint(query, normalized)

    list_markers = (
        "elenca",
        "elencami",
        "lista",
        "tutti",
        "tutte",
        "quali sono",
        "completo",
        "completa",
        "primo",
        "secondo",
        "terzo",
    )
    # Le query di elenco o multi-anno hanno bisogno di più chunk dalla stessa
    # fonte; questo flag allenta la deduplica per URL più avanti nel retrieval.
    lab_list_query = task_type == "lab_equipment" and contains_any(
        normalized,
        (
            "che laboratori",
            "quali laboratori",
            "laboratori possiede",
            "laboratori ha",
            "elenco laboratori",
            "lista laboratori",
            "strutture del diem",
            "strutture possiede",
        ),
    )
    requires_complete_answer = (
        contains_any(normalized, list_markers)
        or len(course_years) >= 2
        or lab_list_query
    )

    entities: dict[str, object] = {}
    filters: dict[str, object] = {}

    if course_hint:
        entities["course"] = course_hint
    if teacher_hint:
        entities["teacher"] = teacher_hint
    if course_years:
        filters["course_years"] = course_years
    if course_level:
        filters["course_level"] = course_level
    if academic_years:
        filters["years"] = academic_years

    source_by_task = {
        "study_plan": ("course_catalogue", "course"),
        "course_catalog": ("course_catalogue", "course", "diem"),
        "office_hours": ("teacher_profile", "directory"),
        "teacher_profile": ("teacher_profile", "directory"),
        "teacher_publications": ("teacher_profile",),
        "lab_equipment": ("diem",),
        "research": ("diem", "teacher_profile"),
        "third_mission": ("diem",),
        "international": ("diem", "course"),
        "phd": ("course", "diem"),
        "official_docs": ("course", "diem"),
        "general": (),
    }

    structured_tasks = {
        "study_plan",
        "course_catalog",
        "office_hours",
        "teacher_publications",
        "lab_equipment",
        "official_docs",
    }

    return QueryPlan(
        task_type=task_type if task_type in TASK_TYPES else "general",
        confidence=confidence,
        entities=entities,
        filters=filters,
        source_families=source_by_task.get(task_type, ()),
        needs_structured_data=task_type in structured_tasks or requires_complete_answer,
        needs_vector_search=True,
        requires_complete_answer=requires_complete_answer,
        reason=reason,
    )
