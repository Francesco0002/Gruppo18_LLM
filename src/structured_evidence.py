from __future__ import annotations

"""
Evidence retriever strutturato.

Questo modulo non produce risposte preconfezionate: seleziona chunk già
indicizzati che hanno forma tabellare/listabile o metadata forti, così l'LLM
può ricevere prove compatte e citabili senza affidarsi solo alla similarità
semantica.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from chunk_metadata import flatten_chunk_metadata
from query_planner import QueryPlan


LAB_EQUIPMENT_MARKERS = (
    "strumentazione",
    "strumenti",
    "attrezzature",
    "dotazione",
    "apparecchiature",
)

OFFICIAL_FINAL_EXAM_MARKERS = (
    "attribuzione del voto",
    "esame finale",
    "prova finale",
    "voto_base",
    "media_pesata",
    "media pesata",
    "valutazione conclusiva",
    "centodecimi",
)


@dataclass(frozen=True)
class StructuredEvidence:
    """Chunk candidato con score spiegabile usato nel trace del retrieval."""

    chunk: dict[str, Any]
    score: float
    reason: str


def normalize_text(text: object) -> str:
    value = unicodedata.normalize("NFKD", str(text or ""))
    value = "".join(char for char in value if not unicodedata.combining(char))
    return " ".join(value.lower().split())


def tokens(text: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-zA-ZÀ-ÿ0-9]+", normalize_text(text))
        if len(token) > 2
    }


def metadata_haystack(chunk: dict[str, Any]) -> str:
    # Haystack fielded ma compatto: combina locator, metadata e body pulito.
    # Non serve per embedding, solo per match strutturato e scoring locale.
    metadata = flatten_chunk_metadata(chunk)
    fields = [
        metadata.get("title"),
        metadata.get("content_title"),
        metadata.get("section_heading"),
        metadata.get("breadcrumb_text"),
        metadata.get("source_url"),
        metadata.get("document_url"),
        metadata.get("chunk_kind"),
        metadata.get("topic_family"),
        metadata.get("entity_name"),
        metadata.get("course_name"),
        metadata.get("course_level"),
        metadata.get("curriculum"),
        metadata.get("course_year"),
        metadata.get("academic_year"),
        metadata.get("cohort"),
        metadata.get("publication_title"),
        metadata.get("publication_authors"),
        chunk.get("locator_text"),
        chunk.get("body_text") or chunk.get("text_for_embedding") or chunk.get("text"),
    ]
    return normalize_text(" ".join(str(field or "") for field in fields))


def int_value(value: object) -> int:
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return 0


def source_url(metadata: dict[str, Any]) -> str:
    return str(metadata.get("source_url") or metadata.get("document_url") or "")


def is_department_structures_source(metadata: dict[str, Any]) -> bool:
    try:
        parsed = urlparse(source_url(metadata))
    except ValueError:
        return False

    return (
        parsed.netloc.lower().endswith("diem.unisa.it")
        and parsed.path.rstrip("/") == "/dipartimento/strutture"
    )


def is_department_structures_index_source(metadata: dict[str, Any]) -> bool:
    try:
        parsed = urlparse(source_url(metadata))
    except ValueError:
        return False

    return (
        is_department_structures_source(metadata)
        and not parsed.query
        and not parsed.fragment
    )


def source_priority(metadata: dict[str, Any], plan: QueryPlan | None = None) -> int:
    # Priorità di affidabilità/fonte: CourseCatalogue è primario per didattica,
    # corsi.unisa.it per pagine corso/documenti, docenti.unisa.it per persone.
    url = normalize_text(source_url(metadata))
    source_family = normalize_text(metadata.get("source_family"))
    chunk_kind = normalize_text(metadata.get("chunk_kind"))
    document_type = normalize_text(metadata.get("document_type"))

    if plan and plan.task_type == "official_docs":
        if document_type == "regolamento" and chunk_kind in {"official_document", "official_document_summary"}:
            return 82
        if chunk_kind in {"official_document", "official_document_summary"}:
            return 68
        if "corsi.unisa.it" in url or source_family == "course":
            return 58
        if "diem.unisa.it" in url or source_family == "diem":
            return 54
        if "coursecatalogue" in url:
            return 30

    if plan and plan.task_type == "lab_equipment":
        if is_department_structures_index_source(metadata):
            return 85 if plan.requires_complete_answer else 55
        if is_department_structures_source(metadata):
            return 70
        if source_family == "diem" or "diem.unisa.it" in url:
            return 65
        if source_family == "teacher_profile" or "docenti.unisa.it" in url:
            return 35

    if "coursecatalogue" in url:
        return 70
    if "corsi.unisa.it" in url:
        return 55
    if "docenti.unisa.it" in url:
        return 50
    if "diem.unisa.it" in url:
        return 45
    return 20


def course_match_score(plan: QueryPlan, haystack: str) -> float:
    course = normalize_text(plan.entities.get("course"))
    if not course:
        return 0.0

    course_tokens = tokens(course)
    if not course_tokens:
        return 0.0

    haystack_tokens = tokens(haystack)
    overlap = len(course_tokens.intersection(haystack_tokens))
    # Richiediamo almeno due token per evitare che "ingegneria informatica"
    # agganci per errore "ingegneria dell'informazione per medicina digitale".
    if len(course_tokens) >= 2 and overlap < 2:
        return 0.0
    return min(24.0, 8.0 * overlap)


def teacher_match_score(plan: QueryPlan, haystack: str) -> float:
    teacher = normalize_text(plan.entities.get("teacher"))
    if not teacher:
        return 0.0

    teacher_tokens = tokens(teacher)
    if not teacher_tokens:
        return 0.0

    haystack_tokens = tokens(haystack)
    overlap = len(teacher_tokens.intersection(haystack_tokens))
    if len(teacher_tokens) >= 2 and overlap < 2:
        return 0.0
    return min(22.0, 9.0 * overlap)


def lab_catalog_query(query: str, plan: QueryPlan) -> bool:
    if plan.task_type != "lab_equipment":
        return False

    normalized_query = normalize_text(query)
    return plan.requires_complete_answer and not any(
        marker in normalized_query for marker in LAB_EQUIPMENT_MARKERS
    )


def trusted_lab_catalog_source(metadata: dict[str, Any]) -> bool:
    source_family = normalize_text(metadata.get("source_family"))
    topic_family = normalize_text(metadata.get("topic_family"))
    section_heading = normalize_text(metadata.get("section_heading"))
    return (
        is_department_structures_index_source(metadata)
        or (
            source_family == "diem"
            and topic_family == "laboratori"
            and "laboratori" in section_heading
            and "?" not in source_url(metadata)
        )
    )


def publication_recency_score(plan: QueryPlan, metadata: dict[str, Any]) -> float:
    if plan.task_type != "teacher_publications":
        return 0.0

    year = int_value(metadata.get("publication_year") or metadata.get("year"))
    if year <= 0:
        return 0.0

    return min(36.0, max(0.0, float(year - 1990)) * 1.2)


def official_document_recency_score(plan: QueryPlan, metadata: dict[str, Any]) -> float:
    if plan.task_type != "official_docs":
        return 0.0

    candidate_years: list[int] = []
    for key in ("year", "academic_year"):
        year = int_value(metadata.get(key))
        if year > 0:
            candidate_years.append(year)

    document_years = metadata.get("document_years")
    if isinstance(document_years, list):
        candidate_years.extend(int_value(year) for year in document_years)

    if not candidate_years:
        return 0.0

    year = max(candidate_years)
    return min(18.0, max(0.0, float(year - 2010)) * 1.25)


def chunk_task_score(plan: QueryPlan, metadata: dict[str, Any], haystack: str) -> tuple[float, str]:
    # Score per tipo di evidenza, non boost globale. Il risultato entra in RRF
    # come una lista separata e rimane ispezionabile tramite structured_reason.
    chunk_kind = str(metadata.get("chunk_kind") or "")
    topic_family = str(metadata.get("topic_family") or "")
    score = 0.0
    reasons: list[str] = []

    if plan.task_type == "study_plan":
        if chunk_kind in {"study_plan", "course_syllabus"}:
            score += 70
            reasons.append(f"{chunk_kind} chunk")
        elif "coursecatalogue" in haystack and re.search(r"\b[123]\s+anno\b", haystack):
            score += 45
            reasons.append("coursecatalogue year section")
    elif plan.task_type == "course_catalog":
        if chunk_kind in {"course_info", "course_syllabus", "study_plan"} or topic_family == "didattica":
            score += 42
            reasons.append("course evidence")
    elif plan.task_type == "office_hours":
        if chunk_kind == "office_hours":
            score += 65
            reasons.append("office hours chunk")
    elif plan.task_type == "teacher_publications":
        if chunk_kind == "publication_summary":
            score += 68
            reasons.append("publication summary")
        elif chunk_kind == "teacher_publications_page":
            score += 35
            reasons.append("publication page")
    elif plan.task_type == "lab_equipment":
        if chunk_kind == "lab_equipment" or topic_family == "laboratori":
            score += 58
            reasons.append("lab/equipment evidence")
            if is_department_structures_index_source(metadata):
                if plan.requires_complete_answer:
                    score += 32
                    reasons.append("official department structures index")
                else:
                    score += 8
                    reasons.append("official department structures index")
            elif is_department_structures_source(metadata):
                score += 18
                reasons.append("official department structures detail")
    elif plan.task_type == "international":
        if chunk_kind == "erasmus" or topic_family == "international":
            score += 48
            reasons.append("international evidence")
    elif plan.task_type == "phd":
        if chunk_kind == "phd" or topic_family == "dottorati":
            score += 48
            reasons.append("phd evidence")
    elif plan.task_type == "third_mission":
        if topic_family == "terza_missione":
            score += 45
            reasons.append("third mission evidence")
    elif plan.task_type == "research":
        if topic_family == "ricerca" or chunk_kind == "teacher_projects":
            score += 40
            reasons.append("research evidence")
    elif plan.task_type == "official_docs":
        if chunk_kind in {"official_document", "official_document_summary"}:
            score += 46
            reasons.append("official document evidence")
            if normalize_text(metadata.get("document_type")) == "regolamento":
                score += 18
                reasons.append("regulation document")
            if any(marker in haystack for marker in OFFICIAL_FINAL_EXAM_MARKERS):
                score += 26
                reasons.append("final exam/vote evidence")
            if re.search(r"\b(?:voto_base|media_pesata|formula|calcolo)\b", haystack):
                score += 10
                reasons.append("calculation evidence")

    return score, ", ".join(reasons)


def course_year_score(plan: QueryPlan, metadata: dict[str, Any], haystack: str) -> float:
    wanted = plan.filters.get("course_years")
    if not isinstance(wanted, list) or not wanted:
        return 0.0

    metadata_year = int_value(metadata.get("course_year"))
    if metadata_year in wanted:
        return 18.0

    if any(re.search(rf"\b{year}\s+anno\b", haystack) for year in wanted):
        return 12.0

    return 0.0


def course_level_score(plan: QueryPlan, haystack: str) -> float:
    wanted_level = normalize_text(plan.filters.get("course_level"))
    if wanted_level not in {"magistrale", "triennale"}:
        return 0.0

    magistrale_markers = (
        "laurea magistrale",
        "lauree magistrali",
        "corso di laurea magistrale",
        "classe lm",
        "lm-",
    )
    triennale_markers = (
        "laurea triennale",
        "corso di laurea in ingegneria informatica",
        "classe l-",
        "classe l ",
    )
    has_magistrale = any(marker in haystack for marker in magistrale_markers)
    has_triennale = any(marker in haystack for marker in triennale_markers) and not has_magistrale

    if wanted_level == "magistrale":
        if has_magistrale:
            return 24.0
        if has_triennale:
            return -38.0
    elif wanted_level == "triennale":
        if has_triennale:
            return 24.0
        if has_magistrale:
            return -38.0

    return 0.0


def structured_retrieve(
    query: str,
    plan: QueryPlan,
    chunks: tuple[dict[str, Any], ...],
    limit: int = 30,
) -> list[StructuredEvidence]:
    if not plan.needs_structured_data and not plan.requires_complete_answer:
        return []

    query_tokens = tokens(query)
    matches: list[StructuredEvidence] = []

    for chunk in chunks:
        metadata = flatten_chunk_metadata(chunk)
        haystack = metadata_haystack(chunk)
        task_score, task_reason = chunk_task_score(plan, metadata, haystack)
        if task_score <= 0:
            continue

        score = task_score
        reason_parts = [task_reason]

        course_score = course_match_score(plan, haystack)
        if course_score:
            score += course_score
            reason_parts.append("course entity match")

        teacher_score = teacher_match_score(plan, haystack)
        if teacher_score:
            score += teacher_score
            reason_parts.append("teacher entity match")

        year_score = course_year_score(plan, metadata, haystack)
        if year_score:
            score += year_score
            reason_parts.append("course year match")

        level_score = course_level_score(plan, haystack)
        if level_score:
            score += level_score
            reason_parts.append("course level match" if level_score > 0 else "course level mismatch")

        recency_score = publication_recency_score(plan, metadata)
        if recency_score:
            score += recency_score
            reason_parts.append("publication recency")

        official_recency_score = official_document_recency_score(plan, metadata)
        if official_recency_score:
            score += official_recency_score
            reason_parts.append("official document recency")

        # I filtri entità sono stretti solo quando l'utente ha nominato
        # esplicitamente corso/docente. Senza entità, il retriever rimane recall-oriented.
        if plan.task_type in {"study_plan", "course_catalog"} and plan.entities.get("course") and not course_score:
            continue

        if plan.task_type in {"office_hours", "teacher_publications", "teacher_profile"} and plan.entities.get("teacher") and not teacher_score:
            continue

        if lab_catalog_query(query, plan) and not trusted_lab_catalog_source(metadata):
            continue

        overlap = len(query_tokens.intersection(tokens(haystack)))
        score += min(14.0, float(overlap))
        score += source_priority(metadata, plan) / 10.0
        score += min(6.0, int_value(metadata.get("academic_year") or metadata.get("year")) / 1000.0)

        matches.append(
            StructuredEvidence(
                chunk=chunk,
                score=score,
                reason="; ".join(part for part in reason_parts if part),
            )
        )

    matches.sort(key=lambda item: item.score, reverse=True)
    return matches[:limit]
