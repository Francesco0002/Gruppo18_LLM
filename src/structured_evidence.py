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

from chunk_metadata import flatten_chunk_metadata
from query_planner import QueryPlan


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


def source_priority(metadata: dict[str, Any]) -> int:
    # Priorità di affidabilità/fonte: CourseCatalogue è primario per didattica,
    # corsi.unisa.it per pagine corso/documenti, docenti.unisa.it per persone.
    url = normalize_text(metadata.get("source_url") or metadata.get("document_url"))
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
            score += 42
            reasons.append("official document evidence")

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

        # I filtri entità sono stretti solo quando l'utente ha nominato
        # esplicitamente corso/docente. Senza entità, il retriever rimane recall-oriented.
        if plan.task_type in {"study_plan", "course_catalog"} and plan.entities.get("course") and not course_score:
            continue

        if plan.task_type in {"office_hours", "teacher_publications", "teacher_profile"} and plan.entities.get("teacher") and not teacher_score:
            continue

        overlap = len(query_tokens.intersection(tokens(haystack)))
        score += min(14.0, float(overlap))
        score += source_priority(metadata) / 10.0
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
