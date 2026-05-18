"""
Regole condivise per classificare i PDF scoperti durante la discovery.

La discovery usa queste regole per decidere quali PDF negati da robots.txt
possono comunque entrare nella pipeline; ingest le riusa per costruire report
coerenti senza mantenere una seconda copia della stessa logica.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse


# Documenti stabili che arricchiscono la knowledge base anche fuori dalle news.
STABLE_DOCUMENT_PDF_KEYWORDS = (
    "regolamento",
    "linee-guida",
    "guida",
    "manifesto",
    "piano-di-studi",
)
# Documenti principali di opportunita correnti, utili per domande su bandi e borse.
MAIN_OPPORTUNITY_PDF_KEYWORDS = (
    "bando",
    "call",
    "premio",
    "borsa",
    "concorso",
)
# Materiali didattici pratici utili a rispondere su tempi, accesso e requisiti.
TEACHING_OPERATIONS_PDF_KEYWORDS = (
    "calendario",
    "schedule",
    "ofa",
    "requirements",
)
# Documenti internazionali che descrivono accordi e mobilita.
INTERNATIONAL_PROGRAM_PDF_KEYWORDS = (
    "accordo",
    "accordi",
    "erasmus",
)
# Documenti di qualita ed esiti dei corsi gia ammessi nello scope.
COURSE_EVIDENCE_PDF_KEYWORDS = (
    "sua-cds",
    "schede-sua",
    "almalaurea",
)
# Allegati o esiti accessori che da soli aggiungono poco alla RAG.
OPPORTUNITY_ATTACHMENT_HINTS = (
    "graduatoria",
    "approvazione-atti",
    "decreto",
    "verbale",
    "differimento",
    "elenco",
    "scorrimento",
    "allegato",
    "domanda",
    "modello",
    "locandina",
    "manifesto-elettorale",
    "scrutino",
    "elettorato",
    "faq",
    "modulo",
    "moduloadesione",
    "presentazione",
    "comunicato",
    "avviso-proroga",
    "proroga",
)
# Sezioni del sito da cui una deroga PDF e informativamente affidabile.
CENTRAL_DOCUMENT_SOURCE_SECTIONS = {
    "didattica",
    "dipartimento",
    "ricerca",
    "international",
}
# Slug di pagine rescue che descrivono calendari didattici utili e non allegati
# generici di news o bandi.
INFORMATIVE_CALENDAR_PARENT_HINTS = (
    "calendario-prove-in-itinere",
    "appelli-di-recupero",
)

REPORTABLE_PDF_KEYWORDS = (
    "bando",
    "graduatoria",
    "regolamento",
    "linee-guida",
    "guida",
    "manifesto",
    "piano-di-studi",
    "calendario",
    "schedule",
    "ofa",
    "requirements",
    "accordi",
    "erasmus",
    "sua-cds",
    "schede-sua",
    "almalaurea",
    "decreto",
)


def pdf_source_section_from_url(url: str) -> str:
    """Raggruppa una pagina sorgente in una sezione leggibile del corpus."""
    path = urlparse(url).path.lower().rstrip("/")
    if path.startswith("/home/bandi"):
        return "home_bandi"
    if path.startswith("/home/news"):
        return "home_news"
    if path.startswith("/home/eventi"):
        return "home_eventi"
    if path.startswith("/didattica") or "/didattica/" in path:
        return "didattica"
    if path.startswith("/ricerca") or "/ricerca/" in path:
        return "ricerca"
    if path.startswith("/dipartimento") or "/dipartimento/" in path:
        return "dipartimento"
    if path.startswith("/international") or "/international/" in path:
        return "international"
    if path.startswith("/terza-missione"):
        return "terza_missione"
    return "other"


def normalized_keyword_text(*parts: str) -> str:
    """Normalizza filename e testo link per cercare keyword multi-parola."""
    text = " ".join(part for part in parts if part).lower()
    return re.sub(r"[\s_]+", "-", text)


def matched_pdf_keywords(pdf_url: str, link_text: str) -> list[str]:
    """Keyword utili alla policy trovate nel filename o nel testo del link."""
    filename = Path(urlparse(pdf_url).path).name
    haystack = normalized_keyword_text(filename, link_text)
    keywords = (
        STABLE_DOCUMENT_PDF_KEYWORDS
        + MAIN_OPPORTUNITY_PDF_KEYWORDS
        + TEACHING_OPERATIONS_PDF_KEYWORDS
        + INTERNATIONAL_PROGRAM_PDF_KEYWORDS
        + COURSE_EVIDENCE_PDF_KEYWORDS
        + ("decreto",)
    )
    return [keyword for keyword in keywords if keyword in haystack]


def is_main_opportunity_text(text: str) -> bool:
    """True per opportunita principali, non per allegati o risultati accessori."""
    # I termini positivi da soli sono troppo larghi: molti allegati di bandi
    # contengono comunque "bando" o "borsa" nel filename.
    return (
        any(keyword in text for keyword in MAIN_OPPORTUNITY_PDF_KEYWORDS)
        and not any(hint in text for hint in OPPORTUNITY_ATTACHMENT_HINTS)
    )


def is_main_opportunity_pdf(pdf_url: str, link_text: str) -> bool:
    """True per bandi/call principali, non per allegati e risultati accessori."""
    filename = Path(urlparse(pdf_url).path).name
    return is_main_opportunity_text(normalized_keyword_text(filename, link_text))


def has_strong_decree_context(pdf_url: str, link_text: str) -> bool:
    """True per decreti centrali descritti da un contesto non generico."""
    filename = Path(urlparse(pdf_url).path).name
    haystack = normalized_keyword_text(filename, link_text)
    if "decreto" not in haystack:
        return False
    normalized_label = normalized_keyword_text(link_text)
    return normalized_label not in {"", "pdf", "documento", "decreto"}


def has_policy_keyword(keywords: list[str], allowed_keywords: tuple[str, ...]) -> bool:
    """True se una famiglia semantica e presente tra le keyword riconosciute."""
    return bool(set(keywords) & set(allowed_keywords))


def is_informative_calendar_parent(parent_url: str) -> bool:
    """True solo per pagine rescue DIEM con un contesto didattico esplicito."""
    parsed = urlparse(parent_url)
    path = parsed.path.lower().rstrip("/")
    return (
        parsed.netloc.lower() == "www.diem.unisa.it"
        and path.startswith("/unisa-rescue-page/dettaglio/")
        and any(hint in path for hint in INFORMATIVE_CALENDAR_PARENT_HINTS)
    )


def pdf_download_exception(
    pdf_url: str,
    parent_url: str,
    link_text: str,
    config: dict,
) -> tuple[bool, str, list[str], str | None]:
    """Decide se un PDF negato da robots puo entrare nelle deroghe ristrette."""
    section = pdf_source_section_from_url(parent_url)
    pdf_domain = urlparse(pdf_url).netloc.lower()
    allowed_pdf_domains = {
        str(domain).lower()
        for domain in config["crawler"].get("pdf_allowed_domains", [])
    }
    keywords = matched_pdf_keywords(pdf_url, link_text)
    # Le condizioni restano separate e leggibili di proposito: ogni ramo
    # rappresenta una famiglia documentale ammessa per una ragione diversa e
    # produce un motivo esplicito usato poi nei report di copertura.
    stable_document = (
        pdf_domain in allowed_pdf_domains
        and section in CENTRAL_DOCUMENT_SOURCE_SECTIONS
        and has_policy_keyword(keywords, STABLE_DOCUMENT_PDF_KEYWORDS)
    )
    teaching_operations_document = (
        pdf_domain in allowed_pdf_domains
        and section == "didattica"
        and has_policy_keyword(keywords, TEACHING_OPERATIONS_PDF_KEYWORDS)
    )
    informative_calendar_document = (
        pdf_domain in allowed_pdf_domains
        and keywords == ["calendario"]
        and is_informative_calendar_parent(parent_url)
    )
    international_program_document = (
        pdf_domain in allowed_pdf_domains
        and section == "international"
        and has_policy_keyword(keywords, INTERNATIONAL_PROGRAM_PDF_KEYWORDS)
    )
    course_evidence_document = (
        pdf_domain in allowed_pdf_domains
        and urlparse(parent_url).netloc.lower()
        == str(config["scope"].get("course_domain", "corsi.unisa.it")).lower()
        and has_policy_keyword(keywords, COURSE_EVIDENCE_PDF_KEYWORDS)
    )
    opportunity_document = (
        pdf_domain in allowed_pdf_domains
        and section == "home_bandi"
        and is_main_opportunity_pdf(pdf_url, link_text)
    )
    central_decree = (
        pdf_domain in allowed_pdf_domains
        and section in CENTRAL_DOCUMENT_SOURCE_SECTIONS
        and has_strong_decree_context(pdf_url, link_text)
    )
    if stable_document:
        return True, section, keywords, "allowed_stable_document"
    if teaching_operations_document:
        return True, section, keywords, "allowed_teaching_operations_document"
    if informative_calendar_document:
        return True, section, keywords, "allowed_informative_calendar_document"
    if international_program_document:
        return True, section, keywords, "allowed_international_program_document"
    if course_evidence_document:
        return True, section, keywords, "allowed_course_evidence_document"
    if opportunity_document:
        return True, section, keywords, "allowed_opportunity_document"
    if central_decree:
        return True, section, keywords, "allowed_central_decree"
    return False, section, keywords, None
