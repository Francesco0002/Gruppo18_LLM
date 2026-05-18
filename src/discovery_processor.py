"""
Processing di un singolo URL durante la discovery.

Questo modulo contiene la logica che trasforma un CrawlItem in un
ProcessedDiscoveryItem:
- controllo robots.txt;
- fetch HTTP leggero;
- gestione redirect e canonical;
- distinzione HTML/PDF;
- estrazione link da HTML;
- registrazione immediata dei PDF linkati.
- propagazione del contesto di scope usato per autorizzare i profili docente.

`discover.py` resta così focalizzato sull'orchestrazione BFS, checkpoint e
progress bar.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from discovery_fetch import DomainRateLimiter, RobotsCache, fetch_discovery_candidate
from discovery_io import make_record
from discovery_models import CrawlItem, FetchResult, ProcessedDiscoveryItem
from html_utils import extract_canonical_from_soup, extract_link_entries_from_soup
from pipeline_types import DiscoveryRecord
from url_filters import can_index_url, can_traverse_url, is_diem_personnel_url, is_pdf_url


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
# Documenti internazionali che descrivono accordi e mobilità.
INTERNATIONAL_PROGRAM_PDF_KEYWORDS = (
    "accordo",
    "accordi",
    "erasmus",
)
# Documenti di qualità ed esiti dei corsi già ammessi nello scope.
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
# Sezioni del sito da cui una deroga PDF è informativamente affidabile.
CENTRAL_DOCUMENT_SOURCE_SECTIONS = {
    "didattica",
    "dipartimento",
    "ricerca",
    "international",
}


def filter_context(
    discovered_from: str,
    allowed_teacher_profiles: set[str] | None = None,
    authorized_directory_bridge: bool = False,
) -> dict[str, object]:
    """Crea il contesto da passare ai filtri che dipendono dalla sorgente."""
    return {
        "discovered_from": discovered_from,
        "allowed_teacher_profiles": allowed_teacher_profiles or set(),
        "authorized_directory_bridge": authorized_directory_bridge,
    }


def pick_document_url(
    canonical_url: str,
    final_url: str,
    config: dict,
    context: dict[str, str],
) -> tuple[str, str | None]:
    """Usa canonical_url come chiave solo se resta nello scope della discovery.

    Se il canonical dichiarato esce dallo scope, deduplichiamo sul final_url:
    è una scelta conservativa per non far collassare pagine in-scope diverse
    su una chiave fuori perimetro.
    """
    if not canonical_url or not canonical_url.startswith(("http://", "https://")):
        return final_url, "no_canonical"

    ok, reason = can_traverse_url(canonical_url, config, context)
    if ok:
        return canonical_url, None

    canonical_domain = urlparse(canonical_url).netloc.lower()
    final_domain = urlparse(final_url).netloc.lower()
    if canonical_domain and canonical_domain != final_domain:
        return final_url, "canonical_out_of_scope"

    return final_url, reason


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


def is_main_opportunity_pdf(pdf_url: str, link_text: str) -> bool:
    """True per bandi/call principali, non per allegati e risultati accessori."""
    filename = Path(urlparse(pdf_url).path).name
    haystack = normalized_keyword_text(filename, link_text)
    return (
        any(keyword in haystack for keyword in MAIN_OPPORTUNITY_PDF_KEYWORDS)
        and not any(hint in haystack for hint in OPPORTUNITY_ATTACHMENT_HINTS)
    )


def has_strong_decree_context(pdf_url: str, link_text: str) -> bool:
    """True per decreti centrali descritti da un contesto non generico."""
    filename = Path(urlparse(pdf_url).path).name
    haystack = normalized_keyword_text(filename, link_text)
    if "decreto" not in haystack:
        return False
    normalized_label = normalized_keyword_text(link_text)
    return normalized_label not in {"", "pdf", "documento", "decreto"}


def has_policy_keyword(keywords: list[str], allowed_keywords: tuple[str, ...]) -> bool:
    """True se una famiglia semantica è presente tra le keyword riconosciute."""
    return bool(set(keywords) & set(allowed_keywords))


def pdf_download_exception(
    pdf_url: str,
    parent_url: str,
    link_text: str,
    config: dict,
) -> tuple[bool, str, list[str], str | None]:
    """Decide se un PDF negato da robots può entrare nelle deroghe ristrette.

    La deroga è volutamente stretta:
    - il link deve arrivare da una pagina già ammessa nello scope del crawl;
    - il PDF deve stare su un dominio PDF esplicitamente ammesso;
    - documenti stabili: sezione centrale + keyword ad alto valore;
    - operatività didattica: calendari, OFA e requisiti da pagine didattiche;
    - internazionalizzazione: accordi ed Erasmus da pagine international;
    - qualità/esiti corso: SUA-CDS e AlmaLaurea linkati da corsi già in scope;
    - opportunità correnti: PDF principale da `home_bandi`, non allegato accessorio;
    - decreti: solo da sezioni centrali e con testo link non generico.
    """
    section = pdf_source_section_from_url(parent_url)
    pdf_domain = urlparse(pdf_url).netloc.lower()
    allowed_pdf_domains = {
        str(domain).lower()
        for domain in config["crawler"].get("pdf_allowed_domains", [])
    }
    keywords = matched_pdf_keywords(pdf_url, link_text)
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
    if international_program_document:
        return True, section, keywords, "allowed_international_program_document"
    if course_evidence_document:
        return True, section, keywords, "allowed_course_evidence_document"
    if opportunity_document:
        return True, section, keywords, "allowed_opportunity_document"
    if central_decree:
        return True, section, keywords, "allowed_central_decree"
    return False, section, keywords, None


def split_document_links(
    links: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    """Separa link HTML da attraversare e PDF da registrare subito."""
    traversal_links: list[str] = []
    pdf_links: list[dict[str, str]] = []

    for link in links:
        if is_pdf_url(link["url"]):
            pdf_links.append(link)
        else:
            traversal_links.append(link["url"])

    return traversal_links, pdf_links


def processed_item(
    record: DiscoveryRecord,
    traversal_links: list[str] | None = None,
    html: str | None = None,
    additional_visited: list[str] | None = None,
    linked_pdf_records: list[DiscoveryRecord] | None = None,
) -> ProcessedDiscoveryItem:
    """Costruisce un risultato di discovery con campi espliciti."""
    return ProcessedDiscoveryItem(
        record=record,
        traversal_links=traversal_links or [],
        html=html,
        additional_visited=additional_visited or [],
        linked_pdf_records=linked_pdf_records or [],
    )


def make_pdf_record(item: CrawlItem, status: str, **extra: object) -> DiscoveryRecord:
    """Crea un record PDF con campi comuni coerenti."""
    final_url = str(extra.pop("final_url", item.url))
    return make_record(
        item,
        "pdf",
        status,
        final_url=final_url,
        document_url=final_url,
        raw_path=None,
        **extra,
    )


def make_fetch_record(
    item: CrawlItem,
    source_type: str,
    status: str,
    final_url: str,
    candidate: FetchResult,
    **extra: object,
) -> DiscoveryRecord:
    """Crea un record per esiti legati a una risposta HTTP già classificata."""
    return make_record(
        item,
        source_type,
        status,
        final_url=final_url,
        document_url=final_url,
        content_type=candidate.content_type,
        mime=candidate.mime,
        raw_path=None,
        **extra,
    )


async def make_linked_pdf_records(
    pdf_links: list[dict[str, str]],
    parent_url: str,
    parent_depth: int,
    robots: RobotsCache | None,
    config: dict,
) -> list[DiscoveryRecord]:
    """Crea record PDF appena il link viene trovato in una pagina HTML.

    La regola ordinaria continua a rispettare robots.txt. L'unica eccezione è
    la whitelist stretta dei documenti ad alto valore linkati da sezioni DIEM
    centrali, utile per non perdere documentazione essenziale per la RAG.
    """
    records: list[DiscoveryRecord] = []

    for pdf_link in pdf_links:
        pdf_url = pdf_link["url"]
        link_text = pdf_link.get("text", "")
        item = CrawlItem(pdf_url, parent_depth + 1, parent_url)
        allowed_by_policy, section, keywords, allowed_reason = pdf_download_exception(
            pdf_url,
            parent_url,
            link_text,
            config,
        )
        status = "pending_download"
        robots_txt_denied = bool(robots and not await robots.can_fetch(pdf_url))
        decision = "allowed_by_robots"
        if robots_txt_denied and allowed_by_policy:
            decision = allowed_reason or "allowed_by_policy"
        elif robots_txt_denied:
            status = "robots_denied"
            decision = "blocked_by_robots"

        records.append(
            make_pdf_record(
                item,
                status,
                final_url=pdf_url,
                discovery_method="html_link",
                link_text=link_text,
                pdf_source_section=section,
                pdf_download_decision=decision,
                pdf_match_keywords=keywords,
                robots_txt_denied=robots_txt_denied,
            )
        )

    return records


async def process_item(
    item: CrawlItem,
    client: httpx.AsyncClient,
    limiter: DomainRateLimiter,
    robots: RobotsCache | None,
    config: dict,
    seen_documents: set[str],
    allowed_teacher_profiles: set[str] | None = None,
) -> ProcessedDiscoveryItem:
    """Scarica un URL e restituisce un risultato di discovery.

    Flusso dei filtri:
    - seed, sitemap e link estratti passano da can_traverse_url;
    - solo il document_url HTML finale passa da can_index_url;
    - i PDF restano nel manifest con status pending_download.

    I PDF linkati da una pagina HTML vengono registrati subito, senza attendere
    che vengano pescati dalla coda BFS.
    """
    authorized_directory_bridge = is_diem_personnel_url(item.discovered_from, config)
    context = filter_context(
        item.discovered_from,
        allowed_teacher_profiles,
        authorized_directory_bridge,
    )

    if is_pdf_url(item.url):
        ok, reason = can_traverse_url(item.url, config, context)
        if not ok:
            return processed_item(
                make_pdf_record(item, "out_of_scope", skip_reason=reason)
            )
        if robots and not await robots.can_fetch(item.url):
            return processed_item(make_pdf_record(item, "robots_denied"))

        return processed_item(make_pdf_record(item, "pending_download"))

    if robots and not await robots.can_fetch(item.url):
        return processed_item(make_record(item, "unknown", "robots_denied"))

    max_bytes = int(config["crawler"].get("max_html_bytes", 5_000_000))
    try:
        candidate = await fetch_discovery_candidate(client, item.url, limiter, max_bytes)
    except httpx.HTTPError as error:
        return processed_item(make_record(item, "html", "failed", error=str(error), raw_path=None))

    final_url = candidate.final_url
    additional_visited = [final_url] if final_url != item.url else []

    ok_final, final_reason = can_traverse_url(final_url, config, context)
    if not ok_final:
        record = make_fetch_record(
            item,
            "other",
            "redirected_out_of_scope",
            final_url=final_url,
            candidate=candidate,
            skip_reason=final_reason,
        )
        return processed_item(record, additional_visited=additional_visited)

    if final_url in seen_documents:
        record = make_fetch_record(
            item,
            "html",
            "duplicate_redirect",
            final_url=final_url,
            candidate=candidate,
            duplicate_of=final_url,
        )
        return processed_item(record, additional_visited=additional_visited)

    if is_pdf_url(final_url) or candidate.mime == "application/pdf":
        record = make_pdf_record(
            item,
            "pending_download",
            final_url=final_url,
            content_type=candidate.content_type,
            mime=candidate.mime,
        )
        return processed_item(record, additional_visited=additional_visited)

    if candidate.html is None:
        status = "too_large" if candidate.too_large else "non_html"
        source_type = "html" if candidate.too_large else "other"
        record = make_fetch_record(
            item,
            source_type,
            status,
            final_url=final_url,
            candidate=candidate,
            content_length=candidate.content_length,
        )
        return processed_item(record, additional_visited=additional_visited)

    soup = BeautifulSoup(candidate.html, "lxml")
    canonical_url = extract_canonical_from_soup(soup, final_url)
    document_url, canonical_skip_reason = pick_document_url(
        canonical_url,
        final_url,
        config,
        context,
    )
    if document_url not in (item.url, final_url):
        additional_visited.append(document_url)

    if document_url in seen_documents:
        record = make_record(
            item,
            "html",
            "duplicate_canonical",
            final_url=final_url,
            canonical_url=canonical_url,
            document_url=document_url,
            indexable=False,
            duplicate_of=document_url,
            raw_path=None,
            content_type=candidate.content_type,
            mime=candidate.mime,
            canonical_skip_reason=canonical_skip_reason,
        )
        return processed_item(record, additional_visited=additional_visited)

    all_links = extract_link_entries_from_soup(
        soup,
        final_url,
        config,
        allowed_teacher_profiles,
        authorized_directory_bridge,
    )
    links, pdf_links = split_document_links(all_links)
    linked_pdf_records = await make_linked_pdf_records(
        pdf_links,
        final_url,
        item.depth,
        robots,
        config,
    )
    indexable, index_reason = can_index_url(document_url, config, context)
    status = "ok" if indexable else "not_indexable"

    record = make_record(
        item,
        "html",
        status,
        final_url=final_url,
        canonical_url=canonical_url,
        document_url=document_url,
        indexable=indexable,
        raw_path=None,
        content_type=candidate.content_type,
        mime=candidate.mime,
        links_found=len(all_links),
        pdf_links_found=len(pdf_links),
        canonical_skip_reason=canonical_skip_reason,
        index_skip_reason=None if indexable else index_reason,
    )
    return processed_item(
        record,
        traversal_links=links,
        html=candidate.html if indexable else None,
        additional_visited=additional_visited,
        linked_pdf_records=linked_pdf_records,
    )
