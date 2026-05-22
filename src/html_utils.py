"""
Utility HTML usate dalla discovery.

Responsabilità:
- risolvere href relativi secondo le convenzioni UNISA;
- leggere il canonical da una BeautifulSoup già disponibile;
- estrarre link attraversabili da una pagina HTML usando il contesto di scope
  corrente, inclusa la whitelist docente.

La decisione se un URL sia nello scope resta in url_filters.py.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup

from url_filters import (
    can_traverse_url,
    decoded_rescue_path_segments,
    directory_person_matricola,
    config_list,
    normalize_url,
)


def resolve_href(base_url: str, href: str) -> str:
    """Risolve un href tenendo conto delle convenzioni dei siti UNISA.

    Diverse pagine UNISA espongono allegati come "uploads/..." senza slash
    iniziale, ma quei path sono di fatto root-relative. urljoin li
    risolverebbe sotto il path corrente, generando URL inesistenti.
    """
    href = href.strip()
    if href.startswith("uploads/"):
        href = "/" + href
    else:
        href = re.sub(r"^/?[^/?#]+/uploads/", "/uploads/", href, count=1)
    if is_plain_relative_href(href):
        href_first_segment = href.split("/", 1)[0].lower()
        if is_course_numeric_alias_segment(href_first_segment):
            href = "/" + href
        elif is_course_rescue_wrapped_path(base_url):
            rescue_segments = decoded_rescue_path_segments(base_url)
            if rescue_segments and href_first_segment != rescue_segments[0]:
                href = f"/{rescue_segments[0]}/{href}"
            else:
                href = "/" + href
    return urljoin(base_url, href)


def is_course_rescue_wrapped_path(url: str) -> bool:
    """True se il base URL e' una rescue page che incapsula un path corso."""
    path = urlparse(url).path.rstrip("/").lower()
    return path.startswith((
        "/unisa-rescue-page/dettaglio/",
        "/unisa-rescue-page/search/",
    ))


def is_plain_relative_href(href: str) -> bool:
    """True per link relativi di navigazione, non query/anchor/asset assoluti."""
    parsed = urlparse(href)
    return (
        not parsed.scheme
        and not parsed.netloc
        and bool(parsed.path)
        and not href.startswith(("/", "#", "?"))
        and not href.startswith(("./", "../"))
    )


def is_course_numeric_alias_segment(segment: str) -> bool:
    """True per alias corso numerici usati come primo segmento da corsi.unisa.it."""
    return bool(re.fullmatch(r"\d{16}", segment))


def canonical_teacher_profile_url(
    url: str,
    base_url: str,
    config: dict,
    authorized_directory_bridge: bool,
) -> str:
    """Converte il link rubrica -> docenti nello URL numerico del profilo.

    Le pagine rubrica dei docenti DIEM espongono spesso link come
    `docenti.unisa.it/nome.cognome`, che sul sito attuale possono rispondere
    con la lista globale dei docenti. Dalla rubrica conosciamo invece la
    matricola, quindi usiamo direttamente `docenti.unisa.it/<matricola>/home`.
    """
    if not authorized_directory_bridge:
        return url

    matricola = directory_person_matricola(base_url, config)
    if not matricola:
        return url

    parsed = urlparse(url)
    teacher_domain = str(
        config.get("scope", {}).get("teacher_domain", "docenti.unisa.it")
    ).lower()
    if parsed.netloc.lower() != teacher_domain:
        return url

    segments = [segment for segment in parsed.path.split("/") if segment]
    if len(segments) != 1 or segments[0].isdigit():
        return url

    return normalize_url(f"https://{teacher_domain}/{matricola}/home")


def extract_canonical_from_soup(soup: BeautifulSoup, fallback_url: str) -> str:
    """Estrae il canonical riusando una soup già parsata."""
    tag = soup.find("link", rel="canonical")
    if tag and tag.get("href"):
        return normalize_url(urljoin(fallback_url, tag["href"].strip()))
    return normalize_url(fallback_url)


def extract_link_entries_from_soup(
    soup: BeautifulSoup,
    base_url: str,
    config: dict,
    allowed_teacher_profiles: set[str] | None = None,
    authorized_directory_bridge: bool = False,
) -> list[dict[str, str]]:
    """Estrae URL traversabili mantenendo anche il testo utile del link.

    Il testo del link serve soprattutto per classificare i PDF: molti allegati
    hanno filename poco parlanti, ma sono descritti chiaramente nell'anchor.
    """
    links: dict[str, dict[str, str]] = {}
    context = {
        "discovered_from": base_url,
        "allowed_teacher_profiles": allowed_teacher_profiles or set(),
        "authorized_directory_bridge": authorized_directory_bridge,
    }

    for tag in soup.find_all("a", href=True):
        absolute_url = resolve_href(base_url, tag["href"])
        if urlparse(absolute_url).scheme not in {"http", "https"}:
            continue

        url = canonical_teacher_profile_url(
            normalize_url(absolute_url),
            base_url,
            config,
            authorized_directory_bridge,
        )
        label_parts = [tag.get_text(" ", strip=True), str(tag.get("title", "")).strip()]
        label = " ".join(part for part in label_parts if part)
        if not is_allowed_diem_doctorate_bandi_detail(url, label, config):
            continue

        ok, _ = can_traverse_url(url, config, context)
        if ok:
            links[url] = {
                "url": url,
                "text": label,
            }

    return list(links.values())


def configured_doctorate_codes(config: dict) -> set[str]:
    """Codici dei dottorati DIEM configurati nello scope dei corsi."""
    return {
        str(path).lower()
        for path in config_list(config, "scope", "allowed_course_paths")
        if str(path).lower().startswith("dot")
    }


def is_diem_doctorate_bandi_detail_url(url: str) -> bool:
    """True per dettagli dei concorsi di dottorato esposti da /home/bandi."""
    parsed = urlparse(url)
    if parsed.netloc.lower() != "www.diem.unisa.it":
        return False
    if parsed.path.rstrip("/").lower() != "/home/bandi":
        return False

    params = parse_qs(parsed.query, keep_blank_values=True)
    return (
        params.get("modulo") == ["226"]
        and "bando" in params
        and "idConcorso" in params
    )


def is_allowed_diem_doctorate_bandi_detail(
    url: str,
    link_text: str,
    config: dict,
) -> bool:
    """Filtra i dettagli dottorato tenendo solo i corsi DIEM configurati."""
    if not is_diem_doctorate_bandi_detail_url(url):
        return True

    allowed_codes = configured_doctorate_codes(config)
    if not allowed_codes:
        return False

    text = link_text.lower()
    return any(code in text for code in allowed_codes)


def extract_links_from_soup(
    soup: BeautifulSoup,
    base_url: str,
    config: dict,
    allowed_teacher_profiles: set[str] | None = None,
    authorized_directory_bridge: bool = False,
) -> list[str]:
    """Compatibilità comoda quando serve soltanto la lista degli URL."""
    return [
        entry["url"]
        for entry in extract_link_entries_from_soup(
            soup,
            base_url,
            config,
            allowed_teacher_profiles,
            authorized_directory_bridge,
        )
    ]
