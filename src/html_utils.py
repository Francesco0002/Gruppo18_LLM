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
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from url_filters import can_traverse_url, decoded_rescue_path_segments, normalize_url


def resolve_href(base_url: str, href: str) -> str:
    """Risolve un href tenendo conto delle convenzioni dei siti UNISA.

    Diverse pagine UNISA espongono allegati come "uploads/..." senza slash
    iniziale, ma quei path sono di fatto root-relative. urljoin li
    risolverebbe sotto il path corrente, generando URL inesistenti.
    """
    href = href.strip()
    if href.startswith("uploads/"):
        href = "/" + href
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

        url = normalize_url(absolute_url)
        ok, _ = can_traverse_url(url, config, context)
        if ok:
            label_parts = [tag.get_text(" ", strip=True), str(tag.get("title", "")).strip()]
            links[url] = {
                "url": url,
                "text": " ".join(part for part in label_parts if part),
            }

    return list(links.values())


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
