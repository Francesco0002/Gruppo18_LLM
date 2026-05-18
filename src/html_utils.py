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

from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from url_filters import can_traverse_url, normalize_url


def resolve_href(base_url: str, href: str) -> str:
    """Risolve un href tenendo conto delle convenzioni dei siti UNISA.

    Diverse pagine UNISA espongono allegati come "uploads/..." senza slash
    iniziale, ma quei path sono di fatto root-relative. urljoin li
    risolverebbe sotto il path corrente, generando URL inesistenti.
    """
    href = href.strip()
    if href.startswith("uploads/"):
        href = "/" + href
    return urljoin(base_url, href)


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
