"""Utility HTML usate dalla discovery."""

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


def extract_links_from_soup(soup: BeautifulSoup, base_url: str, config: dict) -> list[str]:
    """Estrae link traversabili da una pagina HTML."""
    links: list[str] = []
    context = {"discovered_from": base_url}

    for tag in soup.find_all("a", href=True):
        absolute_url = resolve_href(base_url, tag["href"])
        if urlparse(absolute_url).scheme not in {"http", "https"}:
            continue

        url = normalize_url(absolute_url)
        ok, _ = can_traverse_url(url, config, context)
        if ok:
            links.append(url)

    return list(dict.fromkeys(links))
