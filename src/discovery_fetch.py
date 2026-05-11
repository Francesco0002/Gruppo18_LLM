"""
Funzioni di rete usate dalla discovery.

Responsabilità:
- applicare il rate limit per dominio;
- leggere e memorizzare robots.txt;
- scaricare pagine/sitemap con httpx;
- riconoscere HTML, PDF e risposte troppo grandi.

Questo modulo non decide lo scope del progetto: per quello usa le funzioni di
url_filters.py.
"""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from discovery_models import FetchResult
from url_filters import can_traverse_url, normalize_url, parse_mime


HTML_MIMES = {"text/html", "application/xhtml+xml"}


class DomainRateLimiter:
    """Rate limit semplice: aspetta tra due richieste allo stesso dominio."""

    def __init__(self, requests_per_second: float) -> None:
        self.delay = 1 / requests_per_second
        self.last_request: dict[str, float] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    async def wait(self, domain: str) -> None:
        """Rispetta il delay configurato per il dominio."""
        lock = self.locks.setdefault(domain, asyncio.Lock())

        async with lock:
            now = asyncio.get_running_loop().time()
            last = self.last_request.get(domain)

            if last is not None:
                wait_time = self.delay - (now - last)
                if wait_time > 0:
                    await asyncio.sleep(wait_time)

            self.last_request[domain] = asyncio.get_running_loop().time()


class RobotsCache:
    """Carica robots.txt una sola volta per dominio."""

    def __init__(self, user_agent: str) -> None:
        self.user_agent = user_agent
        self.parsers: dict[str, RobotFileParser] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    async def can_fetch(self, url: str) -> bool:
        """True se robots.txt permette di scaricare l'URL."""
        domain = urlparse(url).netloc
        lock = self.locks.setdefault(domain, asyncio.Lock())

        async with lock:
            if domain not in self.parsers:
                parser = RobotFileParser()
                parser.set_url(f"https://{domain}/robots.txt")
                try:
                    await asyncio.to_thread(parser.read)
                except Exception:
                    # Se robots.txt non è raggiungibile, non blocchiamo il crawl.
                    parser.allow_all = True
                self.parsers[domain] = parser

        return self.parsers[domain].can_fetch(self.user_agent, url)


async def fetch_url(
    client: httpx.AsyncClient,
    url: str,
    limiter: DomainRateLimiter,
) -> httpx.Response:
    """Scarica un URL rispettando il rate limit per dominio."""
    await limiter.wait(urlparse(url).netloc)
    response = await client.get(url, follow_redirects=True)
    response.raise_for_status()
    return response


async def fetch_discovery_candidate(
    client: httpx.AsyncClient,
    url: str,
    limiter: DomainRateLimiter,
    max_bytes: int,
) -> FetchResult:
    """Scarica il body solo se la risposta è HTML e sotto soglia."""
    await limiter.wait(urlparse(url).netloc)

    async with client.stream("GET", url, follow_redirects=True) as response:
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        mime = parse_mime(content_type)
        content_length_header = response.headers.get("Content-Length")
        content_length = (
            int(content_length_header)
            if content_length_header and content_length_header.isdigit()
            else None
        )
        too_large = bool(content_length and content_length > max_bytes)

        if too_large or mime not in HTML_MIMES:
            return FetchResult(
                final_url=normalize_url(str(response.url)),
                content_type=content_type,
                mime=mime,
                content_length=content_length,
                too_large=too_large,
                html=None,
            )

        body = await response.aread()
        if len(body) > max_bytes:
            return FetchResult(
                final_url=normalize_url(str(response.url)),
                content_type=content_type,
                mime=mime,
                content_length=len(body),
                too_large=True,
                html=None,
            )

        return FetchResult(
            final_url=normalize_url(str(response.url)),
            content_type=content_type,
            mime=mime,
            content_length=len(body),
            too_large=False,
            html=response.text,
        )


def parse_sitemap(xml_text: str) -> list[str]:
    """Estrae i valori <loc> da una sitemap XML."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    urls: list[str] = []
    for element in root.iter():
        tag = element.tag.split("}")[-1]
        if tag == "loc" and element.text:
            urls.append(element.text.strip())

    return urls


def is_sitemap_url(url: str) -> bool:
    """True se l'URL punta a un file sitemap XML."""
    name = Path(urlparse(url).path.lower()).name
    return name.startswith("sitemap") and name.endswith(".xml")


async def fetch_sitemap_urls(
    client: httpx.AsyncClient,
    config: dict,
    limiter: DomainRateLimiter,
) -> list[str]:
    """Legge le sitemap dei domini configurati e restituisce URL validi."""
    if not config["crawler"].get("use_sitemap", True):
        return []

    sitemap_queue = deque(
        f"https://{domain}/sitemap.xml"
        for domain in config["crawler"]["allowed_domains"]
    )
    seen_sitemaps: set[str] = set()
    found_urls: list[str] = []

    while sitemap_queue and len(seen_sitemaps) < 20:
        sitemap_url = sitemap_queue.popleft()
        if sitemap_url in seen_sitemaps:
            continue

        seen_sitemaps.add(sitemap_url)

        try:
            response = await fetch_url(client, sitemap_url, limiter)
        except httpx.HTTPError:
            continue

        for raw_url in parse_sitemap(response.text):
            url = normalize_url(raw_url)

            if is_sitemap_url(url):
                sitemap_queue.append(url)
                continue

            ok, _ = can_traverse_url(url, config)
            if ok:
                found_urls.append(url)

    return list(dict.fromkeys(found_urls))
