from pathlib import Path
from urllib.parse import urlparse, urljoin, urldefrag
from collections import deque
import json
import re
import requests
from bs4 import BeautifulSoup
import hashlib


BASE_DIR = Path(__file__).resolve().parent.parent

URLS_FILE = BASE_DIR / "data" / "urls.txt"
RAW_HTML_DIR = BASE_DIR / "data" / "raw" / "html"
RAW_PDF_DIR = BASE_DIR / "data" / "raw" / "pdf"
METADATA_FILE = BASE_DIR / "data" / "metadata.json"

ALLOWED_DOMAINS = {
    "www.diem.unisa.it",
    "diem.unisa.it",
    "docenti.unisa.it",
    "corsi.unisa.it",
}

PDF_ALLOWED_DOMAINS = {
    "www.diem.unisa.it",
    "diem.unisa.it",
    "docenti.unisa.it",
    "corsi.unisa.it",
    "www.unisa.it",
    "web.unisa.it",
}

MAX_DEPTH = 2
MAX_PAGES = 300
TIMEOUT = 20

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; DIEM-Chatbot-Project/1.0)"
}


def normalize_url(url: str) -> str:
    """
    Normalizza l'URL:
    - rimuove frammenti #...
    - forza https
    - rimuove slash finale, tranne sulla home
    """
    url, _ = urldefrag(url)
    parsed = urlparse(url.strip())

    scheme = "https"
    netloc = parsed.netloc.lower()
    path = parsed.path

    if path != "/" and path.endswith("/"):
        path = path[:-1]

    normalized = parsed._replace(
        scheme=scheme,
        netloc=netloc,
        path=path
    ).geturl()

    return normalized

def has_noisy_query(url: str) -> bool:
    """
    Riconosce URL con query troppo rumorose per una prima knowledge base.
    """
    parsed = urlparse(url)
    query = parsed.query.lower()

    if not query:
        return False

    noisy_params = [
        "progetto=",
        "tip=",
        "archive=",
        "category=",
        "stato=",
        "incubatore=",
        "avviso=",
    ]

    return any(param in query for param in noisy_params)


def is_allowed_url(url: str) -> bool:
    """
    Accetta solo URL appartenenti ai domini consentiti.
    Esclude pagine tecniche o troppo rumorose.
    """
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        return False

    if is_pdf_url(url):
        if parsed.netloc not in PDF_ALLOWED_DOMAINS:
            return False
    else:
        if parsed.netloc not in ALLOWED_DOMAINS:
            return False

    url_lower = url.lower()

    if "sitemap" in url_lower:
        return False

    if has_noisy_query(url):
        return False

    return True

def is_pdf_url(url: str) -> bool:
    """
    Controlla se l'URL sembra puntare a un PDF.
    """
    path = urlparse(url).path.lower()
    return path.endswith(".pdf")


def safe_filename(url: str, extension: str | None = None) -> str:
    """
    Converte un URL in un nome file molto corto.
    Usa dominio + hash dell'URL.
    """
    parsed = urlparse(url)

    domain = re.sub(r"[^a-zA-Z0-9]+", "_", parsed.netloc).strip("_")
    url_hash = hashlib.md5(url.encode("utf-8")).hexdigest()[:16]

    if extension is None:
        extension = ".html"

    return f"{domain}_{url_hash}{extension}"
def read_seed_urls() -> list[str]:
    """
    Legge gli URL iniziali dal file data/urls.txt.
    Ignora righe vuote e commenti.
    """
    if not URLS_FILE.exists():
        raise FileNotFoundError(f"File non trovato: {URLS_FILE}")

    urls = []

    with URLS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            url = normalize_url(line)

            if is_allowed_url(url):
                urls.append(url)
            else:
                print(f"URL ignorato perché fuori dominio: {url}")

    return urls


def download_url(url: str) -> requests.Response:
    """
    Scarica un URL e restituisce la response.
    """
    response = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
    response.raise_for_status()
    return response


def extract_title(html: str, fallback: str) -> str:
    """
    Estrae il titolo HTML della pagina.
    """
    soup = BeautifulSoup(html, "lxml")
    return soup.title.get_text(strip=True) if soup.title else fallback


def extract_links(html: str, base_url: str) -> set[str]:
    """
    Estrae tutti i link da una pagina HTML e li converte in URL assoluti.
    """
    soup = BeautifulSoup(html, "lxml")
    links = set()

    for tag in soup.find_all("a", href=True):
        href = tag["href"].strip()

        absolute_url = urljoin(base_url, href)
        absolute_url = normalize_url(absolute_url)

        if is_allowed_url(absolute_url):
            links.add(absolute_url)

    return links


def save_html(url: str, html: str) -> dict | None:
    """
    Salva una pagina HTML e restituisce i metadata.
    """
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)

    title = extract_title(html, url)

    if "web login service" in title.lower():
        print("  Pagina login ignorata")
        return None

    filename = safe_filename(url, ".html")
    output_path = RAW_HTML_DIR / filename

    output_path.write_text(html, encoding="utf-8")

    return {
        "url": url,
        "title": title,
        "file": str(output_path.relative_to(BASE_DIR)),
        "type": "html",
    }


def save_pdf(url: str, content: bytes) -> dict:
    """
    Salva un PDF e restituisce i metadata.
    """
    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)

    filename = safe_filename(url, ".pdf")
    output_path = RAW_PDF_DIR / filename

    output_path.write_bytes(content)

    return {
        "url": url,
        "title": filename,
        "file": str(output_path.relative_to(BASE_DIR)),
        "type": "pdf",
    }

def crawl() -> None:
    RAW_HTML_DIR.mkdir(parents=True, exist_ok=True)
    RAW_PDF_DIR.mkdir(parents=True, exist_ok=True)

    seed_urls = read_seed_urls()

    queue = deque((url, 0) for url in seed_urls)
    visited = set()
    metadata = []

    print(f"URL iniziali trovati: {len(seed_urls)}")
    print(f"Profondità massima: {MAX_DEPTH}")
    print()

    while queue:
        if len(visited) >= MAX_PAGES:
            print(f"Limite massimo pagine raggiunto: {MAX_PAGES}")
            break
        
        url, depth = queue.popleft()

        if url in visited:
            continue

        visited.add(url)

        print(f"[depth={depth}] Scarico: {url}")

        try:
            response = download_url(url)
            content_type = response.headers.get("Content-Type", "").lower()

            if is_pdf_url(url) or "application/pdf" in content_type:
                item = save_pdf(url, response.content)
                metadata.append(item)
                print(f"  PDF salvato: {item['file']}")
                continue

            html = response.text
            item = save_html(url, html)
            if item is None:
                continue
            metadata.append(item)
            print(f"  HTML salvato: {item['file']}")

            if depth < MAX_DEPTH:
                links = extract_links(html, url)

                new_links = 0

                for link in links:
                    if link not in visited:
                        queue.append((link, depth + 1))
                        new_links += 1

                print(f"  Sottolink trovati: {len(links)} | nuovi aggiunti: {new_links}")

        except requests.RequestException as error:
            print(f"  Errore download: {error}")
            
        except OSError as error:
            print(f"  Errore file system: {error}")

        except Exception as error:
            print(f"  Errore inatteso: {error}")

    METADATA_FILE.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(f"Crawling completato.")
    print(f"Documenti salvati: {len(metadata)}")
    print(f"Metadata salvati in: {METADATA_FILE.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    crawl()