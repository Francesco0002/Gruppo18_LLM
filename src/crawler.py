from pathlib import Path
from urllib.parse import urlparse
import json
import re
import requests
from bs4 import BeautifulSoup


URLS_FILE = Path("data/urls.txt")
RAW_DIR = Path("data/raw")
METADATA_FILE = Path("data/metadata.json")


def safe_filename(url: str) -> str:
    """
    Converte un URL in un nome file sicuro.
    Esempio:
    https://www.diem.unisa.it/didattica
    diventa:
    www_diem_unisa_it_didattica.html
    """
    parsed = urlparse(url)
    name = parsed.netloc + parsed.path

    if name.endswith("/"):
        name = name[:-1]

    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    name = name.strip("_")

    if not name:
        name = "index"

    return name + ".html"


def read_urls() -> list[str]:
    """
    Legge gli URL dal file data/urls.txt.
    Ignora righe vuote e commenti.
    """
    if not URLS_FILE.exists():
        raise FileNotFoundError(f"File non trovato: {URLS_FILE}")

    urls = []

    with URLS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            if line.startswith("#"):
                continue

            urls.append(line)

    return urls


def download_page(url: str) -> tuple[str, str]:
    """
    Scarica una pagina HTML e restituisce:
    - html
    - titolo della pagina
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; DIEM-Chatbot-Project/1.0)"
    }

    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    html = response.text

    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else url

    return html, title


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    urls = read_urls()
    metadata = []

    print(f"URL trovati: {len(urls)}")

    for index, url in enumerate(urls, start=1):
        print(f"[{index}/{len(urls)}] Scarico: {url}")

        try:
            html, title = download_page(url)
            filename = safe_filename(url)

            output_path = RAW_DIR / filename
            output_path.write_text(html, encoding="utf-8")

            metadata.append({
                "url": url,
                "title": title,
                "file": filename,
                "type": "html"
            })

            print(f"  Salvato in: {output_path}")

        except requests.RequestException as error:
            print(f"  Errore durante il download di {url}: {error}")

    METADATA_FILE.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print(f"\nMetadata salvati in: {METADATA_FILE}")
    print("Crawler completato.")


if __name__ == "__main__":
    main()