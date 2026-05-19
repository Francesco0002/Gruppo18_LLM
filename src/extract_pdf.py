"""
Download ed estrazione Markdown dei PDF scoperti dalla discovery.

Input:
- data/discovered_urls.jsonl, record con type=pdf

Output:
- data/raw_pdf/<sh>/<hash>.pdf
- data/processed/markdown_raw/<sh>/<hash>.md
- data/processed/markdown/<sh>/<hash>.md
- data/processed/manifest.jsonl
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import httpx
import pymupdf4llm
from tqdm import tqdm

from discovery_fetch import SyncDomainRateLimiter
from discovery_io import load_config, project_path, validate_config
from markdown_cleaner import MIN_INDEXABLE_CHARS, clean_markdown
from pipeline_io import (
    append_jsonl_batch,
    content_hash,
    load_jsonl,
    now_iso,
    recent_successful_urls,
    relative_path,
    write_text,
)
from pipeline_types import DiscoveryRecord, ProcessedRecord
from url_filters import can_traverse_url, normalize_url, parse_mime


PDF_SIGNATURE = b"%PDF-"
PDF_MIMES = {"application/pdf", "application/x-pdf"}
DEFAULT_MAX_PDF_BYTES = 100 * 1024 * 1024
PDF_SIGNATURE_BUFFER_BYTES = 1024


@dataclass(frozen=True)
class PdfRuntimeSettings:
    """Parametri effettivi usati dalla fase PDF."""

    max_bytes: int
    skip_recent_days: int
    extraction_workers: int
    max_concurrent_downloads: int
    download_delay_seconds: float
    force_reextract: bool
    extraction_kwargs: dict[str, object]


def load_pdf_runtime_settings(config: dict) -> PdfRuntimeSettings:
    """Costruisce i parametri PDF dal config, con default retrocompatibili."""
    crawler = config["crawler"]
    # Copiamo il dict prima di modificarlo: il config caricato resta la fonte
    # di verita per stats e documentazione del run.
    extraction = dict(crawler.get("pdf_extraction", {}))
    extraction["use_layout"] = bool(extraction.get("use_layout", True))
    return PdfRuntimeSettings(
        max_bytes=int(crawler.get("pdf_max_bytes", DEFAULT_MAX_PDF_BYTES)),
        skip_recent_days=int(crawler.get("pdf_skip_recent_days", 7)),
        extraction_workers=int(crawler.get("pdf_extraction_workers", 4)),
        max_concurrent_downloads=int(crawler.get("pdf_max_concurrent_downloads", 3)),
        download_delay_seconds=float(crawler.get("pdf_download_delay_seconds", 0.0)),
        force_reextract=bool(crawler.get("pdf_force_reextract", False)),
        extraction_kwargs=extraction,
    )


def raw_pdf_path(url_hash: str, config: dict) -> Path:
    """Path del PDF raw."""
    base = project_path(config["paths"].get("raw_pdf_dir", "data/raw_pdf"))
    return base / url_hash[:2] / f"{url_hash}.pdf"


def raw_markdown_path(url_hash: str, config: dict) -> Path:
    """Path del Markdown raw estratto dal PDF."""
    base = project_path(config["paths"].get("processed_raw_markdown_dir", "data/processed/markdown_raw"))
    return base / url_hash[:2] / f"{url_hash}.md"


def markdown_path(url_hash: str, config: dict) -> Path:
    """Path del Markdown pulito e indicizzabile."""
    base = project_path(config["paths"].get("processed_markdown_dir", "data/processed/markdown"))
    return base / url_hash[:2] / f"{url_hash}.md"


def pending_pdf_records(config: dict) -> list[DiscoveryRecord]:
    """Record PDF scoperti in attesa di download."""
    discovered_path = project_path(config["paths"]["discovered_urls_file"])
    return [
        record
        for record in load_jsonl(discovered_path)
        if record.get("type") == "pdf" and record.get("status") == "pending_download"
    ]


def pdf_records(records: list[DiscoveryRecord], recent_urls: set[str]) -> list[DiscoveryRecord]:
    """Record PDF non recenti, deduplicati per URL mantenendo l'ordine."""
    selected: list[DiscoveryRecord] = []
    seen_urls: set[str] = set()
    for record in records:
        url = str(record.get("url", ""))
        if not url or url in recent_urls or url in seen_urls:
            continue
        selected.append(record)
        seen_urls.add(url)
    return selected


def pdf_settings_snapshot(settings: PdfRuntimeSettings) -> dict:
    """Serializza una sola volta i parametri effettivi usati nel run PDF."""
    return {
        "max_bytes": settings.max_bytes,
        "skip_recent_days": settings.skip_recent_days,
        "extraction_workers": settings.extraction_workers,
        "max_concurrent_downloads": settings.max_concurrent_downloads,
        "download_delay_seconds": settings.download_delay_seconds,
        "force_reextract": settings.force_reextract,
        "extraction_kwargs": settings.extraction_kwargs,
    }


def pdf_run_stats(
    *,
    pending_records: list[DiscoveryRecord],
    selected_records: list[DiscoveryRecord],
    ready_for_extraction: int,
    network_downloaded: int,
    reused_raw: int,
    downloaded_bytes: int,
    extracted_ok: int,
    failed: int,
    too_large: int,
    manifest_records: int,
    download_seconds: float,
    extraction_seconds: float,
    elapsed_seconds: float,
    failure_kinds: dict[str, int],
    executor_backend: str | None,
    settings: PdfRuntimeSettings,
    manifest_path: Path,
) -> dict:
    """Compone il payload stats senza duplicare il ramo vuoto e quello operativo."""
    throughput = round((extracted_ok / elapsed_seconds) * 60, 2) if elapsed_seconds else 0.0
    return {
        "pdf_pending_discovered": len(pending_records),
        "skipped_recent": len(pending_records) - len(selected_records),
        "pdf_candidates": len(selected_records),
        # `ready_for_extraction` è più preciso di `downloaded`: include anche raw
        # locali riusati, che non hanno generato traffico di rete in questa run.
        "ready_for_extraction": ready_for_extraction,
        "network_downloaded": network_downloaded,
        "reused_raw": reused_raw,
        "downloaded_bytes": downloaded_bytes,
        "extracted_ok": extracted_ok,
        "failed": failed,
        "too_large": too_large,
        "manifest_records": manifest_records,
        "download_seconds": round(download_seconds, 3),
        "extraction_seconds": round(extraction_seconds, 3),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "throughput_pdf_per_minute": throughput,
        "failure_kinds": failure_kinds,
        "executor_backend": executor_backend,
        "settings": pdf_settings_snapshot(settings),
        "manifest": relative_path(manifest_path),
    }


def pdf_target_is_allowed(record: DiscoveryRecord, final_url: str, config: dict) -> tuple[bool, str]:
    """Rivalida il target finale dopo eventuali redirect HTTP."""
    # I link PDF vengono autorizzati in discovery, ma il server puo redirigere
    # verso una destinazione diversa al momento del download. Rivalidiamo il
    # final URL per non trasformare un link in-scope in un fetch fuori perimetro.
    if "crawler" not in config or "pdf_allowed_domains" not in config["crawler"]:
        return True, "config_not_available"
    context = {"discovered_from": record.get("discovered_from", "seed")}
    return can_traverse_url(final_url, config, context)


def has_pdf_signature(data: bytes) -> bool:
    """Riconosce una signature PDF anche con pochi byte bianchi iniziali."""
    return data.lstrip().startswith(PDF_SIGNATURE)


def existing_raw_pdf_is_valid(path: Path, settings: PdfRuntimeSettings) -> bool:
    """True se un raw gia presente e riutilizzabile senza nuovo download."""
    try:
        stat = path.stat()
    except FileNotFoundError:
        return False
    if stat.st_size <= 0 or stat.st_size > settings.max_bytes:
        return False
    # La sola esistenza del file non basta: dopo crash o interruzioni potrebbe
    # essere rimasto un raw parziale con estensione .pdf ma contenuto invalido.
    with path.open("rb") as file:
        return has_pdf_signature(file.read(1024))


def pdf_error_download(
    record: DiscoveryRecord,
    error_kind: str,
    error: str,
    *,
    raw_pdf_path_value: str | None = None,
) -> dict:
    """Record intermedio per download falliti ma isolati dal resto del batch."""
    return {
        "record": record,
        "status": "failed",
        "raw_pdf_path": raw_pdf_path_value,
        "error_kind": error_kind,
        "error": error,
    }


def http_error_kind(error: httpx.HTTPError) -> str:
    """Classifica errori HTTP/download in categorie stabili per gli stats."""
    if isinstance(error, httpx.TimeoutException):
        return "timeout"
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        if status == 404:
            return "http_404"
        if 400 <= status < 500:
            return f"http_{status}"
        if 500 <= status < 600:
            return "http_5xx"
    return "http_error"


def _raw_pdf_download_result(
    record: DiscoveryRecord,
    output_path: Path,
    *,
    reused_raw: bool,
    network_downloaded: bool,
) -> dict:
    """Payload stabile per un PDF pronto all'estrazione."""
    return {
        "record": record,
        "status": "downloaded",
        "content_length": output_path.stat().st_size,
        "raw_pdf_path": relative_path(output_path),
        "reused_raw": reused_raw,
        "network_downloaded": network_downloaded,
    }


def _too_large_download(record: DiscoveryRecord, content_length: int) -> dict:
    """Payload comune per PDF scartati per dimensione."""
    return {
        "record": record,
        "status": "too_large",
        "content_length": content_length,
        "raw_pdf_path": None,
    }


def _content_length_too_large(headers: httpx.Headers, settings: PdfRuntimeSettings) -> int | None:
    """Ritorna Content-Length quando supera la soglia configurata."""
    content_length = headers.get("Content-Length")
    if content_length and content_length.isdigit() and int(content_length) > settings.max_bytes:
        return int(content_length)
    return None


def _temp_pdf_path(output_path: Path) -> Path:
    """Path temporaneo atomico associato al raw finale."""
    return output_path.with_name(f"{output_path.name}.tmp.{uuid4().hex}")


def _record_pdf_chunk(
    chunk: bytes,
    downloaded: int,
    signature_buffer: bytearray,
    settings: PdfRuntimeSettings,
) -> tuple[int, bool]:
    """Aggiorna contatore byte e signature buffer per uno stream PDF."""
    downloaded += len(chunk)
    if downloaded > settings.max_bytes:
        return downloaded, True
    if len(signature_buffer) < PDF_SIGNATURE_BUFFER_BYTES:
        remaining = PDF_SIGNATURE_BUFFER_BYTES - len(signature_buffer)
        signature_buffer.extend(chunk[:remaining])
    return downloaded, False


def _pdf_validation_error(
    record: DiscoveryRecord,
    content_type: str,
    signature: bytes,
) -> dict | None:
    """Valida MIME e signature evitando drift tra download sync e async."""
    has_signature = has_pdf_signature(signature)
    if content_type not in PDF_MIMES and not has_signature:
        return pdf_error_download(
            record,
            "invalid_pdf_response",
            f"invalid_pdf_response:{content_type or 'missing_content_type'}",
        )
    if not has_signature:
        return pdf_error_download(
            record,
            "invalid_pdf_signature",
            "invalid_pdf_signature",
        )
    return None


def _download_body_to_temp(
    response: object,
    temp_path: Path,
    settings: PdfRuntimeSettings,
) -> tuple[int, bytes, bool]:
    """Scarica una risposta sincrona su temp file e ritorna bytes/signature."""
    downloaded = 0
    signature_buffer = bytearray()
    too_large = False
    # Scriviamo su temp file per non lasciare raw definitivi corrotti se il
    # server interrompe lo stream o il file supera la soglia configurata.
    with temp_path.open("wb") as file:
        for chunk in response.iter_bytes():
            downloaded, too_large = _record_pdf_chunk(chunk, downloaded, signature_buffer, settings)
            if too_large:
                break
            file.write(chunk)
    return downloaded, bytes(signature_buffer), too_large


def download_pdf(
    record: DiscoveryRecord,
    config: dict,
    client: httpx.Client,
    settings: PdfRuntimeSettings | None = None,
    limiter: SyncDomainRateLimiter | None = None,
) -> dict:
    """Download sincrono compatibile con codice legacy e test locali."""
    settings = settings or load_pdf_runtime_settings(config)
    url = normalize_url(record["url"])
    url_hash = record["hash"]
    output_path = raw_pdf_path(url_hash, config)

    # Il riuso dei raw evita download ripetuti nelle riesecuzioni incrementali.
    # `force_reextract` permette comunque di invalidare volontariamente la cache.
    if not settings.force_reextract and existing_raw_pdf_is_valid(output_path, settings):
        return _raw_pdf_download_result(record, output_path, reused_raw=True, network_downloaded=False)

    if limiter is not None:
        limiter.wait(urlparse(url).netloc)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_pdf_path(output_path)
    try:
        with client.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            final_url = normalize_url(str(getattr(response, "url", url)))
            allowed, reason = pdf_target_is_allowed(record, final_url, config)
            if not allowed:
                return pdf_error_download(
                    record,
                    "redirected_out_of_scope",
                    f"redirected_out_of_scope:{reason}",
                )

            too_large_length = _content_length_too_large(response.headers, settings)
            if too_large_length is not None:
                return _too_large_download(record, too_large_length)

            downloaded, signature, too_large = _download_body_to_temp(response, temp_path, settings)
            if too_large:
                return _too_large_download(record, downloaded)

            content_type = parse_mime(response.headers.get("Content-Type", ""))
            validation_error = _pdf_validation_error(record, content_type, signature)
            if validation_error is not None:
                return validation_error

        temp_path.replace(output_path)
        return _raw_pdf_download_result(record, output_path, reused_raw=False, network_downloaded=True)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    finally:
        temp_path.unlink(missing_ok=True)


async def _download_body_to_temp_async(
    response: httpx.Response,
    temp_path: Path,
    settings: PdfRuntimeSettings,
) -> tuple[int, bytes, bool]:
    """Scarica una risposta asincrona su temp file e ritorna bytes/signature."""
    downloaded = 0
    signature_buffer = bytearray()
    too_large = False
    # Stessa logica della variante sincrona, ma usata dalla pipeline reale.
    with temp_path.open("wb") as file:
        async for chunk in response.aiter_bytes():
            downloaded, too_large = _record_pdf_chunk(chunk, downloaded, signature_buffer, settings)
            if too_large:
                break
            file.write(chunk)
    return downloaded, bytes(signature_buffer), too_large


async def download_pdf_async(
    record: DiscoveryRecord,
    config: dict,
    client: httpx.AsyncClient,
    settings: PdfRuntimeSettings,
) -> dict:
    """Scarica un PDF in modo asincrono, atomico e validato."""
    url = normalize_url(record["url"])
    url_hash = record["hash"]
    output_path = raw_pdf_path(url_hash, config)

    # I raw validi gia presenti entrano direttamente nella coda di estrazione:
    # questa e la scorciatoia che rende economiche le run incrementali.
    if not settings.force_reextract and existing_raw_pdf_is_valid(output_path, settings):
        return _raw_pdf_download_result(record, output_path, reused_raw=True, network_downloaded=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _temp_pdf_path(output_path)
    try:
        async with client.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            final_url = normalize_url(str(response.url))
            allowed, reason = pdf_target_is_allowed(record, final_url, config)
            if not allowed:
                return pdf_error_download(
                    record,
                    "redirected_out_of_scope",
                    f"redirected_out_of_scope:{reason}",
                )

            too_large_length = _content_length_too_large(response.headers, settings)
            if too_large_length is not None:
                return _too_large_download(record, too_large_length)

            downloaded, signature, too_large = await _download_body_to_temp_async(
                response,
                temp_path,
                settings,
            )
            if too_large:
                return _too_large_download(record, downloaded)

            content_type = parse_mime(response.headers.get("Content-Type", ""))
            validation_error = _pdf_validation_error(record, content_type, signature)
            if validation_error is not None:
                return validation_error

        temp_path.replace(output_path)
        return _raw_pdf_download_result(record, output_path, reused_raw=False, network_downloaded=True)
    except httpx.HTTPError as error:
        return pdf_error_download(record, http_error_kind(error), str(error))
    except OSError as error:
        return pdf_error_download(record, "filesystem_error", str(error))
    except Exception as error:
        return pdf_error_download(record, "download_error", str(error))
    finally:
        temp_path.unlink(missing_ok=True)


def extract_pdf_markdown(
    pdf_path: str,
    extraction_kwargs: dict[str, object] | None = None,
) -> tuple[str, str | None]:
    """Worker process: converte PDF in Markdown."""
    try:
        markdown = pymupdf4llm.to_markdown(
            str(project_path(pdf_path)),
            **(extraction_kwargs or {}),
        )
        return markdown or "", None
    except Exception as error:
        return "", str(error)


def base_manifest_record(record: DiscoveryRecord, status: str, crawled_at: str) -> ProcessedRecord:
    """Campi comuni dei record manifest PDF."""
    return {
        "source": "pdf",
        "status": status,
        "url": record["url"],
        "document_url": record.get("document_url", record["url"]),
        "hash": record["hash"],
        "last_crawled": crawled_at,
    }


def build_manifest_record(
    download: dict,
    markdown: str = "",
    error: str | None = None,
    error_kind: str | None = None,
    config: dict | None = None,
) -> ProcessedRecord:
    """Crea un record processed per un PDF."""
    record = download["record"]
    crawled_at = now_iso()

    if download["status"] == "too_large":
        manifest_record = base_manifest_record(record, "too_large", crawled_at)
        manifest_record.update(
            content_hash=None,
            raw_content_hash=None,
            raw_markdown_path=None,
            index_markdown_path=None,
            raw_pdf_path=download.get("raw_pdf_path"),
            content_length=download.get("content_length"),
            text_extracted=False,
            indexable=False,
        )
        return manifest_record

    if error or download["status"] == "failed":
        manifest_record = base_manifest_record(record, "failed", crawled_at)
        manifest_record.update(
            content_hash=None,
            raw_content_hash=None,
            raw_markdown_path=None,
            index_markdown_path=None,
            raw_pdf_path=download.get("raw_pdf_path"),
            content_length=download.get("content_length"),
            text_extracted=False,
            indexable=False,
            error=error or download.get("error", "download failed"),
            error_kind=error_kind or download.get("error_kind"),
        )
        return manifest_record

    if config is None:
        raise ValueError("config è richiesto per salvare il Markdown PDF.")

    raw_output_path = write_text(raw_markdown_path(record["hash"], config), markdown)
    metadata = {
        "url": record["url"],
        "document_url": record.get("document_url", record["url"]),
        "source": "pdf",
        "hash": record["hash"],
        "title": None,
        "breadcrumb": [],
        "last_crawled": crawled_at,
    }
    clean_body, index_markdown, quality = clean_markdown(
        markdown,
        source="pdf",
        metadata=metadata,
    )
    output_path = markdown_path(record["hash"], config)
    markdown_output = write_text(output_path, index_markdown)
    text_extracted = (
        len(clean_body.strip()) >= MIN_INDEXABLE_CHARS
        and "empty_structured_pdf" not in quality["clean_warnings"]
    )

    manifest_record = base_manifest_record(record, "ok", crawled_at)
    manifest_record.update(
        content_hash=content_hash(clean_body),
        raw_content_hash=content_hash(markdown),
        raw_markdown_path=raw_output_path,
        index_markdown_path=markdown_output,
        raw_pdf_path=download.get("raw_pdf_path"),
        content_length=download.get("content_length"),
        text_extracted=text_extracted,
        markdown_chars=len(clean_body),
        indexable=text_extracted,
        **quality,
    )
    return manifest_record


class AsyncPdfDownloadLimiter:
    """Limita concorrenza e cadenza dei download per dominio."""

    def __init__(self, max_concurrent_downloads: int, delay_seconds: float) -> None:
        self.max_concurrent_downloads = max_concurrent_downloads
        self.delay_seconds = delay_seconds
        self.semaphores: dict[str, asyncio.Semaphore] = {}
        self.last_started: dict[str, float] = {}
        self.delay_locks: dict[str, asyncio.Lock] = {}
        self.creation_lock = asyncio.Lock()

    async def _domain_state(self, domain: str) -> tuple[asyncio.Semaphore, asyncio.Lock]:
        async with self.creation_lock:
            semaphore = self.semaphores.setdefault(
                domain,
                asyncio.Semaphore(self.max_concurrent_downloads),
            )
            delay_lock = self.delay_locks.setdefault(domain, asyncio.Lock())
        return semaphore, delay_lock

    @asynccontextmanager
    async def slot(self, domain: str):
        """Riserva uno slot di download e rispetta la cadenza configurata."""
        semaphore, delay_lock = await self._domain_state(domain)
        async with semaphore:
            # Separiamo concorrenza e cadenza: il semaforo limita quanti file
            # sono in volo, il lock serializza solo l'istante di avvio per
            # mantenere il delay minimo tra richieste allo stesso dominio.
            async with delay_lock:
                now = asyncio.get_running_loop().time()
                last_started = self.last_started.get(domain)
                if last_started is not None and self.delay_seconds > 0:
                    wait_time = self.delay_seconds - (now - last_started)
                    if wait_time > 0:
                        await asyncio.sleep(wait_time)
                self.last_started[domain] = asyncio.get_running_loop().time()
            yield


def open_extraction_executor(workers: int) -> tuple[Executor, str]:
    """Preferisce processi, con fallback a thread in ambienti limitati."""
    try:
        return ProcessPoolExecutor(max_workers=workers), "process"
    except (OSError, PermissionError):
        # Alcuni ambienti sandboxati non permettono i semaphore richiesti dal
        # process pool. Meglio degradare a thread che far fallire l'intera run.
        return ThreadPoolExecutor(max_workers=workers), "thread"


async def run_extract_pdf_async(config: dict | None = None) -> dict:
    """Esegue download PDF e conversione in Markdown con pipeline sovrapposta."""
    config = config or load_config()
    validate_config(config)
    settings = load_pdf_runtime_settings(config)
    manifest_path = project_path(config["paths"].get("processed_manifest_file", "data/processed/manifest.jsonl"))
    pending_records = pending_pdf_records(config)
    recent_urls = set()
    if not settings.force_reextract:
        recent_urls = recent_successful_urls(manifest_path, "pdf", settings.skip_recent_days)
    selected_records = pdf_records(pending_records, recent_urls)
    # Uscita rapida: evita di aprire executor e client HTTP quando non c'e
    # lavoro effettivo, utile anche su sistemi con limiti stretti ai processi.
    if not selected_records:
        return pdf_run_stats(
            pending_records=pending_records,
            selected_records=selected_records,
            ready_for_extraction=0,
            network_downloaded=0,
            reused_raw=0,
            downloaded_bytes=0,
            extracted_ok=0,
            failed=0,
            too_large=0,
            manifest_records=0,
            download_seconds=0.0,
            extraction_seconds=0.0,
            elapsed_seconds=0.0,
            failure_kinds={},
            executor_backend=None,
            settings=settings,
            manifest_path=manifest_path,
        )

    headers = {"User-Agent": config["crawler"]["user_agent"]}
    timeout = httpx.Timeout(config["crawler"]["timeout"])
    ready_for_extraction = 0
    manifest_records = 0
    network_downloaded = 0
    reused_raw = 0
    downloaded_bytes = 0
    extracted_ok = 0
    failed = 0
    too_large = 0
    error_kinds: Counter[str] = Counter()
    download_seconds = 0.0
    extraction_seconds = 0.0
    stats_lock = asyncio.Lock()
    manifest_lock = asyncio.Lock()
    # Due code separano I/O e CPU: appena un download termina, il PDF passa
    # all'estrazione senza aspettare che l'intero batch di rete si completi.
    download_queue: asyncio.Queue[DiscoveryRecord | None] = asyncio.Queue()
    extraction_queue: asyncio.Queue[dict | None] = asyncio.Queue()
    limiter = AsyncPdfDownloadLimiter(
        settings.max_concurrent_downloads,
        settings.download_delay_seconds,
    )
    started_at = time.perf_counter()

    for record in selected_records:
        download_queue.put_nowait(record)

    async def append_manifest(record: ProcessedRecord) -> None:
        nonlocal manifest_records
        # Il manifest resta append-only, ma ora viene aggiornato man mano che
        # ogni PDF termina; un crash non costringe a ripetere l'intero batch.
        async with manifest_lock:
            append_jsonl_batch(manifest_path, [record])
            manifest_records += 1

    def update_progress(progress: tqdm) -> None:
        """Aggiorna la barra sul numero di PDF arrivati a stato terminale."""
        progress.update(1)
        progress.set_postfix(
            pronti=ready_for_extraction,
            ok=extracted_ok,
            falliti=failed,
            troppo_grandi=too_large,
            rete=network_downloaded,
            riusati=reused_raw,
            refresh=False,
        )

    async def handle_download(download: dict, progress: tqdm) -> None:
        nonlocal ready_for_extraction, network_downloaded, reused_raw
        nonlocal downloaded_bytes, failed, too_large
        status = download["status"]
        if status == "downloaded":
            # Anche un raw riusato e gia pronto per l'estrazione: la pipeline
            # non distingue qui tra cache locale e nuovo download di rete.
            async with stats_lock:
                ready_for_extraction += 1
                downloaded_bytes += int(download.get("content_length") or 0)
                if download.get("network_downloaded"):
                    network_downloaded += 1
                if download.get("reused_raw"):
                    reused_raw += 1
            await extraction_queue.put(download)
            return

        manifest = build_manifest_record(download, error=download.get("error"))
        await append_manifest(manifest)
        async with stats_lock:
            if status == "too_large":
                too_large += 1
            else:
                failed += 1
                error_kinds[str(download.get("error_kind") or "download_failed")] += 1
            update_progress(progress)

    async def download_worker(client: httpx.AsyncClient, progress: tqdm) -> None:
        nonlocal download_seconds
        while True:
            record = await download_queue.get()
            if record is None:
                download_queue.task_done()
                return
            domain = urlparse(record["url"]).netloc
            started = time.perf_counter()
            # Ogni worker puo servire qualunque dominio; il limiter applica il
            # vero vincolo per-host quando piu URL arrivano dallo stesso sito.
            async with limiter.slot(domain):
                download = await download_pdf_async(record, config, client, settings)
            elapsed = time.perf_counter() - started
            async with stats_lock:
                download_seconds += elapsed
            await handle_download(download, progress)
            download_queue.task_done()

    async def extraction_worker(executor: Executor, progress: tqdm) -> None:
        nonlocal extracted_ok, failed, extraction_seconds
        loop = asyncio.get_running_loop()
        while True:
            download = await extraction_queue.get()
            if download is None:
                extraction_queue.task_done()
                return
            started = time.perf_counter()
            # L'estrazione PyMuPDF e bloccante: la spostiamo fuori
            # dall'event loop per mantenere vivi i download concorrenti.
            markdown, error = await loop.run_in_executor(
                executor,
                extract_pdf_markdown,
                download["raw_pdf_path"],
                settings.extraction_kwargs,
            )
            elapsed = time.perf_counter() - started
            manifest = build_manifest_record(
                download,
                markdown=markdown,
                error=error,
                error_kind="extract_failed" if error else None,
                config=config,
            )
            await append_manifest(manifest)
            async with stats_lock:
                extraction_seconds += elapsed
                if manifest["status"] == "ok":
                    extracted_ok += 1
                else:
                    failed += 1
                    error_kinds["extract_failed"] += 1
                update_progress(progress)
            extraction_queue.task_done()

    async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
        executor, executor_backend = open_extraction_executor(settings.extraction_workers)
        with executor:
            # Creiamo abbastanza worker da saturare il limite per dominio,
            # senza aprire task inutili quando il batch e piccolo.
            unique_domains = {urlparse(record["url"]).netloc for record in selected_records}
            download_worker_count = min(
                len(selected_records),
                settings.max_concurrent_downloads * max(len(unique_domains), 1),
            )
            with tqdm(
                total=len(selected_records),
                desc="Estrazione PDF",
                unit="pdf",
                dynamic_ncols=True,
            ) as progress:
                download_workers = [
                    asyncio.create_task(download_worker(client, progress))
                    for _ in range(download_worker_count)
                ]
                extraction_workers = [
                    asyncio.create_task(extraction_worker(executor, progress))
                    for _ in range(settings.extraction_workers)
                ]
                await download_queue.join()
                for _ in download_workers:
                    download_queue.put_nowait(None)
                await asyncio.gather(*download_workers)

                await extraction_queue.join()
                for _ in extraction_workers:
                    extraction_queue.put_nowait(None)
                await asyncio.gather(*extraction_workers)

    elapsed_seconds = time.perf_counter() - started_at
    return pdf_run_stats(
        pending_records=pending_records,
        selected_records=selected_records,
        ready_for_extraction=ready_for_extraction,
        network_downloaded=network_downloaded,
        reused_raw=reused_raw,
        downloaded_bytes=downloaded_bytes,
        extracted_ok=extracted_ok,
        failed=failed,
        too_large=too_large,
        manifest_records=manifest_records,
        download_seconds=download_seconds,
        extraction_seconds=extraction_seconds,
        elapsed_seconds=elapsed_seconds,
        failure_kinds=dict(error_kinds),
        executor_backend=executor_backend,
        settings=settings,
        manifest_path=manifest_path,
    )


def run_extract_pdf() -> dict:
    """Wrapper sincrono usato dalla CLI standalone."""
    return asyncio.run(run_extract_pdf_async())


def main() -> None:
    """Entry point."""
    stats = run_extract_pdf()
    print("Estrazione PDF completata")
    for key, value in stats.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
