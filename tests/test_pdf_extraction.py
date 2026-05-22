from __future__ import annotations

import asyncio
import sys
import tempfile
import unittest
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from extract_pdf import (  # noqa: E402
    AsyncPdfDownloadLimiter,
    build_manifest_record,
    download_pdf_async,
    load_pdf_runtime_settings,
    pdf_records,
)
from ingest import mark_duplicate_documents  # noqa: E402


def base_config(temp_dir: str) -> dict:
    return {
        "crawler": {
            "user_agent": "tests",
            "timeout": 5,
            "allowed_domains": ["www.diem.unisa.it"],
            "pdf_allowed_domains": ["www.diem.unisa.it"],
            "max_total_urls": 10,
            "max_depth": 1,
            "per_domain_limits": {"www.diem.unisa.it": 10},
            "max_concurrent_requests": 1,
            "rate_limit_per_domain_rps": 1,
            "max_html_bytes": 1000,
            "pdf_max_bytes": 1024,
            "pdf_extraction_workers": 2,
            "pdf_max_concurrent_downloads": 3,
            "pdf_download_delay_seconds": 0,
            "pdf_skip_recent_days": 7,
            "pdf_force_reextract": False,
            "pdf_extraction": {
                "use_layout": True,
                "use_ocr": False,
            },
        },
        "paths": {
            "raw_pdf_dir": temp_dir,
        },
        "scope": {
            "diem_domain": "www.diem.unisa.it",
            "teacher_domain": "docenti.unisa.it",
            "course_domain": "corsi.unisa.it",
            "allowed_course_paths": [],
            "allowed_course_codes": [],
        },
    }


def pdf_record(url: str = "https://www.diem.unisa.it/file.pdf") -> dict:
    return {
        "url": url,
        "document_url": url,
        "hash": "abc123",
        "discovered_from": "https://www.diem.unisa.it/didattica",
    }


class PdfExtractionTests(unittest.TestCase):
    def test_runtime_settings_are_loaded_from_config(self) -> None:
        config = base_config("/tmp")
        settings = load_pdf_runtime_settings(config)

        self.assertEqual(settings.max_bytes, 1024)
        self.assertEqual(settings.extraction_workers, 2)
        self.assertEqual(settings.max_concurrent_downloads, 3)
        self.assertEqual(settings.extraction_kwargs["use_ocr"], False)

    def test_pdf_records_are_deduplicated_and_skip_recent_urls(self) -> None:
        records = [pdf_record(), pdf_record(), pdf_record("https://www.diem.unisa.it/other.pdf")]

        selected = pdf_records(records, {"https://www.diem.unisa.it/other.pdf"})

        self.assertEqual([record["url"] for record in selected], ["https://www.diem.unisa.it/file.pdf"])

    def test_duplicate_marking_accepts_mixed_timestamp_shapes(self) -> None:
        records = [
            {
                "source": "pdf",
                "status": "ok",
                "url": "https://www.diem.unisa.it/file.pdf",
                "hash": "pdf",
                "content_hash": "same",
                "index_markdown_path": "pdf.md",
                "last_crawled": "not-a-timestamp",
            },
            {
                "source": "html",
                "status": "ok",
                "url": "https://www.diem.unisa.it/page",
                "hash": "html",
                "content_hash": "same",
                "index_markdown_path": "html.md",
                "last_crawled": "2026-05-18T10:00:00",
            },
        ]

        updated, _current, stats = mark_duplicate_documents(records)

        self.assertEqual(stats["duplicates"], 1)
        self.assertTrue(updated[0]["is_duplicate"])

    def test_empty_pdf_markdown_is_failed_as_no_text_extracted(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            record = build_manifest_record(
                {
                    "record": pdf_record(),
                    "status": "downloaded",
                    "raw_pdf_path": "data/raw_pdf/ab/abc123.pdf",
                    "content_length": 1234,
                },
                markdown="",
                config={
                    "paths": {
                        "processed_raw_markdown_dir": str(base / "markdown_raw"),
                        "processed_markdown_dir": str(base / "markdown"),
                    }
                },
            )

        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error_kind"], "no_text_extracted")
        self.assertFalse(record["text_extracted"])
        self.assertFalse(record["indexable"])
        self.assertIsNone(record["index_markdown_path"])

    def test_short_pdf_markdown_stays_ok_but_not_indexable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            record = build_manifest_record(
                {
                    "record": pdf_record(),
                    "status": "downloaded",
                    "raw_pdf_path": "data/raw_pdf/ab/abc123.pdf",
                    "content_length": 1234,
                },
                markdown="Testo breve ma realmente estratto.",
                config={
                    "paths": {
                        "processed_raw_markdown_dir": str(base / "markdown_raw"),
                        "processed_markdown_dir": str(base / "markdown"),
                    }
                },
            )

        self.assertEqual(record["status"], "ok")
        self.assertFalse(record["text_extracted"])
        self.assertFalse(record["indexable"])
        self.assertIsNotNone(record["index_markdown_path"])


class AsyncPdfExtractionTests(unittest.IsolatedAsyncioTestCase):
    async def test_reuses_existing_valid_raw_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = base_config(temp_dir)
            settings = load_pdf_runtime_settings(config)
            output_path = Path(temp_dir) / "ab" / "abc123.pdf"
            output_path.parent.mkdir(parents=True)
            output_path.write_bytes(b"%PDF-1.7\nbody")

            async with httpx.AsyncClient() as client:
                result = await download_pdf_async(pdf_record(), config, client, settings)

        self.assertEqual(result["status"], "downloaded")
        self.assertTrue(result["reused_raw"])
        self.assertFalse(result["network_downloaded"])

    async def test_untrusted_diem_rescue_pdf_is_rejected_before_raw_cache_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = base_config(temp_dir)
            settings = load_pdf_runtime_settings(config)
            output_path = Path(temp_dir) / "ab" / "abc123.pdf"
            output_path.parent.mkdir(parents=True)
            output_path.write_bytes(b"%PDF-1.7\ncached")
            record = pdf_record(
                "https://www.diem.unisa.it/uploads/rescue/292/14149/rep-196-prot-46039-bando-borsa-savarese-dipmed-2026-bs07.pdf"
            )
            record["discovered_from"] = "https://www.diem.unisa.it/home/bandi?anno=2026&modulo=139&struttura=300400"

            async with httpx.AsyncClient() as client:
                result = await download_pdf_async(record, config, client, settings)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_kind"], "out_of_scope_pdf")
        self.assertIn("diem_rescue_upload_untrusted", result["error"])

    async def test_force_reextract_downloads_even_when_raw_exists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = base_config(temp_dir)
            config["crawler"]["pdf_force_reextract"] = True
            settings = load_pdf_runtime_settings(config)
            output_path = Path(temp_dir) / "ab" / "abc123.pdf"
            output_path.parent.mkdir(parents=True)
            output_path.write_bytes(b"%PDF-1.7\nold")
            requests = 0

            def handler(_request: httpx.Request) -> httpx.Response:
                nonlocal requests
                requests += 1
                return httpx.Response(
                    200,
                    headers={"Content-Type": "application/pdf"},
                    content=b"%PDF-1.7\nnew",
                )

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await download_pdf_async(pdf_record(), config, client, settings)

        self.assertEqual(requests, 1)
        self.assertFalse(result["reused_raw"])
        self.assertTrue(result["network_downloaded"])

    async def test_redirected_pdf_out_of_scope_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = base_config(temp_dir)
            settings = load_pdf_runtime_settings(config)

            def handler(request: httpx.Request) -> httpx.Response:
                if request.url.host == "www.diem.unisa.it":
                    return httpx.Response(
                        302,
                        headers={"Location": "https://evil.example/file.pdf"},
                    )
                return httpx.Response(
                    200,
                    headers={"Content-Type": "application/pdf"},
                    content=b"%PDF-1.7\nbody",
                )

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await download_pdf_async(pdf_record(), config, client, settings)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_kind"], "redirected_out_of_scope")

    async def test_non_pdf_response_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = base_config(temp_dir)
            settings = load_pdf_runtime_settings(config)

            def handler(_request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200,
                    headers={"Content-Type": "text/html"},
                    content=b"<html>nope</html>",
                )

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await download_pdf_async(pdf_record(), config, client, settings)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_kind"], "invalid_pdf_response")

    async def test_http_status_error_is_reported_with_status_kind(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = base_config(temp_dir)
            settings = load_pdf_runtime_settings(config)

            def handler(_request: httpx.Request) -> httpx.Response:
                return httpx.Response(400, content=b"bad request")

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await download_pdf_async(pdf_record(), config, client, settings)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_kind"], "http_400")

    async def test_literal_percent_in_pdf_url_is_escaped_before_download(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = base_config(temp_dir)
            settings = load_pdf_runtime_settings(config)
            requested_urls: list[str] = []

            def handler(request: httpx.Request) -> httpx.Response:
                requested_urls.append(str(request.url))
                return httpx.Response(
                    200,
                    headers={"Content-Type": "application/pdf"},
                    content=b"%PDF-1.7\nbody",
                )

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await download_pdf_async(
                    pdf_record("https://www.diem.unisa.it/uploads/file-(100%-Dipartimento).pdf"),
                    config,
                    client,
                    settings,
                )

        self.assertEqual(result["status"], "downloaded")
        self.assertIn("100%25-Dipartimento", requested_urls[0])

    async def test_too_large_response_is_reported_without_writing_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = base_config(temp_dir)
            settings = load_pdf_runtime_settings(config)

            def handler(_request: httpx.Request) -> httpx.Response:
                return httpx.Response(
                    200,
                    headers={
                        "Content-Type": "application/pdf",
                        "Content-Length": "2048",
                    },
                    content=b"",
                )

            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result = await download_pdf_async(pdf_record(), config, client, settings)

            self.assertFalse(any(Path(temp_dir).rglob("*.pdf")))

        self.assertEqual(result["status"], "too_large")

    async def test_download_limiter_caps_same_domain_parallelism(self) -> None:
        limiter = AsyncPdfDownloadLimiter(max_concurrent_downloads=2, delay_seconds=0)
        active = 0
        max_active = 0

        async def use_slot() -> None:
            nonlocal active, max_active
            async with limiter.slot("www.diem.unisa.it"):
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0.01)
                active -= 1

        await asyncio.gather(*(use_slot() for _ in range(5)))

        self.assertEqual(max_active, 2)
