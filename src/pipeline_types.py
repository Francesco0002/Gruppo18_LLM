"""
Contratti tipizzati dei record JSONL della pipeline.

I record restano normali dict per compatibilita con JSONL, ma questi TypedDict
rendono esplicite le chiavi condivise tra discovery, processing e indexing.
"""

from __future__ import annotations

from typing import Literal, TypedDict

DiscoveryStatus = Literal[
    "ok",
    "not_indexable",
    "pending_download",
    "robots_denied",
    "duplicate_redirect",
    "duplicate_canonical",
    "redirected_out_of_scope",
    "failed",
    "too_large",
    "non_html",
]
DiscoveryType = Literal["html", "pdf", "other", "unknown"]
ProcessedSource = Literal["html", "pdf"]
ProcessedStatus = Literal["ok", "failed", "too_large"]
CleanStatus = Literal["ok", "empty", "warning"]


class DiscoveryRecord(TypedDict, total=False):
    """Record scritto in data/discovered_urls.jsonl."""

    hash: str
    requested_hash: str
    url: str
    requested_url: str
    final_url: str
    canonical_url: str | None
    document_url: str
    domain: str
    depth: int
    discovered_from: str
    type: DiscoveryType
    status: DiscoveryStatus
    indexable: bool
    raw_path: str | None
    duplicate_of: str
    content_type: str
    mime: str
    links_found: int
    pdf_links_found: int
    skip_reason: str
    canonical_skip_reason: str | None
    index_skip_reason: str | None
    discovery_method: str
    error: str


class MarkdownQuality(TypedDict, total=False):
    """Esito della pulizia Markdown conservativa."""

    clean_status: CleanStatus
    clean_warnings: list[str]
    raw_markdown_chars: int
    clean_markdown_chars: int
    removed_chars_ratio: float


class ProcessedRecord(TypedDict, total=False):
    """Record scritto in data/processed/manifest.jsonl."""

    source: ProcessedSource
    status: ProcessedStatus
    url: str
    document_url: str
    hash: str
    content_hash: str | None
    raw_content_hash: str | None
    markdown_path: str | None
    raw_markdown_path: str | None
    clean_markdown_path: str | None
    index_markdown_path: str | None
    raw_html_path: str | None
    raw_pdf_path: str | None
    title: str | None
    breadcrumb: list[str]
    last_crawled: str
    text_extracted: bool
    markdown_chars: int
    raw_markdown_chars: int
    clean_markdown_chars: int
    clean_status: CleanStatus
    clean_warnings: list[str]
    removed_chars_ratio: float
    indexable: bool
    content_length: int
    error: str
    is_duplicate: bool
    duplicate_of: str | None
    duplicate_of_url: str | None
    duplicate_reason: str
