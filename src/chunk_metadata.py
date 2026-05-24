from __future__ import annotations

from typing import Any


CHUNK_METADATA_SCHEMA_VERSION = 2

CORE_METADATA_KEYS = [
    "chunk_id",
    "document_hash",
    "document_content_hash",
    "chunk_index",
    "chunk_count",
    "text_hash",
    "chars",
]

RETRIEVAL_METADATA_KEYS = [
    "source",
    "source_url",
    "document_url",
    "title",
    "breadcrumb",
    "breadcrumb_text",
    "content_title",
    "section_heading",
    "discovered_from",
    "link_text",
    "pdf_source_section",
    "chunk_kind",
    "entity_type",
    "entity_name",
    "source_family",
    "teacher_id",
    "course_id",
    "lab_id",
    "year",
    "document_years",
    "document_type",
    "publication_id",
    "publication_title",
    "publication_year",
    "publication_type",
    "publication_venue",
    "publication_authors",
    "publication_doi",
    "publication_iris_url",
    "publication_order",
]

PROVENANCE_METADATA_KEYS = [
    "index_markdown_path",
    "last_crawled",
]

DEBUG_METADATA_KEYS = [
    "pdf_download_decision",
    "pdf_match_keywords",
    "clean_status",
    "clean_warnings",
]


def compact_dict(values: dict[str, Any]) -> dict[str, Any]:
    """
    Rimuove solo i valori assenti, preservando liste vuote e stringhe vuote
    quando sono semanticamente utili per debug o compatibilità.
    """
    return {
        key: value
        for key, value in values.items()
        if value is not None
    }


def chunk_section(chunk: dict[str, Any], section: str) -> dict[str, Any]:
    value = chunk.get(section)
    return value if isinstance(value, dict) else {}


def chunk_metadata_value(chunk: dict[str, Any], key: str) -> Any:
    """
    Legge un campo metadata supportando sia il nuovo formato sezionato sia i
    chunk legacy con campi metadata al top-level.
    """
    for section in ("retrieval_metadata", "provenance", "debug"):
        section_values = chunk_section(chunk, section)
        if key in section_values:
            return section_values[key]

    return chunk.get(key)


def flatten_chunk_metadata(
    chunk: dict[str, Any],
    *,
    include_debug: bool = True,
) -> dict[str, Any]:
    """
    Produce metadata piatti per Chroma, retrieval, reranker ed evaluation.

    Il file chunk resta separato in retrieval/provenance/debug; qui
    ricostruiamo la vista piatta che il resto della pipeline già consuma.
    """
    metadata: dict[str, Any] = {
        "chunk_metadata_schema_version": chunk.get("chunk_metadata_schema_version"),
    }

    for key in CORE_METADATA_KEYS:
        if key in chunk:
            metadata[key] = chunk[key]

    for section in ("retrieval_metadata", "provenance"):
        metadata.update(chunk_section(chunk, section))

    if include_debug:
        metadata.update(chunk_section(chunk, "debug"))

    legacy_keys = (
        RETRIEVAL_METADATA_KEYS
        + PROVENANCE_METADATA_KEYS
        + (DEBUG_METADATA_KEYS if include_debug else [])
    )

    for key in legacy_keys:
        if key not in metadata and key in chunk:
            metadata[key] = chunk[key]

    return compact_dict(metadata)
