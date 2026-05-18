"""
Pulizia conservativa dei Markdown prodotti da HTML e PDF.

Il cleaner rimuove solo boilerplate evidente e normalizza artefatti ricorrenti,
lasciando intatti contenuti didattici, tabelle, date, email e link utili.
"""

from __future__ import annotations

import re
from typing import Any

import yaml

from pipeline_types import MarkdownQuality, ProcessedSource


# Soglia minima generale; per i PDF tabellari serve anche il controllo semantico
# sotto, perché header e placeholder possono superarla senza contenuto reale.
MIN_INDEXABLE_CHARS = 100
PAGE_COUNTER_RE = re.compile(r"^\s*\d+\s*/\s*\d+\s*$")
EMPTY_TABLE_ROW_RE = re.compile(r"^\|(?:\s*\|)+\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\|\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?$")
EMPTY_UPDATE_MARKER_RE = re.compile(r"Ultimo Aggiornamento:\s*0\s+0000", re.IGNORECASE)
MARKDOWN_EMPHASIS_RE = re.compile(r"[*_`]")
SKIP_LINK_RE = re.compile(r"^\[skip to main content\]\([^)]+\)\s*$", re.IGNORECASE)
GO_TO_LINK_RE = re.compile(r"\[Vai al [^\]]+\]\([^)]+\)", re.IGNORECASE)
CONTROL_LINK_RE = re.compile(r"^\[[‹›×]\]\([^)]+\)\s*$")

FOOTER_STARTERS = (
    "Universita degli Studi di Salerno",
    "Università degli Studi di Salerno",
    "University of Salerno",
    "P.IVA ",
    "P. IVA ",
)
NAVIGATION_LINES = {
    "Condividi",
    "Share",
    "Successiva",
    "Precedente",
    "Torna alla Pagina Precedente",
}


def normalize_br(markdown: str) -> str:
    """Sostituisce gli HTML break residui con spazi singoli."""
    return re.sub(r"\s*<br\s*/?>\s*", " ", markdown, flags=re.IGNORECASE)


def looks_like_repeated_table_header(line: str, previous_lines: set[str]) -> bool:
    """Riconosce header tabellari PDF ripetuti senza toccare la prima occorrenza."""
    if not line.startswith("|"):
        return False
    if "**" not in line:
        return False
    if line not in previous_lines:
        return False
    return line.count("|") >= 3


def table_cells(line: str) -> list[str]:
    """Estrae celle Markdown normalizzate da una riga tabellare."""
    stripped = line.strip().strip("|")
    return [cell.strip() for cell in stripped.split("|")]


def normalized_cell_text(cell: str) -> str:
    """Rimuove markup leggero per classificare header e placeholder."""
    return MARKDOWN_EMPHASIS_RE.sub("", cell).strip()


def is_header_like_table_row(line: str) -> bool:
    """Riconosce righe di intestazione o placeholder senza dati reali."""
    cells = table_cells(line)
    nonempty = [cell for cell in cells if normalized_cell_text(cell)]
    if not nonempty:
        return True
    if all(EMPTY_UPDATE_MARKER_RE.fullmatch(normalized_cell_text(cell)) for cell in nonempty):
        return True
    return all("**" in cell for cell in nonempty)


def is_empty_structured_pdf(markdown: str) -> bool:
    """
    Riconosce PDF tabellari che contengono solo struttura e nessun dato utile.

    La lunghezza grezza non basta: alcuni export generano titolo, header e
    placeholder anche quando la tabella non contiene record.
    """
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    table_lines = [line for line in lines if line.startswith("|")]
    if not table_lines:
        return False

    data_rows = [
        line
        for line in table_lines
        if not TABLE_SEPARATOR_RE.match(line)
        and not EMPTY_TABLE_ROW_RE.match(line)
        and not is_header_like_table_row(line)
    ]
    if data_rows:
        return False

    residual_lines = [
        line
        for line in lines
        if not line.startswith("#")
        and not line.startswith("|")
        and not EMPTY_UPDATE_MARKER_RE.fullmatch(line)
    ]
    residual_chars = len("\n".join(residual_lines).strip())
    return residual_chars < MIN_INDEXABLE_CHARS


def clean_markdown_body(markdown: str, source: ProcessedSource) -> tuple[str, list[str]]:
    """Rimuove boilerplate noto e normalizza layout mantenendo il contenuto."""
    warnings: list[str] = []
    normalized = normalize_br(markdown).replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    cleaned_lines: list[str] = []
    seen_table_headers: set[str] = set()
    removed_boilerplate = 0
    removed_repeated_headers = 0
    skip_separator_after_repeated_header = False
    footer_started = False

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if footer_started:
            removed_boilerplate += 1
            continue

        if not stripped:
            cleaned_lines.append("")
            continue

        if SKIP_LINK_RE.match(stripped):
            removed_boilerplate += 1
            continue

        if stripped in NAVIGATION_LINES or CONTROL_LINK_RE.match(stripped):
            removed_boilerplate += 1
            continue

        if source == "html" and any(stripped.startswith(starter) for starter in FOOTER_STARTERS):
            footer_started = True
            removed_boilerplate += 1
            continue

        without_go_to_links = GO_TO_LINK_RE.sub("", line).strip()
        if without_go_to_links != stripped:
            removed_boilerplate += 1
            if not without_go_to_links:
                continue
            line = without_go_to_links
            stripped = line.strip()

        if PAGE_COUNTER_RE.match(stripped):
            removed_boilerplate += 1
            continue

        if EMPTY_TABLE_ROW_RE.match(stripped):
            removed_boilerplate += 1
            continue

        if source == "pdf" and looks_like_repeated_table_header(stripped, seen_table_headers):
            removed_repeated_headers += 1
            skip_separator_after_repeated_header = True
            continue

        if skip_separator_after_repeated_header and TABLE_SEPARATOR_RE.match(stripped):
            removed_repeated_headers += 1
            skip_separator_after_repeated_header = False
            continue
        skip_separator_after_repeated_header = False

        if source == "pdf" and stripped.startswith("|") and "**" in stripped:
            seen_table_headers.add(stripped)

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    if removed_boilerplate:
        warnings.append(f"removed_boilerplate_lines:{removed_boilerplate}")
    if removed_repeated_headers:
        warnings.append(f"removed_repeated_pdf_table_headers:{removed_repeated_headers}")

    return cleaned, warnings


def build_quality(raw_markdown: str, clean_markdown: str, warnings: list[str]) -> MarkdownQuality:
    """Calcola indicatori leggeri per manifest e front matter."""
    raw_chars = len(raw_markdown)
    clean_chars = len(clean_markdown)
    removed_ratio = round(max(raw_chars - clean_chars, 0) / raw_chars, 4) if raw_chars else 0.0
    clean_status = "ok" if clean_markdown.strip() else "empty"
    if clean_status == "ok" and warnings:
        clean_status = "warning"

    return {
        "clean_status": clean_status,
        "clean_warnings": warnings,
        "raw_markdown_chars": raw_chars,
        "clean_markdown_chars": clean_chars,
        "removed_chars_ratio": removed_ratio,
    }


def yaml_front_matter(metadata: dict[str, Any]) -> str:
    """Serializza metadata leggibili e parsabili in front matter YAML."""
    return yaml.safe_dump(
        metadata,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    ).strip()


def with_front_matter(markdown: str, metadata: dict[str, Any]) -> str:
    """Aggiunge YAML front matter al corpo Markdown pulito."""
    return f"---\n{yaml_front_matter(metadata)}\n---\n\n{markdown.strip()}\n"


def clean_markdown(
    raw_markdown: str,
    *,
    source: ProcessedSource,
    metadata: dict[str, Any],
) -> tuple[str, str, MarkdownQuality]:
    """Ritorna corpo pulito, markdown con front matter e qualita."""
    clean_body, warnings = clean_markdown_body(raw_markdown, source)
    if source == "pdf" and is_empty_structured_pdf(clean_body):
        warnings.append("empty_structured_pdf")
    quality = build_quality(raw_markdown, clean_body, warnings)
    front_matter = dict(metadata)
    front_matter.update(quality)
    return clean_body, with_front_matter(clean_body, front_matter), quality
