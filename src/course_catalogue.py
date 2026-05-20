"""
Ingest dedicato per UNISA CourseCatalogue/Cineca.

Il sito CourseCatalogue e' una SPA Angular: il crawler HTML vede solo
`<app-root>`. Questo step usa gli endpoint JSON pubblici per generare Markdown
indicizzabile per corsi, percorsi, anni di corso e singoli insegnamenti.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from discovery_io import load_config, project_path, short_hash, validate_config
from markdown_cleaner import MIN_INDEXABLE_CHARS, clean_markdown
from pipeline_io import (
    append_jsonl_batch,
    content_hash,
    now_iso,
    recent_successful_urls,
    relative_path,
    write_text,
)
from pipeline_types import ProcessedRecord


DEFAULT_DOMAIN = "unisa.coursecatalogue.cineca.it"
DEFAULT_SKIP_RECENT_DAYS = 7


@dataclass(frozen=True)
class CourseCatalogueSettings:
    """Configurazione risolta per l'ingest CourseCatalogue."""

    enabled: bool
    domain: str
    years: tuple[str, ...]
    course_ids: tuple[str, ...]
    course_entries: tuple[tuple[str, str], ...]
    timeout: float
    skip_recent_days: int
    fetch_teaching_details: bool

    @property
    def base_url(self) -> str:
        return f"https://{self.domain}"


def config_list(config: dict, *keys: str) -> list[Any]:
    """Legge una lista da config, restituendo [] se assente."""
    node: Any = config
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return []
        node = node[key]
    if node is None:
        return []
    if isinstance(node, list):
        return node
    return [node]


def settings_from_config(config: dict) -> CourseCatalogueSettings:
    """Estrae la configurazione CourseCatalogue con fallback conservativi."""
    section = config.get("course_catalogue", {})
    scope = config.get("scope", {})
    years = tuple(str(item) for item in section.get("years", ["2025"]))
    course_ids = tuple(
        str(item)
        for item in (
            section.get("course_ids")
            or scope.get("allowed_course_catalogue_ids")
            or []
        )
    )
    raw_entries = section.get("course_entries") or []
    course_entries = tuple(
        (str(entry["year"]), str(entry["course_id"]))
        for entry in raw_entries
        if isinstance(entry, dict) and entry.get("year") and entry.get("course_id")
    )
    if not course_entries:
        course_entries = tuple(
            (year, course_id)
            for year in years
            for course_id in course_ids
        )
    return CourseCatalogueSettings(
        enabled=bool(section.get("enabled", bool(course_entries))),
        domain=str(section.get("domain", scope.get("course_catalogue_domain", DEFAULT_DOMAIN))).lower(),
        years=years,
        course_ids=course_ids,
        course_entries=course_entries,
        timeout=float(section.get("timeout", config.get("crawler", {}).get("timeout", 20))),
        skip_recent_days=int(section.get("skip_recent_days", DEFAULT_SKIP_RECENT_DAYS)),
        fetch_teaching_details=bool(section.get("fetch_teaching_details", True)),
    )


def raw_markdown_path(url_hash: str, config: dict) -> Path:
    """Path del Markdown raw generato da CourseCatalogue."""
    base = project_path(config["paths"].get("processed_raw_markdown_dir", "data/processed/markdown_raw"))
    return base / url_hash[:2] / f"{url_hash}.md"


def processed_markdown_path(url_hash: str, config: dict) -> Path:
    """Path del Markdown pulito generato da CourseCatalogue."""
    base = project_path(config["paths"].get("processed_markdown_dir", "data/processed/markdown"))
    return base / url_hash[:2] / f"{url_hash}.md"


def clean_text(value: Any) -> str:
    """Normalizza testo CourseCatalogue senza perdere i paragrafi."""
    if value is None:
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def title_text(value: Any) -> str:
    """Normalizza una stringa per titoli/list item."""
    return re.sub(r"\s+", " ", clean_text(value)).strip()


def field(record: dict[str, Any], *names: str) -> str:
    """Restituisce il primo campo testuale non vuoto."""
    for name in names:
        value = clean_text(record.get(name))
        if value:
            return value
    return ""


def bullet(label: str, value: Any) -> str | None:
    """Riga bullet se il valore esiste."""
    text = title_text(value)
    if not text:
        return None
    return f"- **{label}:** {text}"


def is_teaching(node: Any) -> bool:
    """Riconosce un oggetto insegnamento nel JSON CourseCatalogue."""
    return (
        isinstance(node, dict)
        and bool(node.get("cod"))
        and bool(node.get("adCod"))
        and bool(node.get("des_it") or node.get("des_en"))
    )


def teaching_key(teaching: dict[str, Any]) -> tuple[str, str, str, str, str]:
    """Chiave stabile per deduplicare le attivita formative."""
    return (
        str(teaching.get("cod", "")),
        str(teaching.get("adCod", "")),
        str(teaching.get("aa", teaching.get("corso_aa", ""))),
        str(teaching.get("corso_cod", "")),
        str(teaching.get("af_percorso_id", teaching.get("af_percorso_cod", ""))),
    )


def walk_teachings(node: Any) -> Iterable[dict[str, Any]]:
    """Estrae ricorsivamente gli insegnamenti da gruppi/attivita annidati."""
    if is_teaching(node):
        yield node
        return
    if isinstance(node, dict):
        for value in node.values():
            yield from walk_teachings(value)
    elif isinstance(node, list):
        for item in node:
            yield from walk_teachings(item)


def course_title(course: dict[str, Any]) -> str:
    """Titolo italiano/inglese del corso."""
    return title_text(course.get("des_it") or course.get("des_en") or course.get("cdsCod") or course.get("cod"))


def course_url(settings: CourseCatalogueSettings, course: dict[str, Any]) -> str:
    """URL pubblico della pagina corso CourseCatalogue."""
    year = str(course.get("aa") or "")
    course_id = str(course.get("cod") or course.get("cdsId") or "")
    query = {}
    if course.get("ordinamento_aa"):
        query["annoOrdinamento"] = str(course["ordinamento_aa"])
    suffix = f"?{urlencode(query)}" if query else ""
    return f"{settings.base_url}/corsi/{year}/{course_id}{suffix}"


def teaching_url(settings: CourseCatalogueSettings, teaching: dict[str, Any]) -> str:
    """URL pubblico della scheda insegnamento CourseCatalogue."""
    coorte = str(teaching.get("corso_aa") or teaching.get("aa") or "")
    course_id = str(teaching.get("corso_cod") or "")
    teaching_year = str(teaching.get("aa") or teaching.get("corso_aa") or "")
    teaching_id = str(teaching.get("cod") or "")
    ordinamento = str(teaching.get("ordinamento_aa") or "")
    percorso = str(teaching.get("af_percorso_id") or teaching.get("corso_percorso_id") or "")
    query = {"coorte": coorte}
    if teaching.get("schemaId"):
        query["schemaid"] = str(teaching["schemaId"])
    return (
        f"{settings.base_url}/corsi/{coorte}/{course_id}/insegnamenti/"
        f"{teaching_year}/{teaching_id}/{ordinamento}/{percorso}?{urlencode(query)}"
    )


def teaching_api_url(settings: CourseCatalogueSettings, teaching: dict[str, Any]) -> str:
    """Endpoint JSON della scheda insegnamento."""
    year = str(teaching.get("aa") or teaching.get("corso_aa") or "")
    teaching_id = str(teaching.get("cod") or "")
    ordinamento = str(teaching.get("ordinamento_aa") or "")
    percorso_id = str(teaching.get("af_percorso_id") or teaching.get("corso_percorso_id") or "")
    course_id = str(teaching.get("corso_cod") or "")
    return (
        f"{settings.base_url}/api/v1/insegnamento-offerta/"
        f"{year}/{teaching_id}/{ordinamento}/{percorso_id}/{course_id}"
    )


async def fetch_json(client: httpx.AsyncClient, url: str) -> Any:
    """Scarica JSON e solleva per status HTTP non riusciti."""
    response = await client.get(url)
    response.raise_for_status()
    return response.json()


async def fetch_course(
    client: httpx.AsyncClient,
    settings: CourseCatalogueSettings,
    year: str,
    course_id: str,
) -> dict[str, Any]:
    """Scarica un corso CourseCatalogue."""
    payload = await fetch_json(client, f"{settings.base_url}/api/v1/corso/{year}/{course_id}")
    if isinstance(payload, list) and payload:
        return payload[0]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Risposta corso vuota o inattesa per {year}/{course_id}")


async def fetch_teaching_detail(
    client: httpx.AsyncClient,
    settings: CourseCatalogueSettings,
    teaching: dict[str, Any],
) -> dict[str, Any]:
    """Scarica la scheda estesa di un insegnamento."""
    payload = await fetch_json(client, teaching_api_url(settings, teaching))
    if isinstance(payload, dict):
        return payload
    raise ValueError("Risposta insegnamento inattesa")


def collect_course_teachings(course: dict[str, Any]) -> list[dict[str, Any]]:
    """Raccoglie tutti gli insegnamenti presenti nei percorsi del corso."""
    by_key: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    for percorso in course.get("percorsi", []) or []:
        for anno in percorso.get("anni", []) or []:
            for teaching in walk_teachings(anno.get("insegnamenti", [])):
                enriched = dict(teaching)
                enriched.setdefault("corso_aa", course.get("aa"))
                enriched.setdefault("corso_cod", course.get("cod") or course.get("cdsId"))
                enriched.setdefault("corso_des_it", course.get("des_it"))
                enriched.setdefault("corso_des_en", course.get("des_en"))
                enriched.setdefault("course_year_number", anno.get("anno"))
                enriched.setdefault("course_year_label", anno.get("label_it") or anno.get("label_en"))
                enriched.setdefault("percorso_des_it", percorso.get("des_it"))
                enriched.setdefault("percorso_des_en", percorso.get("des_en"))
                enriched.setdefault("schemaId", percorso.get("schemaId") or teaching.get("schemaId"))
                by_key.setdefault(teaching_key(enriched), enriched)
    return list(by_key.values())


def render_roles(course: dict[str, Any]) -> list[str]:
    """Rende ruoli/responsabilita del corso."""
    rows: list[str] = []
    for role in course.get("ruoli_it", []) or []:
        name = title_text(f"{role.get('caricaNome', '')} {role.get('caricaCognome', '')}")
        charge = title_text(role.get("carica"))
        matricola = title_text(role.get("caricaMatricola"))
        if name or charge:
            suffix = f" - matricola {matricola}" if matricola else ""
            rows.append(f"- {name}: {charge}{suffix}")
    return rows


def render_programme_texts(course: dict[str, Any]) -> list[str]:
    """Rende i testi di obiettivi/descrizione del corso."""
    rows: list[str] = []
    for item in course.get("programma_testi_obiettivi_it", []) or []:
        title = title_text(item.get("carattTitolo") or item.get("tipoTestoProgDidCod"))
        text = clean_text(item.get("carattTesto"))
        if title and text:
            rows.append(f"### {title}\n\n{text}")
    return rows


def teaching_summary_line(teaching: dict[str, Any]) -> str:
    """Riga compatta per un insegnamento nel piano."""
    title = title_text(teaching.get("des_it") or teaching.get("des_en"))
    details = [
        value
        for value in (
            f"{teaching.get('crediti')} CFU" if teaching.get("crediti") else "",
            f"{teaching.get('ore')} ore" if teaching.get("ore") else "",
            title_text(teaching.get("periodo_didattico_it")),
            title_text(teaching.get("ssd")),
            title_text(teaching.get("adCod")),
        )
        if value
    ]
    docenti = ", ".join(
        title_text(docente.get("des"))
        for docente in teaching.get("docenti", []) or []
        if title_text(docente.get("des"))
    )
    if docenti:
        details.append(f"docenti: {docenti}")
    suffix = f" ({'; '.join(details)})" if details else ""
    return f"- {title}{suffix}"


def render_study_plan(course: dict[str, Any]) -> list[str]:
    """Rende percorsi, anni e gruppi di attivita formative."""
    rows: list[str] = []
    for percorso in course.get("percorsi", []) or []:
        percorso_title = title_text(percorso.get("des_it") or percorso.get("des_en") or "Percorso")
        rows.append(f"## Percorso: {percorso_title}")
        for anno in percorso.get("anni", []) or []:
            anno_label = title_text(anno.get("label_it") or anno.get("label_en"))
            heading = f"### {anno.get('anno')} anno"
            if anno_label:
                heading += f" - {anno_label}"
            rows.append(heading)
            for group in anno.get("insegnamenti", []) or []:
                label = title_text(group.get("label_it") or group.get("label_en") or group.get("cod"))
                if label:
                    rows.append(f"#### {label}")
                lines = [teaching_summary_line(item) for item in walk_teachings(group)]
                rows.extend(lines or ["- Nessun insegnamento indicato"])
    return rows


def render_course_markdown(course: dict[str, Any]) -> str:
    """Markdown raw per il corso e il piano di studi."""
    title = course_title(course)
    lines = [f"# {title}", ""]
    facts = [
        bullet("Anno offerta", course.get("aa")),
        bullet("ID CourseCatalogue", course.get("cod") or course.get("cdsId")),
        bullet("Codice corso", course.get("cdsCod")),
        bullet("Codicione UNISA", course.get("codicione")),
        bullet("Tipo corso", course.get("tipo_corso_des_it")),
        bullet("Classe", course.get("classe_it") or course.get("classe_cod")),
        bullet("Crediti", course.get("crediti_it")),
        bullet("Durata", course.get("durata_it")),
        bullet("Accesso", course.get("accesso_it")),
        bullet("Sede", course.get("sede_des_it")),
        bullet("Lingua", course.get("lingua_des_it")),
    ]
    lines.extend(item for item in facts if item)
    lines.append("")

    roles = render_roles(course)
    if roles:
        lines.extend(["## Ruoli e responsabilita", *roles, ""])

    programme = render_programme_texts(course)
    if programme:
        lines.extend(["## Descrizione e obiettivi", *programme, ""])

    plan = render_study_plan(course)
    if plan:
        lines.extend(["## Piano di studi", *plan, ""])

    return "\n".join(lines).strip() + "\n"


def render_teacher_list(teaching: dict[str, Any]) -> list[str]:
    """Rende i docenti di un insegnamento."""
    rows: list[str] = []
    for docente in teaching.get("docenti", []) or teaching.get("titolari", []) or []:
        name = title_text(docente.get("des") or f"{docente.get('nome', '')} {docente.get('cognome', '')}")
        parts = [
            f"matricola {docente.get('matricola')}" if docente.get("matricola") else "",
            f"id docente {docente.get('cod')}" if docente.get("cod") else "",
        ]
        details = "; ".join(part for part in parts if part)
        suffix = f" ({details})" if details else ""
        if name:
            rows.append(f"- {name}{suffix}")
    return rows


def render_syllabus_texts(teaching: dict[str, Any]) -> list[str]:
    """Rende i testi syllabus della scheda insegnamento."""
    labels = [
        ("Obiettivi formativi", "obiettivi_formativi_it"),
        ("Prerequisiti", "prerequisiti_it"),
        ("Contenuti", "contenuti_it"),
        ("Metodi didattici", "metodi_didattici_est_it", "metodi_didattici_it"),
        ("Verifica dell'apprendimento", "verifica_apprendimento_it"),
        ("Testi", "testi_it"),
        ("Altre informazioni", "altro_it", "altri_testi_1_it", "altri_testi_2_it", "altri_testi_3_it"),
    ]
    rows: list[str] = []
    texts = teaching.get("testiTotali", []) or []
    for text_block in texts or [teaching]:
        for label, *names in labels:
            value = field(text_block, *names) or field(teaching, *names)
            if value:
                rows.append(f"## {label}\n\n{value}")
    return list(dict.fromkeys(rows))


def render_teaching_markdown(teaching: dict[str, Any]) -> str:
    """Markdown raw per un insegnamento CourseCatalogue."""
    title = title_text(teaching.get("des_it") or teaching.get("des_en") or teaching.get("adCod"))
    lines = [f"# {title}", ""]
    facts = [
        bullet("Corso di studio", teaching.get("corso_des_it")),
        bullet("Codice corso", teaching.get("cdsCod")),
        bullet("Codice insegnamento", teaching.get("adCod")),
        bullet("ID insegnamento", teaching.get("cod")),
        bullet("Anno offerta", teaching.get("aa")),
        bullet("Anno di corso", teaching.get("corso_anno") or teaching.get("course_year_number")),
        bullet("Ordinamento", teaching.get("ordinamento_aa")),
        bullet("Percorso", teaching.get("corso_percorso_des_it") or teaching.get("percorso_des_it")),
        bullet("CFU", teaching.get("crediti")),
        bullet("Ore", (teaching.get("durata") or {}).get("totale") or teaching.get("ore")),
        bullet("Periodo didattico", teaching.get("periodo_didattico_it")),
        bullet("SSD", teaching.get("ssd")),
        bullet("Tipo attivita", teaching.get("tipo_it") or teaching.get("tafDes_it")),
        bullet("Modalita esame", teaching.get("tipoEsaDes_it")),
        bullet("Frequenza", teaching.get("frequenza_it")),
        bullet("Lingua", teaching.get("lingua_des_it")),
        bullet("Dipartimento", teaching.get("dip_des_it")),
    ]
    lines.extend(item for item in facts if item)
    lines.append("")

    teachers = render_teacher_list(teaching)
    if teachers:
        lines.extend(["## Docenti", *teachers, ""])

    syllabus = render_syllabus_texts(teaching)
    if syllabus:
        lines.extend(syllabus)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def processed_record_from_markdown(
    *,
    url: str,
    title: str,
    breadcrumb: list[str],
    raw_markdown: str,
    config: dict,
    extra: dict[str, Any] | None = None,
) -> ProcessedRecord:
    """Scrive Markdown raw/pulito e ritorna il record manifest."""
    crawled_at = now_iso()
    url_hash = short_hash(url)
    metadata = {
        "url": url,
        "document_url": url,
        "source": "course_catalogue",
        "hash": url_hash,
        "title": title,
        "breadcrumb": breadcrumb,
        "last_crawled": crawled_at,
    }
    clean_body, index_markdown, quality = clean_markdown(
        raw_markdown,
        source="course_catalogue",
        metadata=metadata,
    )
    raw_path = write_text(raw_markdown_path(url_hash, config), raw_markdown)
    index_path = write_text(processed_markdown_path(url_hash, config), index_markdown)
    text_extracted = len(clean_body.strip()) >= MIN_INDEXABLE_CHARS
    record: ProcessedRecord = {
        "source": "course_catalogue",
        "status": "ok",
        "url": url,
        "document_url": url,
        "hash": url_hash,
        "content_hash": content_hash(clean_body),
        "raw_content_hash": content_hash(raw_markdown),
        "raw_markdown_path": raw_path,
        "index_markdown_path": index_path,
        "title": title,
        "breadcrumb": breadcrumb,
        "last_crawled": crawled_at,
        "text_extracted": text_extracted,
        "markdown_chars": len(clean_body),
        "indexable": text_extracted,
        **quality,
    }
    if extra:
        record.update(extra)
    return record


def failed_record(url: str, error: Exception | str) -> ProcessedRecord:
    """Record manifest per un errore CourseCatalogue."""
    return {
        "source": "course_catalogue",
        "status": "failed",
        "url": url,
        "document_url": url,
        "hash": short_hash(url),
        "last_crawled": now_iso(),
        "error": str(error),
        "error_kind": error.__class__.__name__ if isinstance(error, Exception) else "error",
    }


async def build_course_records(
    course: dict[str, Any],
    client: httpx.AsyncClient,
    settings: CourseCatalogueSettings,
    config: dict,
    recent_urls: set[str],
) -> tuple[list[ProcessedRecord], dict[str, int]]:
    """Genera record per corso e insegnamenti."""
    stats = {
        "course_documents": 0,
        "teaching_documents": 0,
        "teaching_detail_failed": 0,
        "skipped_recent": 0,
    }
    records: list[ProcessedRecord] = []

    url = course_url(settings, course)
    title = course_title(course)
    if url in recent_urls:
        stats["skipped_recent"] += 1
    else:
        records.append(
            processed_record_from_markdown(
                url=url,
                title=title,
                breadcrumb=["CourseCatalogue", title],
                raw_markdown=render_course_markdown(course),
                config=config,
                extra={
                    "course_catalogue_kind": "course",
                    "course_catalogue_year": str(course.get("aa") or ""),
                    "course_catalogue_course_id": str(course.get("cod") or course.get("cdsId") or ""),
                    "course_catalogue_cds_cod": str(course.get("cdsCod") or ""),
                    "course_catalogue_codicione": str(course.get("codicione") or ""),
                },
            )
        )
        stats["course_documents"] += 1

    for teaching in collect_course_teachings(course):
        summary_url = teaching_url(settings, teaching)
        if summary_url in recent_urls:
            stats["skipped_recent"] += 1
            continue

        detail = dict(teaching)
        if settings.fetch_teaching_details:
            try:
                detail.update(await fetch_teaching_detail(client, settings, teaching))
            except Exception:
                stats["teaching_detail_failed"] += 1
        t_url = teaching_url(settings, detail)
        if t_url in recent_urls:
            stats["skipped_recent"] += 1
            continue
        t_title = title_text(detail.get("des_it") or detail.get("des_en") or detail.get("adCod"))
        records.append(
            processed_record_from_markdown(
                url=t_url,
                title=t_title,
                breadcrumb=["CourseCatalogue", title, "Insegnamenti", t_title],
                raw_markdown=render_teaching_markdown(detail),
                config=config,
                extra={
                    "course_catalogue_kind": "teaching",
                    "course_catalogue_year": str(detail.get("corso_aa") or course.get("aa") or ""),
                    "course_catalogue_teaching_year": str(detail.get("aa") or ""),
                    "course_catalogue_course_id": str(detail.get("corso_cod") or course.get("cod") or ""),
                    "course_catalogue_teaching_id": str(detail.get("cod") or ""),
                    "course_catalogue_teaching_code": str(detail.get("adCod") or ""),
                    "course_catalogue_cds_cod": str(detail.get("cdsCod") or course.get("cdsCod") or ""),
                },
            )
        )
        stats["teaching_documents"] += 1

    return records, stats


async def run_course_catalogue_ingest(config: dict | None = None) -> dict:
    """Esegue l'ingest CourseCatalogue e appende i record al manifest."""
    config = config or load_config()
    validate_config(config)
    settings = settings_from_config(config)
    manifest_path = project_path(config["paths"].get("processed_manifest_file", "data/processed/manifest.jsonl"))

    if not settings.enabled or not settings.course_entries:
        return {
            "enabled": False,
            "courses_configured": len(settings.course_entries),
            "processed_ok": 0,
            "failed": 0,
            "skipped_recent": 0,
            "manifest": relative_path(manifest_path),
        }

    recent_urls = recent_successful_urls(
        manifest_path,
        "course_catalogue",
        settings.skip_recent_days,
    )
    output_records: list[ProcessedRecord] = []
    failed = 0
    skipped_recent = 0
    detail_failed = 0
    course_documents = 0
    teaching_documents = 0

    headers = {"User-Agent": config["crawler"]["user_agent"], "Accept": "application/json"}
    timeout = httpx.Timeout(settings.timeout)
    async with httpx.AsyncClient(headers=headers, timeout=timeout, follow_redirects=True) as client:
        for year, course_id in settings.course_entries:
            fallback_url = f"{settings.base_url}/corsi/{year}/{course_id}"
            try:
                course = await fetch_course(client, settings, year, course_id)
                records, stats = await build_course_records(
                    course,
                    client,
                    settings,
                    config,
                    recent_urls,
                )
                output_records.extend(records)
                detail_failed += stats["teaching_detail_failed"]
                course_documents += stats["course_documents"]
                teaching_documents += stats["teaching_documents"]
                skipped_recent += stats["skipped_recent"]
            except Exception as error:
                failed += 1
                output_records.append(failed_record(fallback_url, error))

    append_jsonl_batch(manifest_path, output_records)
    processed_ok = sum(1 for record in output_records if record.get("status") == "ok")
    return {
        "enabled": True,
        "courses_configured": len(settings.course_entries),
        "processed_ok": processed_ok,
        "failed": failed,
        "course_documents": course_documents,
        "teaching_documents": teaching_documents,
        "teaching_detail_failed": detail_failed,
        "skipped_recent": skipped_recent,
        "manifest": relative_path(manifest_path),
    }


async def main() -> None:
    """Entry point CLI."""
    stats = await run_course_catalogue_ingest()
    print("CourseCatalogue completato")
    for key, value in stats.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
