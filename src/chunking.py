from __future__ import annotations

import re
from pathlib import Path
from statistics import mean, median
from urllib.parse import parse_qsl, urlparse

from chunk_metadata import (
    CHUNK_METADATA_SCHEMA_VERSION,
    chunk_metadata_value,
    compact_dict,
)
from pdf_policy import diem_rescue_upload_scope_reason
from pipeline_io import (
    BASE_DIR,
    content_hash,
    latest_records_by_url,
    load_jsonl,
    write_json,
    write_jsonl_atomic,
)


MANIFEST_PATH = BASE_DIR / "data" / "processed" / "manifest.jsonl"

CHUNKS_DIR = BASE_DIR / "data" / "processed" / "chunks"
CHUNKS_FILE = CHUNKS_DIR / "chunks.jsonl"
STATS_FILE = CHUNKS_DIR / "stats.json"

# Parametri principali
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 250
MIN_CHUNK_CHARS = 150
SYNTHETIC_CHUNK_SIZE = 1400
SYNTHETIC_CHUNK_OVERLAP = 160

# Parametri per documenti molto grandi
LARGE_DOC_THRESHOLD = 50_000
LARGE_DOC_CHUNK_SIZE = 2000
LARGE_DOC_OVERLAP = 300


PROTECTED_SECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\borario\s+di\s+ricevimento\b",
        r"\bricevimento\b",
        r"\bstrumentazione\b",
        r"\bdotazione\b",
        r"\battrezzature\b",
        r"\bcontatti?\b",
        r"\bsede\b",
        r"\bubicazione\b",
        r"\bscadenza\b",
        r"\brequisiti?\b",
        r"\bresponsabil[ei]\b",
        r"\bpubblicazioni?\b",
        r"\bdoi\b",
        r"\biris\b",
    ]
]

PUBLICATION_HEADING_RE = re.compile(
    r"^####\s+(\d+)\s*\[([^\]]+)\]\(([^)]+)\)",
    re.MULTILINE,
)

LAB_EQUIPMENT_RE = re.compile(
    r"\b(?:strumentazione|dotazione|attrezzatur[ae]|apparecchiatur[ae])\b",
    re.IGNORECASE,
)


def project_path(path_value: str) -> Path:
    """
    Converte un path salvato nel manifest in Path assoluto.
    Gestisce sia path Windows con \\ sia path Unix con /.
    """
    normalized = str(path_value).replace("\\", "/")
    return BASE_DIR / normalized


def is_valid_record(record: dict) -> bool:
    """
    Tiene solo i documenti utili per l'indicizzazione.
    """
    if diem_rescue_upload_scope_reason(
        str(record.get("url") or record.get("document_url") or ""),
        str(record.get("discovered_from") or ""),
    ):
        return False

    return (
        record.get("status") == "ok"
        and record.get("indexable") is True
        and record.get("text_extracted") is True
        and record.get("is_duplicate") is not True
        and record.get("duplicate") is not True
        and bool(record.get("index_markdown_path"))
    )


def read_markdown(record: dict) -> str:
    """
    Legge il Markdown pulito del documento.
    """
    markdown_path = project_path(record["index_markdown_path"])

    if not markdown_path.exists():
        raise FileNotFoundError(f"Markdown non trovato: {markdown_path}")

    return markdown_path.read_text(encoding="utf-8")


def remove_front_matter(text: str) -> str:
    """
    Rimuove il front matter YAML iniziale:

    ---
    url: ...
    title: ...
    ---

    I metadata li prendiamo già dal manifest, quindi non devono entrare negli embedding.
    """
    pattern = r"^\s*---\s*\n.*?\n---\s*\n?"
    return re.sub(pattern, "", text, count=1, flags=re.DOTALL).strip()

def remove_existing_context_header(text: str) -> str:
    """
    Rimuove eventuali header contestuali già presenti nel Markdown.
    Serve per evitare doppio [CONTESTO DOCUMENTO] nei chunk finali.
    """
    pattern = (
        r"^\s*\[CONTESTO DOCUMENTO\]\s*\n"
        r"(?:Titolo:.*\n)?"
        r"(?:Titolo contenuto:.*\n)?"
        r"(?:Percorso:.*\n)?"
        r"(?:Fonte:.*\n)?"
        r"(?:Fonte originaria:.*\n)?"
        r"(?:Testo link sorgente:.*\n)?"
        r"(?:Tipo documento:.*\n)?"
        r"(?:Anni documento:.*\n)?"
        r"\s*(?:\[CONTESTO SEZIONE\]\s*\n)?"
        r"(?:Sezione:.*\n)?"
        r"\s*(?:\[CONTENUTO\]\s*\n)?"
    )

    previous = None

    while previous != text:
        previous = text
        text = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE)

    return text.strip()


def normalize_text(text: str) -> str:
    """
    Normalizza il testo senza distruggere la struttura Markdown.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("¿", "'")
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()

def breadcrumb_to_text(breadcrumb: object) -> str:
    """
    Converte il breadcrumb in stringa leggibile.
    """
    if isinstance(breadcrumb, list) and breadcrumb:
        return " > ".join(str(item) for item in breadcrumb)

    return "N/D"


def clean_heading_text(text: str) -> str:
    text = re.sub(r"^#{1,6}\s+", "", text.strip())
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("**", "").replace("*", "")
    return " ".join(text.split()).strip()


def markdown_heading_from_section(section: str) -> str:
    for line in section.splitlines():
        if re.match(r"^#{1,6}\s+", line):
            return clean_heading_text(line)
    return ""


def markdown_heading_level_from_section(section: str) -> int:
    for line in section.splitlines():
        match = re.match(r"^(#{1,6})\s+", line)
        if match:
            return len(match.group(1))
    return 0


def first_content_heading(text: str) -> str:
    for section in split_by_markdown_sections(text):
        heading = markdown_heading_from_section(section)
        if heading:
            return heading
    return ""


def metadata_text(value: object) -> str:
    if value is None:
        return ""

    if isinstance(value, list):
        return " > ".join(str(item) for item in value if str(item).strip())

    if isinstance(value, dict):
        return " ".join(str(item) for item in value.values() if str(item).strip())

    return str(value).strip()


def years_from_record(record: dict) -> list[int]:
    haystack = " ".join(
        metadata_text(record.get(key))
        for key in [
            "url",
            "document_url",
            "discovered_from",
            "link_text",
            "title",
            "breadcrumb",
        ]
    )

    years = sorted({int(match) for match in re.findall(r"\b(?:19|20)\d{2}\b", haystack)})
    return years


def infer_document_type(record: dict) -> str:
    haystack = " ".join(
        metadata_text(record.get(key)).lower()
        for key in [
            "url",
            "document_url",
            "discovered_from",
            "link_text",
            "title",
            "breadcrumb",
            "pdf_download_decision",
        ]
    )

    if "almalaurea" in haystack or "profilo dei laureati" in haystack:
        return "almalaurea"
    if "__regolamenti-cds" in haystack or "regolamento" in haystack:
        return "regolamento"
    if "bando" in haystack or "graduatoria" in haystack or "concorso" in haystack:
        return "bando"
    if "sua-cds" in haystack or "schede-sua" in haystack:
        return "qualita_corso"
    if "calendario" in haystack:
        return "calendario"

    return ""


def query_params_from_url(url: str) -> dict[str, str]:
    try:
        return {
            name.lower(): value
            for name, value in parse_qsl(urlparse(url).query, keep_blank_values=True)
        }
    except ValueError:
        return {}


def first_path_segment(url: str) -> str:
    try:
        path = urlparse(url).path
    except ValueError:
        return ""

    segments = [segment for segment in path.split("/") if segment]
    return segments[0] if segments else ""


def path_segments_from_url(url: str) -> list[str]:
    try:
        path = urlparse(url).path
    except ValueError:
        return []

    return [segment for segment in path.split("/") if segment]


def teacher_name_from_record(record: dict) -> str:
    title = str(record.get("title") or "")
    if "|" in title:
        return title.split("|", 1)[0].strip()

    breadcrumb = record.get("breadcrumb")
    if isinstance(breadcrumb, list):
        for item in breadcrumb:
            item_text = str(item).strip()
            if item_text and item_text.lower() != "docenti":
                if " " in item_text:
                    parts = item_text.split()
                    if len(parts) == 2 and parts[0].isupper():
                        return f"{parts[1]} {parts[0]}".strip()
                    return item_text

    return title.strip()


def entity_metadata_from_record(record: dict) -> dict[str, object]:
    url = str(record.get("url") or record.get("document_url") or "")
    domain = urlparse(url).netloc.lower() if url else ""
    params = query_params_from_url(url)
    segments = path_segments_from_url(url)
    document_type = infer_document_type(record)
    title = str(record.get("title") or "")

    metadata: dict[str, object] = {
        "source_family": record.get("source_family") or "",
        "teacher_id": record.get("teacher_id") or "",
        "course_id": record.get("course_id") or "",
        "lab_id": record.get("lab_id") or params.get("id", ""),
        "document_type": document_type,
    }

    if not metadata["source_family"]:
        if "docenti.unisa.it" in domain:
            metadata["source_family"] = "teacher_profile"
        elif "corsi.unisa.it" in domain or "coursecatalogue" in domain:
            metadata["source_family"] = "course"
        elif "diem.unisa.it" in domain:
            metadata["source_family"] = "diem"
        elif "rubrica.unisa.it" in domain:
            metadata["source_family"] = "directory"

    if not metadata["teacher_id"] and "docenti.unisa.it" in domain:
        metadata["teacher_id"] = first_path_segment(url)

    if not metadata["course_id"] and segments:
        if "corsi.unisa.it" in domain:
            metadata["course_id"] = segments[0]

    if metadata["teacher_id"]:
        metadata["entity_type"] = "teacher"
        metadata["entity_name"] = teacher_name_from_record(record)
    elif metadata["lab_id"] and ("strutture" in url or "laboratori" in url):
        metadata["entity_type"] = "lab"
        metadata["entity_name"] = title
    elif metadata["course_id"]:
        metadata["entity_type"] = "course"
        metadata["entity_name"] = title
    elif document_type:
        metadata["entity_type"] = "document"
        metadata["entity_name"] = title
    else:
        metadata["entity_type"] = ""
        metadata["entity_name"] = title

    years = years_from_record(record)
    metadata["year"] = max(years) if years else ""

    return compact_dict(metadata)


def text_matches_any_pattern(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def is_protected_small_chunk(chunk: str, heading: str) -> bool:
    probe = f"{heading}\n{chunk}"
    return text_matches_any_pattern(probe, PROTECTED_SECTION_PATTERNS)


def is_course_catalogue_record(record: dict) -> bool:
    url = str(record.get("url") or record.get("document_url") or "").lower()
    return "coursecatalogue" in url


def parse_course_year(text: str) -> int | str:
    match = re.search(r"\b([123])\s*(?:°|o)?\s*anno\b", text, re.IGNORECASE)
    if not match:
        return ""
    return int(match.group(1))


def parse_academic_year(record: dict) -> int | str:
    url = str(record.get("url") or record.get("document_url") or "")
    match = re.search(r"/(?:corsi|insegnamenti)/((?:19|20)\d{2})(?:/|\?|$)", url)
    if match:
        return int(match.group(1))

    years = years_from_record(record)
    return max(years) if years else ""


def parse_curriculum_and_cohort(heading: str) -> tuple[str, int | str]:
    cleaned = clean_heading_text(heading)
    match = re.search(
        r"\bpercorso\s*:\s*(.+?)(?:\s*[-–]\s*coorte\s*((?:19|20)\d{2}))?$",
        cleaned,
        re.IGNORECASE,
    )
    if not match:
        return "", ""

    curriculum = " ".join(match.group(1).split()).strip()
    cohort = int(match.group(2)) if match.group(2) else ""
    return curriculum, cohort


def course_name_from_record(record: dict, document_content_title: str = "") -> str:
    title = str(record.get("title") or "").strip()
    candidate = document_content_title or title
    candidate = clean_heading_text(candidate)

    if "|" in candidate:
        candidate = candidate.split("|", 1)[0].strip()

    return " ".join(candidate.split()).strip()


def infer_course_level(record: dict, text: str = "") -> str:
    probe = " ".join(
        metadata_text(record.get(key)).lower()
        for key in ["url", "document_url", "title", "breadcrumb", "link_text"]
    )
    probe = f"{probe} {text[:1200].lower()}"

    if "laurea magistrale" in probe or "lm-" in probe:
        return "laurea_magistrale"
    if "dottorato" in probe or "phd" in probe:
        return "dottorato"
    if "laurea triennale" in probe or "corso di laurea" in probe or "l-" in probe:
        return "laurea_triennale"
    return ""


def topic_family_from_record(record: dict, chunk_kind: str, text: str = "") -> str:
    url = str(record.get("url") or record.get("document_url") or "").lower()
    probe = " ".join(
        [
            url,
            metadata_text(record.get("title")).lower(),
            metadata_text(record.get("breadcrumb")).lower(),
            chunk_kind.lower(),
            text[:800].lower(),
        ]
    )

    if chunk_kind in {"study_plan", "course_syllabus", "course_info", "course_statistic"}:
        return "didattica"
    if chunk_kind in {"office_hours", "teacher_publications_page", "teacher_projects", "publication_summary"}:
        return "docenti"
    if chunk_kind == "lab_equipment" or "/ricerca/laboratori" in url or "/dipartimento/strutture" in url:
        return "laboratori"
    if chunk_kind in {"erasmus"} or "erasmus" in probe or "international" in probe:
        return "international"
    if chunk_kind == "phd" or "dottorato" in probe:
        return "dottorati"
    if "terza-missione" in url or "trasferimento tecnologico" in probe or "public engagement" in probe:
        return "terza_missione"
    if "/ricerca/" in url or "progetti finanziati" in probe or "aree di ricerca" in probe:
        return "ricerca"
    if "bando" in probe or "graduatoria" in probe or "avviso" in probe:
        return "bandi"
    if "almalaurea" in probe or "sua-cds" in probe or "qualita" in probe or "qualità" in probe:
        return "qualita_statistiche"
    if "/dipartimento/" in url:
        return "dipartimento"
    return "generale"


def is_study_plan_record(record: dict, section_heading: str, text: str) -> bool:
    metadata_probe = " ".join(
        metadata_text(record.get(key)).lower()
        for key in [
            "url",
            "document_url",
            "discovered_from",
            "link_text",
            "title",
            "breadcrumb",
        ]
    )
    section_probe = section_heading.lower()
    body_probe = text[:2200].lower()

    if any(
        marker in metadata_probe
        for marker in [
            "__piano-studi-cds",
            "/didattica/piano-di-studi",
            "piano di studi a.a",
            "piano degli studi a.a",
            "manifesto degli studi",
        ]
    ):
        return True

    title_or_section_probe = f"{metadata_probe}\n{section_probe}"
    has_plan_title = any(
        marker in title_or_section_probe
        for marker in [
            "piano di studi",
            "piano degli studi",
            "manifesto degli studi",
        ]
    )
    has_curricular_table_signal = bool(
        re.search(r"\b[123]\s*[°o]\s+anno\b", body_probe)
        or re.search(r"\b(?:ssd|cfu|taf|ambito|curriculum)\b", body_probe)
    )

    return has_plan_title and has_curricular_table_signal


def update_study_plan_context(
    context: dict[str, object],
    record: dict,
    section: str,
    section_heading: str,
) -> dict[str, object]:
    """
    Propaga il contesto logico dei piani di studio ai chunk figli.

    Le pagine CourseCatalogue hanno spesso questa forma:
    ## Piano di studi
    ## Percorso: SOFTWARE - COORTE 2025
    ### 1 anno
    ### 2 anno

    Senza ereditarietà, i chunk "2 anno" e "3 anno" vengono classificati come
    testo generico e il retrieval perde proprio le risposte multi-anno.
    """
    heading = clean_heading_text(section_heading)
    heading_lower = heading.lower()
    level = markdown_heading_level_from_section(section)
    current = dict(context)

    if is_course_catalogue_record(record):
        if "piano di studi" in heading_lower or "piano degli studi" in heading_lower:
            current["inside_study_plan"] = True
            current["study_plan_parent_heading"] = heading or "Piano di studi"
            current.pop("curriculum", None)
            current.pop("course_year", None)

        curriculum, cohort = parse_curriculum_and_cohort(heading)
        if curriculum:
            current["inside_study_plan"] = True
            current["curriculum"] = curriculum
            if cohort:
                current["cohort"] = cohort

        course_year = parse_course_year(heading)
        if course_year:
            current["inside_study_plan"] = True
            current["course_year"] = course_year

        if (
            level in {1, 2}
            and heading
            and not any(marker in heading_lower for marker in ["piano di studi", "piano degli studi", "percorso:"])
            and not parse_course_year(heading)
        ):
            current.pop("inside_study_plan", None)
            current.pop("study_plan_parent_heading", None)
            current.pop("curriculum", None)
            current.pop("course_year", None)

    return current


def study_plan_metadata_from_context(
    context: dict[str, object],
    record: dict,
    section_heading: str,
    chunk_body: str,
    document_content_title: str,
) -> dict[str, object]:
    chunk_is_study_plan = bool(context.get("inside_study_plan")) or is_study_plan_record(
        record,
        section_heading,
        chunk_body,
    )

    if not chunk_is_study_plan:
        return {}

    metadata: dict[str, object] = {
        "course_name": course_name_from_record(record, document_content_title),
        "course_level": infer_course_level(record, chunk_body),
        "academic_year": parse_academic_year(record),
        "study_plan_parent_heading": context.get("study_plan_parent_heading") or "Piano di studi",
        "curriculum": context.get("curriculum") or "",
        "cohort": context.get("cohort") or "",
        "course_year": context.get("course_year") or parse_course_year(f"{section_heading}\n{chunk_body[:300]}"),
    }

    return compact_dict(metadata)


def is_course_syllabus_section(record: dict, section_heading: str, text: str) -> bool:
    if not is_course_catalogue_record(record):
        return False

    probe = f"{section_heading}\n{text[:1800]}".lower()
    return any(
        marker in probe
        for marker in [
            "obiettivi formativi",
            "contenuti",
            "metodi didattici",
            "verifica dell'apprendimento",
            "testi",
            "modalita esame",
            "modalità esame",
            "ssd:",
        ]
    )


def is_lab_or_structure_record(record: dict) -> bool:
    url = str(record.get("url") or record.get("document_url") or "").lower()
    return "/ricerca/laboratori" in url or "/dipartimento/strutture" in url


def is_lab_equipment_section(record: dict, section_heading: str, text: str) -> bool:
    """
    Riconosce sezioni di strumentazione vera, evitando falsi positivi nei corsi.
    """
    if is_course_catalogue_record(record):
        return False

    heading_has_equipment_signal = bool(LAB_EQUIPMENT_RE.search(section_heading))
    if heading_has_equipment_signal:
        return True

    if is_lab_or_structure_record(record):
        return bool(LAB_EQUIPMENT_RE.search(text[:1200]))

    return False


def classify_chunk_kind(
    chunk_body: str,
    section_heading: str,
    record: dict,
) -> str:
    probe = f"{section_heading}\n{chunk_body}".lower()
    url = str(record.get("url") or record.get("document_url") or "").lower()
    document_type = infer_document_type(record)

    if "orario di ricevimento" in probe or "ricevimento" in section_heading.lower():
        return "office_hours"
    if is_lab_equipment_section(record, section_heading, chunk_body):
        return "lab_equipment"
    if document_type == "almalaurea" or "almalaurea" in probe:
        return "course_statistic"
    if is_study_plan_record(record, section_heading, chunk_body):
        return "study_plan"
    if is_course_syllabus_section(record, section_heading, chunk_body):
        return "course_syllabus"
    if document_type in {"bando", "regolamento", "calendario"}:
        return "official_document"
    if "docenti.unisa.it" in url and "/ricerca/pubblicazioni" in url:
        return "teacher_publications_page"
    if "docenti.unisa.it" in url and "/ricerca/progetti" in url:
        return "teacher_projects"
    if "erasmus" in probe or "international" in url:
        return "erasmus"
    if "dottorato" in probe or "phd" in probe:
        return "phd"
    if "offerta formativa" in probe or "piano di studi" in probe:
        return "course_info"

    return "text"


def build_context_header(
    record: dict,
    document_content_title: str = "",
    section_heading: str = "",
) -> str:
    """
    Header contestuale da aggiungere a ogni chunk.

    Serve perché molti chunk possono contenere frasi generiche come:
    - Contatti
    - Didattica
    - Calendario
    - Insegnamenti

    Con l'header, il retriever capisce da quale documento arriva quel testo.
    """
    title = record.get("title") or "Documento senza titolo"
    url = record.get("url") or record.get("document_url") or ""
    breadcrumb_text = breadcrumb_to_text(record.get("breadcrumb"))
    discovered_from = metadata_text(record.get("discovered_from"))
    link_text = metadata_text(record.get("link_text"))
    document_type = infer_document_type(record)
    years = years_from_record(record)

    lines = [
        "[CONTESTO DOCUMENTO]",
        f"Titolo: {title}",
    ]

    if document_content_title and document_content_title.lower() != str(title).lower():
        lines.append(f"Titolo contenuto: {document_content_title}")

    lines.extend(
        [
            f"Percorso: {breadcrumb_text}",
            f"Fonte: {url}",
        ]
    )

    if discovered_from:
        lines.append(f"Fonte originaria: {discovered_from}")

    if link_text:
        lines.append(f"Testo link sorgente: {link_text}")

    if document_type:
        lines.append(f"Tipo documento: {document_type}")

    if years:
        lines.append(f"Anni documento: {', '.join(str(year) for year in years)}")

    if section_heading:
        lines.extend(["", "[CONTESTO SEZIONE]", f"Sezione: {section_heading}"])

    lines.extend(["", "[CONTENUTO]"])

    return "\n".join(lines) + "\n"


def build_locator_text(metadata: dict[str, object]) -> str:
    """
    Rappresentazione compatta per il retrieval di tipo "locator".

    Non sostituisce il body embedding: è un secondo segnale, corto e fielded,
    utile per trovare il documento giusto quando la query nomina corso, docente,
    anno, curriculum, fonte o tipo di informazione.
    """
    fields = [
        ("fonte", metadata.get("source_family")),
        ("topic", metadata.get("topic_family")),
        ("tipo_chunk", metadata.get("chunk_kind")),
        ("titolo", metadata.get("title")),
        ("titolo_contenuto", metadata.get("content_title")),
        ("sezione", metadata.get("section_heading")),
        ("corso", metadata.get("course_name") or metadata.get("entity_name")),
        ("livello", metadata.get("course_level")),
        ("curriculum", metadata.get("curriculum")),
        ("anno_corso", metadata.get("course_year")),
        ("anno_accademico", metadata.get("academic_year")),
        ("coorte", metadata.get("cohort")),
        ("docente", metadata.get("entity_name") if metadata.get("entity_type") == "teacher" else ""),
        ("docente_id", metadata.get("teacher_id")),
        ("laboratorio", metadata.get("entity_name") if metadata.get("entity_type") == "lab" else ""),
        ("tipo_documento", metadata.get("document_type")),
        ("anni_documento", metadata.get("document_years")),
        ("breadcrumb", metadata.get("breadcrumb_text")),
        ("url", metadata.get("source_url") or metadata.get("document_url")),
    ]

    parts = [
        f"{name}={metadata_text(value)}"
        for name, value in fields
        if metadata_text(value)
    ]
    return "; ".join(parts)


def split_by_markdown_sections(text: str) -> list[str]:
    """
    Divide il documento in sezioni Markdown.
    Ogni nuova sezione parte da un heading:

    # Titolo
    ## Sezione
    ### Sottosezione
    """
    lines = text.splitlines()
    sections: list[str] = []
    current: list[str] = []

    heading_pattern = re.compile(r"^#{1,6}\s+")

    for line in lines:
        if heading_pattern.match(line) and current:
            section = "\n".join(current).strip()
            if section:
                sections.append(section)
            current = [line]
        else:
            current.append(line)

    if current:
        section = "\n".join(current).strip()
        if section:
            sections.append(section)

    return sections


def find_best_cut(text: str, start: int, target_end: int) -> int:
    """
    Cerca un punto di taglio naturale prima di target_end.
    Preferisce paragrafi, righe, frasi e spazi.
    """
    min_end = start + int((target_end - start) * 0.55)
    window = text[start:target_end]

    separators = ["\n\n", "\n", ". ", "; ", ", ", " "]

    best_cut = -1

    for sep in separators:
        pos = window.rfind(sep)

        if pos != -1:
            candidate = start + pos + len(sep)

            if candidate >= min_end:
                best_cut = candidate
                break

    if best_cut == -1 or best_cut <= start:
        best_cut = target_end

    return best_cut


def split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Spezza una sezione troppo lunga in chunk.
    Usa tagli naturali quando possibile e mantiene overlap.
    """
    text = text.strip()

    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        target_end = min(start + chunk_size, text_len)

        if target_end >= text_len:
            cut = text_len
        else:
            cut = find_best_cut(text, start, target_end)

        chunk = text[start:cut].strip()

        if chunk:
            chunks.append(chunk)

        if cut >= text_len:
            break

        next_start = max(cut - overlap, 0)

        if next_start <= start:
            next_start = cut

        start = next_start

    return chunks


def is_protected_short_section(chunk: str) -> bool:
    """
    Alcune sezioni sono corte ma importanti e non devono essere eliminate.
    Esempio: orari di ricevimento con una sola riga.
    """
    chunk_lower = chunk.lower()

    protected_markers = [
        "orario di ricevimento",
        "ricevimento",
    ]

    return any(marker in chunk_lower for marker in protected_markers)


def merge_small_chunks(chunks: list[str], min_chars: int, max_chars: int) -> list[str]:
    """
    Unisce chunk troppo piccoli quando possibile.
    Evita chunk inutili da poche parole, ma non elimina sezioni corte importanti
    come gli orari di ricevimento.
    """
    if not chunks:
        return []

    merged: list[str] = []
    buffer = ""

    for chunk in chunks:
        chunk = chunk.strip()

        if not chunk:
            continue

        if not buffer:
            buffer = chunk
            continue

        candidate = buffer + "\n\n" + chunk

        # Se il buffer è piccolo oppure il nuovo chunk è piccolo,
        # proviamo a unirli invece di rischiare di perdere il chunk corto.
        if (len(buffer) < min_chars or len(chunk) < min_chars) and len(candidate) <= max_chars:
            buffer = candidate
        else:
            merged.append(buffer.strip())
            buffer = chunk

    if buffer:
        merged.append(buffer.strip())

    good_chunks = [
        chunk
        for chunk in merged
        if len(chunk) >= min_chars or is_protected_short_section(chunk)
    ]

    if good_chunks:
        return good_chunks

    return [max(merged, key=len)] if merged else []


def compatible_chunk_metadata_for_merge(
    left: dict[str, object],
    right: dict[str, object],
) -> bool:
    """
    Evita di fondere sezioni che rappresentano anni/curricula diversi.
    Il merge dei chunk piccoli è utile, ma non deve collassare 1/2/3 anno
    in un solo chunk indistinguibile.
    """
    protected_keys = ("course_year", "curriculum", "cohort")
    for key in protected_keys:
        left_value = left.get(key)
        right_value = right.get(key)
        if left_value and right_value and left_value != right_value:
            return False
    return True


def merge_small_chunk_records(
    chunks: list[tuple[str, str, dict[str, object]]],
    min_chars: int,
    max_chars: int,
) -> list[tuple[str, str, dict[str, object]]]:
    """
    Versione di merge_small_chunks che preserva l'heading della sezione.
    """
    if not chunks:
        return []

    merged: list[tuple[str, str, dict[str, object]]] = []
    buffer = ""
    buffer_heading = ""
    buffer_metadata: dict[str, object] = {}

    for chunk, heading, metadata in chunks:
        chunk = chunk.strip()

        if not chunk:
            continue

        if not buffer:
            buffer = chunk
            buffer_heading = heading
            buffer_metadata = dict(metadata)
            continue

        candidate = buffer + "\n\n" + chunk

        if (
            len(buffer) < min_chars
            and len(candidate) <= max_chars
            and compatible_chunk_metadata_for_merge(buffer_metadata, metadata)
        ):
            buffer = candidate
            if not buffer_heading:
                buffer_heading = heading
            buffer_metadata = {**dict(metadata), **buffer_metadata}
        else:
            merged.append((buffer.strip(), buffer_heading, buffer_metadata))
            buffer = chunk
            buffer_heading = heading
            buffer_metadata = dict(metadata)

    if buffer:
        merged.append((buffer.strip(), buffer_heading, buffer_metadata))

    good_chunks = [
        item
        for item in merged
        if (
            len(item[0]) >= min_chars
            or is_protected_small_chunk(item[0], item[1])
            or bool(item[2].get("inside_study_plan"))
        )
    ]

    if good_chunks:
        return good_chunks

    return [max(merged, key=lambda item: len(item[0]))] if merged else []


def make_chunk_id(document_hash: str, chunk_index: int, text: str) -> str:
    """
    Crea un ID stabile per il chunk.
    """
    chunk_hash = content_hash(text)[:16]
    return f"{document_hash}_{chunk_index:04d}_{chunk_hash}"


def clean_inline_markdown(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("**", "").replace("*", "")
    text = text.replace("\\", "")
    return " ".join(text.split()).strip()


def compact_summary_field(value: object, max_chars: int) -> str:
    text = " ".join(str(value or "").split()).strip()

    if len(text) <= max_chars:
        return text

    cut = find_best_cut(text, 0, max_chars)
    return text[:cut].rstrip(" ,;.") + " [...]"


def table_cells_from_line(line: str) -> list[str]:
    if "|" not in line:
        return []

    return [
        clean_inline_markdown(cell)
        for cell in line.strip().strip("|").split("|")
        if clean_inline_markdown(cell)
    ]


def split_publication_entries(text: str) -> list[dict[str, object]]:
    matches = list(PUBLICATION_HEADING_RE.finditer(text))
    entries: list[dict[str, object]] = []

    for order, match in enumerate(matches, start=1):
        start = match.start()
        end = matches[order].start() if order < len(matches) else len(text)
        block = text[start:end].strip()
        publication_id = match.group(1).strip()
        title = clean_inline_markdown(match.group(2))
        anchor_url = match.group(3).strip()

        lines = [line.strip() for line in block.splitlines() if line.strip()]
        table_rows = [
            table_cells_from_line(line)
            for line in lines
            if "|" in line and "---" not in line
        ]
        table_rows = [row for row in table_rows if row]

        year = ""
        publication_type = ""
        citation = ""
        authors = ""
        venue = ""

        for row in table_rows:
            row_text = " ".join(row)
            year_match = re.search(r"\b(?:19|20)\d{2}\b", row_text)
            if not year and year_match:
                year = year_match.group(0)
                continue

            if not publication_type and len(row_text) < 90 and not any(
                marker in row_text.lower()
                for marker in ["doi", "codice identificativo", "visualizza"]
            ):
                publication_type = row_text
                continue

            if title.lower() in row_text.lower() and not citation:
                citation = row_text
                continue

            if not authors and ";" in row_text and "doi" not in row_text.lower():
                authors = row_text
                continue

            if not venue and any(month in row_text.lower() for month in [
                "gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno",
                "luglio", "agosto", "settembre", "ottobre", "novembre", "dicembre",
                "january", "february", "march", "april", "june", "july",
                "august", "september", "october", "november", "december",
            ]):
                venue = row_text

        doi_match = re.search(r"10\.\d{4,9}/[^\s\]\)\"|]+", block, re.IGNORECASE)
        iris_match = re.search(r"\(http://hdl\.handle\.net/[^)]+", block)

        entries.append(
            compact_dict(
                {
                    "publication_id": publication_id,
                    "publication_title": title,
                    "publication_anchor_url": anchor_url,
                    "publication_year": int(year) if year else "",
                    "publication_type": publication_type,
                    "publication_venue": venue or citation,
                    "publication_authors": authors,
                    "publication_doi": doi_match.group(0).rstrip(".") if doi_match else "",
                    "publication_iris_url": iris_match.group(0).lstrip("(") if iris_match else "",
                    "publication_order": order,
                    "block": block,
                }
            )
        )

    return entries


def publication_summary_text(record: dict, publication: dict[str, object]) -> str:
    teacher_name = teacher_name_from_record(record)
    lines = [
        "[SCHEDA PUBBLICAZIONE]",
        "Tipo chunk: publication_summary",
        f"Docente: {teacher_name}",
        f"Titolo pubblicazione: {compact_summary_field(publication.get('publication_title'), 500)}",
    ]

    if publication.get("publication_year"):
        lines.append(f"Anno: {publication.get('publication_year')}")
    if publication.get("publication_type"):
        lines.append(f"Tipologia: {compact_summary_field(publication.get('publication_type'), 180)}")
    if publication.get("publication_authors"):
        lines.append(f"Autori: {compact_summary_field(publication.get('publication_authors'), 700)}")
    if publication.get("publication_venue"):
        lines.append(f"Sede o rivista: {compact_summary_field(publication.get('publication_venue'), 700)}")
    if publication.get("publication_doi"):
        lines.append(f"DOI: {publication.get('publication_doi')}")
    if publication.get("publication_iris_url"):
        lines.append(f"IRIS: {publication.get('publication_iris_url')}")

    lines.append(f"Fonte: {record.get('url') or record.get('document_url') or ''}")
    return "\n".join(str(line) for line in lines if str(line).strip())


def office_hours_summary_text(
    record: dict,
    section: str,
    *,
    part_index: int | None = None,
    part_count: int | None = None,
) -> str:
    teacher_name = teacher_name_from_record(record)
    lines = [
        "[SCHEDA RICEVIMENTO]",
        "Tipo chunk: office_hours",
        f"Docente: {teacher_name}",
    ]

    if part_index and part_count and part_count > 1:
        lines.append(f"Parte: {part_index}/{part_count}")

    lines.extend(
        [
            compact_summary_field(clean_inline_markdown(section), 700),
            "",
            section,
        ]
    )
    return "\n".join(lines).strip()


def office_hours_chunk_specs(
    record: dict,
    section: str,
    section_heading: str,
) -> list[dict[str, object]]:
    parts = split_long_text(
        text=section,
        chunk_size=SYNTHETIC_CHUNK_SIZE,
        overlap=SYNTHETIC_CHUNK_OVERLAP,
    )
    part_count = len(parts)
    specs: list[dict[str, object]] = []

    for index, part in enumerate(parts, start=1):
        extra_metadata = {}
        if part_count > 1:
            extra_metadata = {
                "chunk_part_index": index,
                "chunk_part_count": part_count,
            }

        specs.append(
            {
                "body": office_hours_summary_text(
                    record,
                    part,
                    part_index=index,
                    part_count=part_count,
                ),
                "section_heading": section_heading or "Orario di Ricevimento",
                "chunk_kind": "office_hours",
                "extra_metadata": extra_metadata,
            }
        )

    return specs


def lab_equipment_summary_text(
    record: dict,
    section: str,
    section_heading: str,
    *,
    part_index: int | None = None,
    part_count: int | None = None,
) -> str:
    lines = [
        "[SCHEDA STRUMENTAZIONE LABORATORIO]",
        "Tipo chunk: lab_equipment",
        f"Titolo: {record.get('title') or ''}",
        f"Sezione: {section_heading}",
        f"Fonte: {record.get('url') or record.get('document_url') or ''}",
    ]

    if part_index and part_count and part_count > 1:
        lines.append(f"Parte: {part_index}/{part_count}")

    lines.extend(["", section])
    return "\n".join(lines).strip()


def lab_equipment_chunk_specs(
    record: dict,
    section: str,
    section_heading: str,
) -> list[dict[str, object]]:
    parts = split_long_text(
        text=section,
        chunk_size=SYNTHETIC_CHUNK_SIZE,
        overlap=SYNTHETIC_CHUNK_OVERLAP,
    )
    part_count = len(parts)
    specs: list[dict[str, object]] = []

    for index, part in enumerate(parts, start=1):
        extra_metadata = {}
        if part_count > 1:
            extra_metadata = {
                "chunk_part_index": index,
                "chunk_part_count": part_count,
            }

        specs.append(
            {
                "body": lab_equipment_summary_text(
                    record,
                    part,
                    section_heading,
                    part_index=index,
                    part_count=part_count,
                ),
                "section_heading": section_heading or "Strumentazione",
                "chunk_kind": "lab_equipment",
                "extra_metadata": extra_metadata,
            }
        )

    return specs


def short_document_summary_text(record: dict, text: str, chunk_kind: str) -> str:
    title = record.get("title") or "Documento senza titolo"
    snippet = clean_inline_markdown(text[:1800])
    return "\n".join(
        [
            "[SCHEDA DOCUMENTO]",
            f"Tipo chunk: {chunk_kind}",
            f"Titolo: {title}",
            f"Fonte: {record.get('url') or record.get('document_url') or ''}",
            f"Tipo documento: {infer_document_type(record)}",
            "",
            snippet,
        ]
    ).strip()


def synthetic_chunk_specs(record: dict, text: str, sections: list[str]) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    url = str(record.get("url") or record.get("document_url") or "")
    document_type = infer_document_type(record)

    if "docenti.unisa.it" in url and "/ricerca/pubblicazioni" in url:
        for publication in split_publication_entries(text):
            title = str(publication.get("publication_title") or "")
            body = publication_summary_text(record, publication)
            specs.append(
                {
                    "body": body,
                    "section_heading": f"Pubblicazione: {title}",
                    "chunk_kind": "publication_summary",
                    "extra_metadata": {
                        key: publication.get(key)
                        for key in [
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
                    },
                }
            )

    for section in sections:
        section_heading = markdown_heading_from_section(section)
        section_probe = f"{section_heading}\n{section}".lower()

        if "orario di ricevimento" in section_probe or "ricevimento" in section_heading.lower():
            specs.extend(office_hours_chunk_specs(record, section, section_heading))

        if is_lab_equipment_section(record, section_heading, section_probe):
            specs.extend(lab_equipment_chunk_specs(record, section, section_heading))

    if document_type == "almalaurea" or "almalaurea" in url.lower():
        specs.append(
            {
                "body": short_document_summary_text(record, text, "course_statistic"),
                "section_heading": "Statistiche corso",
                "chunk_kind": "course_statistic",
                "extra_metadata": {},
            }
        )

    if document_type in {"bando", "regolamento", "calendario"}:
        specs.append(
            {
                "body": short_document_summary_text(record, text, "official_document_summary"),
                "section_heading": "Documento ufficiale",
                "chunk_kind": "official_document_summary",
                "extra_metadata": {},
            }
        )

    return specs


def chunk_document(record: dict) -> list[dict]:
    """
    Produce i chunk di un singolo documento.
    """
    raw_markdown = read_markdown(record)

    text = remove_front_matter(raw_markdown)
    text = remove_existing_context_header(text)
    text = normalize_text(text)

    if not text:
        return []

    markdown_chars = int(record.get("markdown_chars") or len(text))

    if markdown_chars >= LARGE_DOC_THRESHOLD:
        chunk_size = LARGE_DOC_CHUNK_SIZE
        overlap = LARGE_DOC_OVERLAP
    else:
        chunk_size = CHUNK_SIZE
        overlap = CHUNK_OVERLAP

    sections = split_by_markdown_sections(text)

    preliminary_chunks: list[tuple[str, str, dict[str, object]]] = []
    study_plan_context: dict[str, object] = {}

    for section in sections:
        section_heading = markdown_heading_from_section(section)
        study_plan_context = update_study_plan_context(
            study_plan_context,
            record,
            section,
            section_heading,
        )
        section_metadata = dict(study_plan_context)

        if len(section) <= chunk_size:
            preliminary_chunks.append((section, section_heading, section_metadata))
        else:
            preliminary_chunks.extend(
                (chunk, section_heading, section_metadata)
                for chunk in split_long_text(
                    text=section,
                    chunk_size=chunk_size,
                    overlap=overlap,
                )
            )

    merged_chunks = merge_small_chunk_records(
        chunks=preliminary_chunks,
        min_chars=MIN_CHUNK_CHARS,
        max_chars=chunk_size,
    )

    document_content_title = first_content_heading(text)

    document_hash = str(record.get("hash") or content_hash(text)[:16])
    record_years = years_from_record(record)
    document_type = infer_document_type(record)
    entity_metadata = entity_metadata_from_record(record)
    chunks: list[dict] = []

    chunk_specs: list[dict[str, object]] = [
        {
            "body": chunk_body,
            "section_heading": section_heading,
            "chunk_kind": (
                "study_plan"
                if study_plan_metadata_from_context(
                    section_context,
                    record,
                    section_heading,
                    chunk_body,
                    document_content_title,
                )
                else classify_chunk_kind(chunk_body, section_heading, record)
            ),
            "extra_metadata": study_plan_metadata_from_context(
                section_context,
                record,
                section_heading,
                chunk_body,
                document_content_title,
            ),
        }
        for chunk_body, section_heading, section_context in merged_chunks
    ]
    chunk_specs.extend(synthetic_chunk_specs(record, text, sections))

    for index, chunk_spec in enumerate(chunk_specs):
        chunk_body = str(chunk_spec["body"])
        section_heading = str(chunk_spec.get("section_heading") or "")
        chunk_kind = str(chunk_spec.get("chunk_kind") or "text")
        extra_metadata = chunk_spec.get("extra_metadata")
        if not isinstance(extra_metadata, dict):
            extra_metadata = {}

        context_header = build_context_header(
            record,
            document_content_title=document_content_title,
            section_heading=section_heading,
        )
        chunk_body_clean = chunk_body.strip()
        final_text = context_header + chunk_body_clean
        text_hash = content_hash(final_text)

        chunk_id = make_chunk_id(document_hash, index, final_text)
        retrieval_metadata = compact_dict(
            {
                "source": record.get("source"),
                "source_url": record.get("url"),
                "document_url": record.get("document_url"),
                "title": record.get("title"),
                "breadcrumb": record.get("breadcrumb", []),
                "breadcrumb_text": breadcrumb_to_text(record.get("breadcrumb")),
                "content_title": document_content_title,
                "section_heading": section_heading,
                "discovered_from": record.get("discovered_from"),
                "link_text": record.get("link_text"),
                "pdf_source_section": record.get("pdf_source_section"),
                "chunk_kind": chunk_kind,
                "topic_family": topic_family_from_record(record, chunk_kind, chunk_body_clean),
                **entity_metadata,
                "document_years": record_years,
                "document_type": document_type,
                **extra_metadata,
            }
        )
        locator_text = build_locator_text(retrieval_metadata)
        provenance = compact_dict(
            {
                "index_markdown_path": record.get("index_markdown_path"),
                "last_crawled": record.get("last_crawled"),
            }
        )
        debug = compact_dict(
            {
                "pdf_download_decision": record.get("pdf_download_decision"),
                "pdf_match_keywords": record.get("pdf_match_keywords", []),
                "clean_status": record.get("clean_status"),
                "clean_warnings": record.get("clean_warnings", []),
            }
        )

        chunk = {
            "chunk_id": chunk_id,
            "chunk_metadata_schema_version": CHUNK_METADATA_SCHEMA_VERSION,
            "document_hash": document_hash,
            "document_content_hash": record.get("content_hash"),
            "chunk_index": index,
            "chunk_count": None,
            "retrieval_metadata": retrieval_metadata,
            "provenance": provenance,
            "debug": debug,
            "text": final_text,
            "body_text": chunk_body_clean,
            "locator_text": locator_text,
            "text_for_embedding": chunk_body_clean,
            "text_for_display": final_text,
            "text_hash": text_hash,
            "chars": len(final_text),
        }

        chunks.append(chunk)

    for chunk in chunks:
        chunk["chunk_count"] = len(chunks)

    return chunks


def load_valid_records() -> list[dict]:
    """
    Legge il manifest append-only e restituisce solo lo stato corrente dei documenti validi.
    """
    history_records = load_jsonl(MANIFEST_PATH)
    current_records = latest_records_by_url(history_records)

    valid_records = [record for record in current_records if is_valid_record(record)]

    print(f"Record storici nel manifest: {len(history_records)}")
    print(f"Record correnti: {len(current_records)}")
    print(f"Documenti validi per chunking: {len(valid_records)}")

    return valid_records


def domain_from_url(url: str | None) -> str:
    """
    Estrae il dominio da un URL.
    """
    if not url:
        return "unknown"

    parsed = urlparse(url)
    return parsed.netloc or "unknown"


def build_stats(
    valid_records: list[dict],
    chunks: list[dict],
    failed_documents: list[dict],
) -> dict:
    """
    Crea statistiche del chunking.
    """
    chunk_lengths = [chunk["chars"] for chunk in chunks]

    by_source: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    chunks_by_document: dict[str, int] = {}

    for chunk in chunks:
        source = chunk_metadata_value(chunk, "source") or "unknown"
        by_source[source] = by_source.get(source, 0) + 1

        domain = domain_from_url(chunk_metadata_value(chunk, "source_url"))
        by_domain[domain] = by_domain.get(domain, 0) + 1

        document_hash = chunk.get("document_hash") or "unknown"
        chunks_by_document[document_hash] = chunks_by_document.get(document_hash, 0) + 1

    top_documents_by_chunks = sorted(
        [
            {
                "document_hash": document_hash,
                "chunks": count,
            }
            for document_hash, count in chunks_by_document.items()
        ],
        key=lambda item: item["chunks"],
        reverse=True,
    )[:10]

    return {
        "valid_documents": len(valid_records),
        "total_chunks": len(chunks),
        "failed_documents_count": len(failed_documents),
        "failed_documents": failed_documents,
        "chunk_chars": {
            "min": min(chunk_lengths) if chunk_lengths else 0,
            "max": max(chunk_lengths) if chunk_lengths else 0,
            "avg": round(mean(chunk_lengths), 2) if chunk_lengths else 0,
            "median": median(chunk_lengths) if chunk_lengths else 0,
        },
        "by_source": by_source,
        "by_domain": by_domain,
        "top_documents_by_chunks": top_documents_by_chunks,
        "config": {
            "chunk_metadata_schema_version": CHUNK_METADATA_SCHEMA_VERSION,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "min_chunk_chars": MIN_CHUNK_CHARS,
            "large_doc_threshold": LARGE_DOC_THRESHOLD,
            "large_doc_chunk_size": LARGE_DOC_CHUNK_SIZE,
            "large_doc_overlap": LARGE_DOC_OVERLAP,
        },
        "output_file": str(CHUNKS_FILE.relative_to(BASE_DIR)),
    }


def build_all_chunks() -> None:
    """
    Esegue tutto lo step di chunking.
    """
    valid_records = load_valid_records()

    all_chunks: list[dict] = []
    failed_documents: list[dict] = []

    for record in valid_records:
        try:
            document_chunks = chunk_document(record)
            all_chunks.extend(document_chunks)
        except Exception as error:
            failed_documents.append(
                {
                    "url": record.get("url"),
                    "title": record.get("title"),
                    "hash": record.get("hash"),
                    "error": str(error),
                }
            )

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    write_jsonl_atomic(CHUNKS_FILE, all_chunks)

    stats = build_stats(
        valid_records=valid_records,
        chunks=all_chunks,
        failed_documents=failed_documents,
    )

    write_json(STATS_FILE, stats)

    print()
    print("Chunking completato.")
    print(f"Chunk creati: {len(all_chunks)}")
    print(f"File chunks: {CHUNKS_FILE.relative_to(BASE_DIR)}")
    print(f"File stats: {STATS_FILE.relative_to(BASE_DIR)}")

    if failed_documents:
        print(f"Documenti falliti: {len(failed_documents)}")


def main() -> None:
    build_all_chunks()


if __name__ == "__main__":
    main()
