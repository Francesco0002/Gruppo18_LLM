"""
Filtri URL per la discovery.

Le regole riflettono lo scope dell'assignment:
- pagine sotto www.diem.unisa.it;
- profili docenti DIEM sotto docenti.unisa.it, autorizzati dal personale DIEM;
- corsi DIEM sotto corsi.unisa.it, riconosciuti da percorsi/codici in config;
- consigli didattici DIEM sotto cd.unisa.it, riconosciuti dai percorsi corso;
- PDF referenziati da pagine in scope.
- query parametriche bloccate di default, con allowlist puntuali per archivi
  informativi verificati.

Tabella delle regole principali:

| Dominio / risorsa     | Regola                                                     |
|-----------------------|------------------------------------------------------------|
| www.diem.unisa.it     | Ammesso.                                                   |
| rubrica.unisa.it      | Solo contatti scoperti dal personale DIEM.                 |
| docenti.unisa.it      | Solo profili whitelistati dal personale DIEM.              |
| corsi.unisa.it        | Solo percorsi/codici DIEM configurati.                     |
| cd.unisa.it           | Solo consigli didattici dei corsi DIEM configurati.        |
| uploads PDF           | Rilevati; scaricati solo se robots.txt lo permette.        |
| query `archive`        | Solo valori allowlistati per news/eventi DIEM.              |
"""

from __future__ import annotations

import base64
import binascii
import re
from pathlib import Path
from urllib.parse import parse_qsl, unquote, urlencode, urldefrag, urlparse

from pdf_policy import diem_rescue_upload_scope_reason, is_explicit_diem_bandi_url


TRACKING_QUERY_PREFIXES = ("utm_",)
INVALID_PERCENT_RE = re.compile(r"%(?![0-9A-Fa-f]{2})")
TRACKING_QUERY_PARAMS = {"fbclid", "gclid", "msclkid"}
TEACHER_MAIN_SECTIONS = {
    "home",
    "curriculum",
    "ricerca",
    "didattica",
    "risorse",
}
TEACHER_RESEARCH_SECTIONS = {
    "brevetti",
    "focus",
    "laboratori",
    "premi",
    "premi-ricerca",
    "pubblicazioni",
    "progetti",
    "spin-off",
}
TEACHER_INTERNATIONAL_SECTIONS = {
    "bip",
    "cattedra-unesco",
    "cooperazione-internazionale",
    "doppio-titolo",
    "dottorato-con-tesi-in-cotutela",
    "eramus-teaching-docenza",
    "erasmus",
    "staff-training",
    "traineeship",
    "visiting-professors",
}
# Query bloccate di default: spesso generano filtri, viste tecniche o duplicati.
BLOCKED_QUERY_PARAMS = {
    "archive",
    "category",
    "execution",
    "incubatore",
    "progetto",
    "return",
    "stato",
}
# Eccezioni locali: alcuni parametri sono rumore solo su path specifici.
BLOCKED_QUERY_BY_PATH: dict[str, set[str]] = {}
# Alcune pagine DIEM espongono filtri per tutte le strutture dell'ateneo.
# Manteniamo solo la struttura DIEM, altrimenti la discovery esplode verso
# bandi non pertinenti.
ALLOWED_QUERY_VALUES_BY_PATH_PREFIX = {
    "/home/bandi": {
        "struttura": {"300638"},
        "cdsstruttura": {"300638"},
    },
}
# Allowlist stretta per archivi che aggiungono pagine informative reali.
ALLOWED_QUERY_VALUES_BY_PATH = {
    "/home/eventi": {"archive": {"1", "2"}},
    "/home/news": {"archive": {"1"}},
}
# Parametri informativi ammessi solo dove aprono un vero dettaglio.
ALLOWED_QUERY_PARAMS_BY_PATH = {
    "/ricerca/progetti-finanziati": {"progetto", "stato"},
    "/terza-missione/trasferimento-tecnologico/conto-terzi": {"progetto", "stato"},
}
ALLOWED_DIEM_SPIN_OFF_INCUBATOR_VALUES = {"0", "1"}
ALLOWED_DIEM_STATUS_VALUES = {"0", "1", "attivi", "scaduti", "tutti"}
ALLOWED_DIEM_INTERNATIONAL_STRUCTURE_ID = "300638"
NO_INDEX_QUERY_PARAMS = {
    "page",
    "p",
    "sort",
    "order",
    "orderby",
    "lang",
    "locale",
    "print",
    "sitemap",
}
DIEM_BANDI_STRUCTURE_PARAMS = {"struttura", "cdsstruttura"}
DIEM_BANDI_INDEXABLE_QUERY_PARAMS = {"anno", "categoria", "modulo"} | DIEM_BANDI_STRUCTURE_PARAMS
DIEM_BANDI_STRUCTURE_ID = "300638"

BLOCKED_PATH_PARTS = (
    "/login",
    "/admin",
    "/user/login",
    "/user/register",
    "/user/password",
    "/user/logout",
    "/feed",
    "/rss",
    "/atom",
    "/print",
    "/comment",
    "/saml",
    "/sso",
    "/idp/",
)

BLOCKED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".svg",
    ".webp",
    ".ico",
    ".css",
    ".js",
    ".mjs",
    ".map",
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".eot",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".wmv",
    ".webm",
    ".zip",
    ".tar",
    ".gz",
    ".rar",
    ".7z",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
}


def normalize_url(url: str) -> str:
    """Normalizza un URL per ridurre duplicati banali."""
    stripped = url.strip()
    if not stripped:
        return ""
    url, _ = urldefrag(stripped)
    parsed = urlparse(url)

    path = INVALID_PERCENT_RE.sub("%25", parsed.path or "/")
    lowered_path = path.lower()
    for index_name in ("/index.php", "/index.html", "/index.htm"):
        if lowered_path.endswith(index_name):
            path = path[: -len(index_name)] or "/"
            break

    if path != "/" and path.endswith("/"):
        path = path[:-1]

    query_params = sorted(parse_qsl(parsed.query, keep_blank_values=True))
    query = urlencode(query_params)

    return parsed._replace(
        scheme="https",
        netloc=parsed.netloc.lower(),
        path=path,
        query=query,
    ).geturl()


def parse_mime(content_type: str) -> str:
    """Estrae il MIME principale da Content-Type."""
    return content_type.split(";", 1)[0].strip().lower()


def is_pdf_url(url: str) -> bool:
    """True se il path sembra puntare a un PDF.

    UNISA espone alcuni PDF tramite endpoint senza estensione, ad esempio
    /unisa-rescue-page/pdf/id/..., quindi non basta controllare ".pdf".
    """
    path = urlparse(url).path.lower()
    parts = path_segments(url)
    is_unisa_pdf_endpoint = (
        len(parts) >= 4
        and parts[0] == "unisa-rescue-page"
        and parts[1] == "pdf"
        and parts[2] == "id"
    )
    return path.endswith(".pdf") or is_unisa_pdf_endpoint


def is_metadata_url(url: str) -> bool:
    """robots.txt e sitemap XML servono al crawler, non all'indice."""
    filename = Path(urlparse(url).path.lower()).name
    return filename == "robots.txt" or (
        filename.startswith("sitemap") and filename.endswith(".xml")
    )


def is_english_url(url: str) -> bool:
    """True per pagine in versione inglese da escludere dal corpus italiano."""
    segments = path_segments(url)
    if "en" in segments:
        return True

    params = {
        name.lower(): value.lower()
        for name, value in parse_qsl(urlparse(url).query, keep_blank_values=True)
    }
    return params.get("lang") == "en" or params.get("locale") == "en"


def query_param_names(url: str) -> set[str]:
    """Nomi dei parametri query, in minuscolo."""
    return {
        name.lower()
        for name, _ in parse_qsl(urlparse(url).query, keep_blank_values=True)
    }


def has_tracking_query(url: str) -> bool:
    """True per query di tracking da non crawlare."""
    params = query_param_names(url)
    return bool(params & TRACKING_QUERY_PARAMS) or any(
        name.startswith(TRACKING_QUERY_PREFIXES) for name in params
    )


def has_allowed_query_value(url: str, name: str) -> bool:
    """True per query eccezionalmente utili su path esplicitamente consentiti."""
    path = urlparse(url).path.lower().rstrip("/")
    allowed_values = ALLOWED_QUERY_VALUES_BY_PATH.get(path, {}).get(name)
    if not allowed_values and is_course_news_archive_query(url, name):
        allowed_values = {"1"}
    if not allowed_values:
        return False

    values = [
        value
        for param_name, value in parse_qsl(urlparse(url).query, keep_blank_values=True)
        if param_name.lower() == name
    ]
    return bool(values) and all(value in allowed_values for value in values)


def is_course_news_archive_query(url: str, name: str) -> bool:
    """True per archivi news dei corsi/dottorati su corsi.unisa.it."""
    if name != "archive" or domain_of(url) != "corsi.unisa.it":
        return False
    segments = path_segments(url)
    return bool(segments) and segments[-1] == "news"


def has_allowed_query_param(url: str, name: str) -> bool:
    """True per parametri informativi consentiti solo su path specifici."""
    path = urlparse(url).path.lower().rstrip("/")
    if is_teacher_detail_query_param(url, name):
        return True
    if is_diem_bandi_query_param(url, name):
        return True
    if is_diem_spin_off_query_param(url, name):
        return True
    if is_diem_international_query_param(url, name):
        return True
    return name in ALLOWED_QUERY_PARAMS_BY_PATH.get(path, set())


def is_teacher_detail_query_param(url: str, name: str) -> bool:
    """True per query di dettaglio informative sulle schede docente."""
    segments = path_segments(url)
    if len(segments) < 2:
        return False
    if len(segments) == 2 and segments[1] == "home":
        return name in {"avvisi", "avviso"}
    if len(segments) == 2 and segments[1] == "risorse":
        return name in {"categoria", "risorsa"}
    if len(segments) == 2 and segments[1] == "didattica":
        return name in {"anno", "id"}
    if len(segments) < 3:
        return False
    if segments[1] == "ricerca" and segments[2] == "progetti":
        return name in {"progetto", "ruolo", "stato"}
    if segments[1] == "ricerca" and segments[2] == "spin-off":
        return name in {"id", "incubatore"}
    if segments[1] == "ricerca" and segments[2] == "laboratori":
        return name == "id"
    return False


def is_diem_spin_off_query_param(url: str, name: str) -> bool:
    """True per i filtri informativi della pagina spin-off DIEM."""
    path = urlparse(url).path.lower().rstrip("/")
    if path != "/terza-missione/trasferimento-tecnologico/spin-off":
        return False
    if name != "incubatore":
        return False
    values = query_params(url).get(name, [])
    return bool(values) and all(
        value in ALLOWED_DIEM_SPIN_OFF_INCUBATOR_VALUES
        for value in values
    )


def is_diem_bandi_query_param(url: str, name: str) -> bool:
    """True per sottocategorie bandi DIEM ancorate alla struttura corretta."""
    path = urlparse(url).path.lower().rstrip("/")
    if path != "/home/bandi" or name != "categoria":
        return False
    params = query_params(url)
    structure_values = [
        value
        for structure_name in DIEM_BANDI_STRUCTURE_PARAMS
        for value in params.get(structure_name, [])
    ]
    category_values = params.get("categoria", [])
    return (
        "modulo" in params
        and structure_values == [DIEM_BANDI_STRUCTURE_ID]
        and bool(category_values)
        and all(value.isdigit() for value in category_values)
    )


def is_diem_international_query_param(url: str, name: str) -> bool:
    """True per i filtri delle liste accordi internazionali DIEM."""
    path = urlparse(url).path.lower().rstrip("/")
    if not path.startswith("/international/"):
        return False
    if name not in {"anno", "stato", "struttura"}:
        return False

    params = query_params(url)
    structure_values = params.get("struttura", [])
    if not structure_values or any(
        value != ALLOWED_DIEM_INTERNATIONAL_STRUCTURE_ID
        for value in structure_values
    ):
        return False

    status_values = params.get("stato", [])
    if status_values and any(value not in ALLOWED_DIEM_STATUS_VALUES for value in status_values):
        return False

    return True


def has_blocked_query(url: str) -> bool:
    """True per query tecniche che non portano contenuto utile."""
    params = query_param_names(url)
    path = urlparse(url).path.lower().rstrip("/")
    for prefix, allowed_params in ALLOWED_QUERY_VALUES_BY_PATH_PREFIX.items():
        if path == prefix or path.startswith(prefix + "/"):
            for name, allowed_values in allowed_params.items():
                values = [
                    value
                    for param_name, value in parse_qsl(
                        urlparse(url).query,
                        keep_blank_values=True,
                    )
                    if param_name.lower() == name
                ]
                if values and any(value not in allowed_values for value in values):
                    return True

    blocked_params = {
        param
        for param in params & BLOCKED_QUERY_PARAMS
        if not has_allowed_query_value(url, param)
        and not has_allowed_query_param(url, param)
    }
    if blocked_params:
        return True

    for prefix, blocked_params in BLOCKED_QUERY_BY_PATH.items():
        if path == prefix or path.startswith(prefix + "/"):
            return bool(params & blocked_params)

    return False


def has_noisy_query(url: str) -> bool:
    """True per query utili alla navigazione ma non all'indice."""
    params = query_param_names(url)
    return has_tracking_query(url) or bool(params & NO_INDEX_QUERY_PARAMS)


def is_indexable_diem_bandi_url(url: str) -> bool:
    """True per le liste bandi filtrate sulla sola struttura DIEM."""
    parsed = urlparse(url)
    path = parsed.path.lower().rstrip("/")
    if path != "/home/bandi":
        return False
    if not parsed.query:
        return True
    return is_explicit_diem_bandi_url(url)


def index_policy_skip_reason(url: str, config: dict) -> str | None:
    """Motivo per cui un URL traversabile non deve entrare nel corpus testuale."""
    parsed = urlparse(url)
    path = parsed.path.lower().rstrip("/")
    params = query_param_names(url)
    domain = domain_of(url)

    if domain == scope_value(config, "teacher_domain", "docenti.unisa.it"):
        if not is_teacher_supported_content_url(url, config):
            return "unsupported_teacher_section"
        if is_teacher_publications_landing_url(url):
            return "teacher_publications_landing"
        return None

    if domain == scope_value(config, "diem_domain", "www.diem.unisa.it"):
        if parsed.query and path == "/home/bandi" and not is_indexable_diem_bandi_url(url):
            return "noisy_bandi_query"

    if domain == scope_value(config, "course_domain", "corsi.unisa.it"):
        if parsed.query and path.endswith("/strutture-didattiche/calendario-occupazione-spazi"):
            return "noisy_room_calendar_query"

    if has_noisy_query(url):
        return "noisy_query"
    return None


def has_blocked_path(url: str) -> bool:
    """Blocca login/admin/feed/asset statici."""
    parsed = urlparse(url)
    path = parsed.path.lower()
    host = parsed.netloc.lower()
    if "auth." in host or "login." in host:
        return True
    normalized_path = "/" + "/".join(path_segments(url))
    blocked_paths = {
        "/" + part.strip("/").lower()
        for part in BLOCKED_PATH_PARTS
    }
    if any(
        normalized_path == blocked_path
        or normalized_path.startswith(f"{blocked_path}/")
        for blocked_path in blocked_paths
    ):
        return True
    return Path(path).suffix.lower() in BLOCKED_EXTENSIONS


def config_list(config: dict, section: str, key: str) -> list[str]:
    """Legge una lista dal config, con fallback a []."""
    return list(config.get(section, {}).get(key, []))


def scope_value(config: dict, key: str, default: str) -> str:
    """Legge un valore dalla sezione scope."""
    return str(config.get("scope", {}).get(key, default)).lower()


def domain_of(url: str) -> str:
    """Dominio normalizzato di un URL."""
    return urlparse(url).netloc.lower()


def source_url_from_context(context: dict | None) -> str | None:
    """Estrae discovered_from dal contesto, escludendo i seed."""
    if not context:
        return None
    source_url = context.get("discovered_from")
    if not source_url or source_url == "seed":
        return None
    return normalize_url(str(source_url))


def first_path_segment(url: str) -> str:
    """Primo segmento del path."""
    segments = path_segments(url)
    return segments[0] if segments else ""


def path_segments(url: str) -> list[str]:
    """Segmenti del path normalizzati in minuscolo."""
    return [
        segment.lower()
        for segment in urlparse(url).path.split("/")
        if segment
    ]


def is_diem_url(url: str, config: dict) -> bool:
    """True per il dominio principale DIEM."""
    return domain_of(url) == scope_value(config, "diem_domain", "www.diem.unisa.it")

def is_directory_person_url(url: str, config: dict) -> bool:
    """True solo per pagine personali della rubrica UNISA.

    La rubrica viene indicizzata solo quando parte dal personale DIEM e viene
    usata anche come dominio ponte:
    DIEM personale -> rubrica.unisa.it/persone?matricola=... -> docenti.unisa.it
    """
    parsed = urlparse(url)
    directory_domain = scope_value(config, "directory_domain", "rubrica.unisa.it")

    return (
        parsed.netloc.lower() == directory_domain
        and parsed.path.rstrip("/").lower() == "/persone"
        and "matricola" in query_param_names(url)
    )


def directory_person_matricola(url: str, config: dict) -> str | None:
    """Estrae la matricola da una pagina personale della rubrica UNISA."""
    if not is_directory_person_url(url, config):
        return None

    for name, value in parse_qsl(urlparse(url).query, keep_blank_values=True):
        if name.lower() == "matricola" and value:
            return value
    return None


def configured_course_paths(config: dict) -> set[str]:
    """Percorsi corso ammessi come primo segmento di corsi.unisa.it.

    Esempio: in https://corsi.unisa.it/ingegneria-informatica/didattica,
    il percorso corso è "ingegneria-informatica".
    """
    return {
        str(course_path).lower()
        for course_path in config_list(config, "scope", "allowed_course_paths")
    }


def course_has_allowed_identifier(url: str, config: dict) -> bool:
    """True se un URL corsi.unisa.it contiene percorso o codice corso DIEM."""
    allowed_course_paths = configured_course_paths(config)
    allowed_codes = {
        str(code).lower()
        for code in config_list(config, "scope", "allowed_course_codes")
    }
    allowed_code_prefixes = {
        match.group(1)
        for code in allowed_codes
        if (match := re.match(r"^(\d{5})(?:[a-z]|$)", code))
    }
    allowed_numeric_ids = {
        str(course_id).lower()
        for course_id in config_list(config, "scope", "allowed_course_numeric_ids")
    }
    segments = path_segments(url)
    first_segment = first_path_segment(url)
    return (
        first_segment in allowed_course_paths
        or first_segment in allowed_code_prefixes
        or bool(
            set(segments) & (allowed_codes | allowed_numeric_ids)
        )
    )


def decoded_rescue_path_segments(url: str) -> list[str]:
    """Decodifica il segmento base64 `url/...` delle rescue page UNISA."""
    raw_segments = [segment for segment in urlparse(url).path.split("/") if segment]
    lower_segments = [segment.lower() for segment in raw_segments]
    decoded_segments: list[str] = []
    for index, segment in enumerate(lower_segments[:-1]):
        if segment != "url":
            continue
        encoded = unquote(raw_segments[index + 1])
        padding = "=" * (-len(encoded) % 4)
        try:
            decoded = base64.urlsafe_b64decode(encoded + padding).decode(
                "utf-8", errors="ignore"
            )
        except (ValueError, binascii.Error):
            continue
        decoded_segments.extend(path_segments(decoded))
    return decoded_segments


def course_rescue_page_has_allowed_identifier(url: str, config: dict) -> bool:
    """True per dettagli/news rescue page riferiti a un corso DIEM."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/").lower()
    if not path.startswith((
        "/unisa-rescue-page/dettaglio/",
        "/unisa-rescue-page/search/",
    )):
        return False

    allowed_course_paths = configured_course_paths(config)
    allowed_codes = {
        str(code).lower()
        for code in config_list(config, "scope", "allowed_course_codes")
    }
    allowed_numeric_ids = {
        str(course_id).lower()
        for course_id in config_list(config, "scope", "allowed_course_numeric_ids")
    }
    decoded_segments = decoded_rescue_path_segments(url)
    return bool(decoded_segments) and (
        decoded_segments[0] in allowed_course_paths
        or bool(set(decoded_segments) & (allowed_codes | allowed_numeric_ids))
    )


def teaching_council_has_allowed_identifier(url: str, config: dict) -> bool:
    """True per consigli didattici relativi ai corsi DIEM configurati."""
    first_segment = first_path_segment(url)
    allowed_course_paths = configured_course_paths(config)
    allowed_codes = {
        str(code).lower()
        for code in config_list(config, "scope", "allowed_course_codes")
    }
    allowed_code_prefixes = {
        match.group(1)
        for code in allowed_codes
        if (match := re.match(r"^(\d{5})(?:[a-z]|$)", code))
    }
    return first_segment in (
        allowed_course_paths | allowed_codes | allowed_code_prefixes
    )


def is_malformed_course_rescue_detail_url(url: str, config: dict) -> bool:
    """True per rescue URL nate da link relativi appesi al dettaglio corrente."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/").lower()
    if not path.startswith((
        "/unisa-rescue-page/dettaglio/",
        "/unisa-rescue-page/search/",
    )):
        return False

    allowed_numeric_ids = {
        str(course_id).lower()
        for course_id in config_list(config, "scope", "allowed_course_numeric_ids")
    }
    if not allowed_numeric_ids:
        return False

    raw_segments = [segment.lower() for segment in urlparse(url).path.split("/") if segment]
    encoded_index = raw_segments.index("url") + 1 if "url" in raw_segments else -1
    for index, segment in enumerate(raw_segments):
        if index == encoded_index:
            continue
        if segment in allowed_numeric_ids:
            return True

    if path.startswith("/unisa-rescue-page/search/"):
        for first, second in zip(raw_segments, raw_segments[1:]):
            if first == second and first != "url":
                return True

    return False


def is_known_scope_source(source_url: str | None, config: dict) -> bool:
    """True se la pagina sorgente è già nello scope del progetto."""
    if not source_url:
        return False

    domain = domain_of(source_url)
    teacher_domain = scope_value(config, "teacher_domain", "docenti.unisa.it")
    course_domain = scope_value(config, "course_domain", "corsi.unisa.it")
    teaching_council_domain = scope_value(
        config,
        "teaching_council_domain",
        "cd.unisa.it",
    )

    return (
        is_diem_url(source_url, config)
        or domain == teacher_domain
        or (
            domain == teaching_council_domain
            and teaching_council_has_allowed_identifier(source_url, config)
        )
        or (
            domain == course_domain
            and (
                course_has_allowed_identifier(source_url, config)
                or course_rescue_page_has_allowed_identifier(source_url, config)
            )
        )
    )

def is_diem_personnel_url(url: str | None, config: dict) -> bool:
    """True solo per la pagina DIEM che elenca il personale del dipartimento."""
    if not url or not is_diem_url(url, config):
        return False
    return urlparse(url).path.rstrip("/").lower() == "/dipartimento/personale"


def teacher_profile_key(url: str, config: dict) -> str | None:
    """Identificatore stabile del profilo docente ricavato dal primo segmento."""
    if domain_of(url) != scope_value(config, "teacher_domain", "docenti.unisa.it"):
        return None
    return first_path_segment(url) or None


def teacher_section_from_url(url: str, config: dict) -> str | None:
    """Sezione docente indicizzabile ricavata dal path del profilo."""
    if domain_of(url) != scope_value(config, "teacher_domain", "docenti.unisa.it"):
        return None

    segments = path_segments(url)
    if len(segments) < 2:
        return None
    if (
        len(segments) >= 3
        and segments[1] == "ricerca"
        and segments[2] in TEACHER_RESEARCH_SECTIONS
    ):
        return segments[2]
    if segments[1] in TEACHER_MAIN_SECTIONS:
        return segments[1]
    return None


def query_params(url: str) -> dict[str, list[str]]:
    """Query params normalizzati preservando valori multipli."""
    params: dict[str, list[str]] = {}
    for name, value in parse_qsl(urlparse(url).query, keep_blank_values=True):
        params.setdefault(name.lower(), []).append(value.lower())
    return params


def has_exact_query_params(url: str, expected: dict[str, str]) -> bool:
    """True se la query contiene esattamente i parametri attesi."""
    params = query_params(url)
    return params == {name: [value] for name, value in expected.items()}


def is_teacher_publications_overview_url(url: str) -> bool:
    """True per la vista aggregata esplicita delle pubblicazioni docente."""
    segments = path_segments(url)
    if len(segments) != 3 or segments[1] != "ricerca" or segments[2] != "pubblicazioni":
        return False
    return has_exact_query_params(url, {"anno": "0"})


def is_teacher_publications_landing_url(url: str) -> bool:
    """True per la landing pubblicazioni: attraversabile, non indicizzabile."""
    segments = path_segments(url)
    return (
        len(segments) == 3
        and segments[1] == "ricerca"
        and segments[2] == "pubblicazioni"
        and not urlparse(url).query
    )


def is_teacher_publications_year_url(url: str) -> bool:
    """True per archivi annuali pubblicazioni, esclusi dal corpus DIEM."""
    segments = path_segments(url)
    return (
        len(segments) == 3
        and segments[1] == "ricerca"
        and segments[2] == "pubblicazioni"
        and has_single_numeric_query_param(url, "anno")
    )


def is_teacher_projects_overview_url(url: str) -> bool:
    """True per la vista aggregata dei progetti del docente."""
    segments = path_segments(url)
    if len(segments) != 3 or segments[1] != "ricerca" or segments[2] != "progetti":
        return False
    return not urlparse(url).query or has_exact_query_params(url, {"ruolo": "tutti"})


def is_teacher_projects_filtered_url(url: str) -> bool:
    """True per filtri ruolo/stato dei progetti docente."""
    segments = path_segments(url)
    if len(segments) != 3 or segments[1] != "ricerca" or segments[2] != "progetti":
        return False
    params = query_params(url)
    if not set(params) <= {"ruolo", "stato"}:
        return False
    role_values = params.get("ruolo", [])
    if not role_values or any(
        value not in {"responsabile", "componente", "tutti"}
        for value in role_values
    ):
        return False
    status_values = params.get("stato", [])
    return not status_values or all(value in {"0", "1"} for value in status_values)


def has_single_numeric_query_param(url: str, name: str) -> bool:
    """True se la query contiene un solo parametro numerico con il nome dato."""
    params = query_params(url)
    return set(params) == {name} and len(params[name]) == 1 and params[name][0].isdigit()


def is_teacher_didactics_url(url: str) -> bool:
    """True per la didattica docente corrente o storica."""
    segments = path_segments(url)
    if len(segments) != 2 or segments[1] != "didattica":
        return False
    if not urlparse(url).query:
        return True
    return has_single_numeric_query_param(url, "anno")


def is_teacher_teaching_detail_url(url: str) -> bool:
    """True per la scheda di un singolo insegnamento docente."""
    segments = path_segments(url)
    if len(segments) != 2 or segments[1] != "didattica":
        return False
    params = query_params(url)
    return (
        set(params) == {"anno", "id"}
        and len(params["anno"]) == 1
        and len(params["id"]) == 1
        and params["anno"][0].isdigit()
        and params["id"][0].isdigit()
    )


def is_teacher_lesson_schedule_url(url: str) -> bool:
    """True per l'orario lezioni del docente."""
    segments = path_segments(url)
    if len(segments) != 3 or segments[1] != "didattica" or segments[2] != "orari":
        return False
    return not urlparse(url).query or has_exact_query_params(url, {"include": "docente"})


def is_teacher_project_detail_url(url: str) -> bool:
    """True per una scheda progetto docente."""
    segments = path_segments(url)
    return (
        len(segments) == 3
        and segments[1] == "ricerca"
        and segments[2] == "progetti"
        and has_single_numeric_query_param(url, "progetto")
    )


def is_teacher_spin_off_url(url: str) -> bool:
    """True per la lista o una scheda spin-off docente."""
    segments = path_segments(url)
    if len(segments) != 3 or segments[1] != "ricerca" or segments[2] != "spin-off":
        return False
    if not urlparse(url).query:
        return True
    params = query_params(url)
    if has_single_numeric_query_param(url, "id"):
        return True
    return (
        set(params) == {"incubatore"}
        and len(params["incubatore"]) == 1
        and params["incubatore"][0] in {"0", "1"}
    )


def is_teacher_research_query_url(url: str) -> bool:
    """True per viste di dettaglio o archivio nelle sezioni ricerca docente."""
    segments = path_segments(url)
    if len(segments) != 3 or segments[1] != "ricerca":
        return False
    section = segments[2]
    if section in {"brevetti", "focus"}:
        return has_single_numeric_query_param(url, "id")
    if section == "laboratori":
        return has_single_numeric_query_param(url, "id")
    if section == "premi-ricerca":
        return has_single_numeric_query_param(url, "anno")
    return False


def is_teacher_resources_url(url: str) -> bool:
    """True per risorse docente e relativi dettagli."""
    segments = path_segments(url)
    if len(segments) != 2 or segments[1] != "risorse":
        return False
    if not urlparse(url).query:
        return True
    params = query_params(url)
    if set(params) == {"categoria"}:
        return len(params["categoria"]) == 1 and params["categoria"][0].isdigit()
    return (
        set(params) == {"categoria", "risorsa"}
        and len(params["categoria"]) == 1
        and len(params["risorsa"]) == 1
        and params["categoria"][0].isdigit()
        and params["risorsa"][0].isdigit()
    )


def is_teacher_home_url(url: str) -> bool:
    """True per home docente e avvisi personali."""
    segments = path_segments(url)
    if len(segments) == 1:
        return not urlparse(url).query
    if len(segments) != 2 or segments[1] != "home":
        return False
    if not urlparse(url).query:
        return True
    params = query_params(url)
    if set(params) == {"avvisi"}:
        return params["avvisi"] == ["1"]
    return has_single_numeric_query_param(url, "avviso")


def is_teacher_international_url(url: str) -> bool:
    """True per sottosezioni internazionali dei profili docente."""
    segments = path_segments(url)
    return (
        len(segments) == 3
        and segments[1] == "international"
        and segments[2] in TEACHER_INTERNATIONAL_SECTIONS
        and not urlparse(url).query
    )


def is_teacher_simple_research_section_url(url: str) -> bool:
    """True per sottosezioni ricerca docente senza query."""
    segments = path_segments(url)
    return (
        len(segments) == 3
        and segments[1] == "ricerca"
        and segments[2] in TEACHER_RESEARCH_SECTIONS
        and not urlparse(url).query
    )


def is_teacher_supported_content_url(url: str, config: dict) -> bool:
    """True per le sole pagine docente utili al corpus RAG pre-chunking."""
    if domain_of(url) != scope_value(config, "teacher_domain", "docenti.unisa.it"):
        return False

    segments = path_segments(url)
    if is_teacher_home_url(url):
        return True
    if len(segments) == 2 and segments[1] in {"curriculum", "ricerca"}:
        return not urlparse(url).query
    if is_teacher_resources_url(url):
        return True
    if is_teacher_didactics_url(url):
        return True
    if is_teacher_teaching_detail_url(url):
        return True
    if is_teacher_lesson_schedule_url(url):
        return True
    if is_teacher_publications_landing_url(url):
        return True
    if is_teacher_publications_overview_url(url):
        return True
    if is_teacher_projects_overview_url(url):
        return True
    if is_teacher_projects_filtered_url(url):
        return True
    if is_teacher_project_detail_url(url):
        return True
    if is_teacher_spin_off_url(url):
        return True
    if is_teacher_research_query_url(url):
        return True
    if is_teacher_simple_research_section_url(url):
        return True
    if is_teacher_international_url(url):
        return True
    return False


def allowed_teacher_profiles_from_context(context: dict | None) -> set[str]:
    """Whitelist profili docente propagata dalla discovery ai filtri URL."""
    if not context:
        return set()
    return {str(profile).lower() for profile in context.get("allowed_teacher_profiles", set())}


def has_authorized_directory_bridge(context: dict | None) -> bool:
    """True quando la rubrica corrente deriva dal personale DIEM."""
    return bool(context and context.get("authorized_directory_bridge", False))


def is_directory_link_in_scope(url: str, source_url: str | None, config: dict) -> bool:
    """Permette la rubrica solo dalla pagina personale DIEM."""
    if not is_directory_person_url(url, config):
        return False

    return is_diem_personnel_url(source_url, config)

def is_teacher_link_in_scope(
    url: str,
    source_url: str | None,
    config: dict,
    context: dict | None = None,
) -> bool:
    """Permette solo profili docenti scoperti dal personale DIEM."""
    target_profile = teacher_profile_key(url, config)
    if not target_profile:
        return False
    if not is_teacher_supported_content_url(url, config):
        return False

    allowed_profiles = allowed_teacher_profiles_from_context(context)
    if target_profile in allowed_profiles:
        return True

    if not source_url:
        return False

    # Primo ingresso autorizzato: il profilo viene poi registrato nella whitelist.
    if is_diem_personnel_url(source_url, config):
        return True
    
    # Caso ponte:
    # rubrica.unisa.it/persone?matricola=... -> docenti.unisa.it/nome.cognome
    if is_directory_person_url(source_url, config) and has_authorized_directory_bridge(context):
        return True
    
    if domain_of(source_url) != scope_value(config, "teacher_domain", "docenti.unisa.it"):
        return False

    # Una volta entrati in un profilo docente autorizzato, restiamo confinati
    # allo stesso primo segmento del path. Questo copre i profili numerici
    # esposti da docenti.unisa.it dopo redirect, es. /058553/home -> /058553/curriculum.
    source_profile = teacher_profile_key(source_url, config)
    return bool(source_profile and source_profile == target_profile)


def is_pdf_in_scope(url: str, config: dict, source_url: str | None) -> bool:
    """I PDF sono ammessi solo se su dominio consentito e referenziati dallo scope."""
    domain = domain_of(url)
    allowed_pdf_domains = {
        str(item).lower()
        for item in config.get("crawler", {}).get("pdf_allowed_domains", [])
    }

    if domain not in allowed_pdf_domains:
        return False

    return is_diem_url(url, config) or is_known_scope_source(source_url, config)


def is_in_scope_url(
    url: str,
    config: dict,
    context: dict | None = None,
) -> tuple[bool, str]:
    """Controlla lo scope di dominio/progetto."""
    source_url = source_url_from_context(context)
    domain = domain_of(url)
    allowed_domains = {
        str(item).lower()
        for item in config.get("crawler", {}).get("allowed_domains", [])
    }

    if is_pdf_url(url):
        rescue_skip_reason = diem_rescue_upload_scope_reason(url, source_url, config)
        if rescue_skip_reason:
            return False, rescue_skip_reason
        if is_pdf_in_scope(url, config, source_url):
            return True, "ok"
        return False, "scope_pdf"

    if domain not in allowed_domains:
        return False, "domain"

    teaching_council_domain = scope_value(config, "teaching_council_domain", "cd.unisa.it")
    if domain == teaching_council_domain:
        if teaching_council_has_allowed_identifier(url, config):
            return True, "ok"
        return False, "scope_teaching_council"

    if is_diem_url(url, config):
        return True, "ok"
    
    directory_domain = scope_value(config, "directory_domain", "rubrica.unisa.it")
    if domain == directory_domain:
        if is_directory_link_in_scope(url, source_url, config):
            return True, "ok"
        return False, "scope_directory"

    teacher_domain = scope_value(config, "teacher_domain", "docenti.unisa.it")
    if domain == teacher_domain:
        if is_teacher_link_in_scope(url, source_url, config, context):
            return True, "ok"
        return False, "scope_teacher"

    course_domain = scope_value(config, "course_domain", "corsi.unisa.it")
    if domain == course_domain:
        if is_malformed_course_rescue_detail_url(url, config):
            return False, "malformed_course_rescue"
        if course_has_allowed_identifier(url, config) or (
            domain_of(source_url or "") == course_domain
            and course_has_allowed_identifier(source_url or "", config)
        ) or course_rescue_page_has_allowed_identifier(url, config):
            return True, "ok"
        if (
            urlparse(url).path.rstrip("/").lower().startswith(
                "/unisa-rescue-page/dettaglio/"
            )
            and domain_of(source_url or "") == course_domain
            and (
                course_has_allowed_identifier(source_url or "", config)
                or course_rescue_page_has_allowed_identifier(source_url or "", config)
            )
        ):
            return True, "ok"
        return False, "scope_course"

    return True, "ok"


def can_traverse_url(
    url: str,
    config: dict,
    context: dict | None = None,
) -> tuple[bool, str]:
    """True se l'URL può essere visitato per scoprire link."""
    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        return False, "scheme"
    if is_metadata_url(url):
        return False, "metadata"
    if is_english_url(url):
        return False, "language"

    ok, reason = is_in_scope_url(url, config, context)
    if not ok:
        return False, reason
    if has_blocked_path(url):
        return False, "blocked_path"
    if has_blocked_query(url):
        return False, "blocked_query"
    if has_tracking_query(url):
        return False, "tracking_query"

    return True, "ok"


def can_index_url(
    url: str,
    config: dict,
    context: dict | None = None,
) -> tuple[bool, str]:
    """True se l'URL HTML può diventare documento indicizzabile."""
    if is_pdf_url(url):
        return False, "pdf_not_indexable_here"

    ok, reason = can_traverse_url(url, config, context)
    if not ok:
        return False, reason

    skip_reason = index_policy_skip_reason(url, config)
    if skip_reason:
        return False, skip_reason

    return True, "ok"
