"""
Filtri URL per la discovery.

Le regole riflettono lo scope dell'assignment:
- pagine sotto www.diem.unisa.it;
- profili docenti DIEM sotto docenti.unisa.it, autorizzati dal personale DIEM;
- corsi DIEM sotto corsi.unisa.it, riconosciuti da percorsi/codici in config;
- PDF referenziati da pagine in scope.
- query parametriche bloccate di default, con allowlist puntuali per archivi
  informativi verificati.

Tabella delle regole principali:

| Dominio / risorsa     | Regola                                                     |
|-----------------------|------------------------------------------------------------|
| www.diem.unisa.it     | Ammesso.                                                   |
| docenti.unisa.it      | Solo profili whitelistati dal personale DIEM.              |
| corsi.unisa.it        | Solo percorsi/codici DIEM configurati.                     |
| uploads PDF           | Rilevati; scaricati solo se robots.txt lo permette.        |
| query `archive`        | Solo valori allowlistati per news/eventi DIEM.              |
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urldefrag, urlparse


TRACKING_QUERY_PREFIXES = ("utm_",)
INVALID_PERCENT_RE = re.compile(r"%(?![0-9A-Fa-f]{2})")
TRACKING_QUERY_PARAMS = {"fbclid", "gclid", "msclkid"}
# Query bloccate di default: spesso generano filtri, viste tecniche o duplicati.
BLOCKED_QUERY_PARAMS = {
    "archive",
    "category",
    "execution",
    "incubatore",
    "progetto",
    "return",
    "sitemap",
    "stato",
}
# Eccezioni locali: alcuni parametri sono rumore solo su path specifici.
BLOCKED_QUERY_BY_PATH = {
    "/ricerca/focus": {"anno", "id"},
    "/ricerca/progetti-finanziati": {"tip"},
}
# Allowlist stretta per archivi che aggiungono pagine informative reali.
ALLOWED_QUERY_VALUES_BY_PATH = {
    "/home/eventi": {"archive": {"1", "2"}},
    "/home/news": {"archive": {"1"}},
}
# Parametri informativi ammessi solo dove aprono un vero dettaglio.
ALLOWED_QUERY_PARAMS_BY_PATH = {
    "/ricerca/progetti-finanziati": {"progetto"},
}
NO_INDEX_QUERY_PARAMS = {
    "page",
    "p",
    "sort",
    "order",
    "orderby",
    "lang",
    "locale",
    "print",
}

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
    if not allowed_values:
        return False

    values = [
        value
        for param_name, value in parse_qsl(urlparse(url).query, keep_blank_values=True)
        if param_name.lower() == name
    ]
    return bool(values) and all(value in allowed_values for value in values)


def has_allowed_query_param(url: str, name: str) -> bool:
    """True per parametri informativi consentiti solo su path specifici."""
    path = urlparse(url).path.lower().rstrip("/")
    return name in ALLOWED_QUERY_PARAMS_BY_PATH.get(path, set())


def has_blocked_query(url: str) -> bool:
    """True per query tecniche che non portano contenuto utile."""
    params = query_param_names(url)
    blocked_params = {
        param
        for param in params & BLOCKED_QUERY_PARAMS
        if not has_allowed_query_value(url, param)
        and not has_allowed_query_param(url, param)
    }
    if blocked_params:
        return True

    path = urlparse(url).path.lower().rstrip("/")
    for prefix, blocked_params in BLOCKED_QUERY_BY_PATH.items():
        if path == prefix or path.startswith(prefix + "/"):
            return bool(params & blocked_params)

    return False


def has_noisy_query(url: str) -> bool:
    """True per query utili alla navigazione ma non all'indice."""
    params = query_param_names(url)
    return has_tracking_query(url) or bool(params & NO_INDEX_QUERY_PARAMS)


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

    La rubrica viene usata come dominio ponte:
    DIEM personale -> rubrica.unisa.it/persone?matricola=... -> docenti.unisa.it
    """
    parsed = urlparse(url)
    directory_domain = scope_value(config, "directory_domain", "rubrica.unisa.it")

    return (
        parsed.netloc.lower() == directory_domain
        and parsed.path.rstrip("/").lower() == "/persone"
        and "matricola" in query_param_names(url)
    )

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
    segments = path_segments(url)
    return first_path_segment(url) in allowed_course_paths or bool(
        set(segments) & allowed_codes
    )


def is_known_scope_source(source_url: str | None, config: dict) -> bool:
    """True se la pagina sorgente è già nello scope del progetto."""
    if not source_url:
        return False

    domain = domain_of(source_url)
    teacher_domain = scope_value(config, "teacher_domain", "docenti.unisa.it")
    course_domain = scope_value(config, "course_domain", "corsi.unisa.it")

    return (
        is_diem_url(source_url, config)
        or domain == teacher_domain
        or (domain == course_domain and course_has_allowed_identifier(source_url, config))
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


def allowed_teacher_profiles_from_context(context: dict | None) -> set[str]:
    """Whitelist profili docente propagata dalla discovery ai filtri URL."""
    if not context:
        return set()
    return {str(profile).lower() for profile in context.get("allowed_teacher_profiles", set())}


def has_authorized_directory_bridge(context: dict | None) -> bool:
    """True quando la rubrica corrente deriva dal personale DIEM."""
    return bool(context and context.get("authorized_directory_bridge", False))


def is_directory_link_in_scope(url: str, source_url: str | None, config: dict) -> bool:
    """Permette la rubrica solo come ponte dalla pagina personale DIEM."""
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
    if not source_url:
        return False

    target_profile = teacher_profile_key(url, config)
    if not target_profile:
        return False

    allowed_profiles = allowed_teacher_profiles_from_context(context)
    if target_profile in allowed_profiles:
        return True

    # Primo ingresso autorizzato: il profilo viene poi registrato nella whitelist.
    if is_diem_personnel_url(source_url, config):
        return True
    
    # Caso ponte:
    # rubrica.unisa.it/persone?matricola=... -> docenti.unisa.it/nome.cognome
    if is_directory_person_url(source_url, config) and has_authorized_directory_bridge(context):
        return True
    
    if domain_of(source_url) != scope_value(config, "teacher_domain", "docenti.unisa.it"):
        return False
    
    return False


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
        if is_pdf_in_scope(url, config, source_url):
            return True, "ok"
        return False, "scope_pdf"

    if domain not in allowed_domains:
        return False, "domain"

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
        if course_has_allowed_identifier(url, config):
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

    # La rubrica è solo un ponte verso docenti.unisa.it:
    # la attraversiamo, ma non la indicizziamo.
    if is_directory_person_url(url, config):
        return False, "directory_bridge_not_indexable"

    if has_noisy_query(url):
        return False, "noisy_query"

    return True, "ok"
