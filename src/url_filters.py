"""
Filtri URL per la discovery.

Le regole riflettono lo scope dell'assignment:
- pagine sotto www.diem.unisa.it;
- profili docenti DIEM sotto docenti.unisa.it, raggiunti da pagine in scope;
- corsi DIEM sotto corsi.unisa.it, riconosciuti da slug/codici in config;
- PDF referenziati da pagine in scope.

Tabella delle regole principali:

| Dominio / risorsa     | Regola                                                     |
|-----------------------|------------------------------------------------------------|
| www.diem.unisa.it     | Ammesso.                                                   |
| docenti.unisa.it      | Solo se scoperto da DIEM o da un profilo docente in scope. |
| corsi.unisa.it        | Solo slug/codici DIEM configurati.                         |
| uploads PDF           | Rilevati; scaricati solo se robots.txt lo permette.        |
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qsl, urldefrag, urlparse


TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_PARAMS = {"fbclid", "gclid", "msclkid"}
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
BLOCKED_QUERY_BY_PATH = {
    "/ricerca/focus": {"anno", "id"},
    "/ricerca/progetti-finanziati": {"tip"},
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
    url, _ = urldefrag(url)
    parsed = urlparse(url.strip())

    path = parsed.path or "/"
    lowered_path = path.lower()
    for index_name in ("/index.php", "/index.html", "/index.htm"):
        if lowered_path.endswith(index_name):
            path = path[: -len(index_name)] or "/"
            break

    if path != "/" and path.endswith("/"):
        path = path[:-1]

    return parsed._replace(
        scheme="https",
        netloc=parsed.netloc.lower(),
        path=path,
    ).geturl()


def parse_mime(content_type: str) -> str:
    """Estrae il MIME principale da Content-Type."""
    return content_type.split(";", 1)[0].strip().lower()


def is_pdf_url(url: str) -> bool:
    """True se il path sembra puntare a un PDF.

    UNISA espone alcuni PDF tramite endpoint senza estensione, ad esempio
    /unisa-rescue-page/pdf/id/..., quindi non basta controllare ".pdf".
    """
    path_parts = [part for part in urlparse(url).path.lower().split("/") if part]
    return urlparse(url).path.lower().endswith(".pdf") or "pdf" in path_parts


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


def has_blocked_query(url: str) -> bool:
    """True per query tecniche che non portano contenuto utile."""
    params = query_param_names(url)
    if params & BLOCKED_QUERY_PARAMS:
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
    if any(part in path for part in BLOCKED_PATH_PARTS):
        return True
    return Path(path).suffix in BLOCKED_EXTENSIONS


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
    path = urlparse(url).path.strip("/")
    return path.split("/", 1)[0].lower() if path else ""


def is_diem_url(url: str, config: dict) -> bool:
    """True per il dominio principale DIEM."""
    return domain_of(url) == scope_value(config, "diem_domain", "www.diem.unisa.it")


def course_has_allowed_identifier(url: str, config: dict) -> bool:
    """True se un URL corsi.unisa.it contiene slug o codice corso DIEM."""
    allowed_slugs = {
        str(slug).lower()
        for slug in config_list(config, "scope", "allowed_course_slugs")
    }
    allowed_codes = {
        str(code).lower()
        for code in config_list(config, "scope", "allowed_course_codes")
    }
    path = urlparse(url).path.lower()
    return first_path_segment(url) in allowed_slugs or any(
        code in path for code in allowed_codes
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


def is_teacher_link_in_scope(url: str, source_url: str | None, config: dict) -> bool:
    """Permette ingresso da DIEM e navigazione interna allo stesso profilo docente."""
    if not source_url:
        return False
    if is_diem_url(source_url, config):
        return True
    if domain_of(source_url) != scope_value(config, "teacher_domain", "docenti.unisa.it"):
        return False
    return first_path_segment(source_url) == first_path_segment(url)


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

    teacher_domain = scope_value(config, "teacher_domain", "docenti.unisa.it")
    if domain == teacher_domain:
        if is_teacher_link_in_scope(url, source_url, config):
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
    if has_noisy_query(url):
        return False, "noisy_query"

    return True, "ok"
