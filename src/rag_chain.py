from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv
from groq import Groq
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from pipeline_io import BASE_DIR


load_dotenv(BASE_DIR / ".env", override=True)


from retrieval import RetrievalResult, hybrid_retrieve
from pipeline_io import load_jsonl
from vector_store import CHUNKS_FILE


DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")

DEFAULT_FINAL_K = int(os.getenv("RAG_FINAL_K", "7"))
DEFAULT_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "9000"))

GROQ_TIMEOUT_SECONDS = int(os.getenv("GROQ_TIMEOUT_SECONDS", "60"))
GROQ_JSON_MODE = os.getenv("GROQ_JSON_MODE", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "3"))


ConversationTurn = dict[str, str]


@dataclass
class Source:
    title: str
    url: str
    breadcrumb: str
    chunk_id: str


@dataclass
class RagResponse:
    question: str
    answer: str
    sources: list[Source]
    retrieved_chunks: list[RetrievalResult]


def metadata_to_string(value: Any) -> str:
    """
    Converte valori metadata in stringa leggibile.
    Serve perché alcuni campi possono essere liste, dict o stringhe JSON.
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, list):
        return " > ".join(str(item) for item in value)

    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)

    return str(value)


def get_source_from_result(result: RetrievalResult) -> Source:
    metadata = result.metadata or {}

    title = metadata_to_string(metadata.get("title")) or "Titolo non disponibile"

    url = (
        metadata_to_string(metadata.get("source_url"))
        or metadata_to_string(metadata.get("document_url"))
        or metadata_to_string(metadata.get("url"))
        or "URL non disponibile"
    )

    breadcrumb = (
        metadata_to_string(metadata.get("breadcrumb"))
        or metadata_to_string(metadata.get("breadcrumb_text"))
        or ""
    )

    chunk_id = metadata_to_string(metadata.get("chunk_id")) or result.chunk_id

    return Source(
        title=title,
        url=url,
        breadcrumb=breadcrumb,
        chunk_id=chunk_id,
    )


def clean_display_url(url: str) -> str:
    """
    Pulisce gli URL mostrati all'utente senza modificare i metadati originali.
    Query string e frammenti sono utili internamente, ma rumorosi nelle fonti.
    """
    if not url or url == "URL non disponibile":
        return url

    parts = urlsplit(url)

    if not parts.scheme or not parts.netloc:
        return url

    path = parts.path.rstrip("/") or "/"

    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            path,
            "",
            "",
        )
    )


def get_display_source_from_result(result: RetrievalResult) -> Source:
    source = get_source_from_result(result)

    return Source(
        title=source.title,
        url=clean_display_url(source.url),
        breadcrumb=source.breadcrumb,
        chunk_id=source.chunk_id,
    )


def build_sources(results: list[RetrievalResult]) -> list[Source]:
    """
    Deduplica le fonti per URL pulito, preservando l'ordine dei risultati.
    """
    sources: list[Source] = []
    seen_urls: set[str] = set()

    for result in results:
        source = get_display_source_from_result(result)

        if source.url in seen_urls:
            continue

        seen_urls.add(source.url)
        sources.append(source)

    return sources


def truncate_text(text: str, max_chars: int) -> str:
    text = text.strip()

    if len(text) <= max_chars:
        return text

    return text[:max_chars].rstrip() + "..."


def build_context(
    results: list[RetrievalResult],
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
) -> str:
    """
    Costruisce il contesto da passare all'LLM.
    Limita la lunghezza complessiva per non appesantire modelli piccoli.
    """
    blocks: list[str] = []
    current_chars = 0

    for index, result in enumerate(results, start=1):
        source = get_source_from_result(result)

        header = (
            f"[DOCUMENTO {index}]\n"
            f"Titolo: {source.title}\n"
            f"URL: {source.url}\n"
            f"Percorso: {source.breadcrumb}\n"
            f"Chunk ID: {source.chunk_id}\n"
            f"Contenuto:\n"
        )

        remaining_chars = max_context_chars - current_chars - len(header)

        if remaining_chars <= 300:
            break

        content = truncate_text(result.text, remaining_chars)
        block = header + content

        blocks.append(block)
        current_chars += len(block)

        if current_chars >= max_context_chars:
            break

    return "\n\n---\n\n".join(blocks)


LAB_ALIAS_PATTERNS = (
    (re.compile(r"\b(?:labrob|roblab|laboratorio\s+di\s+robotica)\b", re.IGNORECASE), "Laboratorio di Robotica LabROB del DIEM"),
    (re.compile(r"\b(?:nclab|computazione\s+naturale)\b", re.IGNORECASE), "NCLab Computazione Naturale del DIEM"),
    (re.compile(r"\b(?:lcem|caratterizzazione\s+elettromagnetica)\b", re.IGNORECASE), "LCEM Caratterizzazione Elettromagnetica dei Materiali del DIEM"),
    (re.compile(r"\b(?:mivia|macchine\s+intelligenti)\b", re.IGNORECASE), "MIVIA Macchine Intelligenti per il Riconoscimento di Video, Immagini e Audio del DIEM"),
    (re.compile(r"\b(?:inbit|intelligent\s+bioengineering)\b", re.IGNORECASE), "INBIT Intelligent Bioengineering Technologies del DIEM"),
    (re.compile(r"\b(?:knowmis|knowledge\s+management)\b", re.IGNORECASE), "KnowMIS Knowledge Management and Information Systems del DIEM"),
    (re.compile(r"\b(?:te4de|digital\s+energy)\b", re.IGNORECASE), "TE4DE Tecnologie Elettriche per la Digital Energy del DIEM"),
    (re.compile(r"\b(?:teti|telecomunicazioni)\b", re.IGNORECASE), "TETI Telecomunicazioni e Teoria dell'Informazione del DIEM"),
)


FOLLOW_UP_PATTERNS = [
    r"\bsuo\b",
    r"\bsua\b",
    r"\bsuoi\b",
    r"\bsue\b",
    r"\bquest[oaie]\b",
    r"\bquel(?:lo|la|li|le)?\b",
    r"\btale\b",
    r"\blaboratorio\b",
    r"\bstruttura\b",
    r"\bche strumenti possiede\b",
    r"\bquali strumenti\b",
    r"\bquando scade\b",
    r"\bqual[ie] sono\b",
    r"\bcome funziona\b",
    r"\bchi (?:è|e|sono)\b",
    r"\bdove si trova\b",
    r"\bdi cosa si occupa\b",
    r"\bquanto dura\b",
    r"\bcome si accede\b",
    r"\bcome posso candidarmi\b",
    r"\bquali requisiti\b",
]


SUBJECT_PATTERNS = (
    (re.compile(r"\borari?\s+di\s+ricevimento\b|\bricevimento\b", re.IGNORECASE), "orari di ricevimento dei docenti DIEM"),
    (re.compile(r"\b(?:erasmus|mobilità|mobilita|accordi\s+erasmus|traineeship|learning\s+agreement)\b", re.IGNORECASE), "Erasmus e mobilità internazionale del DIEM"),
    (re.compile(r"\b(?:dottorato|dottorati|phd|doctoral)\b", re.IGNORECASE), "dottorati collegati al DIEM"),
    (re.compile(r"\b(?:progetti?\s+finanziati?|progetti?\s+di\s+ricerca|ricerca|intelligenza\s+artificiale|ia\s+generativa)\b", re.IGNORECASE), "progetti di ricerca e progetti finanziati del DIEM"),
    (re.compile(r"\b(?:bandi?|avvisi?|graduatorie?|selezion[ei]|concorso|concorsi)\b", re.IGNORECASE), "bandi e avvisi del DIEM"),
    (re.compile(r"\b(?:corsi?\s+di\s+laurea|laure[ae]|laurea\s+magistrale|offerta\s+formativa|insegnamenti?|didattica)\b", re.IGNORECASE), "offerta formativa e corsi di laurea del DIEM"),
    (re.compile(r"\b(?:docenti|professori|professore|professoressa|personale|rubrica)\b", re.IGNORECASE), "docenti e personale del DIEM"),
    (re.compile(r"\b(?:laboratori|laboratorio|strutture|aule|centri)\b", re.IGNORECASE), "laboratori e strutture del DIEM"),
    (re.compile(r"\b(?:contatti?|sede|indirizzo|ubicazione|dove\s+si\s+trova)\b", re.IGNORECASE), "sede e contatti del DIEM"),
    (re.compile(r"\b(?:immatricolazioni?|iscrizion[ei]|accesso|requisiti|tolc|ofa|ammissione)\b", re.IGNORECASE), "requisiti di accesso e immatricolazioni dei corsi DIEM"),
    (re.compile(r"\b(?:servizi?|orientamento|tutorato|segreteria)\b", re.IGNORECASE), "servizi e orientamento del DIEM"),
)


def normalize_conversation_history(
    conversation_history: list[ConversationTurn] | None,
    max_messages: int = 12,
) -> list[ConversationTurn]:
    if not conversation_history:
        return []

    normalized: list[ConversationTurn] = []

    for turn in conversation_history[-max_messages:]:
        role = str(turn.get("role") or "").strip().lower()
        content = str(turn.get("content") or "").strip()

        if role not in {"user", "assistant"} or not content:
            continue

        normalized.append({"role": role, "content": content})

    return normalized


def expand_known_aliases(question: str) -> str:
    normalized_question = question.lower()

    for pattern, expansion in LAB_ALIAS_PATTERNS:
        if not pattern.search(question):
            continue

        if expansion.lower() in normalized_question:
            return question

        return f"{question} {expansion}"

    return question


def is_context_dependent_question(question: str) -> bool:
    question_lower = question.lower()
    tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9_]+", question_lower)

    if len(tokens) <= 6:
        return True

    return any(re.search(pattern, question_lower) for pattern in FOLLOW_UP_PATTERNS)


def extract_recent_subject(conversation_history: list[ConversationTurn] | None) -> str:
    for turn in reversed(normalize_conversation_history(conversation_history)):
        content = turn["content"]

        for pattern, subject in LAB_ALIAS_PATTERNS:
            if pattern.search(content):
                return subject

        teacher_match = re.search(
            r"\b(?:prof\.?|professore|professoressa|docente)\s+([A-ZÀ-Ý][a-zà-ÿ]+(?:\s+[A-ZÀ-Ý][a-zà-ÿ]+){1,2})",
            content,
        )
        if teacher_match:
            return f"docente {teacher_match.group(1)} del DIEM"

        for pattern, subject in SUBJECT_PATTERNS:
            if pattern.search(content):
                return subject

    return ""


def build_retrieval_question(
    question: str,
    conversation_history: list[ConversationTurn] | None = None,
) -> str:
    """
    Rende più autonome le domande di follow-up prima del retrieval.
    La riscrittura è deterministica e prudente: se non trova un soggetto
    recente affidabile, lascia la domanda invariata.
    """
    question = question.strip()
    alias_expanded_question = expand_known_aliases(question)

    if alias_expanded_question != question:
        return alias_expanded_question

    if not is_context_dependent_question(question):
        return question

    subject = extract_recent_subject(conversation_history)

    if not subject:
        return question

    return f"{question} {subject}"


def build_conversation_context(
    conversation_history: list[ConversationTurn] | None = None,
    retrieval_question: str = "",
    original_question: str = "",
) -> str:
    lines: list[str] = []
    history = normalize_conversation_history(conversation_history, max_messages=8)

    for turn in history:
        label = "Utente" if turn["role"] == "user" else "Assistente"
        content = truncate_text(turn["content"], 700)
        lines.append(f"{label}: {content}")

    if retrieval_question and retrieval_question != original_question:
        lines.append(f"Domanda contestualizzata per il recupero: {retrieval_question}")

    return "\n".join(lines)


def build_prompt(
    question: str,
    context: str,
    conversation_context: str = "",
) -> str:
    """
    Prompt RAG rigido:
    - usa solo il contesto;
    - non inventa;
    - gestisce fuori dominio;
    - non produce fonti inventate.
    """
    conversation_block = ""
    if conversation_context.strip():
        conversation_block = f"""
CONTESTO CONVERSAZIONALE:
{conversation_context}

Usa il contesto conversazionale solo per capire a quale soggetto si riferisce la domanda.
Non usarlo come fonte fattuale: le informazioni della risposta devono venire dal CONTESTO documentale.
""".strip()

    return f"""
Sei un assistente informativo del DIEM dell'Università di Salerno.

Devi rispondere alla domanda dell'utente usando esclusivamente il CONTESTO fornito.
Non usare conoscenza esterna.
Non inventare informazioni mancanti.
Non inventare date, orari, nomi di docenti, corsi, regolamenti, aule o link.
Se il contesto non contiene informazioni sufficienti, rispondi chiaramente:
"Non ho trovato questa informazione nelle fonti DIEM indicizzate."

Se la domanda riguarda la sede, l’ufficio, la stanza o il laboratorio di un docente e 
nel contesto sono presenti più locali associati a quel docente, non scrivere che 
"l'ufficio si trova" in più luoghi e non scegliere un locale principale se non è 
esplicitamente indicato. Rispondi invece con la formula: "Nelle fonti risultano questi 
locali associati al docente [nome docente]:", poi elenca i locali trovati.

Se la domanda non riguarda il DIEM, i corsi DIEM, i docenti DIEM, i servizi DIEM,
le attività didattiche, di ricerca, internazionali o i documenti ufficiali indicizzati,
rispondi chiaramente: ""La domanda è fuori dal contesto del DIEM.""

Rispondi in italiano, in modo chiaro, completo e strutturato.
Se il contesto contiene più dettagli utili, includili nella risposta.
Per domande che chiedono elenchi, panoramiche o informazioni articolate, usa punti elenco.
Non essere telegrafico: scrivi una risposta utile, con più frasi quando il contesto lo consente.
Quando sono disponibili dettagli su attività, strumenti, responsabili, sedi, date o descrizioni, includili in modo ordinato.
Non essere eccessivamente sintetico, ma non aggiungere informazioni non presenti nel contesto.
Inserisci citazioni inline nel testo, usando i numeri dei documenti: [1], [2].
Ogni affermazione fattuale specifica deve avere almeno una citazione.

Non aggiungere una sezione "Fonti" nella risposta discorsiva.
Restituisci solo un oggetto JSON valido in questo formato:
{{
  "answer": "testo della risposta con citazioni inline",
  "used_sources": [1, 2],
  "inline_citations": [1, 2],
  "no_answer_reason": ""
}}

Inserisci in used_sources solo i numeri dei documenti realmente usati.
Se non hai usato nessun documento perché il contesto è insufficiente o la domanda è fuori dominio,
usa used_sources: [] e inline_citations: [].

Esempio positivo:
DOMANDA: Quali corsi di laurea offre il DIEM?
RISPOSTA JSON:
{{
  "answer": "Il DIEM offre corsi di laurea e laurea magistrale elencati nell'offerta formativa, tra cui Ingegneria Informatica e Ingegneria dell'Informazione per la Medicina Digitale [1].",
  "used_sources": [1],
  "inline_citations": [1],
  "no_answer_reason": ""
}}

Esempio quando il contesto non basta:
{{
  "answer": "Non ho trovato questa informazione nelle fonti DIEM indicizzate.",
  "used_sources": [],
  "inline_citations": [],
  "no_answer_reason": "insufficient_context"
}}

CONTESTO:
{context}

{conversation_block}

DOMANDA UTENTE:
{question}

RISPOSTA:
""".strip()


@retry(
    retry=retry_if_exception_type(Exception),
    stop=stop_after_attempt(GROQ_MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def create_groq_completion(client: Groq, request_kwargs: dict[str, Any]):
    return client.chat.completions.create(**request_kwargs)


def call_groq(
    prompt: str,
    model: str = DEFAULT_GROQ_MODEL,
) -> str:
    """
    Chiama Groq tramite API.
    Richiede GROQ_API_KEY nel file .env.
    """
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY non trovata. "
            "Aggiungila nel file .env, ad esempio: GROQ_API_KEY=gsk_..."
        )

    client = Groq(api_key=api_key)
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": 0.1,
        "top_p": 0.9,
        "stream": False,
        "timeout": GROQ_TIMEOUT_SECONDS,
    }

    if GROQ_JSON_MODE:
        request_kwargs["response_format"] = {"type": "json_object"}

    try:
        completion = create_groq_completion(client, request_kwargs)
    except Exception as exc:
        if "response_format" in request_kwargs:
            request_kwargs.pop("response_format", None)
            try:
                completion = create_groq_completion(client, request_kwargs)
            except Exception as fallback_exc:
                raise RuntimeError(
                    "Non riesco a contattare Groq in questo momento. "
                    "Riprova tra poco."
                ) from fallback_exc
        else:
            raise RuntimeError(
                "Non riesco a contattare Groq in questo momento. "
                "Riprova tra poco."
            ) from exc

    answer = completion.choices[0].message.content

    if not answer:
        raise RuntimeError("Groq ha restituito una risposta vuota.")

    return strip_model_thinking(answer)


def strip_model_thinking(answer: str) -> str:
    """
    Rimuove i blocchi di ragionamento che alcuni modelli reasoning, come Qwen3,
    possono restituire nel formato <think>...</think>.
    """
    without_thinking = re.sub(
        r"<think>.*?</think>",
        "",
        answer,
        flags=re.IGNORECASE | re.DOTALL,
    )

    return without_thinking.strip()


def format_sources(sources: list[Source]) -> str:
    if not sources:
        return "Fonti: nessuna fonte disponibile."

    lines = ["Fonti:"]

    for index, source in enumerate(sources, start=1):
        lines.append(f"{index}. {source.title} — {source.url}")

    return "\n".join(lines)


def parse_used_source_indexes(answer: str) -> tuple[str, list[int]]:
    """
    Estrae la riga tecnica FONTI_USATE: [1, 2] dalla risposta del modello.

    Restituisce:
    - risposta pulita senza riga tecnica;
    - lista degli indici documento usati, in base ai numeri mostrati nel contesto.
    """
    pattern = re.compile(
        r"FONTI_USATE\s*:\s*\[([0-9,\s]*)\]",
        re.IGNORECASE,
    )

    match = pattern.search(answer)

    if not match:
        return answer.strip(), []

    raw_indexes = match.group(1)

    indexes: list[int] = []

    for item in raw_indexes.split(","):
        item = item.strip()

        if item.isdigit():
            indexes.append(int(item))

    cleaned_answer = pattern.sub("", answer).strip()

    return cleaned_answer, indexes


def parse_model_answer(answer: str) -> tuple[str, list[int]]:
    """
    Interpreta prima il JSON mode; se il modello non lo rispetta, usa il
    vecchio formato FONTI_USATE per compatibilità.
    """
    stripped_answer = answer.strip()

    try:
        payload = json.loads(stripped_answer)
    except json.JSONDecodeError:
        return parse_used_source_indexes(stripped_answer)

    if not isinstance(payload, dict):
        return parse_used_source_indexes(stripped_answer)

    clean_answer = str(payload.get("answer") or "").strip()
    raw_sources = payload.get("used_sources") or []

    indexes: list[int] = []
    if isinstance(raw_sources, list):
        for item in raw_sources:
            if isinstance(item, int):
                indexes.append(item)
            elif isinstance(item, str) and item.strip().isdigit():
                indexes.append(int(item.strip()))

    return clean_answer, indexes


GENERIC_TEACHER_WORDS = {
    "professor",
    "professore",
    "professoressa",
    "professori",
    "prof",
    "prof.",
    "docente",
    "docenti",
    "ricevimento",
    "ricevimenti",
    "orario",
    "orari",
    "ore",
    "quando",
    "dove",
    "dici",
    "dire",
    "dimmi",
    "sapere",
    "conoscere",
    "vorrei",
    "voglio",
    "puoi",
    "mi",
    "quali",
    "qual",
    "sono",
    "diem",
    "del",
    "della",
    "dei",
    "degli",
    "delle",
    "di",
    "in",
    "il",
    "lo",
    "la",
    "le",
    "gli",
    "un",
    "una",
    "uno",
}


def filter_sources_by_document_indexes(
    results: list[RetrievalResult],
    used_indexes: list[int],
) -> list[Source]:
    """
    Converte gli indici dichiarati dal modello rispetto ai DOCUMENTI del contesto
    nelle fonti corrispondenti.

    Esempio:
    DOCUMENTO 1 -> Studio
    DOCUMENTO 2 -> Studio
    DOCUMENTO 3 -> Traineeship

    Se il modello restituisce FONTI_USATE: [1, 2],
    la fonte finale deve essere solo Studio, non Studio + Traineeship.
    """
    if not used_indexes:
        return []

    filtered_sources: list[Source] = []
    seen_urls: set[str] = set()

    for index in used_indexes:
        result_position = index - 1

        if not 0 <= result_position < len(results):
            continue

        source = get_display_source_from_result(results[result_position])

        if source.url in seen_urls:
            continue

        seen_urls.add(source.url)
        filtered_sources.append(source)

    return filtered_sources


def simple_tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9_]+", text.lower())

    return {
        token
        for token in tokens
        if len(token) > 1 and token not in GENERIC_TEACHER_WORDS
    }


@lru_cache(maxsize=1)
def load_rag_chunks() -> tuple[dict[str, Any], ...]:
    return tuple(load_jsonl(CHUNKS_FILE))


def result_from_chunk(chunk: dict[str, Any], rank: int, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=str(chunk.get("chunk_id") or chunk.get("text_hash") or f"chunk_{rank}"),
        text=str(chunk.get("text") or ""),
        metadata=dict(chunk),
        source="teacher_lookup",
        rank=rank,
        score=score,
    )


def teacher_tokens_from_question(question: str) -> set[str]:
    return simple_tokenize(question)


def metadata_tokens_for_chunk(chunk: dict[str, Any]) -> set[str]:
    metadata_text = " ".join(
        str(chunk.get(key) or "")
        for key in ["title", "breadcrumb", "breadcrumb_text", "source_url", "document_url"]
    )
    return simple_tokenize(metadata_text)


def find_teacher_profile_office_hours(question: str) -> list[RetrievalResult]:
    teacher_tokens = teacher_tokens_from_question(question)
    if not teacher_tokens:
        return []

    matches: list[RetrievalResult] = []

    for chunk in load_rag_chunks():
        url = str(chunk.get("source_url") or chunk.get("document_url") or "")
        if "docenti.unisa.it" not in url:
            continue

        combined_tokens = metadata_tokens_for_chunk(chunk).union(
            simple_tokenize(str(chunk.get("text") or ""))
        )

        if not teacher_metadata_matches(teacher_tokens, combined_tokens):
            continue

        text_lower = str(chunk.get("text") or "").lower()
        if "ricevimento" not in text_lower:
            continue

        score = 3.0
        if url.rstrip("/").endswith("/home"):
            score += 1.0

        matches.append(result_from_chunk(chunk, rank=len(matches) + 1, score=score))

    matches.sort(key=lambda result: result.score, reverse=True)

    for rank, result in enumerate(matches, start=1):
        result.rank = rank

    return matches[:3]


def find_teacher_personnel_listing(question: str) -> RetrievalResult | None:
    teacher_tokens = teacher_tokens_from_question(question)
    if not teacher_tokens:
        return None

    for chunk in load_rag_chunks():
        title = str(chunk.get("title") or "")
        if title != "Dipartimento | Docenti e Personale":
            continue

        text_tokens = simple_tokenize(str(chunk.get("text") or ""))
        if teacher_metadata_matches(teacher_tokens, text_tokens):
            return result_from_chunk(chunk, rank=1, score=1.0)

    return None


def retrieve_teacher_office_hours(question: str) -> tuple[list[RetrievalResult], RetrievalResult | None]:
    profile_results = find_teacher_profile_office_hours(question)
    if profile_results:
        return profile_results, None

    return [], find_teacher_personnel_listing(question)


def is_office_hours_query(question: str) -> bool:
    question_lower = question.lower()

    return any(
        keyword in question_lower
        for keyword in [
            "ricevimento",
            "orario di ricevimento",
            "orari di ricevimento",
            "riceve",
            "ricevono",
        ]
    )


def is_specific_office_hours_query(question: str) -> bool:
    question_lower = question.lower()

    asks_office_hours = is_office_hours_query(question_lower)

    teacher_name_tokens = simple_tokenize(question_lower)

    return asks_office_hours and bool(teacher_name_tokens)


def is_generic_office_hours_query(question: str) -> bool:
    """
    Riconosce domande tipo:
    - Quali sono gli orari di ricevimento dei docenti?
    - Quando ricevono i docenti?

    In questo caso non bisogna mostrare orari casuali:
    bisogna chiedere il nome del docente.
    """
    question_lower = question.lower()

    if not is_office_hours_query(question_lower):
        return False

    teacher_name_tokens = simple_tokenize(question_lower)

    return not teacher_name_tokens


def is_out_of_scope_university_query(question: str) -> bool:
    """
    Riconosce domande rivolte esplicitamente ad altre università.
    Il chatbot deve rispondere solo su DIEM/UNISA.
    """
    question_lower = question.lower()

    allowed_terms = [
        "diem",
        "unisa",
        "università di salerno",
        "universita di salerno",
        "university of salerno",
    ]

    if any(term in question_lower for term in allowed_terms):
        return False

    other_university_patterns = [
        r"\buniversità di\s+(?!salerno\b)[a-zà-ÿ\s]+",
        r"\buniversita di\s+(?!salerno\b)[a-zà-ÿ\s]+",
        r"\buniversity of\s+(?!salerno\b)[a-zà-ÿ\s]+",
        r"\bpolitecnico di\s+[a-zà-ÿ\s]+",
    ]

    return any(
        re.search(pattern, question_lower)
        for pattern in other_university_patterns
    )
    

def is_out_of_scope_department_query(question: str) -> bool:
    """
    Riconosce domande rivolte esplicitamente a un dipartimento diverso dal DIEM,
    senza mantenere una lista dei dipartimenti UNISA.
    """
    question_lower = question.lower()

    allowed_terms = [
        "diem",
        "dipartimento di ingegneria dell'informazione ed elettrica",
        "dipartimento di ingegneria dell informazione ed elettrica",
    ]

    if any(term in question_lower for term in allowed_terms):
        return False

    if "dipartimento" not in question_lower:
        return False

    asks_department_info = any(
        keyword in question_lower
        for keyword in [
            "corsi",
            "corso",
            "lauree",
            "laurea",
            "docenti",
            "laboratori",
            "strutture",
            "servizi",
            "offerta formativa",
            "dove si trova",
        ]
    )

    mentions_named_department = bool(
        re.search(
            r"\bdipartimento\s+(?:di\s+)?[a-zA-ZÀ-ÿ0-9_-]+",
            question,
            re.IGNORECASE,
        )
    )

    return asks_department_info and mentions_named_department


def teacher_metadata_matches(
    teacher_name_tokens: set[str],
    metadata_tokens: set[str],
) -> bool:
    """
    Verifica se i token del docente richiesto nella query corrispondono
    ai token presenti nei metadati della pagina docente.

    Se la query contiene nome+cognome, richiede almeno 2 token in comune.
    Se contiene un solo token, accetta 1 token in comune.
    """
    matching_tokens = teacher_name_tokens.intersection(metadata_tokens)

    if len(teacher_name_tokens) >= 2:
        return len(matching_tokens) >= 2

    return len(matching_tokens) >= 1


def filter_chunks_for_generation(
    question: str,
    results: list[RetrievalResult],
) -> list[RetrievalResult]:
    """
    Filtra i chunk prima della generazione.

    Se la domanda chiede l'orario di ricevimento di un docente specifico,
    usa solo la pagina personale del docente richiesto.
    """
    if not is_specific_office_hours_query(question):
        return results

    teacher_name_tokens = simple_tokenize(question)
    filtered_results: list[RetrievalResult] = []

    for result in results:
        source = get_source_from_result(result)

        metadata_text = f"{source.title} {source.breadcrumb} {source.url}".lower()
        metadata_tokens = simple_tokenize(metadata_text)

        is_docenti_page = "docenti.unisa.it" in source.url

        if is_docenti_page and teacher_metadata_matches(teacher_name_tokens, metadata_tokens):
            filtered_results.append(result)

    if filtered_results:
        return filtered_results[:1]

    return []


def clean_markdown_text(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("**", "")
    text = text.replace("*", "")
    return " ".join(text.split())


def extract_office_hours_rows(text: str) -> list[tuple[str, str, str]]:
    """
    Estrae righe di orario di ricevimento da tabelle Markdown.

    Gestisce sia tabelle con 3 colonne:
    | Giorno | Orario | Luogo |

    sia tabelle con 2 colonne:
    | Giorno | Orario |
    """
    day_pattern = re.compile(
        r"\b(lunedì|lunedi|martedì|martedi|mercoledì|mercoledi|giovedì|giovedi|venerdì|venerdi|sabato|domenica)\b",
        re.IGNORECASE,
    )

    time_pattern = re.compile(
        r"\b\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2}\b"
    )

    cells = [clean_markdown_text(cell) for cell in text.split("|")]
    cells = [cell.strip() for cell in cells if cell.strip()]

    rows: list[tuple[str, str, str]] = []

    for index in range(len(cells) - 1):
        day_cell = cells[index]
        time_cell = cells[index + 1]

        day_match = day_pattern.search(day_cell)
        time_match = time_pattern.search(time_cell)

        if not day_match or not time_match:
            continue

        day = day_match.group(1)
        time = time_match.group(0)

        place = ""

        # Se esiste una terza cella utile, la usiamo come luogo.
        # Altrimenti lasciamo il luogo vuoto.
        if index + 2 < len(cells):
            possible_place = cells[index + 2].strip()

            is_separator = "---" in possible_place
            is_next_day = bool(day_pattern.search(possible_place))
            is_navigation = possible_place.lower().startswith(
                ("docenti", "ricerca", "materiali", "international")
            )

            if not is_separator and not is_next_day and not is_navigation:
                place = possible_place

        row = (day, time, place)

        if row not in rows:
            rows.append(row)

    return rows


def build_direct_office_hours_answer(
    question: str,
    results: list[RetrievalResult],
) -> str | None:
    """
    Per domande sugli orari di ricevimento di un docente specifico,
    produce una risposta estrattiva e non generativa.

    Questo evita che il modello LLM mischi orari di docenti diversi
    o perda righe della tabella.
    """
    if not is_specific_office_hours_query(question):
        return None

    if not results:
        return None

    result = results[0]
    source = get_source_from_result(result)

    rows = extract_office_hours_rows(result.text)

    if not rows:
        return None

    teacher_name = source.title.replace("| Home", "").strip()

    lines = [f"Gli orari di ricevimento di {teacher_name} sono:"]

    for day, time, place in rows:
        if place:
            lines.append(f"- {day}: {time}, {place}.")
        else:
            lines.append(f"- {day}: {time}.")

    return "\n".join(lines)


def answer_question(
    question: str,
    final_k: int = DEFAULT_FINAL_K,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    model: str = DEFAULT_GROQ_MODEL,
    conversation_history: list[ConversationTurn] | None = None,
) -> RagResponse:
    """
    Pipeline RAG completa:
    domanda -> retrieval ibrido -> eventuale filtro/estrazione -> LLM -> risposta + fonti.
    """
    question = question.strip()

    if not question:
        raise ValueError("La domanda non può essere vuota.")

    retrieval_question = build_retrieval_question(
        question=question,
        conversation_history=conversation_history,
    )
    
    if is_out_of_scope_university_query(question):
        return RagResponse(
            question=question,
            answer="La domanda è fuori dal contesto del DIEM.",
            sources=[],
            retrieved_chunks=[],
        )
        
    if is_out_of_scope_department_query(question):
        return RagResponse(
            question=question,
            answer="La domanda è fuori dal contesto del DIEM.",
            sources=[],
            retrieved_chunks=[],
        )

    # Caso particolare ma generale:
    # se l'utente chiede gli orari di ricevimento dei docenti senza indicare il nome,
    # non ha senso recuperare docenti casuali.
    if is_generic_office_hours_query(retrieval_question):
        return RagResponse(
            question=question,
            answer=(
                "Gli orari di ricevimento sono specifici per ciascun docente. "
                "Indica il nome del docente per cui vuoi conoscere l'orario."
            ),
            sources=[],
            retrieved_chunks=[],
        )

    if is_specific_office_hours_query(retrieval_question):
        profile_chunks, personnel_result = retrieve_teacher_office_hours(retrieval_question)

        if profile_chunks:
            direct_answer = build_direct_office_hours_answer(
                question=retrieval_question,
                results=profile_chunks,
            )

            if direct_answer:
                return RagResponse(
                    question=question,
                    answer=f"{direct_answer} [1]",
                    sources=build_sources(profile_chunks),
                    retrieved_chunks=profile_chunks,
                )

            return RagResponse(
                question=question,
                answer=(
                    "Ho trovato la pagina del docente, ma non sono riuscito a estrarre "
                    "automaticamente gli orari di ricevimento dal testo indicizzato."
                ),
                sources=build_sources(profile_chunks),
                retrieved_chunks=profile_chunks,
            )

        if personnel_result:
            return RagResponse(
                question=question,
                answer=(
                    "Ho trovato il docente nell'elenco del personale DIEM, ma non ho trovato "
                    "nelle fonti indicizzate una pagina personale con gli orari di ricevimento."
                ),
                sources=build_sources([personnel_result]),
                retrieved_chunks=[personnel_result],
            )

        return RagResponse(
            question=question,
            answer="Non ho trovato questo docente nelle fonti DIEM indicizzate.",
            sources=[],
            retrieved_chunks=[],
        )

    retrieved_chunks = hybrid_retrieve(
        query=retrieval_question,
        final_k=final_k,
    )

    retrieved_chunks = filter_chunks_for_generation(
        question=retrieval_question,
        results=retrieved_chunks,
    )

    if not retrieved_chunks:
        return RagResponse(
            question=question,
            answer="Non ho trovato informazioni pertinenti nelle fonti DIEM indicizzate.",
            sources=[],
            retrieved_chunks=[],
        )

    # Per gli orari di ricevimento è più sicuro estrarre direttamente la tabella
    # invece di affidarsi alla generazione del modello.
    direct_answer = build_direct_office_hours_answer(
        question=retrieval_question,
        results=retrieved_chunks,
    )

    if direct_answer:
        sources = build_sources(retrieved_chunks)

        return RagResponse(
            question=question,
            answer=direct_answer,
            sources=sources,
            retrieved_chunks=retrieved_chunks,
        )
        
    # Se la domanda riguarda l'orario di ricevimento di un docente specifico
    # ma non siamo riusciti a estrarre la tabella, non passiamo al modello generativo:
    # meglio evitare risposte inventate o errori di memoria.
    if is_specific_office_hours_query(retrieval_question):
        sources = build_sources(retrieved_chunks)

        return RagResponse(
            question=question,
            answer=(
                "Ho trovato la pagina del docente, ma non sono riuscito a estrarre "
                "automaticamente gli orari di ricevimento dal testo indicizzato."
            ),
            sources=sources,
            retrieved_chunks=retrieved_chunks,
        )

    context = build_context(
        results=retrieved_chunks,
        max_context_chars=max_context_chars,
    )

    conversation_context = build_conversation_context(
        conversation_history=conversation_history,
        retrieval_question=retrieval_question,
        original_question=question,
    )

    prompt = build_prompt(
        question=question,
        context=context,
        conversation_context=conversation_context,
    )

    raw_answer = call_groq(
        prompt=prompt,
        model=model,
    )

    clean_answer, used_source_indexes = parse_model_answer(raw_answer)

    all_sources = build_sources(retrieved_chunks)

    used_sources = filter_sources_by_document_indexes(
        results=retrieved_chunks,
        used_indexes=used_source_indexes,
    )

    # Fallback:
    # se il modello non rispetta il formato FONTI_USATE,
    # mostriamo solo la prima fonte recuperata invece di tutte le fonti rumorose.
    answer_lower = clean_answer.lower()
    
    is_no_source_answer = any(
        phrase in answer_lower
        for phrase in [
            "non ho trovato",
            "fuori dominio",
            "fuori dal contesto",
            "fuori dal dominio",
            "non riguarda il diem",
            "non riguarda le fonti diem",
        ]
    )
    
    

    if not used_sources and all_sources and not is_no_source_answer:
        used_sources = all_sources[:1]

    if is_no_source_answer:
        used_sources = []

    return RagResponse(
        question=question,
        answer=clean_answer,
        sources=used_sources,
        retrieved_chunks=retrieved_chunks,
    )


def answer_question_as_text(
    question: str,
    final_k: int = DEFAULT_FINAL_K,
    model: str = DEFAULT_GROQ_MODEL,
    conversation_history: list[ConversationTurn] | None = None,
) -> str:
    """
    Utility comoda per CLI/Chainlit: restituisce risposta già formattata.

    Se non ci sono fonti, mostra solo la risposta.
    Questo evita righe inutili come:
    "Fonti: nessuna fonte disponibile."
    per fuori dominio, domande generiche o contesto insufficiente.
    """
    response = answer_question(
        question=question,
        final_k=final_k,
        model=model,
        conversation_history=conversation_history,
    )

    if not response.sources:
        return response.answer

    return f"{response.answer}\n\n{format_sources(response.sources)}"
