from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

from dotenv import load_dotenv
from groq import Groq
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from chunk_metadata import flatten_chunk_metadata
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
CONTEXTUALIZER_MAX_HISTORY_MESSAGES = int(os.getenv("RAG_CONTEXTUALIZER_MAX_HISTORY_MESSAGES", "6"))

MISSING_TITLE_PLACEHOLDERS = {
    "",
    "n/d",
    "n.a.",
    "n/a",
    "nd",
    "none",
    "null",
    "titolo non disponibile",
    "titolo non presente",
    "senza titolo",
}


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


def title_from_url(url: str) -> str:
    """
    Crea un'etichetta leggibile quando la fonte non ha un titolo indicizzato.
    Utile soprattutto per PDF, che spesso non espongono metadata title.
    """
    if not url or url == "URL non disponibile":
        return ""

    path = urlsplit(url).path
    slug = unquote(path.rstrip("/").split("/")[-1]).strip()
    if not slug:
        return ""

    slug = re.sub(r"\.(pdf|html?|aspx?)$", "", slug, flags=re.IGNORECASE)
    words = [word for word in re.split(r"[-_\s]+", slug) if word]
    if not words:
        return ""

    keep_uppercase = {"diem", "ofa", "tolc", "tolc-i", "pdf"}
    formatted_words = [
        word.upper() if word.lower() in keep_uppercase else word.capitalize()
        for word in words
    ]
    return " ".join(formatted_words)


def display_title_from_metadata(metadata: dict[str, Any], url: str) -> str:
    title = metadata_to_string(metadata.get("title")).strip()
    if title.lower() not in MISSING_TITLE_PLACEHOLDERS:
        return title

    breadcrumb = (
        metadata_to_string(metadata.get("breadcrumb"))
        or metadata_to_string(metadata.get("breadcrumb_text"))
    ).strip()
    if breadcrumb:
        return breadcrumb.split(" > ")[-1].strip() or breadcrumb

    return title_from_url(url) or "Fonte senza titolo"


def get_source_from_result(result: RetrievalResult) -> Source:
    metadata = result.metadata or {}

    url = (
        metadata_to_string(metadata.get("source_url"))
        or metadata_to_string(metadata.get("document_url"))
        or metadata_to_string(metadata.get("url"))
        or "URL non disponibile"
    )

    title = display_title_from_metadata(metadata, url)

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


def truthy_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


CONTEXT_REFERENCE_PATTERNS = [
    r"\blui\b",
    r"\blei\b",
    r"\bloro\b",
    r"\bsuo\b",
    r"\bsua\b",
    r"\bsuoi\b",
    r"\bsue\b",
    r"\bquest[oaie]\b",
    r"\bquel(?:lo|la|li|le)?\b",
    r"\btale\b",
    r"\bne\b",
]

ELLIPTIC_FOLLOW_UP_PATTERNS = [
    r"^(?:e|invece|anche)\b",
    r"\bquanto dura\b",
    r"\bquanto costa\b",
    r"\bquando scade\b",
    r"\bqual[ei]\s+(?:è|e|sono)\s+la\s+scadenza\b",
    r"\bcome funziona\b",
    r"\bcome si fa\b",
    r"\bcome si accede\b",
    r"\bcome partecipare\b",
    r"\bcome posso partecipare\b",
    r"\bcome posso candidarmi\b",
    r"\ba chi (?:rivolgersi|mi posso rivolgere)\b",
    r"\bdi cosa si occupa\b",
    r"\bche requisiti\b",
    r"\bquali requisiti\b",
    r"\bqual[ei]\s+(?:è|e|sono)\s+i\s+requisiti\b",
    r"\bci sono requisiti\b",
    r"\bchi (?:è|e|sono)\b",
    r"\bdove (?:si trova|trovo)\b",
    r"\bche strumenti possiede\b",
    r"\bquali strumenti\b",
    r"\bprossim[oa]\s+semestre\b",
]

EXPLICIT_ANCHOR_PATTERNS = [
    r"\b(?:requisiti|criteri|scadenza|durata|costo|calendario|sede|studio|ufficio|strumenti|strumentazione)\s+(?:di|del|della|dei|degli|delle|per|a|al|alla|su|sul|sulla)\s+\w+",
    r"\b(?:come funziona|come si accede|dove si trova|dove trovo|chi sono|chi è)\s+(?:il|lo|la|i|gli|le|l'|un|una)?\s*\w+",
    r"\b(?:programma|progetto|bando|corso|docente|professore|professoressa|laboratorio|servizio|seduta|prova|esame)\s+(?:di|del|della|dei|degli|delle|per|a|al|alla|su|sul|sulla)?\s*\w+",
]


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


def has_context_reference(question: str) -> bool:
    question_lower = question.lower()
    return any(re.search(pattern, question_lower) for pattern in CONTEXT_REFERENCE_PATTERNS)


def has_explicit_anchor(question: str) -> bool:
    question_lower = question.lower()
    return any(re.search(pattern, question_lower) for pattern in EXPLICIT_ANCHOR_PATTERNS)


def has_elliptic_followup_shape(question: str) -> bool:
    question_lower = question.lower()
    return any(re.search(pattern, question_lower) for pattern in ELLIPTIC_FOLLOW_UP_PATTERNS)


def has_follow_up_cue(question: str) -> bool:
    question_lower = question.lower().strip()

    follow_up_patterns = [
        r"^e\b",
        r"\binvece\b",
        r"\banche\b",
        r"\bquelli\b",
        r"\bquelle\b",
        r"\bquesto\b",
        r"\bquesta\b",
        r"\bquesti\b",
        r"\bqueste\b",
        r"\bsuo\b",
        r"\bsua\b",
        r"\bsuoi\b",
        r"\bsue\b",
        r"\btale\b",
        r"\blo stesso\b",
        r"\bla stessa\b",
    ]

    return any(re.search(pattern, question_lower) for pattern in follow_up_patterns)


def has_explicit_standalone_topic(question: str) -> bool:
    question_lower = question.lower()

    standalone_topics = [
        "diem",
        "unisa",
        "università di salerno",
        "universita di salerno",
        "dipartimento",
        "corsi",
        "corso",
        "laurea",
        "lauree",
        "triennale",
        "triennali",
        "magistrale",
        "magistrali",
        "offerta formativa",
        "laboratori",
        "laboratorio",
        "strutture",
        "aule",
        "erasmus",
        "accordi",
        "dottorato",
        "dottorati",
        "phd",
        "docenti",
        "professori",
        "personale",
        "sede",
        "contatti",
        "indirizzo",
        "bandi",
        "bando",
        "avvisi",
        "servizi",
        "ricerca",
        "didattica",
    ]

    return any(topic in question_lower for topic in standalone_topics)


def is_context_dependent_question(question: str) -> bool:
    question_lower = question.lower().strip()
    tokens = re.findall(r"[a-zA-ZÀ-ÿ0-9_]+", question_lower)

    if not tokens:
        return False

    if has_context_reference(question):
        return True

    if has_explicit_standalone_topic(question):
        return False

    if has_follow_up_cue(question):
        return True

    if not has_elliptic_followup_shape(question):
        return False

    return not has_explicit_anchor(question)


def latest_substantive_user_turn(
        conversation_history: list[ConversationTurn] | None,
) -> str:
        for turn in reversed(normalize_conversation_history(conversation_history)):
                if turn["role"] == "user" and not is_low_signal_turn(turn["content"]):
                        return turn["content"]

        return ""


def is_low_signal_turn(content: str) -> bool:
        normalized = content.strip().lower()
        return normalized in {
                "ciao",
                "salve",
                "buongiorno",
                "buonasera",
                "ok",
                "okay",
                "grazie",
                "perfetto",
                "va bene",
                "chiaro",
        }


def contextualizer_enabled() -> bool:
        return truthy_env("RAG_CONTEXTUALIZER_ENABLED", default=True)


def format_contextualizer_history(
        conversation_history: list[ConversationTurn] | None,
) -> str:
        history = normalize_conversation_history(
                conversation_history,
                max_messages=CONTEXTUALIZER_MAX_HISTORY_MESSAGES,
        )
        lines: list[str] = []

        for turn in history:
                label = "Utente" if turn["role"] == "user" else "Assistente"
                lines.append(f"{label}: {truncate_text(turn['content'], 500)}")

        return "\n".join(lines)


def build_contextualizer_prompt(
        question: str,
        conversation_history: list[ConversationTurn] | None,
) -> str:
        history_text = format_contextualizer_history(conversation_history)

        return f"""
Sei un modulo di contestualizzazione per un sistema RAG.

Devi decidere se la DOMANDA CORRENTE dipende davvero dalla CRONOLOGIA per essere
capita dal retriever. Non devi rispondere alla domanda.

Regole:
- Se la domanda corrente introduce un soggetto, corso, docente, servizio, documento
    o tema esplicito nuovo, non usare la cronologia: restituisci la domanda invariata.
- Se la domanda corrente è già comprensibile e ricercabile da sola, restituiscila
    invariata.
- Usa la cronologia solo per risolvere pronomi, deittici o ellissi, ad esempio
    "lui", "lei", "suo", "questo", "quello", "come funziona?", "quando scade?",
    "quali sono i requisiti?", "dove si trova?".
- Preferisci sempre l'ultimo scambio utente/assistente. Non recuperare soggetti
    vecchi se nel frattempo l'utente ha cambiato argomento.
- Se non c'è un antecedente chiaro e recente, non inventarlo: restituisci la
    domanda invariata.
- La domanda autonoma deve restare in italiano e deve contenere solo le parole
    necessarie per il retrieval, non una risposta.

Restituisci solo JSON valido:
{{
    "needs_context": true,
    "standalone_question": "domanda autonoma per il retrieval",
    "reason": "breve motivo"
}}

CRONOLOGIA:
{history_text or "Nessuna cronologia utile."}

DOMANDA CORRENTE:
{question}
""".strip()


def sanitize_contextualized_question(
    original_question: str,
    candidate: str,
) -> str:
    candidate = " ".join(candidate.split()).strip()
    if not candidate:
        return original_question

    if len(candidate) > max(400, len(original_question) * 4):
        return original_question

    return candidate


def parse_contextualizer_payload(
    raw_response: str,
    original_question: str,
) -> str:
    try:
        payload = json.loads(strip_model_thinking(raw_response))
    except json.JSONDecodeError:
        return original_question

    if not isinstance(payload, dict):
        return original_question

    needs_context = payload.get("needs_context")
    standalone_question = str(payload.get("standalone_question") or "").strip()

    if needs_context is not True:
        return original_question

    return sanitize_contextualized_question(original_question, standalone_question)


def call_contextualizer_groq(prompt: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY non trovata.")

    client = Groq(api_key=api_key)
    request_kwargs: dict[str, Any] = {
        "model": os.getenv("RAG_CONTEXTUALIZER_MODEL", DEFAULT_GROQ_MODEL),
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 0.1,
        "stream": False,
        "timeout": GROQ_TIMEOUT_SECONDS,
        "response_format": {"type": "json_object"},
    }

    try:
        completion = create_groq_completion(client, request_kwargs)
    except Exception:
        request_kwargs.pop("response_format", None)
        completion = create_groq_completion(client, request_kwargs)

    answer = completion.choices[0].message.content
    if not answer:
        raise RuntimeError("Il contextualizer ha restituito una risposta vuota.")

    return answer


def contextualize_question_with_llm(
    question: str,
    conversation_history: list[ConversationTurn] | None,
) -> str:
    if not contextualizer_enabled() or not normalize_conversation_history(conversation_history):
        return question

    prompt = build_contextualizer_prompt(
        question=question,
        conversation_history=conversation_history,
    )
    raw_response = call_contextualizer_groq(prompt)
    return parse_contextualizer_payload(raw_response, question)


def deterministic_contextualize_question(
    question: str,
    conversation_history: list[ConversationTurn] | None,
) -> str:
    """
    Fallback deterministico quando il contextualizer LLM non è disponibile.

    Non prova a classificare il topic: usa solo segnali linguistici generici
    di anafora/ellissi e l'ultimo turno utente sostanziale.
    """
    if not is_context_dependent_question(question):
        return question

    recent_user_turn = latest_substantive_user_turn(conversation_history)
    if not recent_user_turn:
        return question

    return (
        f"{question} "
        f"Contesto della domanda precedente: {truncate_text(recent_user_turn, 180)}"
    )


def build_retrieval_question(
    question: str,
    conversation_history: list[ConversationTurn] | None = None,
) -> str:
    """
    Rende più autonome le domande di follow-up prima del retrieval.

    Il percorso principale è un contextualizer LLM topic-agnostic, come nei
    history-aware retriever: se la domanda è autonoma resta invariata, se è
    ellittica viene riscritta in domanda standalone. Le euristiche sotto sono
    solo fallback per mantenere il chatbot utilizzabile quando il rewriter non
    è disponibile.
    """
    question = question.strip()
    has_history = bool(normalize_conversation_history(conversation_history))

    try:
        if contextualizer_enabled() and has_history:
            contextualized_question = contextualize_question_with_llm(
                question=question,
                conversation_history=conversation_history,
            )
            return expand_known_aliases(contextualized_question)
    except Exception:
        pass

    alias_expanded_question = expand_known_aliases(question)
    if alias_expanded_question != question:
        return alias_expanded_question
    
     # Follow-up: "invece di Mario Di Mauro" dopo una domanda sulle pubblicazioni.
    if "invece" in question.lower() and recent_context_was_publications(conversation_history):
        new_subject = extract_name_after_instead(question)

        if new_subject:
            return f"quali sono le recenti pubblicazioni del professore {new_subject}?"
    
    if is_office_hours_query(question) and len(teacher_tokens_from_question(question)) >= 2:
        return question

    return deterministic_contextualize_question(
        question=question,
        conversation_history=conversation_history,
    )


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
Se la domanda non specifica un anno accademico o un periodo storico, privilegia le informazioni correnti.
Non citare anni accademici passati, pagine storiche o versioni archiviate se non sono necessari per rispondere.
Se nel contesto sono presenti sia informazioni correnti sia informazioni di anni precedenti, usa solo quelle correnti, salvo richiesta esplicita dell'utente.
Se il contesto non contiene informazioni sufficienti, rispondi chiaramente:
"Non ho trovato questa informazione nelle fonti DIEM indicizzate."

Se la domanda riguarda la sede, l’ufficio, la stanza o il laboratorio di un docente e 
nel contesto sono presenti più locali associati a quel docente, non scrivere che 
"l'ufficio si trova" in più luoghi e non scegliere un locale principale se non è 
esplicitamente indicato. Rispondi invece con la formula: "Nelle fonti risultano questi 
locali associati al docente [nome docente]:", poi elenca i locali trovati.

Se la domanda non riguarda il DIEM, i corsi DIEM, i docenti DIEM, i servizi DIEM,
le attività didattiche, di ricerca, internazionali o i documenti ufficiali indicizzati,
rispondi chiaramente: "La domanda è fuori dal contesto del DIEM."

Rispondi in italiano, in modo chiaro e strutturato.

Se la domanda chiede un elenco, ad esempio "quali sono", "elenca", "quali corsi", "quali docenti", "quali laboratori",
rispondi preferibilmente con una breve frase introduttiva e poi con punti elenco.

Per le domande sui corsi di laurea, lauree triennali, lauree magistrali o offerta formativa:
- usa un elenco puntato;
- inserisci nome del corso, codice/classe se presenti nel contesto;
- non scrivere una risposta discorsiva lunga;
- non aggiungere dettagli secondari, anni storici o informazioni non richieste.

Se il contesto contiene più dettagli utili e pertinenti alla domanda, includili nella risposta.
Non aggiungere dettagli secondari, storici o non richiesti solo perché presenti nel contesto.

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
DOMANDA: Quali sono i corsi di laurea triennale del DIEM?
RISPOSTA JSON:
{{
  "answer": "I corsi di laurea triennale del DIEM sono:\\n- Ingegneria dell'Informazione per la Medicina Digitale, codice IE128L-8 [1];\\n- Ingegneria Informatica, codice IE127L-8 [1].",
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
    "pubblicazione",
    "pubblicazioni",
    "articolo",
    "articoli",
    "paper",
    "lavori",
    "scientifici",
    "recente",
    "recenti",
    "ultima",
    "ultime",
    "ultimo",
    "ultimi",
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


def select_sources_and_remap_citations(
    answer: str,
    results: list[RetrievalResult],
    used_indexes: list[int],
) -> tuple[str, list[Source]]:
    """
    Converte le citazioni del modello rispetto ai DOCUMENTI del contesto
    in citazioni coerenti con la lista finale delle fonti.

    Esempio:
    answer cita [1][6]
    fonti finali diventano:
    1. Fonte del documento 1
    2. Fonte del documento 6

    answer viene riscritto come [1][2].
    """
    if not used_indexes:
        return answer, []

    sources: list[Source] = []
    seen_urls: dict[str, int] = {}
    old_to_new_index: dict[int, int] = {}

    for old_index in used_indexes:
        result_position = old_index - 1

        if not 0 <= result_position < len(results):
            continue

        source = get_display_source_from_result(results[result_position])

        if source.url in seen_urls:
            old_to_new_index[old_index] = seen_urls[source.url]
            continue

        sources.append(source)
        new_index = len(sources)
        seen_urls[source.url] = new_index
        old_to_new_index[old_index] = new_index

    def replace_citation(match: re.Match) -> str:
        old_index = int(match.group(1))
        new_index = old_to_new_index.get(old_index)

        if new_index is None:
            return ""

        return f"[{new_index}]"

    # Rimappa le citazioni rispetto alla lista finale delle fonti.
    remapped_answer = re.sub(r"\[(\d+)\]", replace_citation, answer)

    # Pulisce eventuali citazioni duplicate consecutive, tipo [1][1].
    remapped_answer = re.sub(r"(\[\d+\])(?:\1)+", r"\1", remapped_answer)

    # Normalizza gli spazi, ma conserva gli a capo utili per gli elenchi Markdown.
    lines = [line.rstrip() for line in remapped_answer.splitlines()]
    remapped_answer = "\n".join(lines).strip()

    return remapped_answer, sources


def normalize_markdown_lists(answer: str) -> str:
    """
    Se il modello produce liste inline tipo:
    'sono: - item 1; - item 2.'
    le converte in elenco Markdown con a capo.
    """
    answer = re.sub(r":\s*-\s+", ":\n- ", answer)
    answer = re.sub(r";\s*-\s+", ";\n- ", answer)
    answer = re.sub(r"\.\s*-\s+", ".\n- ", answer)

    return answer.strip()

    
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
    metadata = flatten_chunk_metadata(chunk)

    return RetrievalResult(
        chunk_id=str(chunk.get("chunk_id") or chunk.get("text_hash") or f"chunk_{rank}"),
        text=str(chunk.get("text") or ""),
        metadata=metadata,
        source="teacher_lookup",
        rank=rank,
        score=score,
    )


def teacher_tokens_from_question(question: str) -> set[str]:
    return simple_tokenize(question)


def metadata_tokens_for_chunk(chunk: dict[str, Any]) -> set[str]:
    metadata = flatten_chunk_metadata(chunk)
    metadata_text = " ".join(
        str(metadata.get(key) or "")
        for key in [
            "title",
            "breadcrumb",
            "breadcrumb_text",
            "source_url",
            "document_url",
            "content_title",
            "section_heading",
            "entity_name",
            "teacher_id",
            "publication_title",
            "publication_authors",
        ]
    )
    return simple_tokenize(metadata_text)



def is_publications_query(question: str) -> bool:
    question_lower = question.lower()

    publication_keywords = [
        "pubblicazioni",
        "pubblicazione",
        "articoli",
        "paper",
        "papers",
        "produzione scientifica",
        "lavori scientifici",
    ]

    return any(keyword in question_lower for keyword in publication_keywords)


def find_teacher_publications(question: str, limit: int = 5) -> list[dict[str, str]]:
    """
    Estrae pubblicazioni da chunk docenti.unisa.it/.../ricerca/pubblicazioni.
    Ordina per anno decrescente.
    """
    teacher_tokens = simple_tokenize(question) - {
        "pubblicazioni",
        "pubblicazione",
        "recenti",
        "recente",
        "articoli",
        "articolo",
        "paper",
        "papers",
        "produzione",
        "scientifica",
        "lavori",
        "scientifici",
    }

    if not teacher_tokens:
        return []

    publications: list[dict[str, str]] = []

    pattern = re.compile(
        r"####\s+\d+\[([^\]]+)\]\([^)]+\)\s*\|\s*(\d{4})",
        flags=re.MULTILINE,
    )

    for chunk in load_rag_chunks():
        url = str(chunk.get("source_url") or chunk.get("document_url") or "")
        title = str(chunk.get("title") or "")
        breadcrumb_text = str(chunk.get("breadcrumb_text") or "")
        text = str(chunk.get("text") or "")

        if "docenti.unisa.it" not in url:
            continue

        if "/ricerca/pubblicazioni" not in url:
            continue

        metadata_text = f"{title} {breadcrumb_text} {url}".lower()
        metadata_tokens = simple_tokenize(metadata_text)

        if not teacher_tokens.issubset(metadata_tokens):
            continue

        for pub_title, year in pattern.findall(text):
            publications.append(
                {
                    "title": pub_title.strip(),
                    "year": year.strip(),
                    "url": url,
                    "source_title": title,
                }
            )

    unique: dict[tuple[str, str], dict[str, str]] = {}

    for pub in publications:
        key = (pub["title"], pub["year"])
        unique[key] = pub

    sorted_publications = sorted(
        unique.values(),
        key=lambda pub: int(pub["year"]),
        reverse=True,
    )

    return sorted_publications[:limit]


def teacher_profile_matches_query(question: str, chunk: dict[str, Any]) -> bool:
    """
    Match stretto per le pagine docente.
    Per una query tipo 'orari di ricevimento antonio greco',
    accetta solo pagine il cui titolo/breadcrumb contengono sia 'antonio' sia 'greco'.
    """
    teacher_tokens = teacher_tokens_from_question(question)

    if not teacher_tokens:
        return False

    metadata_text = " ".join(
        str(chunk.get(key) or "")
        for key in ["title", "breadcrumb", "breadcrumb_text", "source_url", "document_url"]
    ).lower()

    metadata_tokens = simple_tokenize(metadata_text)

    if len(teacher_tokens) >= 2:
        return teacher_tokens.issubset(metadata_tokens)

    return bool(teacher_tokens.intersection(metadata_tokens))


def find_teacher_profile_office_hours(question: str) -> list[RetrievalResult]:
    teacher_tokens = teacher_tokens_from_question(question)
    if not teacher_tokens:
        return []

    matches: list[RetrievalResult] = []

    for chunk in load_rag_chunks():
        metadata = flatten_chunk_metadata(chunk)
        url = str(metadata.get("source_url") or metadata.get("document_url") or "")
        if "docenti.unisa.it" not in url:
            continue

        if not teacher_profile_matches_query(question, chunk):
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
        metadata = flatten_chunk_metadata(chunk)
        title = str(metadata.get("title") or "")
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


def is_teacher_publications_query(question: str) -> bool:
    question_lower = question.lower()
    asks_publications = any(
        keyword in question_lower
        for keyword in [
            "pubblicazione",
            "pubblicazioni",
            "articolo",
            "articoli",
            "paper",
            "lavori scientifici",
        ]
    )

    return asks_publications and bool(teacher_tokens_from_question(question_lower))


def publication_year(result: RetrievalResult) -> int:
    value = result.metadata.get("publication_year") or result.metadata.get("year") or 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def publication_order(result: RetrievalResult) -> int:
    value = result.metadata.get("publication_order") or 999_999
    try:
        return int(value)
    except (TypeError, ValueError):
        return 999_999


def find_teacher_publication_summaries(question: str, limit: int = 7) -> list[RetrievalResult]:
    teacher_tokens = teacher_tokens_from_question(question)
    if not teacher_tokens:
        return []

    matches: list[RetrievalResult] = []
    seen_publication_ids: set[str] = set()

    for chunk in load_rag_chunks():
        metadata = flatten_chunk_metadata(chunk)
        if metadata.get("chunk_kind") != "publication_summary":
            continue

        url = str(metadata.get("source_url") or metadata.get("document_url") or "")
        if "docenti.unisa.it" not in url or "/ricerca/pubblicazioni" not in url:
            continue

        combined_tokens = metadata_tokens_for_chunk(chunk).union(
            simple_tokenize(str(chunk.get("text") or ""))
        )

        if not teacher_metadata_matches(teacher_tokens, combined_tokens):
            continue

        publication_id = str(metadata.get("publication_id") or chunk.get("chunk_id") or "")
        if publication_id and publication_id in seen_publication_ids:
            continue
        if publication_id:
            seen_publication_ids.add(publication_id)

        result = result_from_chunk(
            chunk,
            rank=len(matches) + 1,
            score=float(publication_year_from_metadata(metadata)) + (1 / publication_order_from_metadata(metadata)),
        )
        matches.append(result)

    matches.sort(
        key=lambda result: (
            publication_year(result),
            -publication_order(result),
        ),
        reverse=True,
    )

    for rank, result in enumerate(matches, start=1):
        result.rank = rank

    return matches[:limit]


def publication_year_from_metadata(metadata: dict[str, Any]) -> int:
    value = metadata.get("publication_year") or metadata.get("year") or 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def publication_order_from_metadata(metadata: dict[str, Any]) -> int:
    value = metadata.get("publication_order") or 999_999
    try:
        return int(value)
    except (TypeError, ValueError):
        return 999_999


def build_direct_publications_answer(
    question: str,
    results: list[RetrievalResult],
) -> str | None:
    if not is_teacher_publications_query(question) or not results:
        return None

    first_metadata = results[0].metadata or {}
    first_source = get_source_from_result(results[0])
    teacher_name = (
        metadata_to_string(first_metadata.get("entity_name")).strip()
        or first_source.title.split("|", 1)[0].strip()
        or "il docente indicato"
    )

    lines = [f"Le pubblicazioni più recenti di {teacher_name} che ho trovato sono:"]
    seen_publication_ids: set[str] = set()

    for result in results:
        metadata = result.metadata or {}
        publication_id = metadata_to_string(metadata.get("publication_id")) or result.chunk_id
        if publication_id in seen_publication_ids:
            continue
        seen_publication_ids.add(publication_id)

        title = metadata_to_string(metadata.get("publication_title")).strip()
        if not title:
            title_match = re.search(r"Titolo pubblicazione:\s*(.+)", result.text)
            title = title_match.group(1).strip() if title_match else "Titolo non disponibile"

        year = metadata_to_string(metadata.get("publication_year")).strip()
        publication_type = metadata_to_string(metadata.get("publication_type")).strip()
        venue = metadata_to_string(metadata.get("publication_venue")).strip()
        doi = metadata_to_string(metadata.get("publication_doi")).strip()

        if venue and title and venue.lower().startswith(title.lower()):
            venue = venue[len(title):].strip(" .")

        details = [value for value in [publication_type, venue] if value]
        if doi:
            details.append(f"DOI: {doi}")

        year_prefix = f"{year}: " if year else ""
        details_suffix = f" ({'; '.join(details)})" if details else ""
        lines.append(f"- {year_prefix}{title}{details_suffix} [1]")

    if len(lines) == 1:
        return None

    return "\n".join(lines)


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

    return asks_office_hours and len(teacher_name_tokens) >= 2


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


def extract_inline_citation_indexes(answer: str) -> list[int]:
    """
    Estrae gli indici citati davvero nel testo della risposta, ad esempio [1], [2].
    Serve per evitare di mostrare fonti che il modello dichiara in used_sources
    ma che non cita realmente nella risposta.
    """
    indexes = re.findall(r"\[(\d+)\]", answer)
    return sorted({int(index) for index in indexes})


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



def recent_context_was_publications(conversation_history: list[ConversationTurn] | None) -> bool:
    """
    Capisce se negli ultimi turni si parlava di pubblicazioni.
    Serve per follow-up tipo: "invece di Mario Di Mauro".
    """
    if not conversation_history:
        return False

    recent_turns = conversation_history[-4:]

    text = " ".join(
        str(turn.get("content") or "")
        for turn in recent_turns
    ).lower()

    publication_markers = [
        "pubblicazioni",
        "pubblicazione",
        "paper",
        "papers",
        "articoli",
        "produzione scientifica",
        "lavori scientifici",
    ]

    return any(marker in text for marker in publication_markers)


def extract_name_after_instead(question: str) -> str:
    """
    Estrae il nuovo soggetto da follow-up tipo:
    - invece di Mario Di Mauro
    - e Mario Di Mauro invece?
    - invece Fabio Postiglione
    """
    text = question.strip()
    text = re.sub(r"[?!.]+$", "", text).strip()

    text = re.sub(r"^\s*e\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\binvece\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*di\s+", "", text, flags=re.IGNORECASE)

    text = re.sub(r"\s+", " ", text).strip()

    return text


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
        
    if is_publications_query(retrieval_question):
        publications = find_teacher_publications(retrieval_question, limit=5)

        if publications:
            source_url = publications[0]["url"]
            source_title = publications[0]["source_title"] or "Pubblicazioni docente"

            lines = [
                "Le pubblicazioni più recenti trovate nelle fonti indicizzate sono:"
            ]

            for pub in publications:
                lines.append(f"- {pub['year']} — {pub['title']} [1]")

            answer = "\n".join(lines)

            return RagResponse(
                question=question,
                answer=answer,
                sources=[
                    Source(
                        title=source_title,
                        url=clean_display_url(source_url),
                        breadcrumb="",
                        chunk_id="",
                    )
                ],
                retrieved_chunks=[],
            )

    if is_teacher_publications_query(retrieval_question):
        publication_chunks = find_teacher_publication_summaries(
            retrieval_question,
            limit=max(final_k, 7),
        )
        direct_publications_answer = build_direct_publications_answer(
            question=retrieval_question,
            results=publication_chunks,
        )

        if direct_publications_answer:
            return RagResponse(
                question=question,
                answer=direct_publications_answer,
                sources=build_sources(publication_chunks),
                retrieved_chunks=publication_chunks,
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

    conversation_context = ""
    if retrieval_question != question:
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

    # Usa come fonte di verità le citazioni realmente presenti nel testo.
    # Esempio: se la risposta contiene solo [1], mostriamo solo la fonte 1,
    # anche se il modello in used_sources ha scritto [1, 2, 5].
    inline_citation_indexes = extract_inline_citation_indexes(clean_answer)

    if inline_citation_indexes:
        used_source_indexes = inline_citation_indexes

    all_sources = build_sources(retrieved_chunks)

    clean_answer, used_sources = select_sources_and_remap_citations(
        answer=clean_answer,
        results=retrieved_chunks,
        used_indexes=used_source_indexes,
    )
    clean_answer = normalize_markdown_lists(clean_answer)
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
