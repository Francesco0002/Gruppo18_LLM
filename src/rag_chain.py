from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from pipeline_io import BASE_DIR


load_dotenv(BASE_DIR / ".env", override=True)


from retrieval import RetrievalResult, hybrid_retrieve


DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")

DEFAULT_FINAL_K = int(os.getenv("RAG_FINAL_K", "3"))
DEFAULT_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "6000"))

GROQ_TIMEOUT_SECONDS = int(os.getenv("GROQ_TIMEOUT_SECONDS", "60"))


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


def build_sources(results: list[RetrievalResult]) -> list[Source]:
    """
    Deduplica le fonti per URL, preservando l'ordine dei risultati.
    """
    sources: list[Source] = []
    seen_urls: set[str] = set()

    for result in results:
        source = get_source_from_result(result)

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


def build_prompt(question: str, context: str) -> str:
    """
    Prompt RAG rigido:
    - usa solo il contesto;
    - non inventa;
    - gestisce fuori dominio;
    - non produce fonti inventate.
    """
    return f"""
Sei un assistente informativo del DIEM dell'Università di Salerno.

Devi rispondere alla domanda dell'utente usando esclusivamente il CONTESTO fornito.
Non usare conoscenza esterna.
Non inventare informazioni mancanti.
Non inventare date, orari, nomi di docenti, corsi, regolamenti, aule o link.
Se il contesto non contiene informazioni sufficienti, rispondi chiaramente:
"Non ho trovato questa informazione nelle fonti DIEM indicizzate."

Se la domanda non riguarda il DIEM, i corsi DIEM, i docenti DIEM, i servizi DIEM,
le attività didattiche, di ricerca, internazionali o i documenti ufficiali indicizzati,
rispondi chiaramente: ""La domanda è fuori dal contesto del DIEM.""

Se la domanda chiede gli orari di ricevimento dei docenti in generale senza indicare
un docente specifico, chiedi all'utente di specificare il nome del docente.

Rispondi in italiano, in modo chiaro e sintetico.

Non aggiungere una sezione "Fonti" nella risposta discorsiva.
Alla fine della risposta aggiungi obbligatoriamente una riga tecnica nel formato:
FONTI_USATE: [1, 2]

Inserisci solo i numeri dei documenti realmente usati per formulare la risposta.
Se non hai usato nessun documento perché il contesto è insufficiente o la domanda è fuori dominio, scrivi:
FONTI_USATE: []

CONTESTO:
{context}

DOMANDA UTENTE:
{question}

RISPOSTA:
""".strip()


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

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=0.1,
            top_p=0.9,
            stream=False,
            timeout=GROQ_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        raise RuntimeError(f"Errore durante la chiamata a Groq: {exc}") from exc

    answer = completion.choices[0].message.content

    if not answer:
        raise RuntimeError("Groq ha restituito una risposta vuota.")

    return answer.strip()


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
    "quali",
    "qual",
    "sono",
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

        source = get_source_from_result(results[result_position])

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


def is_office_hours_query(question: str) -> bool:
    question_lower = question.lower()

    return any(
        keyword in question_lower
        for keyword in [
            "ricevimento",
            "orario di ricevimento",
            "orari di ricevimento",
        ]
    )


def is_specific_office_hours_query(question: str) -> bool:
    question_lower = question.lower()

    asks_office_hours = is_office_hours_query(question_lower)

    has_teacher_reference = any(
        keyword in question_lower
        for keyword in [
            "professor",
            "professore",
            "professoressa",
            "prof.",
            "prof",
            "docente",
        ]
    )

    teacher_name_tokens = simple_tokenize(question_lower)

    return asks_office_hours and has_teacher_reference and bool(teacher_name_tokens)


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
) -> RagResponse:
    """
    Pipeline RAG completa:
    domanda -> retrieval ibrido -> eventuale filtro/estrazione -> LLM -> risposta + fonti.
    """
    question = question.strip()

    if not question:
        raise ValueError("La domanda non può essere vuota.")
    
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
    if is_generic_office_hours_query(question):
        return RagResponse(
            question=question,
            answer=(
                "Gli orari di ricevimento sono specifici per ciascun docente. "
                "Indica il nome del docente per cui vuoi conoscere l'orario."
            ),
            sources=[],
            retrieved_chunks=[],
        )

    retrieved_chunks = hybrid_retrieve(
        query=question,
        final_k=final_k,
    )

    retrieved_chunks = filter_chunks_for_generation(
        question=question,
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
        question=question,
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
    if is_specific_office_hours_query(question):
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

    prompt = build_prompt(question=question, context=context)

    raw_answer = call_groq(
        prompt=prompt,
        model=model,
    )

    clean_answer, used_source_indexes = parse_used_source_indexes(raw_answer)

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
    )

    if not response.sources:
        return response.answer

    return f"{response.answer}\n\n{format_sources(response.sources)}"