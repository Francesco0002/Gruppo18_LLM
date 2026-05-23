from __future__ import annotations
import chainlit as cl

# Importiamo le costanti e la funzione reale dal tuo file rag_chain.py
from rag_chain import DEFAULT_FINAL_K, DEFAULT_GROQ_MODEL, answer_question_as_text

# Il nome dell'autore DEVE corrispondere (senza spazi strani o simboli se non necessari, "DIEM Bot" va benissimo)
BOT_NAME = "DIEM Bot"
MAX_CONVERSATION_MESSAGES = 12

WELCOME_MESSAGE = """
👋 **Benvenuto nel chatbot DIEM**

Posso aiutarti a cercare informazioni nelle fonti ufficiali del DIEM, ad esempio:

- corsi di laurea e lauree magistrali;
- accordi Erasmus;
- dottorati collegati al DIEM;
- laboratori e strutture;
- orari di ricevimento dei docenti;
- bandi e informazioni istituzionali.

Scrivi una domanda per iniziare.
"""


def get_conversation_history() -> list[dict[str, str]]:
    history = cl.user_session.get("conversation_history") or []

    if not isinstance(history, list):
        return []

    clean_history: list[dict[str, str]] = []

    for item in history[-MAX_CONVERSATION_MESSAGES:]:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()

        if role in {"user", "assistant"} and content:
            clean_history.append({"role": role, "content": content})

    return clean_history


def save_conversation_turn(question: str, answer: str) -> None:
    history = get_conversation_history()
    history.extend(
        [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    )
    cl.user_session.set(
        "conversation_history",
        history[-MAX_CONVERSATION_MESSAGES:],
    )


@cl.on_chat_start
async def start():
    """
    Questo evento si attiva quando l'utente apre l'interfaccia web.
    """
    cl.user_session.set("model", DEFAULT_GROQ_MODEL)
    cl.user_session.set("final_k", DEFAULT_FINAL_K)
    cl.user_session.set("conversation_history", [])
    
    # Avendo messo i loghi dentro src/public/ nominati logo_dark.png e logo_light.png,
    # Chainlit li applica già come logo globale dell'applicazione.
    
    actions = [
        cl.Action(
            name="example_question",
            payload={"question": "Quali sono i corsi di laurea del DIEM?"},
            label="Corsi di laurea",
        ),
        cl.Action(
            name="example_question",
            payload={"question": "Quali accordi Erasmus per studio sono disponibili al DIEM?"},
            label="Erasmus per studio",
        ),
        cl.Action(
            name="example_question",
            payload={"question": "Quali dottorati sono collegati al DIEM?"},
            label="Dottorati",
        ),
    ]
    
    # Mandiamo il messaggio impostando l'autore personalizzato
    await cl.Message(
        content=WELCOME_MESSAGE,
        author=BOT_NAME,
        actions=actions,
    ).send()

@cl.action_callback("example_question")
async def on_example_question(action: cl.Action):
    question = action.payload.get("question", "")

    if not question:
        return

    history = get_conversation_history()

    answer = await cl.make_async(answer_question_as_text)(
        question=question,
        final_k=DEFAULT_FINAL_K,
        model=DEFAULT_GROQ_MODEL,
        conversation_history=history,
    )

    save_conversation_turn(question, answer)

    await cl.Message(
        content=answer,
        author=BOT_NAME,
    ).send()

@cl.on_message
async def main(message: cl.Message):
    """
    Questo evento si attiva ogni volta che l'utente invia un messaggio in chat.
    """
    question = message.content.strip()
    
    if not question:
        return
    
    model = cl.user_session.get("model") or DEFAULT_GROQ_MODEL
    final_k = cl.user_session.get("final_k") or DEFAULT_FINAL_K
    history = get_conversation_history()

    # Feedback visivo del RAG
    async with cl.Step(name="Ricerca nelle fonti DIEM", type="tool") as step:
        step.input = question
        step.output = "Recupero dei documenti rilevanti e generazione della risposta..."
        
        try:
            answer = await cl.make_async(answer_question_as_text)(
                question=question,
                final_k=final_k,
                model=model,
                conversation_history=history,
            )
            step.output = "Risposta generata!"
            save_conversation_turn(question, answer)
        except Exception as exc:
            step.output = f"Errore: {exc}"
            answer = (
                "⚠️ Si è verificato un errore durante l'elaborazione della domanda.\n\n"
                f"Dettaglio tecnico: `{exc}`"
            )

    # Risposta finale del bot
    await cl.Message(
        content=answer,
        author=BOT_NAME,
    ).send()