from __future__ import annotations
import chainlit as cl

# Importiamo le costanti e la funzione reale dal tuo file rag_chain.py
from rag_chain import DEFAULT_FINAL_K, DEFAULT_OLLAMA_MODEL, answer_question_as_text

# Il nome dell'autore DEVE corrispondere (senza spazi strani o simboli se non necessari, "DIEM Bot" va benissimo)
BOT_NAME = "DIEM Bot"

@cl.on_chat_start
async def start():
    """
    Questo evento si attiva quando l'utente apre l'interfaccia web.
    """
    cl.user_session.set("model", DEFAULT_OLLAMA_MODEL)
    cl.user_session.set("final_k", DEFAULT_FINAL_K)
    
    # Avendo messo i loghi dentro src/public/ nominati logo_dark.png e logo_light.png,
    # Chainlit li applica già come logo globale dell'applicazione.
    
    # Mandiamo il messaggio impostando l'autore personalizzato
    await cl.Message(
        content="🤖 **DIEM Chatbot pronto!**\nChiedimi pure qualsiasi cosa sulle fonti ufficiali DIEM.",
        author=BOT_NAME
    ).send()


@cl.on_message
async def main(message: cl.Message):
    """
    Questo evento si attiva ogni volta che l'utente invia un messaggio in chat.
    """
    question = message.content.strip()
    model = cl.user_session.get("model")
    final_k = cl.user_session.get("final_k")

    if not question:
        return

    # Feedback visivo del RAG
    async with cl.Step(name="RAG Retrieval", type="tool") as step:
        step.input = f"Ricerca dei {final_k} chunk più rilevanti nel Vector Store..."
        step.output = "Fonti recuperate con successo."
    
    # Generazione dell'LLM
    async with cl.Step(name="LLM Generation", type="llm") as step:
        step.input = f"Elaborazione della risposta con il modello {model}..."
        
        try:
            answer = await cl.make_async(answer_question_as_text)(
                question=question,
                final_k=final_k,
                model=model,
            )
            step.output = "Risposta generata!"
        except Exception as exc:
            step.output = f"Errore: {exc}"
            answer = f"⚠️ Si è verificato un errore durante l'elaborazione: {exc}"

    # Risposta finale del bot
    await cl.Message(content=answer, author=BOT_NAME).send()