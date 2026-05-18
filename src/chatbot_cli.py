from __future__ import annotations

import argparse

from rag_chain import DEFAULT_FINAL_K, DEFAULT_GROQ_MODEL, answer_question_as_text

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chatbot RAG per domande sulle fonti ufficiali DIEM."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_GROQ_MODEL,
        help="Nome del modello Groq da usare.",
    )

    parser.add_argument(
        "--final-k",
        type=int,
        default=DEFAULT_FINAL_K,
        help="Numero di chunk recuperati da passare al modello.",
    )

    args = parser.parse_args()

    print("DIEM Chatbot RAG")
    print(f"Modello Groq: {args.model}")
    print(f"Chunk usati per risposta: {args.final_k}")
    print("Scrivi 'exit', 'quit' o 'q' per uscire.")
    print("-" * 80)

    while True:
        question = input("\nTu: ").strip()

        if question.lower() in {"exit", "quit", "q"}:
            print("Chatbot terminato.")
            break

        if not question:
            continue

        try:
            answer = answer_question_as_text(
                question=question,
                final_k=args.final_k,
                model=args.model,
            )
        except Exception as exc:
            print()
            print(f"Errore: {exc}")
            continue

        print()
        print("Bot:")
        print(answer)


if __name__ == "__main__":
    main()