import json
import os
import sys
import re
from pathlib import Path
from ragas.run_config import RunConfig

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

load_dotenv(ROOT / ".env")

from rag_chain import answer_question  # noqa: E402

from openai import OpenAI
from ragas.llms import llm_factory

from langchain_huggingface import HuggingFaceEmbeddings

from ragas import EvaluationDataset, evaluate

from ragas.metrics import (
    ContextPrecision,
    ContextRecall,
    ResponseRelevancy,
    Faithfulness,
    FactualCorrectness,
)


GOLDEN_FILE = ROOT / "eval" / "golden_questions_rag_eval.jsonl"
RESULTS_DIR = ROOT / "eval" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OUT_JSON = RESULTS_DIR / "rag_report.json"
OUT_CSV = RESULTS_DIR / "rag_report.csv"
OUT_MD = RESULTS_DIR / "rag_report.md"
DATASET_JSON = RESULTS_DIR / "rag_dataset.json"


def save_dataset_rows(rows: list[dict]) -> None:
    DATASET_JSON.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_dataset_rows() -> list[dict]:
    if not DATASET_JSON.exists():
        return []

    return json.loads(DATASET_JSON.read_text(encoding="utf-8"))


def to_ragas_dataset(rows: list[dict]) -> EvaluationDataset:
    return EvaluationDataset.from_list(
        [
            {
                "user_input": row["user_input"],
                "retrieved_contexts": row["retrieved_contexts"],
                "response": row["response"],
                "reference": row["reference"],
            }
            for row in rows
        ]
    )
    
    
def clean_for_ragas(text: str) -> str:
    text = re.sub(r"\[\d+(?:,\s*\d+)*\]", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text.strip()


def build_metrics():
    mode = os.getenv("RAGAS_METRICS", "stable").strip().lower()

    if mode == "stable":
        return [
            ContextPrecision(),
            ContextRecall(),
            ResponseRelevancy(strictness=1),
        ]

    if mode == "faithfulness":
        return [
            Faithfulness(),
        ]

    if mode == "factual":
        return [
            FactualCorrectness(mode="f1"),
        ]

    if mode == "all":
        return [
            ContextPrecision(),
            ContextRecall(),
            ResponseRelevancy(strictness=1),
            Faithfulness(),
            FactualCorrectness(mode="f1"),
        ]

    raise ValueError(f"RAGAS_METRICS non valido: {mode}")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_chunk_text(chunk) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("text") or chunk.get("page_content") or "")

    return str(
        getattr(chunk, "text", "")
        or getattr(chunk, "page_content", "")
        or ""
    )


def build_dataset_rows(limit: int | None = None, force: bool = False) -> list[dict]:
    golden_rows = read_jsonl(GOLDEN_FILE)

    cached_rows = [] if force else load_dataset_rows()
    cached_by_id = {row["id"]: row for row in cached_rows if "id" in row}

    output_rows = []

    for row in golden_rows:
        if row.get("expected_behavior", "answer") != "answer":
            continue

        row_id = row.get("id")
        question = row["question"]
        reference = row.get("reference") or row.get("expected_answer")

        if not reference:
            print(f"SKIP {row_id}: manca reference")
            continue

        if row_id in cached_by_id:
            print(f"Cache: {row_id} - {question}")
            output_rows.append(cached_by_id[row_id])
        else:
            print(f"Genero: {row_id} - {question}")

            rag_result = answer_question(question)

            retrieved_contexts = [
                get_chunk_text(chunk)
                for chunk in getattr(rag_result, "retrieved_chunks", [])
            ]

            retrieved_contexts = [
                ctx.strip()
                for ctx in retrieved_contexts
                if ctx and ctx.strip()
            ]

            max_contexts = int(os.getenv("RAGAS_MAX_CONTEXTS", "3"))
            max_context_chars = int(os.getenv("RAGAS_MAX_CONTEXT_CHARS", "1200"))

            retrieved_contexts = [
                ctx[:max_context_chars]
                for ctx in retrieved_contexts[:max_contexts]
            ]

            dataset_row = {
                "id": row_id,
                "category": row.get("category"),
                "user_input": question,
                "retrieved_contexts": retrieved_contexts,
                "response": clean_for_ragas(rag_result.answer),
                "reference": reference,
            }

            cached_by_id[row_id] = dataset_row
            output_rows.append(dataset_row)

            save_dataset_rows(list(cached_by_id.values()))

        if limit is not None and len(output_rows) >= limit:
            break

    if not output_rows:
        raise ValueError("Nessun esempio valido trovato per la RAG evaluation.")

    save_dataset_rows(output_rows)

    return output_rows

def write_markdown(summary: dict, rows: pd.DataFrame) -> None:
    lines = ["# RAG Evaluation", ""]

    lines.append("## Summary")
    lines.append("")

    for key, value in summary.items():
        lines.append(f"- `{key}`: {value:.4f}")

    lines.append("")
    lines.append("## Interpretazione")
    lines.append("")
    lines.append(
        "- `context_precision`: misura quanto il contesto recuperato è pulito e rilevante."
    )
    lines.append(
        "- `context_recall`: misura se il contesto recuperato contiene le informazioni necessarie per rispondere."
    )
    lines.append(
        "- `answer_relevancy` / `response_relevancy`: misura se la risposta risponde davvero alla domanda."
    )
    lines.append(
        "- `faithfulness`: misura se la risposta è supportata dal contesto recuperato."
    )
    lines.append(
        "- `factual_correctness`: misura la sovrapposizione fattuale tra risposta generata e risposta di riferimento."
    )

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    limit_env = os.getenv("RAG_EVAL_LIMIT", "").strip()
    limit = int(limit_env) if limit_env else None

    mode = os.getenv("RAG_EVAL_MODE", "evaluate").strip().lower()
    force_build = os.getenv("RAG_EVAL_FORCE", "false").strip().lower() == "true"

    if mode == "build":
        dataset_rows = build_dataset_rows(limit=limit, force=force_build)
        print(f"\nDataset RAG salvato in: {DATASET_JSON}")
        print(f"Esempi salvati: {len(dataset_rows)}")
        return

    if mode == "evaluate":
        dataset_rows = load_dataset_rows()

        if not dataset_rows:
            print("Dataset cache non trovato. Lo genero ora...")
            dataset_rows = build_dataset_rows(limit=limit, force=force_build)

        if limit is not None:
            dataset_rows = dataset_rows[:limit]

        dataset = to_ragas_dataset(dataset_rows)
    else:
        raise ValueError("RAG_EVAL_MODE deve essere 'build' oppure 'evaluate'.")

    judge_model = os.getenv("RAGAS_JUDGE_MODEL", "llama-3.3-70b-versatile")
    print(f"RAGAS judge model: {judge_model}")
    print(f"RAGAS metrics mode: {os.getenv('RAGAS_METRICS', 'stable')}")
    embedding_model_name = os.getenv(
        "RAGAS_EMBEDDING_MODEL",
        "intfloat/multilingual-e5-small",
    )

    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
        timeout=300.0,
        max_retries=5,
    )

    judge_llm = llm_factory(
        judge_model,
        provider="openai",
        client=client,
        adapter="instructor",
        temperature=0.0,
        max_tokens=4096,
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        encode_kwargs={"normalize_embeddings": True},
    )

    result = evaluate(
        dataset=dataset,
        metrics=build_metrics(),
        llm=judge_llm,
        embeddings=embeddings,
        run_config=RunConfig(
            timeout=300,
            max_retries=2,
            max_workers=1,
        ),
    )

    df = result.to_pandas()
    df.insert(0, "id", [row.get("id") for row in dataset_rows[: len(df)]])
    df.insert(1, "category", [row.get("category") for row in dataset_rows[: len(df)]])
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    numeric_cols = df.select_dtypes(include="number").columns
    summary = {col: float(df[col].mean()) for col in numeric_cols}

    OUT_JSON.write_text(
        json.dumps(
            {
                "golden_file": str(GOLDEN_FILE),
                "summary": summary,
                "rows": df.to_dict(orient="records"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_markdown(summary, df)

    print("\nRAG evaluation completata.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nReport salvati in:\n- {OUT_JSON}\n- {OUT_CSV}\n- {OUT_MD}")


if __name__ == "__main__":
    main()