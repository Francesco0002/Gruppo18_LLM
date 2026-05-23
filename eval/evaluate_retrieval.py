from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


from pipeline_io import load_jsonl  
from retrieval import hybrid_retrieve  


DEFAULT_GOLDEN_FILE = ROOT_DIR / "eval" / "golden_questions_retrieval_eval_only.jsonl"
DEFAULT_OUTPUT_JSON = ROOT_DIR / "eval" / "results" / "retrieval_report.json"
DEFAULT_OUTPUT_MD = ROOT_DIR / "eval" / "results" / "retrieval_report.md"


def url_for_result(result) -> str:
    return str(
        result.metadata.get("source_url")
        or result.metadata.get("document_url")
        or result.metadata.get("url")
        or ""
    )


def is_relevant(result, expected_source_groups: list[list[str]]) -> bool:
    """
    Un risultato è rilevante se contiene tutti i pattern di almeno un gruppo.

    Esempio:
    [["docenti.unisa.it/005501"], ["mario", "vento", "ricevimento"]]

    È rilevante se trova:
    - docenti.unisa.it/005501
    oppure
    - mario + vento + ricevimento
    """
    if not expected_source_groups:
        return False

    metadata = result.metadata or {}

    url = url_for_result(result)
    title = str(metadata.get("title") or "")
    breadcrumb = str(metadata.get("breadcrumb") or metadata.get("breadcrumb_text") or "")
    text = str(getattr(result, "text", "") or "")

    haystack = f"{url} {title} {breadcrumb} {text}".lower()

    for group in expected_source_groups:
        normalized_group = [str(item).lower() for item in group if str(item).strip()]

        if normalized_group and all(item in haystack for item in normalized_group):
            return True

    return False


def matched_expected_group_indexes(result, expected_source_groups: list[list[str]]) -> set[int]:
    """
    Restituisce gli indici dei gruppi attesi coperti da un risultato.

    Ogni gruppo rappresenta una fonte/pattern atteso.
    Un gruppo è coperto se tutti i suoi pattern compaiono in URL, titolo,
    breadcrumb o testo del risultato.
    """
    metadata = result.metadata or {}

    url = url_for_result(result)
    title = str(metadata.get("title") or "")
    breadcrumb = str(metadata.get("breadcrumb") or metadata.get("breadcrumb_text") or "")
    text = str(getattr(result, "text", "") or "")

    haystack = f"{url} {title} {breadcrumb} {text}".lower()

    matched: set[int] = set()

    for group_index, group in enumerate(expected_source_groups):
        normalized_group = [str(item).lower() for item in group if str(item).strip()]

        if normalized_group and all(item in haystack for item in normalized_group):
            matched.add(group_index)

    return matched
  

def dcg(relevances: list[int]) -> float:
    return sum(
        relevance / math.log2(index + 2)
        for index, relevance in enumerate(relevances)
    )


def evaluate_query(item: dict[str, Any], k_values: tuple[int, ...]) -> dict[str, Any]:
    question = str(item["question"])    
    expected_groups = item.get("expected_source_groups") or []
    max_k = max(k_values)
    
    if not expected_groups:
        return {
            "id": item.get("id"),
            "question": question,
            "expected_source_groups": expected_groups,
            "skipped": True,
            "reason": "no_expected_source",
        }
    
    results = hybrid_retrieve(question, final_k=max_k)
    relevances = [1 if is_relevant(result, expected_groups) else 0 for result in results]
    
    first_relevant_rank = None
    for index, relevance in enumerate(relevances, start=1):
        if relevance:
            first_relevant_rank = index
            break

    per_k = {}
    for k in k_values:
        top_results = results[:k]
        top_relevances = relevances[:k]

        covered_groups: set[int] = set()

        for result in top_results:
            covered_groups.update(
                matched_expected_group_indexes(result, expected_groups)
            )

        # Hit@k: almeno una fonte attesa trovata nei primi k.
        per_k[f"hit@{k}"] = 1.0 if covered_groups else 0.0

        # Recall@k: quante fonti/pattern attesi sono stati coperti nei primi k.
        per_k[f"recall@{k}"] = (
            len(covered_groups) / len(expected_groups)
            if expected_groups
            else 0.0
        )

        # Precision@k: quanti risultati nei primi k sono rilevanti.
        per_k[f"precision@{k}"] = sum(top_relevances) / k if k > 0 else 0.0

        # nDCG@k: premia i risultati rilevanti messi più in alto.
        ideal_relevant_count = min(sum(relevances), k)
        ideal_relevances = [1] * ideal_relevant_count
        ideal_dcg = dcg(ideal_relevances)

        per_k[f"ndcg@{k}"] = dcg(top_relevances) / ideal_dcg if ideal_dcg else 0.0        
    
    return {
        "id": item.get("id"),
        "question": question,
        "expected_source_groups": expected_groups,
        "skipped": False,
        "mrr@10": 1.0 / first_relevant_rank if first_relevant_rank and first_relevant_rank <= 10 else 0.0,
        **per_k,
        "top_results": [
            {
                "rank": result.rank,
                "score": result.score,
                "title": result.metadata.get("title"),
                "url": url_for_result(result),
                "relevant": bool(relevance),
            }
            for result, relevance in zip(results, relevances, strict=False)
        ],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, float]:
    usable = [row for row in rows if not row.get("skipped")]
    if not usable:
        return {}

    metric_names = [
        key
        for key in usable[0]
        if (
            key.startswith("hit@")
            or key.startswith("recall@")
            or key.startswith("precision@")
            or key.startswith("ndcg@")
            or key == "mrr@10"
        )
    ]

    return {
        metric: round(sum(float(row.get(metric, 0.0)) for row in usable) / len(usable), 4)
        for metric in metric_names
    } | {"evaluated": float(len(usable)), "skipped": float(len(rows) - len(usable))}


def run_eval(name: str, reranker_enabled: bool, golden_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Esegue una variante dell'esperimento cambiando solo RERANKER_ENABLED.

    Questo permette un confronto A/B locale: stessa pipeline, stesso indice,
    una run senza cross-encoder e una con cross-encoder.
    """
    previous = os.environ.get("RERANKER_ENABLED")
    os.environ["RERANKER_ENABLED"] = "true" if reranker_enabled else "false"

    try:
        rows = [
            evaluate_query(item, k_values=(5, 10))
            for item in golden_rows
        ]
    finally:
        if previous is None:
            os.environ.pop("RERANKER_ENABLED", None)
        else:
            os.environ["RERANKER_ENABLED"] = previous

    return {
        "name": name,
        "reranker_enabled": reranker_enabled,
        "summary": summarize(rows),
        "rows": rows,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = ["# Retrieval Evaluation", ""]

    for run in report["runs"]:
        lines.append(f"## {run['name']}")
        lines.append("")
        for metric, value in run["summary"].items():
            lines.append(f"- `{metric}`: {value}")
        lines.append("")

    if len(report["runs"]) == 2:
        baseline, reranked = report["runs"]
        base_hit = baseline["summary"].get("hit@5", 0.0)
        new_hit = reranked["summary"].get("hit@5", 0.0)
        delta = new_hit - base_hit
        lines.append(f"Hit@5 delta: `{delta:.4f}`")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Valuta il retrieval DIEM su un golden set JSONL.")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_FILE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="Valuta solo la configurazione corrente invece del confronto baseline/reranker.",
    )
    args = parser.parse_args()

    golden_rows = load_jsonl(args.golden)

    if args.current_only:
        runs = [run_eval("current", os.getenv("RERANKER_ENABLED", "true") != "false", golden_rows)]
    else:
        runs = [
            run_eval("baseline_no_neural_reranker", False, golden_rows),
            run_eval("neural_reranker", True, golden_rows),
        ]

    report = {"golden_file": str(args.golden), "runs": runs}

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, args.output_md)

    for run in runs:
        print(run["name"], run["summary"])


if __name__ == "__main__":
    main()
