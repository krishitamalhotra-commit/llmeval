"""Pretty-printing and export utilities for evaluation results."""

import json
import csv


def print_report(result: dict) -> None:
    """Print a single evaluation result to the console in a readable format."""
    print("=" * 50)
    if "rouge" in result:
        r = result["rouge"]
        print(f"ROUGE-1: {r['rouge1']}  ROUGE-2: {r['rouge2']}  ROUGE-L: {r['rougeL']}")
    if "bertscore" in result:
        b = result["bertscore"]
        if "error" in b:
            print(f"BERTScore: unavailable ({b['error']})")
        else:
            print(f"BERTScore F1: {b['f1']}  (P: {b['precision']}, R: {b['recall']})")
    if "hallucination" in result:
        h = result["hallucination"]
        if "error" in h:
            print(f"Hallucination check: unavailable ({h['error']})")
        else:
            print(
                f"Hallucination Risk: {h['hallucination_risk'].upper()}  "
                f"(entailment: {h['entailment_score']}, contradiction: {h['contradiction_score']})"
            )
    if "performance" in result:
        p = result["performance"]
        print(
            f"Latency: {p['latency_ms']} ms  |  Tokens: {p['total_tokens']}  |  Cost: ${p['cost_usd']}"
        )
    print("=" * 50)


def export_json(results: list, filepath: str) -> None:
    """Export a list of evaluation results to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)


def export_csv(results: list, filepath: str) -> None:
    """Export a list of evaluation results to a flattened CSV file."""
    if not results:
        return

    flattened = [_flatten(r) for r in results]
    fieldnames = sorted(set().union(*(row.keys() for row in flattened)))

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened)


def _flatten(d: dict, parent_key: str = "") -> dict:
    """Flatten a nested dict using dot-separated keys, e.g. rouge.rouge1."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten(v, new_key))
        else:
            items[new_key] = v
    return items
