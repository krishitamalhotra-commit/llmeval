"""ROUGE score computation between a prediction and reference text."""

from rouge_score import rouge_scorer


_SCORER = None


def _get_scorer():
    global _SCORER
    if _SCORER is None:
        _SCORER = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
    return _SCORER


def compute_rouge(prediction: str, reference: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L F1 scores.

    Args:
        prediction: The generated text from the LLM/agent.
        reference: The human-written or ground-truth reference text.

    Returns:
        Dict with rouge1, rouge2, rougeL F1 scores (0.0-1.0).
    """
    if not prediction or not reference:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

    scorer = _get_scorer()
    scores = scorer.score(reference, prediction)

    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }
