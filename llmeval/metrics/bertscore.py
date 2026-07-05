"""BERTScore computation for semantic similarity between prediction and reference."""

import warnings

_BERT_SCORER_AVAILABLE = None


def compute_bertscore(prediction: str, reference: str, lang: str = "en") -> dict:
    """Compute BERTScore precision, recall, and F1 between prediction and reference.

    BERTScore captures semantic similarity better than ROUGE for paraphrased or
    reworded text, since it compares contextual embeddings rather than exact
    n-gram overlap.

    Args:
        prediction: The generated text from the LLM/agent.
        reference: The human-written or ground-truth reference text.
        lang: Language code for the underlying model (default "en").

    Returns:
        Dict with precision, recall, f1 scores (0.0-1.0). Returns zeros with
        an "error" key if bert-score / torch is not available or fails to load.
    """
    if not prediction or not reference:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    try:
        from bert_score import score as bert_score_fn
    except ImportError:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "error": "bert-score not installed. Run: pip install bert-score",
        }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            P, R, F1 = bert_score_fn(
                [prediction], [reference], lang=lang, verbose=False
            )
        return {
            "precision": round(P.item(), 4),
            "recall": round(R.item(), 4),
            "f1": round(F1.item(), 4),
        }
    except Exception as e:  # pragma: no cover - defensive against model/runtime issues
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "error": str(e)}
