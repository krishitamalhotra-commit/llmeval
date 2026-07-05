"""Hallucination detection via Natural Language Inference (NLI) entailment.

Approach
--------
Rather than relying on an expensive LLM-as-judge call, this module uses a
small pretrained NLI model to check whether the generated response is
*entailed by* the provided source context. If the response contradicts the
context or introduces unsupported claims, the NLI model will score it as
"contradiction" or "neutral" rather than "entailment" — which we treat as
a hallucination signal.

This is the same family of technique used by tools like SelfCheckGPT: cheap,
local, and explainable, at the cost of being less nuanced than a full
LLM-as-judge pass on long, multi-claim responses.
"""

_NLI_PIPELINE = None
_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"


def _get_pipeline():
    global _NLI_PIPELINE
    if _NLI_PIPELINE is None:
        from transformers import pipeline
        _NLI_PIPELINE = pipeline("text-classification", model=_MODEL_NAME, top_k=None)
    return _NLI_PIPELINE


def _parse_scores(raw) -> dict:
    """Normalise NLI pipeline output across transformers versions.

    Different transformers versions return different shapes:
      v4.x  top_k=None : [[{"label": "entailment", "score": 0.9}, ...]]
      v5.x  top_k=None : [{"label": "entailment", "score": 0.9}, ...]
      single input     : {"label": "entailment", "score": 0.9}   (rare)
    """
    # Unwrap outer list if present: [[{...}]] -> [{...}]
    if isinstance(raw, list) and len(raw) > 0 and isinstance(raw[0], list):
        raw = raw[0]

    # Single dict shape -> wrap in list
    if isinstance(raw, dict):
        raw = [raw]

    return {item["label"].lower(): item["score"] for item in raw}


def detect_hallucination(response: str, source_context: str) -> dict:
    """Estimate hallucination risk by checking entailment against source context.

    Args:
        response: The LLM/agent-generated response to check.
        source_context: The retrieved/source text the response should be grounded in
            (e.g. retrieved RAG chunks, tool outputs, or reference documents).

    Returns:
        Dict with:
            entailment_score (float 0-1): probability the response is entailed
                by the source context.
            contradiction_score (float 0-1): probability of contradiction.
            hallucination_risk (str): "low" | "medium" | "high"
        Returns an "error" key if the model fails to load.
    """
    if not response or not source_context:
        return {
            "entailment_score": 0.0,
            "contradiction_score": 0.0,
            "hallucination_risk": "unknown",
            "error": "response and source_context are both required",
        }

    try:
        nli = _get_pipeline()
    except ImportError:
        return {
            "entailment_score": 0.0,
            "contradiction_score": 0.0,
            "hallucination_risk": "unknown",
            "error": "transformers not installed. Run: pip install transformers torch",
        }

    try:
        # NLI models expect (premise, hypothesis). The source context is the
        # premise (assumed true); the response is the hypothesis we're testing.
        raw = nli({"text": source_context, "text_pair": response})
        scores = _parse_scores(raw)

        entailment = scores.get("entailment", 0.0)
        contradiction = scores.get("contradiction", 0.0)

        if entailment >= 0.6:
            risk = "low"
        elif contradiction >= 0.4:
            risk = "high"
        else:
            risk = "medium"

        return {
            "entailment_score": round(entailment, 4),
            "contradiction_score": round(contradiction, 4),
            "hallucination_risk": risk,
        }
    except Exception as e:
        return {
            "entailment_score": 0.0,
            "contradiction_score": 0.0,
            "hallucination_risk": "unknown",
            "error": str(e),
        }