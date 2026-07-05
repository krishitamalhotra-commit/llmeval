"""Main Evaluator class that orchestrates all evaluation metrics."""

from .metrics import (
    compute_rouge,
    compute_bertscore,
    detect_hallucination,
    compute_cost_and_latency,
)


class Evaluator:
    """Evaluate LLM application outputs across quality, grounding, and cost dimensions.

    Example:
        >>> evaluator = Evaluator()
        >>> result = evaluator.evaluate(
        ...     prediction="The capital of France is Paris.",
        ...     reference="Paris is the capital of France.",
        ...     source_context="France's capital city is Paris.",
        ...     latency_ms=842,
        ...     prompt_tokens=120,
        ...     completion_tokens=15,
        ... )
    """

    def __init__(self, enable_bertscore: bool = True, enable_hallucination: bool = True):
        """
        Args:
            enable_bertscore: If False, skips BERTScore computation (faster, no model load).
            enable_hallucination: If False, skips hallucination/NLI check (faster, no model load).
        """
        self.enable_bertscore = enable_bertscore
        self.enable_hallucination = enable_hallucination

    def evaluate(
        self,
        prediction: str,
        reference: str = None,
        source_context: str = None,
        latency_ms: float = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost_per_1k_prompt: float = 0.0,
        cost_per_1k_completion: float = 0.0,
    ) -> dict:
        """Run the full evaluation suite on a single LLM output.

        Args:
            prediction: The LLM/agent-generated response (required).
            reference: Ground-truth or human-written reference, for ROUGE/BERTScore.
                Skipped if not provided.
            source_context: Retrieved context the response should be grounded in,
                for hallucination detection. Skipped if not provided.
            latency_ms: Latency of the LLM call in milliseconds.
            prompt_tokens: Number of prompt tokens used.
            completion_tokens: Number of completion tokens generated.
            cost_per_1k_prompt: USD per 1K prompt tokens (0 for local/open-source models).
            cost_per_1k_completion: USD per 1K completion tokens (0 for local/open-source models).

        Returns:
            Dict combining all computed metrics. Metrics whose required inputs
            were not provided are simply omitted from the result.
        """
        result = {}

        if reference:
            result["rouge"] = compute_rouge(prediction, reference)
            if self.enable_bertscore:
                result["bertscore"] = compute_bertscore(prediction, reference)

        if source_context and self.enable_hallucination:
            result["hallucination"] = detect_hallucination(prediction, source_context)

        result["performance"] = compute_cost_and_latency(
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_per_1k_prompt=cost_per_1k_prompt,
            cost_per_1k_completion=cost_per_1k_completion,
        )

        return result

    def evaluate_batch(self, samples: list) -> list:
        """Evaluate a list of samples.

        Args:
            samples: List of dicts, each with the same keys accepted by `evaluate()`.

        Returns:
            List of evaluation result dicts, in the same order as the input.
        """
        return [self.evaluate(**sample) for sample in samples]
