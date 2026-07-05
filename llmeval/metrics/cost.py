"""Latency and cost tracking utilities.

Token counts and pricing are supplied manually by the caller, since this
library is designed to be provider-agnostic (works the same whether you're
calling OpenAI, Anthropic, or a self-hosted Llama model where there is no
API cost at all).
"""


def compute_cost_and_latency(
    latency_ms: float = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_per_1k_prompt: float = 0.0,
    cost_per_1k_completion: float = 0.0,
) -> dict:
    """Compute total tokens, total cost, and pass through latency.

    Args:
        latency_ms: Time taken for the LLM call, in milliseconds. Optional.
        prompt_tokens: Number of input/prompt tokens used.
        completion_tokens: Number of output/completion tokens generated.
        cost_per_1k_prompt: USD cost per 1,000 prompt tokens (0 for local models).
        cost_per_1k_completion: USD cost per 1,000 completion tokens (0 for local models).

    Returns:
        Dict with latency_ms, prompt_tokens, completion_tokens, total_tokens,
        and cost_usd.
    """
    total_tokens = prompt_tokens + completion_tokens
    cost_usd = (
        (prompt_tokens / 1000.0) * cost_per_1k_prompt
        + (completion_tokens / 1000.0) * cost_per_1k_completion
    )

    return {
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": round(cost_usd, 6),
    }
