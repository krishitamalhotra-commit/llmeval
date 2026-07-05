"""Unit tests for llmeval metrics. Run with: pytest tests/"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llmeval.metrics.rouge import compute_rouge
from llmeval.metrics.cost import compute_cost_and_latency
from llmeval.evaluator import Evaluator


def test_rouge_identical_text():
    result = compute_rouge("the cat sat on the mat", "the cat sat on the mat")
    assert result["rouge1"] == 1.0
    assert result["rougeL"] == 1.0


def test_rouge_empty_inputs():
    result = compute_rouge("", "something")
    assert result["rouge1"] == 0.0


def test_rouge_partial_overlap():
    result = compute_rouge("the cat sat on the mat", "a dog sat on the rug")
    assert 0.0 < result["rouge1"] < 1.0


def test_cost_calculation_basic():
    result = compute_cost_and_latency(
        latency_ms=500,
        prompt_tokens=1000,
        completion_tokens=500,
        cost_per_1k_prompt=0.01,
        cost_per_1k_completion=0.03,
    )
    assert result["total_tokens"] == 1500
    assert result["cost_usd"] == 0.025  # (1000/1000*0.01) + (500/1000*0.03)
    assert result["latency_ms"] == 500


def test_cost_calculation_local_model_zero_cost():
    result = compute_cost_and_latency(
        latency_ms=300, prompt_tokens=200, completion_tokens=50
    )
    assert result["cost_usd"] == 0.0
    assert result["total_tokens"] == 250


def test_evaluator_without_reference_skips_rouge():
    evaluator = Evaluator(enable_bertscore=False, enable_hallucination=False)
    result = evaluator.evaluate(prediction="hello world", latency_ms=100)
    assert "rouge" not in result
    assert "performance" in result
    assert result["performance"]["latency_ms"] == 100


def test_evaluator_with_reference_includes_rouge():
    evaluator = Evaluator(enable_bertscore=False, enable_hallucination=False)
    result = evaluator.evaluate(
        prediction="the cat sat on the mat",
        reference="the cat sat on the mat",
    )
    assert "rouge" in result
    assert result["rouge"]["rouge1"] == 1.0


def test_evaluator_batch():
    evaluator = Evaluator(enable_bertscore=False, enable_hallucination=False)
    samples = [
        {"prediction": "hello", "reference": "hello"},
        {"prediction": "world", "reference": "world"},
    ]
    results = evaluator.evaluate_batch(samples)
    assert len(results) == 2
    assert results[0]["rouge"]["rouge1"] == 1.0


if __name__ == "__main__":
    test_rouge_identical_text()
    test_rouge_empty_inputs()
    test_rouge_partial_overlap()
    test_cost_calculation_basic()
    test_cost_calculation_local_model_zero_cost()
    test_evaluator_without_reference_skips_rouge()
    test_evaluator_with_reference_includes_rouge()
    test_evaluator_batch()
    print("All tests passed!")
