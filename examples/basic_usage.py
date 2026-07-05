"""Basic usage example for llmeval.

This simulates evaluating a response from a local Llama model where:
- We have a reference answer (for ROUGE/BERTScore)
- We have the retrieved RAG context (for hallucination detection)
- We manually measured latency and token counts (since it's a local model, cost = $0)
"""

from llmeval import Evaluator, print_report

evaluator = Evaluator()

# Simulated RAG pipeline output
source_context = (
    "The Eiffel Tower was completed in 1889 and stands 330 meters tall. "
    "It was designed by engineer Gustave Eiffel for the 1889 World's Fair in Paris."
)
reference_answer = "The Eiffel Tower was finished in 1889 and is 330 meters tall."

# Example 1: A grounded, accurate response
good_response = "The Eiffel Tower was completed in 1889 and is 330 meters tall."

print("Example 1: Grounded response")
result_good = evaluator.evaluate(
    prediction=good_response,
    reference=reference_answer,
    source_context=source_context,
    latency_ms=412,
    prompt_tokens=180,
    completion_tokens=20,
    cost_per_1k_prompt=0.0,   # local Llama model -> no API cost
    cost_per_1k_completion=0.0,
)
print_report(result_good)

# Example 2: A hallucinated response (fabricates a fact not in the context)
hallucinated_response = "The Eiffel Tower was completed in 1925 and is 450 meters tall."

print("\nExample 2: Hallucinated response")
result_bad = evaluator.evaluate(
    prediction=hallucinated_response,
    reference=reference_answer,
    source_context=source_context,
    latency_ms=398,
    prompt_tokens=180,
    completion_tokens=18,
)
print_report(result_bad)

# Batch evaluation + export example
print("\nBatch evaluation:")
samples = [
    {
        "prediction": good_response,
        "reference": reference_answer,
        "source_context": source_context,
        "latency_ms": 412,
        "prompt_tokens": 180,
        "completion_tokens": 20,
    },
    {
        "prediction": hallucinated_response,
        "reference": reference_answer,
        "source_context": source_context,
        "latency_ms": 398,
        "prompt_tokens": 180,
        "completion_tokens": 18,
    },
]
batch_results = evaluator.evaluate_batch(samples)

from llmeval import export_json, export_csv

export_json(batch_results, "results.json")
export_csv(batch_results, "results.csv")
print("Exported results.json and results.csv")
