# llm-inspector

![tests](https://github.com/your-username/llm-inspector/actions/workflows/tests.yml/badge.svg)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![license](https://img.shields.io/badge/license-MIT-green)

A lightweight, provider-agnostic evaluation toolkit for LLM application outputs.

Most LLM evaluation libraries assume you're calling a hosted API and want LLM-as-judge style grading. `llm-inspector` is built for a different (and very common) scenario: you're running open-source models (Llama, Mistral, etc.) locally or self-hosted, and you need **fast, free, reproducible** quality metrics without making another paid API call just to evaluate your first one.

## Features

- **ROUGE-1 / ROUGE-2 / ROUGE-L** — n-gram overlap against a reference answer
- **BERTScore** — semantic similarity using contextual embeddings (catches paraphrases ROUGE misses)
- **Hallucination detection** — NLI-based entailment check against retrieved/source context
- **Latency & cost tracking** — provider-agnostic; works for OpenAI/Anthropic pricing or zero-cost local models
- **Batch evaluation** — evaluate many samples and export to JSON/CSV in one call
- **Zero external API calls** — everything runs locally; no API key required

## Installation

```bash
pip install -e .
```

Or once published to PyPI:

```bash
pip install llm-inspector
```

### Optional dependencies

BERTScore and hallucination detection require `transformers` + `torch`, which are heavier installs. If you only need ROUGE and cost/latency tracking, you can skip them:

```python
evaluator = Evaluator(enable_bertscore=False, enable_hallucination=False)
```

## Quickstart

```python
from llm-inspector import Evaluator, print_report

evaluator = Evaluator()

result = evaluator.evaluate(
    prediction="The Eiffel Tower was completed in 1889 and is 330 meters tall.",
    reference="The Eiffel Tower was finished in 1889 and is 330 meters tall.",
    source_context="The Eiffel Tower was completed in 1889 and stands 330 meters tall...",
    latency_ms=412,
    prompt_tokens=180,
    completion_tokens=20,
    cost_per_1k_prompt=0.0,       # 0 for local/open-source models
    cost_per_1k_completion=0.0,
)

print_report(result)
```

```
==================================================
ROUGE-1: 0.8889  ROUGE-2: 0.75  ROUGE-L: 0.8889
BERTScore F1: 0.973  (P: 0.971, R: 0.975)
Hallucination Risk: LOW  (entailment: 0.91, contradiction: 0.02)
Latency: 412 ms  |  Tokens: 200  |  Cost: $0.0
==================================================
```

## Why NLI-based hallucination detection?

Most hallucination checks either:
1. Use a full LLM-as-judge call (accurate, but costs money and adds latency for every evaluation), or
2. Use crude keyword/fact overlap (fast and free, but flags valid paraphrases as hallucinations)

`llm-inspector` uses a small pretrained **Natural Language Inference (NLI)** model (`cross-encoder/nli-deberta-v3-base`) to check whether the generated response is *entailed by* the source context it was supposedly grounded in. This is the same family of approach used by tools like SelfCheckGPT — local, free, and fast enough to run on every single generation, at the cost of being less nuanced on long, multi-claim responses than a full LLM-as-judge pass.

## Batch evaluation and export

```python
samples = [
    {"prediction": "...", "reference": "...", "source_context": "...", "latency_ms": 400, "prompt_tokens": 100, "completion_tokens": 20},
    {"prediction": "...", "reference": "...", "source_context": "...", "latency_ms": 380, "prompt_tokens": 95,  "completion_tokens": 18},
]

results = evaluator.evaluate_batch(samples)

from llm-inspector import export_json, export_csv
export_json(results, "results.json")
export_csv(results, "results.csv")
```

## API Reference

### `Evaluator(enable_bertscore=True, enable_hallucination=True)`

### `evaluator.evaluate(...) -> dict`

| Argument | Type | Description |
|---|---|---|
| `prediction` | `str` | The LLM/agent output (required) |
| `reference` | `str` | Ground-truth text, enables ROUGE + BERTScore |
| `source_context` | `str` | Retrieved/source text, enables hallucination check |
| `latency_ms` | `float` | Call latency in milliseconds |
| `prompt_tokens` | `int` | Input token count |
| `completion_tokens` | `int` | Output token count |
| `cost_per_1k_prompt` | `float` | USD per 1K prompt tokens (0 for local models) |
| `cost_per_1k_completion` | `float` | USD per 1K completion tokens (0 for local models) |

### `evaluator.evaluate_batch(samples: list) -> list`

Same arguments as `evaluate()`, passed as a list of dicts.

## Project structure

```
llm-inspector/
├── llm-inspector/
│   ├── metrics/
│   │   ├── rouge.py          # ROUGE-1/2/L
│   │   ├── bertscore.py      # BERTScore
│   │   ├── hallucination.py  # NLI entailment-based hallucination check
│   │   └── cost.py           # Latency + token/cost tracking
│   ├── evaluator.py          # Main Evaluator orchestrator
│   └── report.py             # Console + JSON/CSV reporting
├── tests/
├── examples/
└── pyproject.toml
```

## Running tests

```bash
pip install -r requirements-dev.txt  # or just pytest
pytest tests/
```

## Roadmap

- [ ] BLEU and METEOR scores
- [ ] Multi-claim hallucination detection (sentence-level NLI for long responses)
- [ ] Built-in adapters for OpenAI/Anthropic usage objects (auto-fill token counts)
- [ ] Streamlit dashboard for visualizing batch evaluation results

## License

MIT

---

## How to interpret the scores

### ROUGE (0.0 – 1.0)
Measures **word overlap** between the generated answer and a reference answer.
Higher = more overlap with the expected answer.

| Score | Meaning |
|---|---|
| 0.8 – 1.0 | Excellent — near-identical wording to reference |
| 0.6 – 0.8 | Good — most key points covered |
| 0.4 – 0.6 | Moderate — partially correct, some gaps |
| 0.0 – 0.4 | Poor — significant mismatch with reference |

**Important caveat:** ROUGE penalises valid paraphrases. An answer can be factually correct and well-written but score 0.4 simply because it used different words than the reference. Always read BERTScore alongside ROUGE.

**In your pipeline results:**
- RAG question scored 0.69 → Good (the answer closely followed the source text)
- BERT question scored 0.39 → Moderate (the answer was correct but phrased differently from the reference)

---

### BERTScore F1 (0.0 – 1.0)
Measures **semantic similarity** using contextual embeddings. Captures meaning even when wording differs — a paraphrase of the reference will still score high.

| Score | Meaning |
|---|---|
| 0.95 – 1.0 | Excellent — semantically near-identical |
| 0.90 – 0.95 | Good — same meaning, different words |
| 0.85 – 0.90 | Moderate — mostly correct, some semantic gaps |
| Below 0.85 | Poor — meaning has diverged significantly |

**In your pipeline results:** All three answers scored 0.90–0.93 → consistently Good. This confirms the BERT question's low ROUGE (0.39) was a paraphrase issue, not a factual one.

---

### Hallucination Risk (low / medium / high)
Based on NLI entailment — checks whether the generated answer is logically supported by the retrieved context.

| Risk | Entailment Score | Meaning |
|---|---|---|
| **low** | ≥ 0.60 | Answer is well-supported by the context |
| **medium** | 0.40 – 0.60 | Partially supported; some claims may not be grounded |
| **high** | contradiction ≥ 0.40 | Answer contradicts or significantly departs from context |

**Rule of thumb:** In a RAG system, you want hallucination risk = **low** for all responses. A **high** result means the model is fabricating facts not present in the retrieved documents.

---

### Latency (ms)
End-to-end time for the LLM call.

| Range | Meaning |
|---|---|
| < 500ms | Fast (good for real-time applications) |
| 500ms – 1500ms | Acceptable for most use cases |
| > 1500ms | Slow — consider caching or a smaller model |

**In your pipeline results:** Individual agent calls were 537–686ms (acceptable). Total pipeline latency was 1.5–2.1 seconds across 3 agents — expected for a sequential multi-agent setup.

---

### Cost (USD)
With `cost_per_1k_prompt=0.0` (local/Groq free tier), this will always be $0.0. To track costs for paid APIs set the rates:

```python
# GPT-4o pricing example
evaluator.evaluate(
    ...,
    cost_per_1k_prompt=0.005,       # $5 per 1M input tokens
    cost_per_1k_completion=0.015,   # $15 per 1M output tokens
)
```