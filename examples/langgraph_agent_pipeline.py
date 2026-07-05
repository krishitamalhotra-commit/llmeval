"""
LangGraph Multi-Agent RAG Pipeline — evaluated with llmeval
=============================================================

Architecture:
    User Question
        │
        ▼
  [Agent 1: Retriever]   — picks the most relevant context chunk from a
                           small in-memory knowledge base
        │
        ▼
  [Agent 2: Answerer]    — generates a grounded answer using only the
                           retrieved context
        │
        ▼
  [Agent 3: Critic]      — reviews the answer and flags unsupported claims
        │
        ▼
  [llmeval Evaluator]    — ROUGE, BERTScore, NLI hallucination check,
                           latency & token tracking for every agent call

Usage:
    export GROQ_API_KEY=your_key_here
    python examples/langgraph_agent_pipeline.py
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from llmeval import Evaluator, print_report, export_json, export_csv

# ---------------------------------------------------------------------------
# 0.  Groq client setup
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    print("ERROR: Set your GROQ_API_KEY environment variable first.")
    print("  export GROQ_API_KEY=gsk_...")
    sys.exit(1)

MODEL = "openai/gpt-oss-20b"   # fast, free on Groq

def make_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=MODEL,
        temperature=0.0,   # deterministic for eval reproducibility
    )

# ---------------------------------------------------------------------------
# 1.  In-memory knowledge base  (simulates RAG document store)
# ---------------------------------------------------------------------------

KNOWLEDGE_BASE = [
    {
        "id": "doc1",
        "text": (
            "The transformer architecture was introduced in the 2017 paper "
            "'Attention Is All You Need' by Vaswani et al. It relies entirely "
            "on self-attention mechanisms and eliminates recurrence and convolutions. "
            "Transformers have become the foundation of modern LLMs such as GPT and BERT."
        ),
    },
    {
        "id": "doc2",
        "text": (
            "BERT (Bidirectional Encoder Representations from Transformers) was "
            "released by Google in 2018. It is pre-trained using masked language "
            "modelling and next-sentence prediction tasks. BERT learns bidirectional "
            "context, unlike GPT which is unidirectional (left-to-right)."
        ),
    },
    {
        "id": "doc3",
        "text": (
            "GPT-4 was released by OpenAI in March 2023. It is a multimodal model "
            "capable of accepting both text and image inputs. GPT-4 significantly "
            "outperforms GPT-3.5 on professional and academic benchmarks."
        ),
    },
    {
        "id": "doc4",
        "text": (
            "Retrieval-Augmented Generation (RAG) is a technique that combines a "
            "retrieval component with a generative model. The retriever fetches "
            "relevant documents from a knowledge base, which are then passed as "
            "context to the generator, reducing hallucination and improving factual accuracy."
        ),
    },
]

# Reference answers for llmeval (what a human expert would say)
REFERENCE_ANSWERS = {
    "What is the transformer architecture?": (
        "The transformer architecture was introduced in 2017 in 'Attention Is All You Need'. "
        "It uses self-attention mechanisms and is the basis for models like GPT and BERT."
    ),
    "How is BERT different from GPT?": (
        "BERT uses bidirectional context via masked language modelling, while GPT "
        "processes text left-to-right (unidirectional)."
    ),
    "What is RAG and why is it useful?": (
        "RAG combines a retriever with a generative model. The retriever fetches relevant "
        "documents which are given as context to the generator, reducing hallucinations."
    ),
}

# ---------------------------------------------------------------------------
# 2.  LangGraph State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    question: str
    retrieved_context: Optional[str]
    retrieved_doc_id: Optional[str]
    answer: Optional[str]
    critique: Optional[str]
    # per-agent timing (ms) and token counts for llmeval
    retriever_latency_ms: Optional[float]
    retriever_prompt_tokens: Optional[int]
    retriever_completion_tokens: Optional[int]
    answerer_latency_ms: Optional[float]
    answerer_prompt_tokens: Optional[int]
    answerer_completion_tokens: Optional[int]
    critic_latency_ms: Optional[float]
    critic_prompt_tokens: Optional[int]
    critic_completion_tokens: Optional[int]

# ---------------------------------------------------------------------------
# 3.  Agent nodes
# ---------------------------------------------------------------------------

def retriever_agent(state: AgentState) -> AgentState:
    """Agent 1 — Given the user question, pick the most relevant KB document."""
    print("\n[Agent 1: Retriever] selecting relevant context...")

    llm = make_llm()
    doc_list = "\n".join(
        f"[{d['id']}] {d['text'][:120]}..." for d in KNOWLEDGE_BASE
    )
    messages = [
        SystemMessage(content=(
            "You are a document retriever. Given a question and a list of documents, "
            "reply with ONLY the document ID (e.g. 'doc1') that is most relevant. "
            "No explanation, no punctuation — just the ID."
        )),
        HumanMessage(content=f"Question: {state['question']}\n\nDocuments:\n{doc_list}"),
    ]

    t0 = time.time()
    response = llm.invoke(messages)
    latency_ms = (time.time() - t0) * 1000

    doc_id = response.content.strip().lower()
    matched = next((d for d in KNOWLEDGE_BASE if d["id"] == doc_id), KNOWLEDGE_BASE[0])

    usage = response.usage_metadata or {}
    return {
        **state,
        "retrieved_doc_id": matched["id"],
        "retrieved_context": matched["text"],
        "retriever_latency_ms": round(latency_ms, 1),
        "retriever_prompt_tokens": usage.get("input_tokens", 0),
        "retriever_completion_tokens": usage.get("output_tokens", 0),
    }


def answerer_agent(state: AgentState) -> AgentState:
    """Agent 2 — Generate an answer grounded strictly in the retrieved context."""
    print("[Agent 2: Answerer] generating answer...")

    llm = make_llm()
    messages = [
        SystemMessage(content=(
            "You are a factual question-answering assistant. "
            "Answer the question using ONLY the provided context. "
            "Be concise (2-3 sentences). Do not add information not present in the context."
        )),
        HumanMessage(content=(
            f"Context:\n{state['retrieved_context']}\n\n"
            f"Question: {state['question']}"
        )),
    ]

    t0 = time.time()
    response = llm.invoke(messages)
    latency_ms = (time.time() - t0) * 1000

    usage = response.usage_metadata or {}
    return {
        **state,
        "answer": response.content.strip(),
        "answerer_latency_ms": round(latency_ms, 1),
        "answerer_prompt_tokens": usage.get("input_tokens", 0),
        "answerer_completion_tokens": usage.get("output_tokens", 0),
    }


def critic_agent(state: AgentState) -> AgentState:
    """Agent 3 — Review the answer and flag any unsupported or hallucinated claims."""
    print("[Agent 3: Critic] reviewing answer for hallucinations...")

    llm = make_llm()
    messages = [
        SystemMessage(content=(
            "You are a strict fact-checker. Given a context and an answer, "
            "identify any claims in the answer that are NOT supported by the context. "
            "If the answer is fully supported, say 'VERDICT: GROUNDED'. "
            "If there are unsupported claims, say 'VERDICT: HALLUCINATION' and list the issues briefly."
        )),
        HumanMessage(content=(
            f"Context:\n{state['retrieved_context']}\n\n"
            f"Answer:\n{state['answer']}"
        )),
    ]

    t0 = time.time()
    response = llm.invoke(messages)
    latency_ms = (time.time() - t0) * 1000

    usage = response.usage_metadata or {}
    return {
        **state,
        "critique": response.content.strip(),
        "critic_latency_ms": round(latency_ms, 1),
        "critic_prompt_tokens": usage.get("input_tokens", 0),
        "critic_completion_tokens": usage.get("output_tokens", 0),
    }

# ---------------------------------------------------------------------------
# 4.  Build the LangGraph
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("retriever", retriever_agent)
    graph.add_node("answerer", answerer_agent)
    graph.add_node("critic",   critic_agent)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever", "answerer")
    graph.add_edge("answerer",  "critic")
    graph.add_edge("critic",    END)

    return graph.compile()

# ---------------------------------------------------------------------------
# 5.  Run the pipeline + evaluate with llmeval
# ---------------------------------------------------------------------------

def run_and_evaluate(question: str, reference: str) -> dict:
    """Run the full agent pipeline for one question and evaluate each agent."""
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print('='*60)

    # --- run the graph ---
    graph = build_graph()
    final_state = graph.invoke({"question": question})

    print(f"\nRetrieved: [{final_state['retrieved_doc_id']}]")
    print(f"Answer:    {final_state['answer']}")
    print(f"Critique:  {final_state['critique']}")

    # --- evaluate with llmeval ---
    evaluator = Evaluator(enable_bertscore=True, enable_hallucination=True)

    print("\n--- llmeval: Answerer Agent ---")
    answerer_eval = evaluator.evaluate(
        prediction=final_state["answer"],
        reference=reference,
        source_context=final_state["retrieved_context"],
        latency_ms=final_state["answerer_latency_ms"],
        prompt_tokens=final_state["answerer_prompt_tokens"],
        completion_tokens=final_state["answerer_completion_tokens"],
    )
    print_report(answerer_eval)

    # evaluate the critic's output too (against what a good critique looks like)
    print("--- llmeval: Critic Agent (latency / cost only) ---")
    critic_eval = evaluator.evaluate(
        prediction=final_state["critique"],
        latency_ms=final_state["critic_latency_ms"],
        prompt_tokens=final_state["critic_prompt_tokens"],
        completion_tokens=final_state["critic_completion_tokens"],
    )
    print_report(critic_eval)

    return {
        "question": question,
        "retrieved_doc_id": final_state["retrieved_doc_id"],
        "answer": final_state["answer"],
        "critique": final_state["critique"],
        "answerer_eval": answerer_eval,
        "critic_eval": critic_eval,
        # total pipeline latency
        "total_latency_ms": round(
            (final_state["retriever_latency_ms"] or 0)
            + (final_state["answerer_latency_ms"] or 0)
            + (final_state["critic_latency_ms"] or 0),
            1,
        ),
        "total_tokens": (
            (final_state["retriever_prompt_tokens"] or 0)
            + (final_state["retriever_completion_tokens"] or 0)
            + (final_state["answerer_prompt_tokens"] or 0)
            + (final_state["answerer_completion_tokens"] or 0)
            + (final_state["critic_prompt_tokens"] or 0)
            + (final_state["critic_completion_tokens"] or 0)
        ),
    }


def print_summary(all_results: list) -> None:
    """Print a pipeline-level summary across all questions."""
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print('='*60)
    for r in all_results:
        rouge1 = r["answerer_eval"].get("rouge", {}).get("rouge1", "N/A")
        bs_f1  = r["answerer_eval"].get("bertscore", {}).get("f1", "N/A")
        h_risk = r["answerer_eval"].get("hallucination", {}).get("hallucination_risk", "N/A")
        verdict = "GROUNDED" if "GROUNDED" in r["critique"] else "HALLUCINATION"
        print(
            f"  Q: {r['question'][:45]:<45} | "
            f"ROUGE-1: {rouge1}  BERTScore: {bs_f1}  "
            f"Hall: {h_risk:<6}  Critic: {verdict}  "
            f"Latency: {r['total_latency_ms']}ms  Tokens: {r['total_tokens']}"
        )


if __name__ == "__main__":
    questions = list(REFERENCE_ANSWERS.items())

    all_results = []
    for question, reference in questions:
        result = run_and_evaluate(question, reference)
        all_results.append(result)

    print_summary(all_results)

    # export full results
    export_json(all_results, "pipeline_results.json")
    export_csv(
        [r["answerer_eval"] for r in all_results],
        "answerer_eval_results.csv",
    )
    print("\nExported: pipeline_results.json, answerer_eval_results.csv")
