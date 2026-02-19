"""Evaluation script: NDCG@5 per retrieval method + faithfulness & answer relevance.

Usage:
    python3 eval.py
"""

from __future__ import annotations

import json

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics import ndcg_score

from compare import baseline_search, llm_rerank_search, doc_id
from eval_data import EVAL_QUERIES
from rag import ask
from rerank_search import search as cross_encoder_search


# ---------------------------------------------------------------------------
# NDCG helpers
# ---------------------------------------------------------------------------

def compute_ndcg(query_data: dict, retrieved_docs: list, k: int = 5) -> float:
    """Compute NDCG@k for a single query given retrieved documents.

    ``retrieved_docs`` is a list of (doc, ...) tuples in ranked order.
    ``query_data["relevant_docs"]`` maps doc_id → relevance (0-3).
    """
    relevance_map = query_data["relevant_docs"]

    # Build the gain vector in retrieval order
    gains = []
    for item in retrieved_docs[:k]:
        doc = item[0]
        did = doc_id(doc)
        gains.append(relevance_map.get(did, 0))

    # Pad to k if fewer results
    while len(gains) < k:
        gains.append(0)

    # ndcg_score expects 2-D arrays
    true_relevance = np.array([gains])
    # Ideal ordering: sorted descending (already captured by ndcg_score)
    # Scores: use position-based descending values so ndcg_score treats
    # the *order* we provide as the predicted ranking.
    predicted_scores = np.array([list(range(k, 0, -1))], dtype=float)

    # Edge case: all zeros → NDCG is 0
    if true_relevance.sum() == 0:
        return 0.0

    return float(ndcg_score(true_relevance, predicted_scores, k=k))


# ---------------------------------------------------------------------------
# LLM-as-judge helpers (follow vector_search.py:score_relevance pattern)
# ---------------------------------------------------------------------------

def score_faithfulness(client: OpenAI, answer: str, source_chunks: list[str]) -> dict:
    """Rate whether every claim in the answer is grounded in the source chunks (1-5)."""
    context = "\n\n---\n\n".join(chunk[:1000] for chunk in source_chunks)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a faithfulness judge. Given an answer and source context, "
                    "rate how well the answer is grounded in the provided context on a "
                    "scale of 1-5:\n"
                    "1 = Mostly fabricated, not supported by context\n"
                    "2 = Significant unsupported claims\n"
                    "3 = Partially grounded, some claims unsupported\n"
                    "4 = Mostly grounded, minor extrapolations\n"
                    "5 = Fully grounded in context\n\n"
                    "Respond with JSON only: {\"score\": <int>, \"reason\": \"<brief justification>\"}"
                ),
            },
            {
                "role": "user",
                "content": f"Answer:\n{answer}\n\nSource context:\n{context}",
            },
        ],
    )
    try:
        content = response.choices[0].message.content or ""
        return json.loads(content)
    except (json.JSONDecodeError, IndexError):
        return {"score": 3, "reason": "Failed to parse LLM response"}


def score_answer_relevance(client: OpenAI, query: str, answer: str) -> dict:
    """Rate whether the answer addresses the query (1-5)."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an answer-relevance judge. Given a query and an answer, "
                    "rate how well the answer addresses the query on a scale of 1-5:\n"
                    "1 = Completely off-topic\n"
                    "2 = Barely addresses the query\n"
                    "3 = Partially addresses the query\n"
                    "4 = Mostly addresses the query\n"
                    "5 = Fully and directly addresses the query\n\n"
                    "Respond with JSON only: {\"score\": <int>, \"reason\": \"<brief justification>\"}"
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nAnswer:\n{answer}",
            },
        ],
    )
    try:
        content = response.choices[0].message.content or ""
        return json.loads(content)
    except (json.JSONDecodeError, IndexError):
        return {"score": 3, "reason": "Failed to parse LLM response"}


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_eval() -> None:
    load_dotenv()
    client = OpenAI()

    k = 5
    ndcg_results: dict[str, list[float]] = {"Baseline": [], "LLM": [], "CE": []}
    faith_scores: list[int] = []
    relevance_scores: list[int] = []

    query_labels: list[str] = []
    ndcg_rows: list[tuple[str, float, float, float]] = []
    quality_rows: list[tuple[str, int, int]] = []

    for entry in EVAL_QUERIES:
        query = entry["query"]
        short_query = query[:45] + "..." if len(query) > 45 else query
        query_labels.append(short_query)

        # --- NDCG@5 for each retrieval method ---
        baseline = baseline_search(query, k=k)
        ndcg_b = compute_ndcg(entry, baseline, k=k)
        ndcg_results["Baseline"].append(ndcg_b)

        llm = llm_rerank_search(query, k=k)
        ndcg_l = compute_ndcg(entry, llm, k=k)
        ndcg_results["LLM"].append(ndcg_l)

        ce = cross_encoder_search(query, k=k)
        ndcg_c = compute_ndcg(entry, ce, k=k)
        ndcg_results["CE"].append(ndcg_c)

        ndcg_rows.append((short_query, ndcg_b, ndcg_l, ndcg_c))

        # --- Faithfulness & answer relevance (single RAG call) ---
        rag_result = ask(query, k=k)
        answer = rag_result["answer"]
        chunks = [doc.page_content for doc in rag_result["source_documents"]]

        faith = score_faithfulness(client, answer, chunks)
        ans_rel = score_answer_relevance(client, query, answer)

        faith_scores.append(faith["score"])
        relevance_scores.append(ans_rel["score"])
        quality_rows.append((short_query, faith["score"], ans_rel["score"]))

    # --- Print NDCG results ---
    print("\nNDCG@5 (per retrieval method):")
    header = f"  {'Query':<48} {'Baseline':>8} {'LLM':>8} {'CE':>8}"
    print(header)
    print(f"  {'-'*48} {'-'*8} {'-'*8} {'-'*8}")
    for label, b, l, c in ndcg_rows:
        print(f"  {label:<48} {b:>8.3f} {l:>8.3f} {c:>8.3f}")

    avg_b = np.mean(ndcg_results["Baseline"])
    avg_l = np.mean(ndcg_results["LLM"])
    avg_c = np.mean(ndcg_results["CE"])
    print(f"  {'AVERAGE':<48} {avg_b:>8.3f} {avg_l:>8.3f} {avg_c:>8.3f}")

    # --- Print answer quality results ---
    print(f"\nANSWER QUALITY (baseline retriever → GPT-4o-mini):")
    header = f"  {'Query':<48} {'Faith.':>8} {'Ans.Rel.':>8}"
    print(header)
    print(f"  {'-'*48} {'-'*8} {'-'*8}")
    for label, f, r in quality_rows:
        print(f"  {label:<48} {f:>5}/5 {r:>5}/5")

    avg_f = np.mean(faith_scores)
    avg_r = np.mean(relevance_scores)
    print(f"  {'AVERAGE':<48} {avg_f:>5.1f}/5 {avg_r:>5.1f}/5")


if __name__ == "__main__":
    run_eval()
