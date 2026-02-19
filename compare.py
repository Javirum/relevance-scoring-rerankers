"""Side-by-side comparison of ranking methods.

Runs three rankings for a given query:
  1. Baseline — pure cosine similarity (no reranking)
  2. LLM rerank — GPT-4o-mini relevance scoring
  3. Cross-encoder rerank — cross-encoder/ms-marco-MiniLM-L-6-v2
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv
from openai import OpenAI

from retrieve import parse_filters, retrieve
from rerank_search import search as cross_encoder_search
from vector_search import score_relevance, combined_score

TOP_K = 5


def baseline_search(query: str, k: int = TOP_K, filter: dict | None = None) -> list[tuple]: # type: ignore
    """Return results sorted by cosine similarity only."""
    results = retrieve(query, k=k, filter=filter)
    results.sort(key=lambda x: x[1], reverse=True)
    return [(doc, sim) for doc, sim in results]


def llm_rerank_search(query: str, k: int = TOP_K, filter: dict | None = None) -> list[tuple]: # type: ignore
    """Return results reranked by LLM relevance scoring."""
    client = OpenAI()
    results = retrieve(query, k=k, filter=filter)

    scored = []
    for doc, similarity in results:
        relevance = score_relevance(client, query, doc.page_content)
        score = combined_score(similarity, relevance["score"])
        scored.append((doc, similarity, relevance, score))

    scored.sort(key=lambda x: x[3], reverse=True)
    return scored


def doc_id(doc) -> str:
    """Short identifier for a document chunk."""
    meta = doc.metadata
    source = meta.get("source", "unknown")
    company = meta.get("company", "")
    return f"{source}|{company}" if company else source


def main() -> None:
    load_dotenv()

    args, filter = parse_filters(sys.argv[1:])

    if args:
        query = " ".join(args)
    else:
        query = input("Enter search query: ").strip()
        if not query:
            print("No query provided.")
            return

    print(f"\nQuery: {query}")
    if filter:
        active = ", ".join(f"{k}={v['$eq']}" for k, v in filter.items())
        print(f"Filters: {active}")
    print()

    # 1. Baseline
    print("=" * 60)
    print("BASELINE (cosine similarity)")
    print("=" * 60)
    baseline = baseline_search(query, filter=filter)
    baseline_order = []
    for rank, (doc, sim) in enumerate(baseline, 1):
        did = doc_id(doc)
        baseline_order.append(did)
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"  [{rank}] similarity={sim:.2f} | {did}")
        print(f"      {preview}...")

    # 2. LLM rerank
    print()
    print("=" * 60)
    print("LLM RERANK (GPT-4o-mini)")
    print("=" * 60)
    llm = llm_rerank_search(query, filter=filter)
    llm_order = []
    for rank, (doc, sim, rel, score) in enumerate(llm, 1):
        did = doc_id(doc)
        llm_order.append(did)
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"  [{rank}] combined={score:.2f} | similarity={sim:.2f} | relevance={rel['score']}/5 | {did}")
        print(f"      {preview}...")

    # 3. Cross-encoder rerank
    print()
    print("=" * 60)
    print("CROSS-ENCODER RERANK (ms-marco-MiniLM-L-6-v2)")
    print("=" * 60)
    ce = cross_encoder_search(query, filter=filter)
    ce_order = []
    for rank, (doc, sim, rerank_score) in enumerate(ce, 1):
        did = doc_id(doc)
        ce_order.append(did)
        preview = doc.page_content[:120].replace("\n", " ")
        print(f"  [{rank}] rerank_score={rerank_score:.4f} | similarity={sim:.2f} | {did}")
        print(f"      {preview}...")

    # Summary
    print()
    print("=" * 60)
    print("RANK COMPARISON")
    print("=" * 60)
    print(f"  {'Doc':<40} {'Baseline':>8} {'LLM':>8} {'CE':>8}")
    print(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*8}")

    all_ids = []
    for did in baseline_order + llm_order + ce_order:
        if did not in all_ids:
            all_ids.append(did)

    for did in all_ids:
        b_rank = baseline_order.index(did) + 1 if did in baseline_order else "-"
        l_rank = llm_order.index(did) + 1 if did in llm_order else "-"
        c_rank = ce_order.index(did) + 1 if did in ce_order else "-"
        label = did[:40]
        print(f"  {label:<40} {str(b_rank):>8} {str(l_rank):>8} {str(c_rank):>8}")


if __name__ == "__main__":
    main()
