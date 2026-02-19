"""Vector search with cross-encoder reranking.

Retrieves the top-k most similar chunks from Pinecone, then rescores
each (query, chunk) pair using a cross-encoder model and reranks by
the cross-encoder score.
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from retrieve import parse_filters, retrieve

TOP_K = 5
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def search(query: str, k: int = TOP_K, filter: dict | None = None) -> list[tuple]:
    """Retrieve chunks and rerank with a cross-encoder.

    Returns a list of (doc, similarity, rerank_score) sorted by rerank_score desc.
    """
    cross_encoder = HuggingFaceCrossEncoder(model_name=CROSS_ENCODER_MODEL)

    results = retrieve(query, k=k, filter=filter)

    scored = []
    for doc, similarity in results:
        rerank_score = cross_encoder.score([(query, doc.page_content)])[0]
        scored.append((doc, similarity, float(rerank_score)))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


def display(query: str, scored_results: list[tuple]) -> None:
    """Print reranked results."""
    print(f"\nQuery: {query}")
    print(f"Top {len(scored_results)} results (reranked by cross-encoder):\n")

    for rank, (doc, similarity, rerank_score) in enumerate(scored_results, 1):
        meta = doc.metadata
        source = meta.get("source", "unknown")
        doc_type = meta.get("doc_type", "unknown")
        company = meta.get("company", "")

        print(f"[{rank}] rerank_score={rerank_score:.4f} | similarity={similarity:.2f}")

        header = f"    {doc_type} | {source}"
        if company:
            header += f" | {company}"
        print(header)

        preview = doc.page_content[:200].replace("\n", " ")
        print(f"    {preview}...")
        print()


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

    scored_results = search(query, filter=filter)
    display(query, scored_results)


if __name__ == "__main__":
    main()
