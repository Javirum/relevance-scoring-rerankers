"""Shared Pinecone retrieval helper.

Embeds a query with OpenAI text-embedding-3-small and returns the top-k
most similar chunks from the Pinecone index.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

PINECONE_INDEX = "rag-relevance-scoring"

ALLOWED_FILTER_KEYS = {
    "doc_type", "source", "company", "size", "sector", "implementation_status",
}


def parse_filters(args: list[str]) -> tuple[list[str], dict | None]:
    """Extract ``--filter key=value`` pairs from CLI args.

    Returns (remaining_args, filter_dict).  *remaining_args* has the
    ``--filter`` flags removed so they don't pollute the query string.
    *filter_dict* is ``None`` when no filters were supplied.
    """
    remaining: list[str] = []
    filters: dict = {}
    i = 0
    while i < len(args):
        if args[i] == "--filter" and i + 1 < len(args):
            key, _, value = args[i + 1].partition("=")
            if key not in ALLOWED_FILTER_KEYS:
                raise SystemExit(
                    f"Unknown filter key '{key}'. "
                    f"Allowed keys: {', '.join(sorted(ALLOWED_FILTER_KEYS))}"
                )
            if not value:
                raise SystemExit(f"Missing value for filter key '{key}'")
            filters[key] = {"$eq": value}
            i += 2
        else:
            remaining.append(args[i])
            i += 1
    return remaining, filters or None


def retrieve(
    query: str, k: int = 5, filter: dict | None = None,
) -> list[tuple[Document, float]]:
    """Return top-k (document, similarity_score) pairs from Pinecone."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX,
        embedding=embeddings,
    )
    return vectorstore.similarity_search_with_score(query, k=k, filter=filter)
