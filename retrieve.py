"""Shared Pinecone retrieval helper.

Embeds a query with OpenAI text-embedding-3-small and returns the top-k
most similar chunks from the Pinecone index.  Supports optional HyDE
(Hypothetical Document Embeddings) to bridge the query–document gap.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI

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


def generate_hypothetical_doc(client: OpenAI, query: str) -> str:
    """Generate a hypothetical document passage that would answer *query*.

    Used by HyDE to shift the query embedding into document space.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Given a question, write a short passage (~150 words) that "
                    "would answer it, as if from a formal report or academic paper."
                ),
            },
            {"role": "user", "content": query},
        ],
    )
    return response.choices[0].message.content or query


def retrieve(
    query: str, k: int = 5, filter: dict | None = None, hyde: bool = False,
) -> list[tuple[Document, float]]:
    """Return top-k (document, similarity_score) pairs from Pinecone.

    When *hyde* is ``True``, a hypothetical answer passage is generated and
    embedded instead of the raw query (Hypothetical Document Embeddings).
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX,
        embedding=embeddings,
    )

    search_text = query
    if hyde:
        client = OpenAI()
        search_text = generate_hypothetical_doc(client, query)

    return vectorstore.similarity_search_with_score(search_text, k=k, filter=filter)
