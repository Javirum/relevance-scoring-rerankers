"""Vector search with LLM relevance reranking.

Embeds a natural-language query with OpenAI text-embedding-3-small,
retrieves the top-k most similar chunks from Pinecone, scores each
chunk for relevance using GPT-4o-mini, and reranks by a combined score.
"""

from __future__ import annotations

import json
import sys

from dotenv import load_dotenv
from openai import OpenAI

from retrieve import parse_filters, retrieve

TOP_K = 5
SIMILARITY_WEIGHT = 0.5
RELEVANCE_WEIGHT = 0.5


def score_relevance(client: OpenAI, query: str, chunk_text: str) -> dict:
    """Ask GPT-4o-mini to rate chunk relevance to the query (1-5)."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a relevance judge. Given a user query and a text chunk, "
                    "rate how relevant the chunk is to the query on a scale of 1-5:\n"
                    "1 = Not relevant at all\n"
                    "2 = Slightly relevant\n"
                    "3 = Moderately relevant\n"
                    "4 = Very relevant\n"
                    "5 = Perfectly relevant\n\n"
                    "Respond with JSON only: {\"score\": <int>, \"reason\": \"<brief justification>\"}"
                ),
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nChunk:\n{chunk_text[:1000]}",
            },
        ],
    )
    try:
        content = response.choices[0].message.content or ""
        return json.loads(content)
    except (json.JSONDecodeError, IndexError):
        return {"score": 3, "reason": "Failed to parse LLM response"}


def combined_score(similarity: float, relevance: int) -> float:
    """Combine cosine similarity (0-1) and relevance score (1-5) into a single score."""
    relevance_normalized = relevance / 5.0
    return SIMILARITY_WEIGHT * similarity + RELEVANCE_WEIGHT * relevance_normalized


def search(query: str, k: int = TOP_K, filter: dict | None = None) -> None:
    client = OpenAI()

    results = retrieve(query, k=k, filter=filter)

    scored_results = []
    for doc, similarity in results:
        relevance = score_relevance(client, query, doc.page_content)
        score = combined_score(similarity, relevance["score"])
        scored_results.append((doc, similarity, relevance, score))

    scored_results.sort(key=lambda x: x[3], reverse=True)

    print(f"\nQuery: {query}")
    print(f"Top {k} results (reranked by combined score):\n")

    for rank, (doc, similarity, relevance, score) in enumerate(scored_results, 1):
        meta = doc.metadata
        source = meta.get("source", "unknown")
        doc_type = meta.get("doc_type", "unknown")
        company = meta.get("company", "")

        print(f"[{rank}] combined={score:.2f} | similarity={similarity:.2f} | relevance={relevance['score']}/5")
        print(f"    Reason: {relevance['reason']}")

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

    search(query, filter=filter)


if __name__ == "__main__":
    main()
