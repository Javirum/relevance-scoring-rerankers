"""Baseline vector search against the Pinecone RAG index.

Embeds a natural-language query with OpenAI text-embedding-3-small,
retrieves the top-k most similar chunks, and displays results.
"""

import sys

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

PINECONE_INDEX = "rag-relevance-scoring"
TOP_K = 5


def search(query: str, k: int = TOP_K) -> None:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX,
        embedding=embeddings,
    )

    results = vectorstore.similarity_search_with_score(query, k=k)

    print(f"\nQuery: {query}")
    print(f"Top {k} results:\n")

    for rank, (doc, score) in enumerate(results, 1):
        meta = doc.metadata
        source = meta.get("source", "unknown")
        doc_type = meta.get("doc_type", "unknown")
        company = meta.get("company", "")

        header = f"[{rank}] score={score:.4f} | {doc_type} | {source}"
        if company:
            header += f" | {company}"
        print(header)

        preview = doc.page_content[:200].replace("\n", " ")
        print(f"    {preview}...")
        print()


def main() -> None:
    load_dotenv()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter search query: ").strip()
        if not query:
            print("No query provided.")
            return

    search(query)


if __name__ == "__main__":
    main()
