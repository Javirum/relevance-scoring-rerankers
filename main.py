"""RAG pipeline with Pinecone retrieval and GPT-4o-mini generation."""

from __future__ import annotations

import sys

from dotenv import load_dotenv

from retrieve import parse_filters
from rag import ask


def main() -> None:
    load_dotenv()

    args, filter = parse_filters(sys.argv[1:])

    hyde = "--hyde" in args
    args = [a for a in args if a != "--hyde"]

    query = " ".join(args) if args else input("Enter question: ").strip()
    if not query:
        print("No query provided.")
        return

    result = ask(query, filter=filter, hyde=hyde)
    print(result["answer"])


if __name__ == "__main__":
    main()
