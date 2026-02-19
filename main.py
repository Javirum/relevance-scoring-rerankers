"""RAG pipeline with Pinecone retrieval and GPT-4o-mini generation."""

from __future__ import annotations

import sys

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from retrieve import PINECONE_INDEX, parse_filters


def main() -> None:
    load_dotenv()

    args, filter = parse_filters(sys.argv[1:])

    query = " ".join(args) if args else input("Enter question: ").strip()
    if not query:
        print("No query provided.")
        return

    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5, **({"filter": filter} if filter else {})},
    )

    chain = RetrievalQA.from_chain_type(llm=ChatOpenAI(model="gpt-4o-mini"), retriever=retriever)

    result = chain.invoke(query)
    print(result["result"])


if __name__ == "__main__":
    main()
