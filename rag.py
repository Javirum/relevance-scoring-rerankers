"""Reusable RAG function backed by Pinecone + GPT-4o-mini."""

from __future__ import annotations

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from retrieve import PINECONE_INDEX


def ask(
    query: str, k: int = 5, filter: dict | None = None,
) -> dict:
    """Run retrieval-augmented generation and return answer + sources.

    Returns ``{"query": str, "answer": str, "source_documents": list[Document]}``.
    """
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k, **({"filter": filter} if filter else {})},
    )

    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        retriever=retriever,
        return_source_documents=True,
    )

    result = chain.invoke(query)
    return {
        "query": query,
        "answer": result["result"],
        "source_documents": result["source_documents"],
    }
