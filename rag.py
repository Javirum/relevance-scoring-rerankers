"""Reusable RAG function backed by Pinecone + GPT-4o-mini."""

from __future__ import annotations

from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from openai import OpenAI

from retrieve import PINECONE_INDEX, generate_hypothetical_doc


def ask(
    query: str, k: int = 5, filter: dict | None = None, hyde: bool = False,
) -> dict:
    """Run retrieval-augmented generation and return answer + sources.

    When *hyde* is ``True``, a hypothetical answer passage is generated and
    used as the retrieval query instead of the raw user query.

    Returns ``{"query": str, "answer": str, "source_documents": list[Document]}``.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX,
        embedding=embeddings,
    )

    search_query = query
    if hyde:
        client = OpenAI()
        search_query = generate_hypothetical_doc(client, query)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k, **({"filter": filter} if filter else {})},
    )

    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        retriever=retriever,
        return_source_documents=True,
    )

    result = chain.invoke(search_query)
    return {
        "query": query,
        "answer": result["result"],
        "source_documents": result["source_documents"],
    }
