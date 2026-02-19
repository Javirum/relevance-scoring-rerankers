"""Chunk documents and generate embeddings for RAG pipeline.

Reads extracted .md files from docs/, chunks them with metadata,
generates embeddings via OpenAI, and stores in Pinecone.
"""

import re
from pathlib import Path

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

DOCS_DIR = Path(__file__).parent / "docs"

PDF_FILE = "Living_Repository_AI_Literacy_Practices_Update_16042025_UqmogIt2HpLVokdcuzJL4mDvHk8_112203.md"
AUDIO_FILES = [
    "Red_Lines_and_Risks_in_the_AI_Act.md",
    "The_Blueprint_For_Trustworthy_AI.md",
]

PINECONE_INDEX = "rag-relevance-scoring"
EMBEDDING_DIM = 1536


# ---------------------------------------------------------------------------
# PDF chunking – split by company section
# ---------------------------------------------------------------------------

def chunk_pdf(text: str) -> list[Document]:
    """Split the AI Literacy Practices PDF by company entry."""
    # Find all "On the organisation" anchors and extract company sections
    anchor_pattern = re.compile(r"^On the organisation\s*$", re.MULTILINE)
    anchors = list(anchor_pattern.finditer(text))

    if not anchors:
        raise ValueError("No company sections found in PDF")

    # Track implementation status section boundaries
    section_pattern = re.compile(
        r"^(I{1,3}V?\.)\s*(Fully implemented practices|Partially rolled\s*-?\s*out practices|Planned practices)",
        re.MULTILINE | re.IGNORECASE,
    )
    status_positions: list[tuple[int, str]] = []
    for m in section_pattern.finditer(text):
        status_label = m.group(2).strip().lower()
        if "fully" in status_label:
            status_positions.append((m.start(), "fully implemented"))
        elif "partially" in status_label:
            status_positions.append((m.start(), "partially rolled-out"))
        elif "planned" in status_label:
            status_positions.append((m.start(), "planned"))

    def _status_for_offset(offset: int) -> str:
        current = "unknown"
        for pos, st in status_positions:
            if pos <= offset:
                current = st
            else:
                break
        return current

    documents: list[Document] = []

    for chunk_idx, anchor in enumerate(anchors):
        # Company name is on the line before "On the organisation"
        preceding = text[:anchor.start()].rstrip()
        company_name = preceding.split("\n")[-1].strip()

        # Section runs from company name start to next company's name start (or EOF)
        section_start = preceding.rfind("\n") + 1
        if chunk_idx + 1 < len(anchors):
            next_preceding = text[:anchors[chunk_idx + 1].start()].rstrip()
            section_end = next_preceding.rfind("\n") + 1
        else:
            section_end = len(text)

        section = text[section_start:section_end]

        # Extract metadata
        size_match = re.search(r"Size:\s*(.+?)(?:\n|$)", section)
        sector_match = re.search(r"Sector:\s*(.+?)(?:\n|$)", section)

        size = size_match.group(1).strip() if size_match else "Unknown"
        sector = sector_match.group(1).strip() if sector_match else "Unknown"

        # Handle typo in source doc where "Size: ICT" is used instead of "Sector: ICT"
        if sector == "Unknown" and size_match:
            all_size_lines = re.findall(r"Size:\s*(.+?)(?:\n|$)", section)
            for val in all_size_lines:
                val = val.strip()
                if not re.match(r"(Micro|Small|Medium|Large)", val):
                    sector = val
                    break

        impl_status = _status_for_offset(section_start)

        # Clean up page header noise
        cleaned = re.sub(
            r"\n\d+\s*\n\s*Living Repository of\s*\nAI Literacy Practice\s*s\s*–\s*v\.\s*\d{2}\.\d{2}\.\d{4}\s*\n",
            "\n",
            section,
        )

        documents.append(Document(
            page_content=cleaned,
            metadata={
                "source": PDF_FILE,
                "doc_type": "pdf",
                "chunk_index": chunk_idx,
                "company": company_name,
                "size": size,
                "sector": sector,
                "implementation_status": impl_status,
            },
        ))

    return documents


# ---------------------------------------------------------------------------
# Audio/podcast chunking – fixed-size with overlap
# ---------------------------------------------------------------------------

def chunk_audio(text: str, source: str) -> list[Document]:
    """Split podcast transcripts into fixed-size chunks (~500 tokens, 100 overlap)."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="text-embedding-3-small",
        chunk_size=500,
        chunk_overlap=100,
    )
    chunks = splitter.split_text(text)

    return [
        Document(
            page_content=chunk,
            metadata={
                "source": source,
                "doc_type": "audio",
                "chunk_index": i,
            },
        )
        for i, chunk in enumerate(chunks)
    ]


# ---------------------------------------------------------------------------
# Pinecone setup
# ---------------------------------------------------------------------------

def get_or_create_index(pc: Pinecone, index_name: str) -> None:
    """Create the Pinecone index if it doesn't already exist."""
    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        print(f"Creating Pinecone index '{index_name}' ...")
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print("Index created.")
    else:
        print(f"Pinecone index '{index_name}' already exists.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Read docs
    pdf_text = (DOCS_DIR / PDF_FILE).read_text()
    print(f"Read PDF doc: {len(pdf_text):,} chars")

    # 2. Chunk PDF
    pdf_docs = chunk_pdf(pdf_text)
    print(f"PDF chunks: {len(pdf_docs)} (one per company)")

    # 3. Chunk audio transcripts
    audio_docs: list[Document] = []
    for audio_file in AUDIO_FILES:
        text = (DOCS_DIR / audio_file).read_text()
        print(f"Read audio doc: {audio_file} ({len(text):,} chars)")
        chunks = chunk_audio(text, audio_file)
        audio_docs.extend(chunks)
        print(f"  -> {len(chunks)} chunks")

    all_docs = pdf_docs + audio_docs
    print(f"\nTotal chunks: {len(all_docs)}")

    # Print summary
    print("\n--- Chunk Summary ---")
    for doc in pdf_docs[:3]:
        m = doc.metadata
        print(f"  [{m['doc_type']}] {m['company'][:30]:<30} | {m['size'][:25]:<25} | {m['sector'][:20]:<20} | {m['implementation_status']}")
    if len(pdf_docs) > 3:
        print(f"  ... and {len(pdf_docs) - 3} more PDF chunks")
    for doc in audio_docs[:2]:
        m = doc.metadata
        print(f"  [{m['doc_type']}] {m['source'][:50]:<50} | chunk {m['chunk_index']}")
    if len(audio_docs) > 2:
        print(f"  ... and {len(audio_docs) - 2} more audio chunks")

    # 4. Initialize Pinecone
    pc = Pinecone()
    get_or_create_index(pc, PINECONE_INDEX)

    # 5. Embed + upsert via LangChain
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("\nEmbedding and upserting to Pinecone ...")
    vectorstore = PineconeVectorStore.from_documents(
        documents=all_docs,
        embedding=embeddings,
        index_name=PINECONE_INDEX,
    )

    # 6. Verify
    index = pc.Index(PINECONE_INDEX)
    stats = index.describe_index_stats()
    print(f"\nPinecone index stats: {stats.total_vector_count} vectors stored")
    print("Done!")


if __name__ == "__main__":
    main()
