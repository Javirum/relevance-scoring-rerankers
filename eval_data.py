"""Ground-truth evaluation dataset.

Each entry contains a query, manually annotated relevance labels for known
documents (0-3 scale), and a reference answer for answer-quality scoring.

Relevance scale:
  0 = Not relevant
  1 = Marginally relevant
  2 = Moderately relevant
  3 = Highly relevant

Doc IDs follow the ``compare.py:doc_id()`` format:
  - PDF docs: ``"{source}|{company}"``
  - Audio docs: ``"{source}"``
"""

from __future__ import annotations

EVAL_QUERIES: list[dict] = [
    # --- 3 queries from COMPARISON.md ---
    {
        "query": "What AI literacy practices do large companies implement?",
        "relevant_docs": {
            "Workday-2024-Annual-Report.pdf|Workday": 3,
            "Smals-Research-AI-Literacy.pdf|Smals": 2,
            "Collibra-10-K-2024.pdf|Collibra": 2,
            "AI-and-Partners-AI-Literacy.pdf|AI & Partners": 1,
            "Palantir-10-K-2024.pdf|Palantir": 1,
        },
        "reference_answer": (
            "Large companies implement AI literacy practices such as company-wide "
            "AI training programs, internal AI academies, hands-on workshops for "
            "employees, responsible AI guidelines, and dedicated AI literacy teams "
            "to ensure all staff understand how to work with AI tools effectively."
        ),
    },
    {
        "query": "What are the risks of emotion recognition AI?",
        "relevant_docs": {
            "Red-Lines-Trustworthy-AI.pdf|": 2,
            "Trustworthy-AI-in-Practice.pdf|": 2,
            "Red-Lines-Trustworthy-AI.pdf|vulnerability": 3,
            "Red-Lines-Trustworthy-AI.pdf|criminal": 1,
            "Red-Lines-Trustworthy-AI.pdf|ranking": 1,
        },
        "reference_answer": (
            "Emotion recognition AI risks include exploiting human vulnerabilities, "
            "unreliable detection across cultures and contexts, privacy violations "
            "through biometric data collection, potential for discriminatory "
            "outcomes, and manipulation of individuals based on inferred emotions."
        ),
    },
    {
        "query": "How should companies train employees on AI tools?",
        "relevant_docs": {
            "Booking.com-10-K-2024.pdf|Booking.com": 3,
            "Studio-Deussen-AI-Adoption.pdf|Studio Deussen": 3,
            "Smals-Research-AI-Literacy.pdf|Smals": 1,
            "IBM-10-K-2024.pdf|IBM": 1,
            "EnBW-Annual-Report-2024.pdf|EnBW": 1,
        },
        "reference_answer": (
            "Companies should train employees on AI tools through structured "
            "training programs, hands-on workshops, internal AI champions, "
            "role-specific curricula, and continuous learning opportunities that "
            "cover both technical skills and responsible AI use."
        ),
    },
    # --- 4 new queries ---
    {
        "query": "What regulatory frameworks govern AI in the European Union?",
        "relevant_docs": {
            "Trustworthy-AI-in-Practice.pdf|": 3,
            "Red-Lines-Trustworthy-AI.pdf|": 2,
            "Smals-Research-AI-Literacy.pdf|Smals": 2,
            "AI-and-Partners-AI-Literacy.pdf|AI & Partners": 1,
            "Collibra-10-K-2024.pdf|Collibra": 1,
        },
        "reference_answer": (
            "The EU AI Act is the primary regulatory framework, classifying AI "
            "systems by risk level (unacceptable, high, limited, minimal) and "
            "imposing requirements on transparency, human oversight, and data "
            "governance. It complements existing regulations like GDPR."
        ),
    },
    {
        "query": "How do companies measure ROI of AI investments?",
        "relevant_docs": {
            "Workday-2024-Annual-Report.pdf|Workday": 2,
            "Palantir-10-K-2024.pdf|Palantir": 2,
            "IBM-10-K-2024.pdf|IBM": 2,
            "Booking.com-10-K-2024.pdf|Booking.com": 1,
            "Collibra-10-K-2024.pdf|Collibra": 1,
        },
        "reference_answer": (
            "Companies measure AI ROI through productivity gains, cost reduction "
            "metrics, revenue growth attributed to AI features, customer "
            "satisfaction improvements, and operational efficiency benchmarks."
        ),
    },
    {
        "query": "What are best practices for responsible AI governance?",
        "relevant_docs": {
            "Trustworthy-AI-in-Practice.pdf|": 3,
            "Red-Lines-Trustworthy-AI.pdf|": 2,
            "Smals-Research-AI-Literacy.pdf|Smals": 2,
            "AI-and-Partners-AI-Literacy.pdf|AI & Partners": 2,
            "Workday-2024-Annual-Report.pdf|Workday": 1,
        },
        "reference_answer": (
            "Best practices include establishing AI ethics boards, conducting "
            "impact assessments, ensuring transparency and explainability, "
            "implementing bias audits, maintaining human oversight, and creating "
            "clear accountability structures for AI systems."
        ),
    },
    {
        "query": "How is AI used in financial reporting and compliance?",
        "relevant_docs": {
            "Collibra-10-K-2024.pdf|Collibra": 3,
            "Palantir-10-K-2024.pdf|Palantir": 2,
            "IBM-10-K-2024.pdf|IBM": 2,
            "Workday-2024-Annual-Report.pdf|Workday": 1,
            "EnBW-Annual-Report-2024.pdf|EnBW": 1,
        },
        "reference_answer": (
            "AI is used in financial reporting for automated data validation, "
            "anomaly detection in transactions, regulatory compliance monitoring, "
            "risk assessment, and streamlining audit processes through natural "
            "language processing of financial documents."
        ),
    },
]
