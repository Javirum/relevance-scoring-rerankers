# Retrieval Quality Comparison: Baseline vs Reranking

## Methodology

Four ranking methods were evaluated on the same Pinecone index:

1. **Baseline** — pure cosine similarity (no reranking)
2. **HyDE** — Hypothetical Document Embeddings (GPT-4o-mini generates a hypothetical answer passage, which is embedded instead of the raw query)
3. **LLM Rerank** — GPT-4o-mini relevance scoring (1-5) combined 50/50 with similarity
4. **Cross-Encoder Rerank** — `cross-encoder/ms-marco-MiniLM-L-6-v2`

Baseline, LLM Rerank, and Cross-Encoder retrieve the same top-5 chunks and reorder them. HyDE retrieves a *different* set of chunks because it searches with a different embedding (the hypothetical passage), so it can surface documents the other methods miss entirely.

---

## Query 1: "What AI literacy practices do large companies implement?"

### Results

| Rank | Baseline | HyDE | LLM Rerank | Cross-Encoder |
|------|----------|------|------------|---------------|
| 1 | Smals (sim=0.64) | Smals (sim=0.68) | Workday (combined=0.82, rel=5/5) | Smals (rerank=3.79) |
| 2 | Workday (sim=0.64) | AI & Partners (sim=0.67) | Smals (combined=0.62, rel=3/5) | Workday (rerank=3.51) |
| 3 | AI & Partners (sim=0.64) | **Credo AI (sim=0.66)** | Collibra (combined=0.61, rel=3/5) | Palantir (rerank=2.27) |
| 4 | Collibra (sim=0.63) | Workday (sim=0.65) | AI & Partners (combined=0.52, rel=2/5) | Collibra (rerank=0.85) |
| 5 | Palantir (sim=0.62) | **Dedalus Healthcare (sim=0.65)** | Palantir (combined=0.51, rel=2/5) | AI & Partners (rerank=0.58) |

### Analysis

**Winner: LLM Rerank.** The baseline had nearly identical similarity scores (0.64 for the top 3), so it couldn't meaningfully differentiate. The LLM understood the query intent and promoted Workday (which has richer AI literacy content) to rank 1 with a perfect 5/5 relevance score. The cross-encoder agreed on the same set but ordered differently.

**HyDE effect:** HyDE boosted all similarity scores (~0.64 → ~0.68) and surfaced two new documents — Credo AI and Dedalus Healthcare — that the other methods missed entirely, while dropping Collibra and Palantir. This shows HyDE's ability to expand the candidate set by embedding in "document space" rather than "query space".

---

## Query 2: "What are the risks of emotion recognition AI?"

### Results

| Rank | Baseline | HyDE | LLM Rerank | Cross-Encoder |
|------|----------|------|------------|---------------|
| 1 | Red Lines intro (sim=0.52) | Trustworthy AI — echo chambers (sim=0.58) | Red Lines intro (combined=0.46, rel=2/5) | Red Lines — vulnerability chunk (rerank=-0.66) |
| 2 | Trustworthy AI (sim=0.50) | Red Lines — vulnerability (sim=0.57) | Trustworthy AI (combined=0.45, rel=2/5) | Red Lines intro (rerank=-1.91) |
| 3 | Red Lines — vulnerability (sim=0.48) | **Red Lines — predictive profiling (sim=0.57)** | Red Lines — vulnerability (combined=0.44, rel=2/5) | Red Lines — AI ranking (rerank=-4.88) |
| 4 | Red Lines — criminal (sim=0.46) | **Trustworthy AI — inference (sim=0.55)** | Red Lines — AI ranking (combined=0.43, rel=2/5) | Red Lines — criminal (rerank=-8.11) |
| 5 | Red Lines — AI ranking (sim=0.46) | **Trustworthy AI — risks vs opportunities (sim=0.54)** | Red Lines — criminal (combined=0.33, rel=1/5) | Trustworthy AI (rerank=-8.83) |

### Analysis

**Winner: Cross-Encoder.** It promoted the chunk discussing AI exploiting human vulnerabilities (the most topically related to emotion recognition risks) from rank 3 to rank 1. The LLM rerank was honest about low overall relevance (all scored 1-2/5) but didn't meaningfully reorder. All methods show low scores, correctly signaling the corpus lacks a direct hit for emotion recognition.

**HyDE effect:** HyDE significantly improved similarity scores (~0.52 → ~0.58) and brought in 3 new chunks — predictive profiling, AI inference risks, and a risks-vs-opportunities analysis — that are arguably more relevant to "emotion recognition risks" than the baseline's generic AI Act chunks. By generating a passage about privacy, bias, and manipulation risks, HyDE shifted the search toward chunks discussing those specific themes.

---

## Query 3: "How should companies train employees on AI tools?"

### Results

| Rank | Baseline | HyDE | LLM Rerank | Cross-Encoder |
|------|----------|------|------------|---------------|
| 1 | Booking.com (sim=0.61) | Smals (sim=0.66) | Booking.com (combined=0.80, rel=5/5) | Smals (rerank=2.34) |
| 2 | Smals (sim=0.60) | EnBW (sim=0.66) | Studio Deussen (combined=0.69, rel=4/5) | IBM (rerank=2.17) |
| 3 | IBM (sim=0.59) | Studio Deussen (sim=0.63) | Smals (combined=0.50, rel=2/5) | EnBW (rerank=1.12) |
| 4 | EnBW (sim=0.58) | **Credo AI (sim=0.63)** | IBM (combined=0.50, rel=2/5) | Booking.com (rerank=1.02) |
| 5 | Studio Deussen (sim=0.58) | **AI & Partners (sim=0.63)** | EnBW (combined=0.49, rel=2/5) | Studio Deussen (rerank=-3.02) |

### Analysis

**Winner: LLM Rerank.** It made the sharpest relevance distinctions: Booking.com got 5/5 (strong training program content), Studio Deussen 4/5 (practical hands-on training focus), while Smals/IBM/EnBW got 2/5 (contain AI info but aren't specifically about employee training). The cross-encoder actually degraded quality here by demoting the two most relevant results to ranks 4 and 5.

**HyDE effect:** HyDE boosted similarity scores (~0.61 → ~0.66) and replaced Booking.com and IBM with Credo AI and AI & Partners. Notably, HyDE dropped Booking.com (the LLM Rerank's top pick at 5/5 relevance) out of the top-5 entirely, suggesting that the hypothetical passage about "structured training programs" and "mentorship" pushed the embedding toward more generic organizational AI content rather than the specific training-focused documents.

---

## Summary

| Query | Baseline | HyDE | LLM Rerank | Cross-Encoder |
|-------|----------|------|------------|---------------|
| AI literacy (large companies) | Adequate | Good | **Best** | Good |
| Emotion recognition risks | Adequate | **Best** | Adequate | Good |
| Employee AI training | Adequate | Mixed | **Best** | Worse |

### Key Takeaways

- **LLM Rerank** excels when the query requires understanding intent and content quality — it can read the chunks and judge semantic relevance, not just lexical overlap. Wins 2 out of 3 queries.
- **HyDE** consistently boosts similarity scores (typically +0.04–0.06) and surfaces documents the other methods miss entirely. It shines on vague or conceptual queries (emotion recognition) where the raw query is far from document language. However, the hypothetical passage can drift from the user's actual intent (e.g., dropping Booking.com for the training query), making it less reliable for precise queries.
- **Cross-Encoder** excels at fine-grained query-passage matching (emotion recognition query) but can miss higher-level intent.
- **Baseline** is a safe default but can't differentiate when similarity scores are clustered (e.g., 0.64 vs 0.64 vs 0.64).
- **HyDE + Reranking** could be a strong combination: HyDE expands the candidate pool, then LLM or cross-encoder reranking filters for true relevance. This is not yet implemented but is a natural next step.
