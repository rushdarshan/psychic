# Iteration 2 — Post-Experiment Research Notes

Context: ran 6 changes (sub-span JD, TF-weighted lexical, BM25 hybrid REVERTED, Hadamard, cross-encoder, validation fix). Cross-encoder was the only one that observably moved top-10. This note targets the 5 highest-leverage follow-ups.

## Overall Outcome

| Metric | Iter1 (best) | Iter2 | Verdict |
|---|---|---|---|
| Top-10 overlap | — | 10/10 same set | No change |
| Kendall-Tau (top-100) | — | 0.9149 | Near identical |
| Max score | 0.7560 | 0.7535 | Marginally lower |
| Honeypot leak | 1% | 1% | Same |
| Runtime | **238.9s** ✅ | **315.0s** ❌ | Over budget |

**Conclusion:** Iter2 changes (sub-span re-weighting, preamble stripping) had no measurable impact on ranking output. Cross-encoder signal (25%) + shipped-system evidence (25%) dominate the formula; tweaking semantic similarity sub-weights within its 25% allocation couldn't move the needle. **Final submission should use Iter1 output.**

## 1. Reranker swap — `mixedbread-ai/mxbai-rerank-base-v2`

**Hypothesis:** MS-MARCO MiniLM has domain mismatch for resume-JD ranking. A stronger OOD reranker should improve top-10 quality.

**Status: REVERTED** — mxbai-rerank-base-v2 (Qwen2-based, ~184M params) was too slow on available CPU. Model load + 2000-pair inference exceeded 600s timeout. Sticking with `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params, ~27s inference).

**Evidence the concern is real:**
- HF Forum (Jan 2026): "the specific checkpoints you tested are trained for MS MARCO passage ranking, which is a different distribution than [your task]."
- jina-reranker-v3 paper (arXiv 2509.25085, Oct 2025): MS-MARCO-MiniLM-class rerankers are now the weakest on BEIR OOD benchmark (8–10 nDCG points below mxbai/jina-v3).

**If budget allows:** reduce RERANK_TOP_K to 500 and try mxbai again (~8s inference).

## 2. Hadamard product — DCN-V2 backing

**Status: RETAINED.** Research confirmed the signal is standard, not fragile.

**Citations now in deck:**
- DCN-V2 (Google, KDD 2021): replaced inner-product with Hadamard in cross networks; used in YouTube Ads production
- NFM (He & Chua, SIGIR 2017): Hadamard as principled extension of FM for CTR ranking
- Chrysos et al., IEEE TPAMI 2025 (52 cites): Hadamard products as canonical query-document fusion in industrial recsys

**Implementation improvement:** sub-embeddings now L2-normalized before Hadamard, bounding each dimension-level score to [-1, 1].

## 3. JD sub-span re-weighting + preamble stripping

**Status: IMPLEMENTED.**
- Weights changed from [0.3, 0.4, 0.3] → [0.25, 0.30, 0.45]
- Sub-span 1 now strips location/company/preamble lines before embedding
- Lukauskas et al. (MDPI Applied Sciences 2023) validates requirements section carries strongest discriminative signal → supports 0.45 weight on ideal-candidate sub-span

## 4. Label-free proxy evaluation

**Status: IMPLEMENTED** as `ranker/scripts/proxy_eval.py`.

Three metrics:
1. **Pseudo-positive AUC** — top-3 as weak positives, 3000 random as negatives; AUC = 1.000
2. **Honeypot leak rate** — top-100 honeypot rate; currently 1.0% (under 10% spec)
3. **Rank stability (--stability flag)** — Kendall-τ after 10% JD token dropout

The pseudo-AUC metric is trivially high (1.0) for any ranker that separates top-3 from random. More useful when comparing two ranker versions against the same pseudo-label set.

## 5. bm25s — feasible BM25 alternative

bm25s (Lù 2024, HF Blog March 2026) is 500× faster than rank_bm25 (pure Numpy sparse-matrix indexing vs Python loops). Was not implemented because:
- TF-weighted lexical coverage (item 2 from Iter 1) already approximates BM25's term-frequency signal
- Cross-encoder reranking provides stronger relevance signal than BM25 hybrid
- Running BM25 just for a "we tested it" deck line isn't worth 30 min

## Sources (deck-quotable)
- Jina Reranker v3 — arXiv:2509.25085, Oct 2025
- HF Forum, Cross-Encoder Training thread — Jan 2026
- Chrysos et al., Hadamard Product in Deep Learning — IEEE TPAMI 2025
- DCN-V2 — Wang et al., KDD 2021
- NFM — He & Chua, SIGIR 2017
- Lukauskas et al., MDPI Applied Sciences 2023
- bm25s, Lù 2024 + HF Blog March 2026
- Brenndoerfer, Reranking: Cross-Encoders blog — Jan 2026

## Decisions Not Taken
- **ColBERTv2:** skipped — index footprint (1.2 GB for 100K × 384 × 32 bits) exceeds 16 GB RAM budget with model + candidates loaded. Jina v3 (Oct 2025) shows single-vector + CE reranker matches ColBERT quality.
- **bm25s + RRF:** skipped — TF-weighted lexical coverage provides similar term-frequency signal at zero cost, and CE reranking dominates BM25 for relevance.
