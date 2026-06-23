# Iteration 2 — Detailed Retrospective

## What we changed, what worked, what didn't, and why

---

### 1. 🔴 mxbai Reranker Swap — REVERTED

**What:** Swapped `cross-encoder/ms-marco-MiniLM-L-6-v2` (22M params) → `mixedbread-ai/mxbai-rerank-base-v2` (Qwen2-based, ~184M params, 8K context) with `activation_fn=torch.nn.Sigmoid()` for calibrated 0-1 output.

**Rationale:** Research confirmed MS MARCO checkpoints have domain mismatch for resume-JD tasks (HF Forum Jan 2026). mxbai-rerank-v2 scores +8-9 nDCG@10 OOD on BEIR vs MiniLM (Jina v3 paper, arXiv 2509.25085). Sigmoid activation would eliminate the need for per-batch min-max normalization.

**Code:** `rank.py` — changed `CROSS_ENCODER_MODEL` string, added `import torch.nn`, added `activation_fn=torch.nn.Sigmoid()` to `CrossEncoder()`, removed `ce_min/ce_max/ce_norm` normalization block.

**Result:** ❌ **REVERTED.** Model loading completed, but inference on 2000 pairs did not finish within 600s timeout. The Qwen2 backbone is significantly heavier than MiniLM on CPU.

**Issue:** The research estimate of "~30s for 2000 pairs on CPU" was optimistic for this hardware (AMD Ryzen 5, no GPU). Actual MiniLM inference took 27s on the same CPU. The Qwen2 model likely requires 4-5× more compute per forward pass.

**Fix applied:** Reverted `CROSS_ENCODER_MODEL` back to `cross-encoder/ms-marco-MiniLM-L-6-v2`. Removed `torch.nn` import. Restored min-max normalization block. Code now back to Iter1 baseline for reranker.

---

### 2. 🟢 Hadamard Product — Citations Added to Deck

**What:** Updated `submission_metadata.yaml` methodology section with three formal citations supporting Hadamard product as principled industrial practice. Also added L2-normalization of sub-embeddings before Hadamard to bound output to [-1, 1].

**Rationale:** User flagged concern that Hadamard was "fragile" with "unbounded magnitude." Research confirmed:
- DCN-V2 (Google, KDD 2021): Hadamard replaced inner-product in YouTube Ads cross networks
- NFM (He & Chua, SIGIR 2017): Hadamard as principled FM extension for CTR ranking, beats factored inner-product
- Chrysos et al., IEEE TPAMI 2025 (52 citations): Hadamard products as canonical query-document fusion operation

**Code:** `submission_metadata.yaml` — added 3 citations with full venue/year. `semantic.py` — no code change needed (L2-norm already applied in `embed_texts` via `normalize_embeddings=True`; sub-embeddings are already unit vectors).

**Result:** ✅ **RETAINED.** Pure narrative win. No runtime cost. The L2-normalization was already happening (embeddings are normalized before storage). The Hadamard interaction is already bounded by the unit-norm property of the embeddings.

---

### 3. 🟡 Sub-Span Re-weighting + Preamble Stripping — NEUTRAL

**What:** Two changes to the JD sub-span decomposition:
1. **Weights:** [0.3, 0.4, 0.3] → [0.25, 0.30, 0.45] (boosted ideal-candidate/requirements sub-span)
2. **Preamble strip:** Sub-span 1 (role context) now removes lines starting with Location:, Opening:, Department:, Company:, About, Let's be honest before embedding

**Rationale:** Lukauskas et al. (MDPI Applied Sciences 2023) validated that the "Requirements" section carries the most discriminative skills signal in JD segmentation. The preamble (location + company blurb) was flagged as diluting the role signal.

**Code:** `semantic.py` — `JD_SUB_WEIGHTS` changed, `_strip_jd_preamble()` function added, called in `build_jd_sub_texts()` for i==0.

**Result:** 🟡 **NEUTRAL — RETAINED but no observable effect.** Top-10 candidate set identical (10/10), Kendall-Tau 0.9149 with Iter1. Semantic similarity contributes only 25% of total formula, and within that the re-weighted sub-spans changed scores by <0.003. The preamble strip removed ~2 sentences from a multi-paragraph sub-span.

**Issue:** The semantic similarity signal is structurally limited in impact. Cross-encoder (25%) + shipped-system evidence (25%) = 50% of the formula, and these two dominate candidate ordering. Tweaking weights within the semantic 25% slice only matters for candidates tied on everything else.

**Could have done better:** Tested preamble stripping without re-weighting, or tested weight extremes like [0.1, 0.2, 0.7] to force a visible effect.

---

### 4. 🟢 proxy_eval.py — New Label-Free Eval Harness

**What:** Created `ranker/scripts/proxy_eval.py` with three label-free quality metrics.

**Rationale:** Without labeled NDCG data, cannot objectively measure improvement. Research recommended three proxy metrics for triangulation.

**Implementation:**
1. **Pseudo-positive AUC** — scores top-3 as positives, 3000 random candidates as negatives via full ranker pipeline, computes `sklearn.metrics.roc_auc_score`. Takes ~66s (scoring 3000 candidates with embeddings). AUC = 1.000 (trivially high — any ranker separates its top-3 from random).
2. **Honeypot leak rate** — checks `compute_honeypot_score` for all top-100 candidates. Currently 1.0%. Under 10% spec.
3. **Rank stability** (optional, `--stability`) — drops 10% of JD tokens, re-embeds, re-ranks all 100K, computes Kendall-Tau vs original. Takes ~3 min. Uses `scipy.stats.kendalltau`.

**Usage:** `python scripts/proxy_eval.py submission.csv --candidates ../candidates.jsonl --embeddings embeddings`

**Issues:**
1. **Pseudo-AUC meaningless at absolute level** — AUC=1.0 because top-3 have scores 0.68-0.76 while random average is 0.58-0.62. Any reasonable ranker gets 1.0. Metric is only useful for comparing two ranker versions (delta-AUC).
2. **Stability check is slow** — re-ranks all 100K candidates (~3 min). Could be optimized to score only top-2000.
3. **Honeypot metric is reliable** — directly checks the official spec requirement.

---

### 5. ⏭️ bm25s — Intentionally Skipped

**What:** Did not implement bm25s (500× faster BM25 alternative to the rank_bm25 library that blew our budget).

**Rationale:** 
- TF-weighted lexical coverage (Iter1) already approximates BM25's term-frequency signal at zero runtime cost
- Cross-encoder reranking provides stronger relevance signal than any BM25 hybrid
- Implementing BM25 solely to write "we tested a fast BM25 alternative" in the deck was not worth 30 min

**Would revisit if:** Labeled data shows that candidates ranking well on lexical coverage but poorly on embedding similarity are being under-ranked by our formula. That would justify a RRF (Reciprocal Rank Fusion) hybrid with `bm25s`.

---

## Overall Iteration 2 Summary

| Change | Retained? | Impact on output | Runtime impact | Why |
|---|---|---|---|---|
| mxbai reranker swap | ❌ Reverted | — | N/A (timed out) | Qwen2 too heavy for CPU |
| Hadamard citations | ✅ Retained | No code impact | 0s | Pure narrative improvement |
| Sub-span re-weight [0.25/0.30/0.45] | ✅ Retained | Negligible (Kendall-Tau 0.915) | 0s | Semantic weight too small to matter |
| Preamble stripping | ✅ Retained | Negligible | 0s | Removed ~2 sentences from long text |
| proxy_eval.py | ✅ Created | New capability | ~66s (eval only) | Label-free quality metrics |
| bm25s | ⏭️ Skipped | — | — | TF-lexical + CE already cover signal |

**Bottom line:** Iter2 was a net zero on ranking quality. The cross-encoder reranker from Iter1 was and remains the single change that observably improved the output. The sub-span re-weighting and preamble stripping are academically sound but structurally incapable of affecting the outcome given the current formula weights.

**Recommendation:** Final submission = `submission_reranker.csv` (Iter1, 238.9s, validated). Iter2 changes kept in codebase for documentation purposes but not reflected in final output.
