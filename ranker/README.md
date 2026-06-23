# Redrob Candidate Ranker

Candidate ranking system for the [Redrob Intelligent Candidate Discovery & Ranking Challenge](https://redrob.com/hackathon). Ranks 100K candidates against a job description using a transparent, interpretable formula.

## Quick Start

```bash
pip install -r requirements.txt
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python scripts/verify_submission.py
```

## Two-Stage Architecture (L1 + L2)

Inspired by LinkedIn MUSE (Multi-granular Semantic Embeddings, KDD'20)
and Indeed UBM (SIGIR'19).

### Stage 1: Semantic Retrieval (L1)

Precomputed `all-MiniLM-L6-v2` embeddings (384-dim) for all candidates.
The JD is decomposed into **3 concept sub-spans** (role context,
responsibilities, ideal candidate) and embedded separately —
inspired by the question-driven embedding pattern from AAAI'24
career-path prediction. Each candidate receives **3 cosine similarities**
fused via learned weights, providing richer semantic signal than a
single blob-to-blob cosine.

```bash
python precompute.py --candidates ./candidates.jsonl
```

Output saved to `./embeddings/`. Ranking step loads these instead of
re-embedding at runtime.

### Stage 2: Multi-Attribute Scoring (L2)

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Must complete within **5 minutes on CPU, 16GB RAM, no network**. Uses
precomputed embeddings if available; falls back to rule-only scoring
otherwise.

## Scoring Formula

```
fit_score = 0.25 × semantic_similarity       (3 sub-span cosine fusion)
          + 0.05 × hadamard_similarity       (element-wise sub-emb × cand-emb)
          + 0.05 × jd_lexical_coverage       (TF-weighted keyword coverage)
          + 0.25 × shipped_system_evidence
          + 0.15 × experience_band_fit
          + 0.10 × location_fit
          − 0.15 × disqualifier_penalties

final_score = fit_score × availability_multiplier
              − honeypot_penalty
```

After L2 scoring, top 2000 candidates are re-ranked via a cross-encoder:

```
final_score = 0.75 × L2_score + 0.25 × cross_encoder_score
```

### Components

| Component | Weight | What it measures |
|-----------|--------|------------------|
| Semantic similarity | 25% | Multi-span cosine fusion: JD decomposed into role context (30%), responsibilities (40%), ideal candidate (30%) — richer than single cosine |
| Hadamard similarity | 5% | Element-wise product of sub-emb x cand-emb, summed — captures dimension-level alignment beyond cosine angle |
| JD lexical coverage | 5% | TF-weighted coverage of 27 JD-relevant ranking/NLP/ML keywords — hybrid dense+sparse signal |
| Shipped-system evidence | 25% | Rule-based detection of ranking/search/recsys at product companies |
| Experience band fit | 15% | 6-8 years total, ≥4 years at product companies |
| Location fit | 10% | Noida/Pune/India + relocation willingness |
| Disqualifier penalties | −15% | Consulting-only, pure research, stale architect, CV/speech without NLP, job-hopping |
| Availability multiplier | ×0.4–1.0 | Recency, response rate, notice period, interview completion, open to work |

## Verification

```bash
python scripts/verify_submission.py                    # sample data (50 candidates)
python scripts/verify_submission.py --full             # full 100K
```

Checks three things:
1. **Benchmark runtime** — scores all candidates, confirms <300s
2. **Honeypot rate** — checks top-100 for suspicious profiles (<10% required)
3. **Keyword-stuffer traps** — confirms off-domain titles with AI buzzwords land outside top 50

### Disqualifier Rules (from JD)

| Rule | Penalty |
|------|---------|
| Career entirely at consulting firms (TCS/Infosys/Wipro/Accenture/Cognizant/Capgemini) | Strong |
| Pure research, zero production deployment | Strong |
| Senior title but no code commits in current role | Moderate |
| Primary background in CV/speech/robotics without NLP/IR | Moderate |
| Job-hopping pattern (<18mo stints) | Moderate |
| 5+ years closed-source with no external validation | Mild |

### Honeypot Detection

Cross-checks for consistency anomalies:
- `years_of_experience` vs summed `duration_months`
- `proficiency == "expert"` with `<6mo` duration
- 5+ expert skills all at 0 endorsements
- 8+ advanced/expert skills with >50% under 12 months

Applied as a penalty, not a hard filter — scoring naturally avoids most honeypots, the penalty catches edge cases.

### Reasoning Generation

Reasoning text is populated from actual computed sub-scores per candidate:

```
"{years}y exp, {title} at {company}. {applied_ml_years}. {jd_keyword_coverage}. {shipped_evidence}. {location}. {concerns}. {availability}."
```

No templated filler. Each field reflects the candidate's actual features.

## Project Structure

```
ranker/
├── rank.py                 # CLI entry point (+ cross-encoder reranker)
├── precompute.py           # Offline embedding computation
├── jd_rules.py             # JD-derived rules + precompiled regex patterns
├── features.py             # Candidate feature extraction
├── scoring.py              # Weighted scoring + Hadamard + availability multiplier
├── semantic.py             # Sentence-transformer embedder
├── honeypot.py             # Honeypot consistency checks
├── reasoning.py            # Reasoning formatting utilities
├── validate.py             # Wrapper around official validate_submission.py
├── scripts/
│   ├── verify_submission.py  # Three-check verification
│   └── benchmark.py          # Throughput benchmark
├── sandbox/
│   └── app.py              # Streamlit sandbox (HF Spaces)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── submission_metadata.yaml   # Hackathon metadata
```

## File sizes

| File | Lines | Purpose |
|------|-------|---------|
| `jd_rules.py` | 143 | JD config + precompiled patterns |
| `features.py` | 197 | Feature extraction |
| `scoring.py` | 264 | Scoring formula + Hadamard + reasoning |
| `honeypot.py` | 72 | Honeypot detection |
| `semantic.py` | 40 | Embedding pipeline |
| `rank.py` | 230 | Main entry point + cross-encoder reranker |
| `precompute.py` | 77 | Embedding precomputation |

## Performance Notes

- All keyword matching uses **precompiled alternation regex patterns** compiled once at module load, not per-call — avoids recompilation overhead at 100K scale.
- Pattern matching uses `\b` word boundaries to avoid false positives from substring matches (e.g., `"ship"` vs `"leadership"`).
- Pure-Python scoring (no embeddings) benchmarks at ~60s for 100K candidates on CPU.
- Embedding step (MiniLM) adds ~15 min precomputation; ranking step with cached embeddings adds ~10s.
