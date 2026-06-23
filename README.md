# Redrob Candidate Ranker

Intelligent candidate-job matching for the Redrob Hiring Challenge. Ranks 100K candidates against a Senior AI Engineer job description using a transparent, hand-weighted formula with cross-encoder reranking.

**Final submission:** `submission_reranker.csv` — validated (0 errors), 1.0% honeypot, 238.9s runtime, NDCG@10 = 0.9434.

## Architecture

| Layer | Component | What it does |
|-------|-----------|-------------|
| L1 | Semantic retrieval | 3 sub-span cosine + Hadamard interaction (all-MiniLM-L6-v2, 384-dim) |
| L2 | Multi-attribute scoring | 7-component weighted formula (shipped evidence, experience, location, JD lexical coverage, disqualifiers, availability, honeypot penalty) |
| L3 | Cross-encoder reranking | Top-2000 re-ranked via ms-marco-MiniLM-L-6-v2, blended 25% with L2 score |

```
fit = 0.25×semantic + 0.05×hadamard + 0.05×lexical + 0.25×shipped + 0.15×exp + 0.10×loc − 0.15×disq
final = fit × availability − honeypot_penalty
blended = 0.75×final + 0.25×cross_encoder
```

[Full methodology →](ranker/README.md)

## Project Structure

```
├── ranker/                  # Core ranking system
│   ├── rank.py             # Entry point + cross-encoder
│   ├── scoring.py          # Scoring formula
│   ├── semantic.py         # Embedding + JD decomposition
│   ├── features.py         # 200+ feature extraction
│   ├── honeypot.py         # Consistency cross-checks
│   ├── jd_rules.py         # Keyword rules + title classifier
│   ├── sandbox/            # Streamlit demo app
│   └── scripts/            # grid_search, proxy_eval, verify, benchmark
├── candidates.jsonl         # 100K candidate profiles (gitignored)
├── embeddings.npy/          # Precomputed embeddings (gitignored)
├── labeling_sheet.csv       # 70 human-labeled candidates
├── Deck.pdf                 # Presentation deck
└── submission_reranker.csv  # Final output (gitignored)
```

## Quick Start

```bash
cd ranker
pip install -r requirements.txt
python precompute.py --candidates ../candidates.jsonl
python rank.py --candidates ../candidates.jsonl --embed-dir ../embeddings.npy --out ../submission.csv
python scripts/verify_submission.py --full
```

## Key Results

- **NDCG@10:** 0.9434 (grid-search confirmed optimal on 70 labels)
- **Honeypot rate:** 1.0% (threshold: <10%)
- **Runtime:** 238.9s (budget: <300s)
- **Validation:** 0 errors on official validator

## Requirements

- Python 3.11+
- `sentence-transformers`, `numpy` (see `ranker/requirements.txt`)
- CPU only during ranking; GPU optional for embedding precomputation
- No network access required during ranking
