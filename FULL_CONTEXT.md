# Redrob Candidate Ranker — Full Context

## Problem
Rank 100K candidates from `candidates.jsonl` (464 MB) against a JD (Senior AI Engineer at Redrob AI), output top 100 to `submission.csv`. Must run in ≤300s on ≤16GB RAM, no network during ranking, no LLM calls during ranking.

## Project Structure (working copy is `ranker/`)

```
ranker/
  rank.py                 # CLI entry point (--candidates, --out, --embed-dir, --no-embedding)
  precompute.py           # GPU-accelerated offline embedding computation
  scoring.py              # Weighted formula + reasoning builder
  features.py             # 200+ feature extraction from candidate data
  jd_rules.py             # JD-derived rules, precompiled regex patterns, title classifier, ML role detection
  honeypot.py             # Consistency checks for honeypot detection
  semantic.py             # all-MiniLM-L6-v2 embedder, cosine_similarity
  submission_metadata.yaml # Portal metadata
  embeddings/
    candidate_embeddings.npy  # (100000, 384) float32 — GPU precomputed
    candidate_ids.pkl         # Ordered candidate IDs matching embeddings
    embedding_metadata.json   # Model, dim, count info
  scripts/
    benchmark.py              # Rule-only timing benchmark
    build_labeling_sheet.py   # Stratified 70-candidate sampling for weight optimization
    grid_search_weights.py    # Grid search over weight combos for max NDCG@10
    verify_submission.py      # 3-check verification (runtime, honeypot, traps)
    peek_top10.py             # Quick top-10 peek
  sandbox/
    app.py                    # Streamlit sandbox for HF Spaces
submission.csv            # Correct output (root, not ranker/)
Deck.pdf                  # 6-slide deck for Hack2skill
labeling_sheet.csv        # 70 stratified candidates, label_0to3 column unfilled
embeddings.npy/           # Dir (misnamed from --out prefix match) with original GPU output
```

There's also a stale copy at `redrob_ranker/redrob/` from earlier setup — not the working code. Delete before submission.

## How the Ranking Works

### 1. Precompute (offline, run once)
`python ranker/precompute.py --candidates candidates.jsonl --output ranker/embeddings`
- Loads 100K candidates
- Builds candidate text from headline + summary + all career descriptions
- Encodes with `all-MiniLM-L6-v2` (384-dim), normalized embeddings
- GPU RTX 4050: **3 min 20 sec** (CPU was 107 min — 35x speedup)
- Saves: `candidate_embeddings.npy`, `candidate_ids.pkl`, `embedding_metadata.json`

### 2. Ranking (the ≤300s step)
`python ranker/rank.py --candidates candidates.jsonl --out submission.csv`
- Loads candidates + precomputed embeddings + JD embedding (from docx or fallback text)
- Scores each of 100K candidates using 5-component formula
- Outputs top 100 sorted by score descending
- Runtime with embeddings: **274s (4m34s)** — within 300s budget

### 3. Scoring Formula (`scoring.py`)

```
fit_score = 0.35 * semantic_similarity
          + 0.25 * shipped_system_evidence
          + 0.15 * experience_band_fit  (includes applied_ml_years boost)
          + 0.10 * location_fit
          - 0.15 * disqualifier_penalties_sum

final_score = fit_score * availability_multiplier - honeypot_penalty
final_score = max(final_score, 0.0)
```

**Components:**

| Component | Weight | How it's computed |
|-----------|--------|-------------------|
| Semantic similarity | 35% | Cosine sim between candidate text embedding and JD embedding (all-MiniLM-L6-v2, 384-dim, normalized). Captures profile-to-JD relevance beyond keywords. |
| Shipped-system evidence | 25% | Regex matching for ranking/search/recsys keywords (precompiled word-boundary patterns). Confidence scaled by duration at the company, boosted 0.2 if at product company. |
| Experience band | 15% | 6-8yr = 1.0, 5-9yr = 0.8, 4-12yr = 0.5, else 0.2. +0.25 if applied_ml_at_product >= 4yr, +0.10 if >= 2yr, +0.05 if applied_ml_total >= 3yr. +0.10 if product_company_years >= 4yr. Capped at 1.0. |
| Location fit | 10% | Noida/Pune = 1.0, remote India = 0.8, other India metro = 0.7, remote global = 0.7, other India = 0.5, outside India = 0.0. Boosted to min 0.5 if willing_to_relocate or already in India. |
| Disqualifier penalties | -15% | Consulting-only (-1.0), pure research (-0.7), stale architect (-0.6), CV/speech without NLP (-0.5), job-hopping (-1.0), off-domain title (-0.6). Capped at -2.0 total. |

**Availability multiplier** (multiplicative, range 0.4-1.0):
- 35% recency (no activity in 180+ days decays)
- 25% recruiter response rate
- 15% open-to-work flag
- 15% interview completion rate
- 10% notice period (30-60d normal, >60d penalized)

**Honeypot penalty** (subtractive):
- Suspicious: 0.5 + score*0.1
- Non-suspicious: score*0.05

### 4. JD Rule Engine (`jd_rules.py`)

All regex patterns are **precompiled at module load** as alternation regexes using `_alternation()`:
```python
pattern = re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b", re.IGNORECASE)
```

Key patterns:
- `_SHIPPED_RE`: 15 keywords (ranking, search, recommendation, recsys, retrieval, ndcg, mrr, etc.)
- `_SHIPPED_TECH_RE`: 14 tech keywords (FAISS, Pinecone, Qdrant, sentence-transformers, etc.)
- `_NLP_IR_RE`: 19 NLP keywords (NLP, BERT, transformer, RAG, fine-tuning, etc.)
- `_CV_RE`: 15 CV/speech keywords (YOLO, CNN, speech recognition, SLAM, etc.)
- `_RESEARCH_RE`: 8 research terms (paper, publication, conference, lab, etc.)
- `_PRODUCTION_RE`: 10 production terms (deployed, "shipped a", A/B test, serving, etc.)
- `_ML_TITLE_RE`: 20 ML title tokens (machine learning, data scientist, NLP, etc.)
- `_TECH_TITLE_RE`: 11 tech title tokens (software engineer, backend, data engineer, etc.)
- `_OFF_DOMAIN_TITLE_RE`: 15 off-domain title tokens (graphic designer, accountant, etc.)
- `_ML_CAREER_TITLE_RE`: 16 ML career title tokens (detects ML-specific roles in career history)
- `_ML_ROLE_COMBINED_RE`: Combined pattern of ML career titles + shipped + NLP keywords (efficient 1-regex ML role check)

### 5. Title Domain Classifier (`jd_rules.py:compute_title_domain_penalty`)
- If title hits any ML or tech token → 0.0 penalty (good)
- If title hits any off-domain token → 0.6 penalty (bad — keyword stuffer trap)
- Otherwise → 0.3 penalty (uncertain/ambiguous)

### 6. ML Role Detection (`jd_rules.py:is_ml_role`)
Checks if a career-history entry is an ML-specific role:
- Title matches `_ML_CAREER_TITLE_RE` → True (includes "machine learning", "data scientist", "recommendation systems", "nlp engineer", etc.)
- Description has ≥2 hits from combined `_ML_ROLE_COMBINED_RE` (ML titles + shipped keywords + NLP keywords) → True
- Used by `compute_applied_ml_years()` to calculate total and product-company applied ML years

### 7. Honeypot Detection (`honeypot.py`)
Three checks, score accumulates (threshold 2.5 = suspicious):
1. **Years mismatch**: career_duration vs profile.years_of_experience ratio <0.3 or >2.5 → +1.5
2. **Expert skills without tenure**: "expert" proficiency but <6mo duration → +1.0 each; 5+ expert skills all with 0 endorsements → +2.0
3. **Desc says years, duration <12mo**: career entry description mentions "X years" but duration_months <12 → +1.0

### 8. Feature Extraction (`features.py`)
Extracts 30+ features from candidate JSON:
- `years_exp`, `product_company_years`, `applied_ml_years`, `applied_ml_years_at_product`, `location_fit`
- `shipped_system_evidence` (0-1 confidence)
- `consulting_penalty, pure_research_penalty, stale_architect_penalty, cv_speech_penalty, job_hopping_penalty, title_domain_penalty`
- `nlp_ir_exposure`, `seniority_score`, `ai_ml_months`
- Redrob signals: open_to_work, last_active_date, recruiter_response_rate, interview_completion_rate, notice_period_days

## Performance Results

```
=== PERFORMANCE SUMMARY ===
Top score:          0.7291
Bottom (rank 100):  0.6144
Spread:             0.1147
Average top 100:    0.6457
Median top 100:     0.6427
Std deviation:      0.0232

ML/AI titles in top 100:  44/100
Product company names:     37/100

Score distribution:
  [0.60-0.62):  13  #############
  [0.62-0.64):  30  ##############################
  [0.64-0.66):  36  ####################################
  [0.66-0.68):  15  ###############
  [0.68-0.70):   4  ####
  [0.70-0.73):   2  ##

Honeypot rate:      0/100 = 0.0%  (threshold < 10%)
Off-domain top 50:  0
Budget (300s):      YES (274s actual)
```

### Top 10 (with reasoning showing applied ML years)
| Rank | ID | Score | Title | Company |
|------|-----|-------|-------|---------|
| 1 | CAND_0018499 | 0.7291 | Senior Machine Learning Engineer | Zomato |
| 2 | CAND_0046525 | 0.7129 | Senior Machine Learning Engineer | Genpact AI |
| 3 | CAND_0067866 | 0.6971 | Senior Software Engineer (ML) | Tech Mahindra |
| 4 | CAND_0053605 | 0.6960 | Senior Software Engineer (ML) | Verloop.io |
| 5 | CAND_0083879 | 0.6932 | Machine Learning Engineer | Ola |
| 6 | CAND_0010257 | 0.6823 | Senior Data Scientist | Google |
| 7 | CAND_0041669 | 0.6798 | Recommendation Systems Engineer | CRED |
| 8 | CAND_0052328 | 0.6793 | Recommendation Systems Engineer | Amazon |
| 9 | CAND_0006567 | 0.6790 | Senior AI Engineer | Meta |
| 10 | CAND_0008425 | 0.6779 | Senior NLP Engineer | Ola |

All top 10 are legitimate ML/AI roles at product companies. Reasoning now includes "{X}y applied ML ({Y}y at product co)" for explainability.

## Runtime Benchmarks
| Step | Time | Notes |
|------|------|-------|
| Precompute (CPU, no CUDA) | 107 min | Died at 94% — no partial save |
| Precompute (GPU RTX 4050) | 3 min 20 sec | 35x speedup |
| Ranking with embeddings | 274s | 100K candidates, all components including applied_ml_years |
| Ranking rule-only | 241s | Benchmark without semantic component |
| **Total reproduce** | **~4.5 min** | Precompute once, then rank command |

## Key Decisions and Rationale

### Why hand-weighted formula instead of learned ranker
- **No labeled training data exists** — 0 labels across 100K candidates
- **Defensible in interview**: every weight has a reasoning trace tied to JD requirements
- **Transparent**: judges can read the reasoning output for any candidate
- A learned ranker on 0 labels would be an undertrained model that's hard to justify
- Weight optimization via pseudo-labeled 70-candidate set + grid search is the pragmatic middle ground (not yet done — labels still needed)

### Why all-MiniLM-L6-v2
- 384-dim embeddings — small enough for 100K × 384 = ~150 MB in float32
- Fast inference, good general semantic quality
- Sentence-transformers ecosystem, standard choice

### Why precompute embeddings
- Ranking step must be ≤300s, no network
- Loading model + encoding 100K inline would blow the budget
- Precompute runs once, ranking loads cached numpy + pickle

### Why honeypot as penalty not hard filter
- Spec says "naturally avoid" — penalty aligns with nudging vs. hard exclusion
- Hard filter risks false positives disqualifying legitimate candidates

### Why no LLM-as-reranker
- Would violate compute budget (LLM calls are slow + need network)
- Nondeterministic — submission must reproduce exactly

## Bugs Fixed

### Batch 1 (before FULL_CONTEXT.md creation)

### 1. Title-domain substring bug (critical)
**File**: `jd_rules.py:compute_title_domain_penalty` (moved from `features.py`)
**Original**: `any(tok in title.lower() for tok in OFF_DOMAIN_TITLE_TOKENS)`
**Problem**: "ai" matched inside "Retail", "ship" matched "Worship", "live" matched "Delivered", "scal" matched "university of california"
**Fix**: Precompiled `\b(?:phrase1|phrase2)\b` word-boundary alternation regexes
**Impact**: "Retail Analyst" correctly → 0.6 penalty (was 0.0), "Research Analyst" → 0.3 (was 0.0), "Search Engineer" → 0.0 (no false positive)

### 2. Regex recompilation performance regression
**File**: `jd_rules.py` (module level)
**Original**: Recompiled regex patterns inside function calls (100K × N calls)
**Fix**: Precompile 6 alternation patterns at module load time
**Impact**: Restored benchmark runtime to original

### 3. Missing encoding="utf-8" on open() calls
**Files**: 3 locations across `precompute.py`, `sandbox/app.py`
**Problem**: Windows default encoding (cp1252) corrupts Unicode characters
**Fix**: Added `encoding="utf-8"` to all open() calls — verified across all 9 .py files

### 4. Unicode chars in print() breaking Windows console
**Files**: All .py files
**Problem**: Em-dash (—), checkmark (⚠), emoji broke PowerShell cp1252 console
**Fix**: Replaced with ASCII equivalents (--, [WARN], [OK])

### Batch 2 (code audit on 2026-06-23)

### 5. Missing applied_ml_years_at_product feature (HIGH)
**File**: `features.py:38-42`, `jd_rules.py`, `scoring.py`
**Problem**: JD explicitly asks for "4-5 years applied ML at product companies." Our `product_company_years` counted ALL roles at product companies (backend, QA, data eng) as equal to ML-specific roles. Backend engineer at Google got same product credit as ranking engineer at Amazon.
**Fix**: Added `is_ml_role()` function in `jd_rules.py` that checks career entry titles (precompiled `_ML_CAREER_TITLE_RE`) and descriptions (combined `_ML_ROLE_COMBINED_RE` with ≥2 keyword hits). Added `compute_applied_ml_years()` returning (total_applied_years, product_applied_years). Integrated into `scoring.py` with +0.25 boost for ≥4y applied ML at product, +0.10 for ≥2y, +0.05 for ≥3y total. Reasoning now shows "Xy applied ML (Yy at product co)".
**Impact**: More accurate ranking of ML-vs-non-ML roles. The scores at top band unchanged (already at cap), but mid-band candidates with dedicated ML experience now rank higher.

### 6. Empty career_history → consulting penalty (MEDIUM)
**File**: `jd_rules.py:137`
**Problem**: `is_consulting_only([])` returned True, penalizing candidates with no career data as "consulting-only."
**Fix**: Changed to `return False` — no data should not penalize.

### 7. "Remote" not recognized in location (MEDIUM)
**File**: `features.py:115-132`
**Problem**: JD says location "flexible." Remote India candidates got 0.5 (same as small town). Remote outside India got 0.0.
**Fix**: Added `remote_variants = {"remote", "work from home", "hybrid"}`. Remote India → 0.8, remote global → 0.7.

### 8. Stale architect `"code"` substring match (LOW)
**File**: `features.py:195`
**Problem**: `"code" in desc` matched "encode", "codebase", "decode". Gave false credit to non-coding architects (benign, but inaccurate).
**Fix**: Tightened to `" code "` and `" code,"` with surrounding spaces.

### 9. Dead code FRAMEWORK_TOURIST_PATTERNS (LOW)
**File**: `jd_rules.py:59-63`
**Problem**: Defined but never referenced anywhere.
**Fix**: Removed entirely. Replaced slot with `ML_CAREER_TITLE_TOKENS`.

### 10. is_ml_role performance regression (PERF)
**File**: `jd_rules.py`
**Problem**: `is_ml_role` ran 4 regex calls per career entry (1 title + 3 description patterns). Added +53s to ranking runtime.
**Fix**: Combined 3 desc patterns into `_ML_ROLE_COMBINED_RE`. Title check exits early (1 regex). Only 2 regex calls max. Runtime went 298s → 274s.

## What Still Needs Doing

### 1. Label labeling_sheet.csv
File: `ranker/labeling_sheet.csv`
- 70 candidates stratified: top 25 + ranks 26-150 (30) + random tail (10) + trap titles (5)
- Fill `label_0to3` column: 0 = bad fit → 3 = perfect fit
- Read raw candidate data (profile, career_history, skills, education) blind to current scores
- ~10 min work

### 2. Run grid search
Command: `python ranker/scripts/grid_search_weights.py`
- Tests ~87 weight combos (semantic 25-45%, shipped 15-35%, exp 5-25%, loc 5-15%, dis 10-25%)
- Maximizes NDCG@10 against your labels
- If improvement >0.01 NDCG, updates `scoring.py` weights + `submission_metadata.yaml`

### 3. Verify after changes
`python ranker/scripts/verify_submission.py --full`

### 4. Personalize metadata
File: `ranker/submission_metadata.yaml`
- Replace name, email, phone, GitHub repo, HF Spaces link

### 5. Check Deck.pdf renders
File: `Deck.pdf`
- 6 slides: problem framing, architecture, why no learned ranker, concrete bug catches, verification results, compute/reproducibility
- Verify uploads correctly to Hack2skill portal

### 6. Two codebases
- `ranker/` is the working copy (all changes go here)
- `redrob_ranker/redrob/` is stale — delete or ignore before submission
- The correct `submission.csv` is at root level (matches reproduce command path)

## Candidate Data Schema
```json
{
  "candidate_id": "CAND_0000001",
  "profile": {
    "anonymized_name": "...",
    "headline": "Backend Engineer | SQL, Spark, Cloud",
    "summary": "Software professional with 6.9 years...",
    "location": "Toronto",
    "country": "Canada",
    "years_of_experience": 6.9,
    "current_title": "Backend Engineer",
    "current_company": "Mindtree",
    "current_company_size": "10001+",
    "current_industry": "IT Services"
  },
  "career_history": [
    {
      "company": "Mindtree",
      "title": "Backend Engineer",
      "description": "...",
      "duration_months": 24,
      "is_current": true,
      "industry": "IT Services",
      "start_date": "2022-01",
      "end_date": ""
    }
  ],
  "education": [{ "degree": "B.Tech", "field": "Computer Science", ... }],
  "skills": [{ "name": "Python", "proficiency": "advanced", "duration_months": 48 }],
  "certifications": [],
  "languages": [],
  "redrob_signals": {
    "notice_period_days": 60,
    "willing_to_relocate": false,
    "open_to_work_flag": false,
    "last_active_date": "2025-03-01",
    "recruiter_response_rate": 0.0,
    "interview_completion_rate": 0.0
  }
}
```

## JD Context (summarized from docx)
Role: Senior AI Engineer at Redrob AI
- Need: ML engineering + applied NLP, embeddings, retrieval, ranking, LLMs
- Ideal: 6-8 yrs exp, 4-5 yrs applied ML at product companies, shipped ranking/search/recsys
- Location: Noida, Pune, or willing to relocate (flexible)
- **Disqualifies**: consulting-only profiles, pure researchers (papers without production), stale architects (don't code anymore), CV/speech without NLP/IR, job-hoppers (<18mo stints >50%), framework tourists (tutorial-only experience)
- Honeypot: naturally avoided by candidates genuinely fitting

## Commands Cheat Sheet
```powershell
# Precompute embeddings (GPU)
python ranker/precompute.py --candidates candidates.jsonl --output ranker/embeddings

# Run ranking
python ranker/rank.py --candidates candidates.jsonl --out submission.csv

# Rule-only benchmark  
python ranker/scripts/benchmark.py

# Verify submission
python ranker/scripts/verify_submission.py --full

# Build labeling sheet
python ranker/scripts/build_labeling_sheet.py

# Grid search weights (after labeling)
python ranker/scripts/grid_search_weights.py

# Peek top 10
python ranker/scripts/peek_top10.py
```
