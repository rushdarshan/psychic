"""Streamlit sandbox for the Redrob Candidate Ranker.

Accepts up to 100 candidates, runs ranking end-to-end, and displays the
top-10 with reasoning. Meets the hackathon's sandbox requirement (Section 10.5).
"""
import json
import os
import sys
import tempfile
from io import StringIO

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

SAMPLE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..",
    "candidates.jsonl",
)

st.set_page_config(
    page_title="Redrob Candidate Ranker",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource
def load_sample():
    if os.path.exists(SAMPLE_PATH):
        samples = []
        with open(SAMPLE_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples[:50]
    return []


def run_ranking(candidates):
    from scoring import compute_candidate_score

    results = []
    progress = st.progress(0, "Scoring candidates...")
    for i, c in enumerate(candidates):
        r = compute_candidate_score(c, None, None, use_embeddings=False)
        results.append(r)
        progress.progress((i + 1) / len(candidates))
    results.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    return results[:100]


def main():
    st.title("🔍 Redrob Candidate Ranker")
    st.markdown(
        "Rank candidates against a Senior AI Engineer job description. "
        "Upload a `candidates.jsonl` file (≤100 candidates) or use the built-in sample."
    )

    tab1, tab2 = st.tabs(["Rank Candidates", "About"])

    with tab1:
        with st.expander("Job Description", expanded=False):
            st.markdown("""
**Senior AI Engineer — Founding Team** at Redrob AI (Series A).

**Signal priority (in order):**
1. Shipped ranking/search/recsys system to real users at scale
2. 6-8 yrs total, 4-5 yrs applied ML at product companies
3. Noida/Pune or willing to relocate
4. Active on platform / clear job-market signal
            """)

        source = st.radio(
            "Candidate source",
            ["Sample (50 candidates)", "Upload JSONL"],
            horizontal=True,
        )

        candidates = []
        if source == "Sample (50 candidates)":
            candidates = load_sample()
            if not candidates:
                st.error(f"Sample file not found: {SAMPLE_PATH}")
            else:
                st.info(f"Loaded {len(candidates)} sample candidates")
        else:
            uploaded = st.file_uploader(
                "Upload candidates.jsonl",
                type=["jsonl"],
                help="Max 100 candidates for sandbox mode",
            )
            if uploaded:
                text = uploaded.read().decode("utf-8")
                for line in text.strip().split("\n"):
                    if line.strip():
                        candidates.append(json.loads(line))
                if len(candidates) > 100:
                    candidates = candidates[:100]
                    st.warning("Truncated to first 100 candidates")
                st.info(f"Loaded {len(candidates)} candidates")

        if candidates and st.button("Run Ranking", type="primary"):
            results = run_ranking(candidates)
            st.success(f"Ranked {len(candidates)} candidates")

            st.subheader("Top 10")
            rows = []
            for r in results[:10]:
                rows.append({
                    "Rank": r.get("_rank"),
                    "Candidate ID": r["candidate_id"],
                    "Score": f"{r['score']:.4f}",
                    "Reasoning": r["reasoning"],
                })
            for i, r in enumerate(results[:10]):
                r["_rank"] = i + 1
            for rank, r in enumerate(results[:10], 1):
                r["_rank"] = rank
                st.markdown(
                    f"**{rank}.** `{r['candidate_id']}` — **{r['score']:.4f}**  \n"
                    f"{r['reasoning']}"
                )

            with st.expander("Show all 100"):
                all_rows = []
                for rank, r in enumerate(results[:100], 1):
                    all_rows.append({
                        "Rank": rank,
                        "Candidate ID": r["candidate_id"],
                        "Score": f"{r['score']:.4f}",
                        "Reasoning": r["reasoning"],
                    })
                st.dataframe(all_rows, use_container_width=True, hide_index=True)

            csv_lines = ["candidate_id,rank,score,reasoning"]
            for rank, r in enumerate(results[:100], 1):
                reasoning = r["reasoning"].replace('"', "'")
                csv_lines.append(f"{r['candidate_id']},{rank},{r['score']:.4f},{reasoning}")
            csv_text = "\n".join(csv_lines)
            st.download_button(
                label="Download submission.csv",
                data=csv_text,
                file_name="submission.csv",
                mime="text/csv",
            )

    with tab2:
        st.markdown("""
### How scoring works

A transparent, hand-weighted formula — no black-box model:

```
fit = 0.35 × semantic_sim + 0.25 × shipped_evidence
    + 0.15 × exp_band + 0.10 × location
    - 0.15 × disqualifier_penalties

final = fit × availability_multiplier - honeypot_penalty
```

### Key signals

| Signal | What it detects |
|--------|-----------------|
| Shipped system | Career descriptions matching ranking/search/recsys at product companies |
| Experience band | 6-8 years total, ≥4 at product companies |
| Disqualifier | Consulting-only, pure research, stale architect, CV-only, job-hopping |
| Availability | Recency, response rate, notice period, interview completion |

### Submission format

Output matches the exact `submission.csv` schema: `candidate_id,rank,score,reasoning`.

### Compute

Rule-only mode — no LLM calls, no GPU, no network. Fits 5-min/16GB CPU budget on 100K.
        """)


if __name__ == "__main__":
    main()
