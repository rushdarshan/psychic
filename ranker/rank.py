import argparse
import csv
import json
import os
import pickle
import sys
import time

import numpy as np

from semantic import (
    load_model, build_candidate_text, embed_texts, build_jd_text,
    build_jd_sub_texts,
)
from scoring import compute_candidate_score

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K = 2000
RERANK_BLEND_WEIGHT = 0.25


AUTO_DETECT_PATHS = [
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "[PUB] India_runs_data_and_ai_challenge",
        "India_runs_data_and_ai_challenge",
        "India_runs_data_and_ai_challenge",
        "candidates.jsonl",
    ),
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "[PUB] India_runs_data_and_ai_challenge",
        "India_runs_data_and_ai_challenge",
        "candidates.jsonl",
    ),
    "candidates.jsonl",
    "../candidates.jsonl",
]


PRE_COMPUTED_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "embeddings"
)


def find_file(paths):
    for p in paths:
        expanded = os.path.expanduser(p)
        if os.path.exists(expanded):
            return os.path.normpath(expanded)
    return None


def load_candidates(path):
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def load_precomputed_embeddings(embed_dir, candidates):
    emb_path = os.path.join(embed_dir, "candidate_embeddings.npy")
    ids_path = os.path.join(embed_dir, "candidate_ids.pkl")
    if not os.path.exists(emb_path) or not os.path.exists(ids_path):
        return None, None
    embeddings = np.load(emb_path)
    with open(ids_path, "rb") as f:
        stored_ids = pickle.load(f)
    id_to_idx = {cid: i for i, cid in enumerate(stored_ids)}
    ordered = np.zeros((len(candidates), embeddings.shape[1]), dtype=np.float32)
    ordered.fill(float("nan"))
    for i, c in enumerate(candidates):
        idx = id_to_idx.get(c["candidate_id"])
        if idx is not None:
            ordered[i] = embeddings[idx]
    return ordered, stored_ids


def _find_jd_docx():
    return find_file([
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "[PUB] India_runs_data_and_ai_challenge",
            "India_runs_data_and_ai_challenge",
            "India_runs_data_and_ai_challenge",
            "job_description.docx",
        ),
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "[PUB] India_runs_data_and_ai_challenge",
            "India_runs_data_and_ai_challenge",
            "job_description.docx",
        ),
    ])


def _load_jd_data():
    """Load JD as a structured dict with section keys.

    Attempts to parse the docx paragraph-by-paragraph, guessing
    section boundaries from headers. Falls back to flat text under
    a generic 'description' key.
    """
    jd_docx = _find_jd_docx()
    if jd_docx is None:
        return {
            "description": (
                "Senior AI Engineer role at Redrob AI. "
                "Need ML engineering experience with embeddings, "
                "retrieval, ranking, LLMs."
            ),
            "ideal_candidate": "",
            "what_we_need": "",
            "key_responsibilities": "",
        }

    try:
        import docx
        doc = docx.Document(jd_docx)
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    except ImportError:
        return {
            "description": (
                "Senior AI Engineer role at Redrob AI. "
                "Need ML engineering experience with embeddings, "
                "retrieval, ranking, LLMs."
            ),
            "ideal_candidate": "",
            "what_we_need": "",
            "key_responsibilities": "",
        }

    section_headers = {
        "description": [
            "about", "overview", "company",
            "let's be honest about this role",
            "job description:",
            "the vibe check",
        ],
        "what_we_need": [
            "what we need", "requirements", "qualifications",
            "what we're looking for", "skills",
            "what we mean by",
            "the skills inventory",
            "things you absolutely need",
            "things we'd like you to have",
            "things we explicitly do not want",
        ],
        "key_responsibilities": [
            "responsibilities", "what you'll do",
            "key responsibilities", "role",
            "what you'd actually be doing",
            "on location, comp, and logistics",
        ],
        "ideal_candidate": [
            "ideal candidate", "about you", "who you are",
            "how to read between the lines",
        ],
    }

    jd_data = {k: [] for k in section_headers}
    current_section = "description"

    for para in paragraphs:
        para_lower = para.lower()
        matched = False
        for section, headers in section_headers.items():
            if any(para_lower.startswith(h) or para_lower == h
                   for h in headers):
                current_section = section
                matched = True
                break
        if not matched:
            jd_data[current_section].append(para)

    return {k: "\n".join(v) for k, v in jd_data.items()}


def load_jd_text():
    jd_data = _load_jd_data()
    return " ".join(v for v in jd_data.values() if v)


def main():
    parser = argparse.ArgumentParser(
        description="Redrob Candidate Ranker - Rank 100K candidates against a JD"
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help="Path to candidates.jsonl (auto-detected if not provided)",
    )
    parser.add_argument(
        "--out",
        default="submission.csv",
        help="Output CSV path (default: submission.csv)",
    )
    parser.add_argument(
        "--embed-dir",
        default=PRE_COMPUTED_DIR,
        help="Directory with precomputed embeddings (default: ./embeddings)",
    )
    parser.add_argument(
        "--no-embedding",
        action="store_true",
        help="Skip embedding-based similarity (fall back to rule-based only)",
    )
    args = parser.parse_args()

    t_start = time.time()
    print("=" * 60)
    print("Redrob Candidate Ranker")
    print("=" * 60)

    candidates_path = args.candidates
    if candidates_path is None:
        candidates_path = find_file(AUTO_DETECT_PATHS)
    if candidates_path is None:
        print("Error: candidates.jsonl not found.")
        print("Provide --candidates path or place candidates.jsonl in working directory.")
        sys.exit(1)
    print(f"Candidates: {candidates_path}")

    print("Loading candidates...")
    candidates = load_candidates(candidates_path)
    print(f"Loaded {len(candidates)} candidates")

    model = None
    jd_embedding = None
    jd_sub_embeddings = None
    candidate_embeddings = None

    if not args.no_embedding:
        print("Loading precomputed embeddings...")
        if os.path.exists(args.embed_dir):
            candidate_embeddings, stored_ids = load_precomputed_embeddings(
                args.embed_dir, candidates
            )
        if candidate_embeddings is not None:
            print(f"Loaded precomputed embeddings: {candidate_embeddings.shape}")
            print("Loading embedding model for JD...")
            model = load_model()

            jd_text = load_jd_text()
            jd_emb = embed_texts(model, [jd_text])
            jd_embedding = jd_emb[0]

            jd_sub_texts = build_jd_sub_texts(_load_jd_data())
            if jd_sub_texts:
                print(f"JD decomposed into {len(jd_sub_texts)} sub-spans")
                jd_sub_embeddings = embed_texts(
                    model, jd_sub_texts, batch_size=4
                )
                for i, text in enumerate(jd_sub_texts):
                    print(f"  Sub-span {i+1}: {text[:60]}...")
            else:
                print("No JD sub-spans available, using single embedding")

            print("JD embedding computed")
        else:
            print("No precomputed embeddings found. Running with rule-based scoring only.")
            print("(Run precompute.py first to enable semantic similarity)")

    print("Scoring candidates...")
    results = []
    for i, c in enumerate(candidates):
        cand_emb = None
        if candidate_embeddings is not None and i < len(candidate_embeddings):
            emb = candidate_embeddings[i]
            if not np.any(np.isnan(emb)):
                cand_emb = emb
        use_emb = cand_emb is not None and jd_embedding is not None
        result = compute_candidate_score(
            c, jd_embedding, cand_emb,
            use_embeddings=use_emb,
            jd_sub_embeddings=jd_sub_embeddings,
        )
        results.append(result)

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t_start
            print(f"  Scored {i + 1}/{len(candidates)} ({elapsed:.1f}s)")

    if not args.no_embedding:
        print(f"Cross-encoder reranking top {RERANK_TOP_K}...")
        t_rerank = time.time()
        top_k = sorted(results, key=lambda r: -r["score"])[:RERANK_TOP_K]
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(CROSS_ENCODER_MODEL)

        pairs = []
        cand_map = {c["candidate_id"]: c for c in candidates}
        for r in top_k:
            c = cand_map.get(r["candidate_id"])
            if c:
                c_text = build_candidate_text(c)
                pairs.append((jd_text, c_text))
            else:
                pairs.append((jd_text, ""))

        ce_scores = reranker.predict(pairs, batch_size=32, show_progress_bar=True)
        ce_scores = np.array(ce_scores)
        ce_min, ce_max = ce_scores.min(), ce_scores.max()
        if ce_max > ce_min:
            ce_norm = (ce_scores - ce_min) / (ce_max - ce_min)
        else:
            ce_norm = np.zeros_like(ce_scores)

        ce_map = {}
        for r, ce_n in zip(top_k, ce_norm):
            ce_map[r["candidate_id"]] = float(ce_n)

        for r in results:
            ce_val = ce_map.get(r["candidate_id"])
            if ce_val is not None:
                r["ce_score"] = float(ce_val)
                blended = (
                    (1 - RERANK_BLEND_WEIGHT) * r["score"]
                    + RERANK_BLEND_WEIGHT * ce_val
                )
                r["score"] = round(blended, 4)
        print(f"Reranking done in {time.time() - t_rerank:.1f}s")

    results.sort(key=lambda r: (-r["score"], r["candidate_id"]))

    top_100 = results[:100]
    output_path = args.out
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, r in enumerate(top_100, start=1):
            writer.writerow([
                r["candidate_id"],
                rank,
                f"{r['score']:.4f}",
                r["reasoning"],
            ])

    elapsed = time.time() - t_start
    print(f"\nDone! Wrote {output_path}")
    print(f"Top score: {top_100[0]['score']:.4f}, Bottom: {top_100[-1]['score']:.4f}")
    print(f"Total time: {elapsed:.1f}s")

    print("\nTop 10:")
    for r in top_100[:10]:
        print(f"  {r['candidate_id']}: {r['score']:.4f} | {r['reasoning'][:80]}...")


if __name__ == "__main__":
    main()
