"""Label-free proxy evaluation for candidate ranker.

Metrics updated after Iter2 experiments (Kendall tau 0.915 vs Iter1).
Three metrics that triangulate ranking quality without ground-truth labels:
  (a) Pseudo-positive AUC    — top-3 as weak positives, 3000 random as negatives
  (b) Honeypot leak rate     — % of honeypots in top-100
  (c) Rank stability (opt)   — Kendall-τ after perturbing 10% of JD tokens

Usage:
    python scripts/proxy_eval.py submission.csv --candidates ../candidates.jsonl --embeddings ../embeddings
    python scripts/proxy_eval.py submission.csv --candidates ../candidates.jsonl --embeddings ../embeddings --stability
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from honeypot import compute_honeypot_score
from semantic import load_model, embed_texts, build_jd_text, build_jd_sub_texts
from scoring import compute_candidate_score
from rank import load_precomputed_embeddings


def load_candidates(path):
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def load_submission(path):
    candidates, scores = [], []
    with open(path, "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",", 3)
            candidates.append(parts[0])
            scores.append(float(parts[2]))
    return candidates, scores


def pseudo_auc(sub_ids, sub_scores, all_candidates, candidate_embeddings,
               jd_embedding, jd_sub_embeddings, n_neg=3000):
    top3 = set(sub_ids[:3])
    neg_pool = [c for c in all_candidates if c["candidate_id"] not in top3]
    neg_sample = random.sample(neg_pool, min(n_neg, len(neg_pool)))

    neg_scores = []
    for c in neg_sample:
        cid = c["candidate_id"]
        idx = next(i for i, x in enumerate(all_candidates)
                   if x["candidate_id"] == cid)
        cand_emb = None
        if candidate_embeddings is not None and idx < len(candidate_embeddings):
            emb = candidate_embeddings[idx]
            if not np.any(np.isnan(emb)):
                cand_emb = emb
        r = compute_candidate_score(
            c, jd_embedding, cand_emb,
            use_embeddings=cand_emb is not None,
            jd_sub_embeddings=jd_sub_embeddings,
        )
        neg_scores.append(r["score"])

    labels = [1, 1, 1] + [0] * len(neg_scores)
    all_scores = list(sub_scores[:3]) + neg_scores

    labels = np.array(labels, dtype=np.float64)
    scores = np.array(all_scores, dtype=np.float64)

    if len(np.unique(scores)) == 1:
        return 0.5
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(labels, scores))


def honeypot_leak_rate(top_candidate_ids, all_candidates):
    cand_map = {c["candidate_id"]: c for c in all_candidates}
    hits = 0
    for cid in top_candidate_ids:
        c = cand_map.get(cid)
        if c and compute_honeypot_score(c).get("score", 0) > 0:
            hits += 1
    return hits / len(top_candidate_ids)


def rank_stability(sub_ids, jd_data, all_candidates, candidate_embeddings):
    jd_text = build_jd_text(jd_data)
    tokens = jd_text.split()
    k = max(1, len(tokens) // 10)
    drop = set(random.sample(range(len(tokens)), k))
    perturbed = " ".join(t for i, t in enumerate(tokens) if i not in drop)

    model = load_model()
    pert_emb = embed_texts(model, [perturbed])[0]

    jd_sub_texts = build_jd_sub_texts(jd_data)
    jd_sub_embeddings = None
    if jd_sub_texts:
        jd_sub_embeddings = embed_texts(model, jd_sub_texts, batch_size=4)

    pert_scores = []
    for i, c in enumerate(all_candidates):
        cand_emb = None
        if candidate_embeddings is not None and i < len(candidate_embeddings):
            emb = candidate_embeddings[i]
            if not np.any(np.isnan(emb)):
                cand_emb = emb
        r = compute_candidate_score(
            c, pert_emb, cand_emb,
            use_embeddings=cand_emb is not None,
            jd_sub_embeddings=jd_sub_embeddings,
        )
        pert_scores.append(r["score"])
    pert_order = [all_candidates[i]["candidate_id"]
                  for i in np.argsort(-np.array(pert_scores))[:100]]

    orig_rank = {cid: i for i, cid in enumerate(sub_ids[:100])}
    pert_rank = {cid: i for i, cid in enumerate(pert_order)}
    common = [cid for cid in sub_ids[:100] if cid in pert_rank]
    if len(common) < 2:
        return 0.0
    from scipy.stats import kendalltau
    tau, _ = kendalltau(
        [orig_rank[c] for c in common],
        [pert_rank[c] for c in common],
    )
    return float(tau)


def _get_jd_data():
    jd_paths = [
        os.path.join(os.path.dirname(__file__), "..",
                     "[PUB] India_runs_data_and_ai_challenge",
                     "India_runs_data_and_ai_challenge",
                     "India_runs_data_and_ai_challenge",
                     "job_description.docx"),
        os.path.join(os.path.dirname(__file__), "..",
                     "[PUB] India_runs_data_and_ai_challenge",
                     "job_description.docx"),
    ]
    for p in jd_paths:
        if os.path.exists(p):
            try:
                import docx
                doc = docx.Document(p)
                paras = [t.text.strip() for t in doc.paragraphs if t.text.strip()]
                return {"description": "\n".join(paras)}
            except Exception:
                return {"description": "Senior AI Engineer role at Redrob AI."}
    return {"description": "Senior AI Engineer role at Redrob AI."}


def main():
    parser = argparse.ArgumentParser(description="Label-free proxy evaluation")
    parser.add_argument("submission", help="Path to submission CSV")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--embeddings", help="Path to precomputed embeddings directory")
    parser.add_argument("--stability", action="store_true", help="Run rank stability check (~3 min)")
    args = parser.parse_args()

    print("Loading candidates...")
    all_candidates = load_candidates(args.candidates)
    print(f"  Loaded {len(all_candidates)} candidates")

    print("Loading embeddings...")
    candidate_embeddings = None
    jd_embedding, jd_sub_embeddings = None, None
    if args.embeddings:
        try:
            base = os.path.dirname(os.path.dirname(__file__))
            emb_dir = os.path.normpath(os.path.join(base, args.embeddings))
            loaded, stored = load_precomputed_embeddings(emb_dir, all_candidates)
            if loaded is None:
                raise FileNotFoundError(f"embeddings not found at {emb_dir}")
            candidate_embeddings = loaded
            print(f"  Loaded embeddings: {candidate_embeddings.shape}")

            jd_data = _get_jd_data()
            model = load_model()
            jd_text = build_jd_text(jd_data)
            jd_embedding = embed_texts(model, [jd_text])[0]
            jd_sub_texts = build_jd_sub_texts(jd_data)
            if jd_sub_texts:
                jd_sub_embeddings = embed_texts(model, jd_sub_texts, batch_size=4)
            print("  JD embedding computed")
        except Exception as e:
            print(f"  Warning: could not load embeddings ({e}), using 0.0 for missing")

    print("Loading submission...")
    sub_ids, sub_scores = load_submission(args.submission)
    print(f"  Loaded {len(sub_ids)} ranked entries")

    print("\n--- Proxy Metrics ---")

    if candidate_embeddings is not None:
        t0 = time.time()
        auc = pseudo_auc(
            sub_ids, sub_scores, all_candidates, candidate_embeddings,
            jd_embedding, jd_sub_embeddings, n_neg=3000,
        )
        print(f"(a) Pseudo-positive AUC:    {auc:.4f}  ({time.time()-t0:.1f}s)")
        print(f"    >0.70 = top-3 scores separate from random scores")
    else:
        print("(a) Pseudo-positive AUC:    SKIP (need --embeddings)")

    t0 = time.time()
    leak = honeypot_leak_rate(sub_ids[:100], all_candidates)
    print(f"(b) Honeypot leak (top-100): {leak:.1%}  ({time.time()-t0:.1f}s)")
    print(f"    <10% required by spec; 0% = no adversaries leaked in")

    if args.stability and candidate_embeddings is not None:
        print("\n(c) Rank stability under JD perturbation...")
        t0 = time.time()
        jd_data = _get_jd_data()
        tau = rank_stability(sub_ids, jd_data, all_candidates, candidate_embeddings)
        print(f"(c) Kendall-τ (JD perturb): {tau:.4f}  ({time.time()-t0:.1f}s)")
        print(f"    >0.80 = ranking stable under 10% JD token dropout")

    print("\nDone.")


if __name__ == "__main__":
    main()
