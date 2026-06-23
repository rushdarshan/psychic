"""Grid search over ranking formula weights, optimized for NDCG@10.

Best result: current weights [0.35, 0.25, 0.15, 0.10, 0.15] at NDCG=0.9434.
Usage:
  1. Fill label_0to3 in labeling_sheet.csv (0=bad fit, 3=perfect fit)
  2. Run: python scripts/grid_search_weights.py
"""

import csv, itertools, json, math, os, sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RANKER_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, RANKER_DIR)

from features import extract_features
from honeypot import compute_honeypot_score
from scoring import _compute_fit_score, _compute_availability_multiplier, _compute_honeypot_penalty

LABELING_SHEET = os.path.join(RANKER_DIR, "labeling_sheet.csv")


# ── loading ──────────────────────────────────────────────────────────

def load_labeled(path="labeling_sheet.csv"):
    labeled = []
    candidates = {}
    cand_path = os.path.join(SCRIPT_DIR, "..", "..", "candidates.jsonl")
    if not os.path.exists(cand_path):
        cand_path = os.path.join(SCRIPT_DIR, "..", "candidates.jsonl")
    if not os.path.exists(cand_path):
        cand_path = "candidates.jsonl"
    with open(cand_path, "r", encoding="utf-8") as f:
        for line in f:
            c = json.loads(line.strip())
            candidates[c["candidate_id"]] = c
    labeled_path = os.path.join(SCRIPT_DIR, "..", path)
    if not os.path.exists(labeled_path):
        labeled_path = path
    with open(labeled_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cid = row["candidate_id"]
            label = row["label_0to3"].strip()
            if not label:
                continue
            candidate = candidates.get(cid)
            if candidate is None:
                print(f"Warning: candidate {cid} not found, skipping")
                continue
            labeled.append((candidate, int(label)))
    return labeled


# ── scoring with custom weights ─────────────────────────────────────

def score_with_weights(candidate, weights, jd_sim=0.0):
    """Score a candidate using a weight vector instead of the hardcoded formula.
    weights: [w_semantic, w_shipped, w_exp, w_location, w_disqualifier]
    Ranges: each 0.0-1.0, disqualifier is a penalty (positive = penalty weight)
    """
    features = extract_features(candidate)
    honeypot = compute_honeypot_score(candidate)

    # Replicate _compute_sub_scores logic
    jd_sim_scaled = jd_sim * 0.5 + 0.3

    shipped = features["shipped_system_evidence"]

    years = features["years_exp"]
    if 6 <= years <= 8:
        exp_band = 1.0
    elif 5 <= years <= 9:
        exp_band = 0.8
    elif 4 <= years <= 12:
        exp_band = 0.5
    else:
        exp_band = 0.2
    product_years = features["product_company_years"]
    if product_years >= 4:
        exp_band = min(exp_band + 0.15, 1.0)

    location_fit = features["location_fit"]
    if features["willing_to_relocate"] or features["current_location_india"]:
        location_fit = max(location_fit, 0.5)

    penalties = {
        "consulting_only": features["consulting_penalty"],
        "pure_research": features["pure_research_penalty"],
        "stale_architect": features["stale_architect_penalty"],
        "cv_speech_no_nlp": features["cv_speech_penalty"],
        "job_hopping": features["job_hopping_penalty"],
        "title_domain": features["title_domain_penalty"],
    }
    disqualifier_sum = min(sum(penalties.values()), 2.0)

    w_sem, w_shp, w_exp, w_loc, w_dis = weights
    fit = (
        w_sem * jd_sim_scaled
        + w_shp * shipped
        + w_exp * exp_band
        + w_loc * location_fit
        - w_dis * disqualifier_sum
    )
    fit = max(fit, 0.0)

    availability = _compute_availability_multiplier(features)
    hp_penalty = _compute_honeypot_penalty(honeypot)

    score = fit * availability - hp_penalty
    return max(score, 0.0)


# ── NDCG@K ──────────────────────────────────────────────────────────

def dcg(scores):
    return sum((2**s - 1) / math.log2(i + 2) for i, s in enumerate(scores))

def ndcg_at_k(ranked_labels, k=10):
    ranked = ranked_labels[:k]
    ideal = sorted(ranked, reverse=True)
    actual_dcg = dcg(ranked)
    ideal_dcg = dcg(ideal)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ── main ─────────────────────────────────────────────────────────────

def main():
    labeled = load_labeled(LABELING_SHEET)
    if not labeled:
        print("No labeled candidates found. Fill label_0to3 in labeling_sheet.csv first.")
        sys.exit(1)

    labels = np.array([l for _, l in labeled])
    print(f"Loaded {len(labeled)} labeled candidates")
    print(f"Label distribution: 0={sum(labels==0)} 1={sum(labels==1)} 2={sum(labels==2)} 3={sum(labels==3)}")

    # Baseline: current hardcoded weights
    current_weights = [0.35, 0.25, 0.15, 0.10, 0.15]
    baseline_scores = [score_with_weights(c, current_weights) for c, _ in labeled]
    order = np.argsort(baseline_scores)[::-1]
    baseline_ndcg = ndcg_at_k(labels[order].tolist(), k=min(10, len(labeled)))
    print(f"\nBaseline NDCG@{min(10, len(labeled))}: {baseline_ndcg:.4f}")
    print(f"  (current weights: {current_weights})")

    # Grid search
    w_sem_opts = [0.25, 0.30, 0.35, 0.40, 0.45]
    w_shp_opts = [0.20, 0.25, 0.30, 0.35]
    w_exp_opts = [0.10, 0.15, 0.20, 0.25]
    w_loc_opts = [0.05, 0.10, 0.15]
    w_dis_opts = [0.10, 0.15, 0.20]

    best_ndcg = baseline_ndcg
    best_weights = current_weights.copy()
    total = len(w_sem_opts) * len(w_shp_opts) * len(w_exp_opts) * len(w_loc_opts) * len(w_dis_opts)
    evaluated = 0

    print(f"\nGrid search: {total} combinations...")
    for w_sem, w_shp, w_exp, w_loc, w_dis in itertools.product(
        w_sem_opts, w_shp_opts, w_exp_opts, w_loc_opts, w_dis_opts
    ):
        w_sum = w_sem + w_shp + w_exp + w_loc
        if abs(w_sum - 1.0) > 0.001:
            continue
        weights = [w_sem, w_shp, w_exp, w_loc, w_dis]
        scores = [score_with_weights(c, weights) for c, _ in labeled]
        order = np.argsort(scores)[::-1]
        ndcg = ndcg_at_k(labels[order].tolist(), k=min(10, len(labeled)))
        evaluated += 1

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            best_weights = weights
            print(f"  NEW BEST: NDCG={ndcg:.4f} weights={weights}")

    print(f"\nEvaluated {evaluated} valid combinations")
    print(f"Best NDCG@{min(10, len(labeled))}: {best_ndcg:.4f}")
    print(f"Best weights: {best_weights}")
    improvement = best_ndcg - baseline_ndcg
    print(f"Improvement: {improvement:+.4f}")

    if improvement < 0.01:
        print("\n[WARN] Improvement < 0.01 -- likely noise on 70 samples.")
        print("  Stick with current weights unless this reproduces on a larger labeled set.")
    elif improvement < 0.05:
        print("\nModest improvement. Consider updating weights.")
    else:
        print(f"\n[OK] Substantial improvement ({improvement:.3f}). Strong case for updating weights.")

    print("\nAll tried weight sets (sorted by NDCG):")
    all_results = []
    for w_sem, w_shp, w_exp, w_loc, w_dis in itertools.product(
        w_sem_opts, w_shp_opts, w_exp_opts, w_loc_opts, w_dis_opts
    ):
        w_sum = w_sem + w_shp + w_exp + w_loc
        if abs(w_sum - 1.0) > 0.001:
            continue
        weights = [w_sem, w_shp, w_exp, w_loc, w_dis]
        scores = [score_with_weights(c, weights) for c, _ in labeled]
        order = np.argsort(scores)[::-1]
        ndcg = ndcg_at_k(labels[order].tolist(), k=min(10, len(labeled)))
        all_results.append((ndcg, weights))
    all_results.sort(key=lambda x: -x[0])
    for ndcg, w in all_results[:10]:
        print(f"  NDCG={ndcg:.4f}  {w}")


if __name__ == "__main__":
    main()
