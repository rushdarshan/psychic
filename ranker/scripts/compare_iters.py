"""Compare iteration 1 vs iteration 2 submissions."""
import csv
import json
import sys

sys.path.insert(0, ".")
from honeypot import compute_honeypot_score


def load_sub(path):
    rows = []
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)
        for r in reader:
            rows.append((r[0], float(r[2])))
    return rows


it1 = load_sub("../submission_reranker.csv")
it2 = load_sub("../submission_iter2.csv")

print("=== Top-10 comparison ===")
print(f"{'Rank':<6} {'Iter1':<30} {'Score':<8} {'Iter2':<30} {'Score':<8}")
for i in range(10):
    print(f"{i+1:<6} {it1[i][0]:<30} {it1[i][1]:<8.4f} {it2[i][0]:<30} {it2[i][1]:<8.4f}")

set1 = set(c for c, _ in it1[:10])
set2 = set(c for c, _ in it2[:10])
print(f"\nTop-10 candidate overlap: {len(set1 & set2)}/10")

rk1 = {c: i for i, (c, _) in enumerate(it1[:100])}
rk2 = {c: i for i, (c, _) in enumerate(it2[:100])}
common = [c for c in rk1 if c in rk2]
from scipy.stats import kendalltau
tau, _ = kendalltau([rk1[c] for c in common], [rk2[c] for c in common])
print(f"Kendall-Tau (top-100): {tau:.4f}")

s1_vals = [s for _, s in it1]
s2_vals = [s for _, s in it2]
print(f"\nIter1 scores: min={min(s1_vals):.4f} max={max(s1_vals):.4f} mean={sum(s1_vals)/len(s1_vals):.4f}")
print(f"Iter2 scores: min={min(s2_vals):.4f} max={max(s2_vals):.4f} mean={sum(s2_vals)/len(s2_vals):.4f}")

cands = [json.loads(l) for l in open("../candidates.jsonl")]
cmap = {c["candidate_id"]: c for c in cands}
for name, rows in [("Iter1", it1), ("Iter2", it2)]:
    hp = sum(
        1 for c, _ in rows[:100] if compute_honeypot_score(cmap[c]).get("score", 0) > 0
    )
    print(f"{name} honeypots in top-100: {hp}")

print(f"\nIter1 runtime: 238.9s")
print(f"Iter2 runtime: 315.0s")
