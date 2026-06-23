import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scoring import compute_candidate_score, _build_reasoning
from features import extract_features
from scoring import _compute_sub_scores, _compute_fit_score, _compute_availability_multiplier


def main():
    samples_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..",
        "candidates.jsonl",
    )
    samples = []
    with open(samples_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    samples = samples[:50]

    results = []
    for c in samples:
        result = compute_candidate_score(c, None, None, use_embeddings=False)
        results.append(result)

    results.sort(key=lambda r: (-r["score"], r["candidate_id"]))

    print("Top 10 by score:")
    print("-" * 100)
    for r in results[:10]:
        print(f"{r['candidate_id']} | {r['score']:.4f} | {r['reasoning'][:120]}")

    print("\n\nBottom 5:")
    print("-" * 100)
    for r in results[-5:]:
        print(f"{r['candidate_id']} | {r['score']:.4f} | {r['reasoning'][:120]}")


if __name__ == "__main__":
    main()
