"""Benchmark scoring throughput on a configurable sample."""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scoring import compute_candidate_score


def main():
    sample_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..",
        "[PUB] India_runs_data_and_ai_challenge",
        "India_runs_data_and_ai_challenge",
        "sample_candidates.json",
    )

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    with open(sample_path, encoding="utf-8") as f:
        all_candidates = json.load(f)
    candidates = all_candidates[:limit]
    print(f"Candidates: {len(candidates)}")

    t0 = time.perf_counter()
    for i, c in enumerate(candidates):
        r = compute_candidate_score(c, None, None, use_embeddings=False)
        if (i + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed
            print(f"  {i+1}/{len(candidates)} ({elapsed:.1f}s, {rate:.0f}/s)")
    elapsed = time.perf_counter() - t0
    rate = len(candidates) / elapsed
    print(f"\n{len(candidates)} candidates in {elapsed:.1f}s ({rate:.0f}/s)")
    print(f"Projected 100K: {100000 / rate:.0f}s ({100000 / rate / 60:.1f}min)")


if __name__ == "__main__":
    main()
