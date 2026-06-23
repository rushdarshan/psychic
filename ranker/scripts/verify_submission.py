"""Verification script for Redrob Candidate Ranker submission.

Includes runtime benchmark, honeypot audit, and trap-candidate detection.
Checks:
1. Wall-clock time for scoring (rule-only mode)
2. Honeypot rate in the actual top-100 output
3. Keyword-stuffer trap candidates outside top-50

Usage:
  python scripts/verify_submission.py                     # sample data
  python scripts/verify_submission.py --full              # full 100K
  python scripts/verify_submission.py --candidates ./path --submission ./path
"""

import argparse
import csv
import json
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RANKER_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, RANKER_DIR)

DATA_DIR = os.path.join(RANKER_DIR, "..")


def find_file(name, search_dirs):
    for d in search_dirs:
        path = os.path.join(d, name)
        if os.path.exists(path):
            return os.path.normpath(path)
    return None


def load_candidates(path):
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def load_submission(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def check_1_benchmark_runtime(candidates_path, print=print):
    print("=" * 60)
    print("CHECK 1: Benchmark scoring runtime (rule-only)")
    print("=" * 60)

    candidates = load_candidates(candidates_path)
    print(f"Candidates: {len(candidates)}")

    from scoring import compute_candidate_score

    t0 = time.time()
    results = []
    for i, c in enumerate(candidates):
        r = compute_candidate_score(c, None, None, use_embeddings=False)
        results.append(r)
        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            print(f"  {i + 1}/{len(candidates)} ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    rate = len(candidates) / elapsed
    print(f"Total: {elapsed:.1f}s  ({rate:.0f} cand/s)")
    print(f"Budget 300s: {'PASS' if elapsed < 300 else 'FAIL'}")
    return elapsed


def check_2_honeypot_rate(submission_path, candidates_path, print=print):
    print("=" * 60)
    print("CHECK 2: Honeypot rate in top 100")
    print("=" * 60)

    all_candidates = {c["candidate_id"]: c for c in load_candidates(candidates_path)}
    submission = load_submission(submission_path)

    from honeypot import compute_honeypot_score

    flagged = []
    for row in submission:
        cid = row["candidate_id"]
        c = all_candidates.get(cid)
        if c:
            hp = compute_honeypot_score(c)
            if hp["is_suspicious"]:
                flagged.append((cid, row["rank"], hp["score"]))

    rate = len(flagged) / len(submission)
    print(f"Honeypots: {len(flagged)}/{len(submission)} ({rate*100:.1f}%)")
    print(f"<10% threshold: {'PASS' if rate < 0.1 else 'FAIL'}")
    for cid, rank, score in flagged:
        print(f"  {cid} (rank {rank}, score {score:.1f})")
    return rate


def check_3_trap_candidates(submission_path, candidates_path, print=print):
    print("=" * 60)
    print("CHECK 3: Keyword-stuffer trap candidates")
    print("=" * 60)

    TRAP_TITLES = {
        "marketing manager", "graphic designer", "hr manager",
        "accountant", "customer support", "content writer",
        "sales executive", "operations manager", "civil engineer",
        "mechanical engineer", "business analyst",
    }
    AI_KEYWORDS = {
        "machine learning", "deep learning", "nlp", "llm",
        "gpt", "neural network", "tensorflow", "pytorch",
        "artificial intelligence", "data science",
        "bert", "transformer", "rag", "embedding",
    }

    all_candidates = {c["candidate_id"]: c for c in load_candidates(candidates_path)}
    submission = load_submission(submission_path)

    top_50_ids = {row["candidate_id"] for row in submission[:50]}
    top_100_ids = {row["candidate_id"] for row in submission}
    rank_of = {row["candidate_id"]: int(row["rank"]) for row in submission}

    trap_in_top_50 = 0
    keyword_stuffers = []

    for cid, c in all_candidates.items():
        if cid not in top_100_ids:
            continue
        title = c.get("profile", {}).get("current_title", "").lower()
        skill_names = {s.get("name", "").lower() for s in c.get("skills", [])}

        is_off = any(t in title for t in TRAP_TITLES)
        ai_count = len(skill_names & AI_KEYWORDS)

        if is_off:
            r = rank_of.get(cid, 999)
            if r <= 50:
                trap_in_top_50 += 1
            if ai_count >= 3:
                keyword_stuffers.append((cid, r, title, ai_count))

    print(f"Off-domain titles in top 50: {trap_in_top_50}")
    print(f"Keyword-stuffers (off-domain + 3+ AI skills): {len(keyword_stuffers)}")

    if keyword_stuffers:
        for cid, r, title, n in sorted(keyword_stuffers, key=lambda x: x[1]):
            flag = " ** TOP 50 **" if r <= 50 else ""
            print(f"  {cid} (rank {r}): {title} ({n} AI skills){flag}")

    stuffers_ok = all(r > 50 for _, r, _, _ in keyword_stuffers) or not keyword_stuffers
    top50_ok = trap_in_top_50 < 5
    print(f"Stuffers outside top 50: {'PASS' if stuffers_ok else 'WARN'}")
    print(f"Off-domain rate <5/50: {'PASS' if top50_ok else 'WARN'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="Run against full 100K")
    parser.add_argument("--candidates", help="Path to candidates.jsonl")
    parser.add_argument("--submission", help="Path to submission.csv")
    args = parser.parse_args()

    search_dirs = [RANKER_DIR, DATA_DIR, os.getcwd()]

    if args.candidates:
        cand_path = args.candidates
    elif args.full:
        cand_path = find_file("candidates.jsonl", search_dirs)
        if not cand_path:
            print("Full dataset not found", flush=True)
            sys.exit(1)
    else:
        cand_path = find_file("sample_candidates.json", search_dirs)
        if not cand_path:
            cand_path = find_file("candidates.jsonl", search_dirs)
        print(f"Using: {os.path.basename(cand_path)}", flush=True)

    if not cand_path or not os.path.exists(cand_path):
        print("Error: candidates file not found", flush=True)
        sys.exit(1)

    sub_path = args.submission
    if not sub_path:
        sub_path = find_file("submission.csv", search_dirs)

    if not sub_path:
        print("Generating submission from scratch...", flush=True)
        from scoring import compute_candidate_score
        candidates = load_candidates(cand_path)
        results = []
        for i, c in enumerate(candidates):
            r = compute_candidate_score(c, None, None, use_embeddings=False)
            results.append(r)
        results.sort(key=lambda r: (-r["score"], r["candidate_id"]))
        sub_path = os.path.join(RANKER_DIR, "submission.csv")
        with open(sub_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["candidate_id", "rank", "score", "reasoning"])
            for rank, r in enumerate(results[:100], start=1):
                w.writerow([r["candidate_id"], rank, f"{r['score']:.4f}", r["reasoning"]])
        print(f"Wrote: {sub_path}", flush=True)

    print(f"Candidates: {cand_path}", flush=True)
    print(f"Submission: {sub_path}", flush=True)
    print(flush=True)

    t = check_1_benchmark_runtime(cand_path)
    print(flush=True)
    check_3_trap_candidates(sub_path, cand_path)
    print(flush=True)
    check_2_honeypot_rate(sub_path, cand_path)

    print("\n" + "=" * 60, flush=True)
    print(f"SUMMARY: runtime {t:.1f}s, all checks complete.", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
