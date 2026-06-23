import csv, sys, os


def validate_submission(path):
    errors = []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    expected_headers = {"candidate_id", "rank", "score", "reasoning"}
    actual_headers = set(rows[0].keys()) if rows else set()
    if actual_headers != expected_headers:
        errors.append(f"Headers: got {actual_headers}, expected {expected_headers}")

    if len(rows) != 100:
        errors.append(f"Row count: {len(rows)}, expected 100")

    ranks = [int(r["rank"]) for r in rows]
    if ranks != list(range(1, 101)):
        errors.append("Ranks must be 1-100 in order")

    ids = [r["candidate_id"] for r in rows]
    if len(set(ids)) != 100:
        errors.append("Duplicate candidate IDs found")

    cid_prefix_ok = all(c.startswith("CAND_") for c in ids)
    if not cid_prefix_ok:
        errors.append("candidate_id must start with CAND_")

    for r in rows:
        try:
            float(r["score"])
        except ValueError:
            errors.append(f"Invalid score for {r['candidate_id']}: {r['score']}")

    return errors


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "submission.csv"
    errors = validate_submission(path)
    if errors:
        print(f"Validation failed ({len(errors)} issue(s)):\n")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("Submission is valid.")


if __name__ == "__main__":
    main()
