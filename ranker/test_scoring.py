import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features import extract_features
from honeypot import compute_honeypot_score
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

    print(f"{'ID':<15} {'Name':<20} {'Title':<25} {'Fit':>6} {'Avail':>6} {'Final':>8} {'Ship':>6} {'Loc':>6} {'Pen':>6}")
    print("-" * 100)

    for c in samples:
        features = extract_features(c)
        sub = _compute_sub_scores(features, 0.5)
        fit = _compute_fit_score(sub)
        avail = _compute_availability_multiplier(features)
        final = fit * avail
        hid = c["candidate_id"]
        name = c["profile"]["anonymized_name"]
        title = c["profile"]["current_title"]
        print(
            f"{hid:<15} {name:<20} {title:<25} "
            f"{fit:>6.3f} {avail:>6.3f} {final:>8.4f} "
            f"{sub['shipped_system_evidence_score']:>6.2f} {sub['location_fit']:>6.2f} {sub['disqualifier_penalty_sum']:>6.2f}"
        )

    print("\nHoneypot checks:")
    for c in samples:
        hp = compute_honeypot_score(c)
        if hp["checks"]:
            print(f"  {c['candidate_id']} ({c['profile']['anonymized_name']}): score={hp['score']}, suspicious={hp['is_suspicious']}")
            for chk in hp["checks"][:3]:
                print(f"    - {chk}")


if __name__ == "__main__":
    main()
