import csv, json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
random.seed(42)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RANKER_DIR = os.path.join(SCRIPT_DIR, "..")
PROJECT_DIR = os.path.join(RANKER_DIR, "..")

def find_candidates():
    paths = [
        os.path.join(PROJECT_DIR, "candidates.jsonl"),
        os.path.join(RANKER_DIR, "candidates.jsonl"),
        "candidates.jsonl",
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("candidates.jsonl not found")

def find_submission():
    paths = [
        os.path.join(RANKER_DIR, "submission.csv"),
        os.path.join(PROJECT_DIR, "submission.csv"),
        "submission.csv",
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("submission.csv not found")

def load_candidates(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def load_submission(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def get_candidate_map(candidates):
    return {c["candidate_id"]: c for c in candidates}

TRAP_TITLES = {
    "marketing manager", "graphic designer", "hr manager",
    "accountant", "customer support", "content writer",
    "sales executive", "operations manager", "civil engineer",
    "mechanical engineer", "business analyst",
}

candidates = load_candidates(find_candidates())
submission = load_submission(find_submission())
cmap = get_candidate_map(candidates)

# Stratified sampling
top25 = [r["candidate_id"] for r in submission[:25]]
mid30 = [r["candidate_id"] for r in submission[25:150] if r not in top25][:30]
top150_ids = {r["candidate_id"] for r in submission[:150]}
rest = [c["candidate_id"] for c in candidates if c["candidate_id"] not in top150_ids]
random10 = random.sample(rest, min(10, len(rest)))

# Traps: find off-domain titles outside top 100
selected_ids = set(top25) | set(mid30) | set(random10)

traps = []
for c in candidates:
    if c["candidate_id"] in selected_ids:
        continue
    title = c.get("profile", {}).get("current_title", "").lower()
    if any(t in title for t in TRAP_TITLES):
        traps.append(c["candidate_id"])
        if len(traps) >= 5:
            break

selected = {cid: "top25" for cid in top25}
selected.update({cid: "mid" for cid in mid30})
selected.update({cid: "random" for cid in random10})
selected.update({cid: "trap" for cid in traps})

# Build CSV
ROWS = []
for cid, stratum in selected.items():
    c = cmap[cid]
    p = c.get("profile", {})
    career = c.get("career_history", [])
    current_role = next((e for e in career if e.get("is_current")), career[-1] if career else {})
    ROWS.append({
        "candidate_id": cid,
        "current_title": p.get("current_title", ""),
        "current_company": p.get("current_company", ""),
        "years_of_experience": p.get("years_of_experience", ""),
        "location": p.get("location", ""),
        "summary": (p.get("summary", "") or "").replace('"', "'"),
        "current_role_description": (current_role.get("description", "") or "").replace('"', "'"),
        "label_0to3": "",
        "_stratum": stratum,
    })

FIELDS = ["candidate_id", "current_title", "current_company", "years_of_experience",
          "location", "summary", "current_role_description", "label_0to3"]
with open(os.path.join(RANKER_DIR, "labeling_sheet.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS)
    w.writeheader()
    for r in ROWS:
        w.writerow({k: r[k] for k in FIELDS})

print(f"Wrote {len(ROWS)} candidates to labeling_sheet.csv")
print(f"  Top 25: {len(top25)}")
print(f"  Ranks 26-150: {len(mid30)}")
print(f"  Random rest: {len(random10)}")
print(f"  Traps: {len(traps)}")
