import csv
with open("submission.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        r = int(row["rank"])
        cid = row["candidate_id"]
        score = row["score"]
        reason = row["reasoning"][:90]
        print(f"{r:3d} | {cid} | {score} | {reason}")
        if r >= 10:
            break
