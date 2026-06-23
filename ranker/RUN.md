# Redrob Candidate Ranker

## Pre-computation (embeddings)
```
python precompute.py --candidates ../candidates.jsonl
```

## Ranking
```
python rank.py --candidates ../candidates.jsonl --embed-dir ../embeddings.npy --out ../submission.csv
```

## Validation
```
python validate.py <submission.csv>
```
