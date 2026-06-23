import argparse
import json
import pickle
import os
import sys
import hashlib

from semantic import load_model, embed_texts, build_candidate_text, EMBEDDING_DIM
import numpy as np


def load_candidates(path):
    candidates = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    return candidates


def compute_embeddings(candidates, batch_size=64):
    model = load_model()
    texts = [build_candidate_text(c) for c in candidates]
    print(f"Computing embeddings for {len(texts)} candidates...")
    embeddings = embed_texts(model, texts, batch_size=batch_size)
    print(f"Done. Shape: {embeddings.shape}")
    return embeddings


def save_embeddings(embeddings, output_dir, candidates_path):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "candidate_embeddings.npy")
    np.save(path, embeddings)
    print(f"Saved embeddings: {path}")

    with open(os.path.join(output_dir, "candidate_ids.pkl"), "wb") as f:
        ids = []
        with open(candidates_path, "r", encoding="utf-8") as cf:
            for line in cf:
                line = line.strip()
                if line:
                    ids.append(json.loads(line)["candidate_id"])
        pickle.dump(ids, f)
    print(f"Saved {len(ids)} candidate IDs to pickle")

    metadata = {
        "model": "all-MiniLM-L6-v2",
        "dim": EMBEDDING_DIM,
        "num_candidates": embeddings.shape[0],
        "source": candidates_path,
    }
    with open(os.path.join(output_dir, "embedding_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata")


def main():
    parser = argparse.ArgumentParser(
        description="Precompute candidate embeddings"
    )
    parser.add_argument(
        "--candidates",
        default=None,
        help="Path to candidates.jsonl. Auto-detects if not provided.",
    )
    parser.add_argument(
        "--output",
        default="embeddings",
        help="Directory to save embeddings (default: ./embeddings)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding (default: 64)",
    )
    args = parser.parse_args()

    if args.candidates is None:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        candidates_path = os.path.join(data_dir, "candidates.jsonl")
    else:
        candidates_path = args.candidates

    if not os.path.exists(candidates_path):
        print(f"Error: candidates file not found: {candidates_path}")
        sys.exit(1)

    candidates = load_candidates(candidates_path)
    print(f"Loaded {len(candidates)} candidates from {candidates_path}")

    embeddings = compute_embeddings(candidates, args.batch_size)
    save_embeddings(embeddings, args.output, candidates_path)


if __name__ == "__main__":
    main()
