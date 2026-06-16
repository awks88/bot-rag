"""
search.py — поиск по готовому векторному индексу.
"""

from pathlib import Path
import json
import sys

import faiss
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "index"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_index():
    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    metadata = json.loads((INDEX_DIR / "metadata.json").read_text(encoding="utf-8"))
    return index, metadata

def search(query, index, metadata, k=3):
    model = SentenceTransformer(MODEL_NAME)
    q_vec = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).astype("float32")

    scores, ids = index.search(q_vec, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        chunk = metadata[idx]
        results.append({"score": float(score), **chunk})
    return results


def main():
    index, metadata = load_index()
    queries = sys.argv[1:] or [
        "Who piloted the Stormhawk?",
        "What destroyed the Void Core?",
        "Tell me about the Synth Flux",
    ]
    for query in queries:
        print(f"\n=== Запрос: {query!r} ===")
        for r in search(query, index, metadata, k=3):
            snippet = r["text"].replace("\n", " ")[:200]
            print(f"[{r['score']:.3f}] {r['source']} (чанк {r['chunk_index']})")
            print(f"        {snippet}...")


if __name__ == "__main__":
    main()
