"""
build_index.py — строит векторный индекс базы знаний.
"""

from pathlib import Path
import json
import time

import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "knowledge_base"
OUT_DIR = ROOT / "index"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def load_documents(kb_dir: Path) -> list[dict]:
    docs = []
    for path in sorted(kb_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        title = lines[0].lstrip("# ").strip() if lines else path.stem
        docs.append({"source": path.name, "title": title, "text": text})
    return docs


def split_into_chunks(docs: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in docs:
        for i, piece in enumerate(splitter.split_text(doc["text"])):
            chunks.append({
                "id": f"{doc['source']}::{i}",
                "source": doc["source"],
                "title": doc["title"],
                "chunk_index": i,
                "text": piece,
            })
    return chunks


def main():
    t0 = time.time()
    OUT_DIR.mkdir(exist_ok=True)

    print("1) Читаю документы...")
    docs = load_documents(KB_DIR)
    print(f"   документов: {len(docs)}")

    print("2) Режу на чанки...")
    chunks = split_into_chunks(docs)
    print(f"   чанков: {len(chunks)}")

    print(f"3) Загружаю модель {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    print("4) Генерирую эмбеддинги...")
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")
    dim = embeddings.shape[1]
    print(f"   получили {embeddings.shape[0]} векторов по {dim} чисел")

    print("5) Строю FAISS-индекс...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(OUT_DIR / "faiss.index"))
    (OUT_DIR / "metadata.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    elapsed = time.time() - t0
    manifest = {
        "model": MODEL_NAME,
        "embedding_dim": dim,
        "num_documents": len(docs),
        "num_chunks": len(chunks),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "build_seconds": round(elapsed, 1),
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nГотово за {elapsed:.1f} c. Индекс лежит в {OUT_DIR}/")

if __name__ == "__main__":
    main()
