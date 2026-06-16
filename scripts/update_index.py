"""
update_index.py — автоматическое инкрементальное обновление индекса.
"""
from pathlib import Path
import json
import time
import hashlib
from datetime import datetime

import faiss
from sentence_transformers import SentenceTransformer

import build_index as bi

STATE_FILE = bi.OUT_DIR / "index_state.json"
LOG_FILE = bi.ROOT / "logs" / "update.log"


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {}


def save_state(files: dict) -> None:
    STATE_FILE.write_text(
        json.dumps({"files": files}, ensure_ascii=False, indent=2), encoding="utf-8")


def log(msg: str) -> None:
    LOG_FILE.parent.mkdir(exist_ok=True)
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def add_new_documents(new_files: list[str]) -> int:
    index = faiss.read_index(str(bi.OUT_DIR / "faiss.index"))
    metadata = json.loads((bi.OUT_DIR / "metadata.json").read_text(encoding="utf-8"))

    docs = [d for d in bi.load_documents(bi.KB_DIR) if d["source"] in new_files]
    chunks = bi.split_into_chunks(docs)

    model = SentenceTransformer(bi.MODEL_NAME)
    emb = model.encode([c["text"] for c in chunks],
                       normalize_embeddings=True, convert_to_numpy=True).astype("float32")

    index.add(emb) 
    metadata.extend(chunks) 

    faiss.write_index(index, str(bi.OUT_DIR / "faiss.index"))
    (bi.OUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(chunks)


def main():
    t0 = time.time()
    log("=== запуск обновления ===")
    try:
        current = {p.name: file_hash(p) for p in sorted(bi.KB_DIR.glob("*.md"))}
        previous = load_state().get("files", {})

        if not previous:
            log("нет состояния -> полная пересборка (базовая точка)")
            bi.main()
            save_state(current)
            index = faiss.read_index(str(bi.OUT_DIR / "faiss.index"))
            log(f"index initialized: {len(current)} files, total chunks={index.ntotal}, "
                f"{time.time() - t0:.1f}s, 0 errors")
            return

        new     = sorted(f for f in current if f not in previous)
        changed = sorted(f for f in current if f in previous and current[f] != previous[f])
        deleted = sorted(f for f in previous if f not in current)

        if not (new or changed or deleted):
            log("изменений нет, индекс актуален. 0 errors")
            return

        if changed or deleted:
            log(f"изменены: {changed or '—'}, удалены: {deleted or '—'} -> полная пересборка")
            bi.main()
            added = 0
            mode = "rebuild"
        else:
            added = add_new_documents(new)
            mode = "incremental"

        save_state(current)
        index = faiss.read_index(str(bi.OUT_DIR / "faiss.index"))
        log(f"index updated ({mode}): {len(new)} added, {len(changed)} changed, "
            f"{len(deleted)} removed, +{added} chunks, total chunks={index.ntotal}, "
            f"{time.time() - t0:.1f}s, 0 errors")
    except Exception as e:
        log(f"ОШИБКА: {e}")
        raise


if __name__ == "__main__":
    main()
