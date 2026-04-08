"""Demo: search by user_prompt over video metadata (RAG), output timecode."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import ollama
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
METADATA_PATH = ARTIFACTS_DIR / "video_metadata.json"
BATCH_SIZE = 100
TOP_K = 3


def get_embeddings(texts, model="bge-m3", batch_size=BATCH_SIZE):
    if isinstance(texts, str):
        texts = [texts]
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = ollama.embed(model=model, input=batch)
        out.append(np.array(resp["embeddings"], dtype=np.float32))
    return np.vstack(out)


def sec_to_mmss(sec: float) -> str:
    m = int(sec) // 60
    s = int(sec) % 60
    return f"{m}:{s:02d}"


def main():
    parser = argparse.ArgumentParser(description="Поиск по видео: user_prompt -> timecode.")
    parser.add_argument("query", nargs="*", help="Поисковый запрос (или передайте одной строкой).")
    parser.add_argument("-k", type=int, default=1, help="Сколько кандидатов вывести (по умолчанию 1).")
    args = parser.parse_args()

    user_prompt = " ".join(args.query).strip()
    if not user_prompt:
        print("Укажите запрос: python demo_search.py <user_prompt>", file=sys.stderr)
        sys.exit(1)

    if not METADATA_PATH.exists():
        print(f"Файл метаданных не найден: {METADATA_PATH}", file=sys.stderr)
        print("Сначала запустите: python build_metadata.py", file=sys.stderr)
        sys.exit(1)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entries", data) if isinstance(data, dict) else data
    if not entries:
        print("Метаданные пусты.", file=sys.stderr)
        sys.exit(1)

    texts = [f"[{e['type']}] {e['text']}" for e in entries]
    chunk_emb = get_embeddings(texts)
    query_emb = get_embeddings(user_prompt)

    sims = cosine_similarity(query_emb, chunk_emb).squeeze()
    top_indices = np.argsort(-sims)[: max(args.k, 1)]

    for rank, idx in enumerate(top_indices, 1):
        e = entries[idx]
        mid = (e["start_sec"] + e["end_sec"]) / 2
        print(f"timecode_sec: {int(mid)}")
        print(f"timecode: {sec_to_mmss(mid)}")
        if args.k > 1:
            snippet = (e["text"] or "")[:80]
            if len(e.get("text", "")) > 80:
                snippet += "..."
            print(f"match: [{e['type']}] {snippet}")
        if rank < len(top_indices):
            print("---")


if __name__ == "__main__":
    main()
