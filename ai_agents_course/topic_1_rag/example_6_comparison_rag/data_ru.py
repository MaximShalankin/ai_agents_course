"""Загрузка mirage-bench/ru: парсинг prompt -> вопрос, контексты, эталон для метрик."""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.common import MIRAGE_BENCH_DIR

RU_DIR = MIRAGE_BENCH_DIR / "ru"
DEV_SMALL_GLOB = "dev.small-*-of-*.parquet"


def _parse_prompt(prompt: str):
    q_start = prompt.find("Question:") + len("Question:")
    ctx_start = prompt.find("Contexts:")
    inst_start = prompt.find("Instruction:")
    if ctx_start == -1 or inst_start == -1:
        return None, None
    question = prompt[q_start:ctx_start].strip()
    contexts_block = prompt[ctx_start + len("Contexts:") : inst_start].strip()
    return question, contexts_block


def _extract_passages(contexts_block: str):
    """Возвращает список (doc_id, text)."""
    parts = re.split(r"\n(?=\[\d)", contexts_block)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        match = re.match(r"\[([^\]]+)\]\s*", p)
        if match:
            doc_id = match.group(1)
            text = p[match.end() :].strip()
            out.append((doc_id, text))
    return out


def load_ru_dev_small(n=None, seed=42):
    """Загружает dev.small ru. n: если задано, берётся n случайных примеров (для скорости)."""
    files = sorted(RU_DIR.glob(DEV_SMALL_GLOB))
    if not files:
        raise FileNotFoundError(f"No {DEV_SMALL_GLOB} in {RU_DIR}")
    df = pd.read_parquet(files[0])
    rows = []
    for _, r in df.iterrows():
        q, ctx_block = _parse_prompt(r["prompt"])
        if q is None:
            continue
        passages = _extract_passages(ctx_block)
        context_ref = "\n\n".join(t for _, t in passages)
        rows.append({"query_id": r["query_id"], "question": q, "context_ref": context_ref, "passages": passages})
    out = pd.DataFrame(rows)
    if n is not None and len(out) > n:
        out = out.sample(n=n, random_state=seed).reset_index(drop=True)
    return out


def build_corpus(rows_df):
    """Из DataFrame с колонкой passages строит списки ids и texts (уникальные по id)."""
    id_to_text = {}
    for passages in rows_df["passages"]:
        for doc_id, text in passages:
            id_to_text[doc_id] = text
    ids = list(id_to_text.keys())
    texts = [id_to_text[i] for i in ids]
    return ids, texts
