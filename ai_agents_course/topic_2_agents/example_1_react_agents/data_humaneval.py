"""Загрузка openai/openai_humaneval с Hugging Face, выбор 20 случайных сэмплов для агентов."""
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def load_humaneval_samples(n: int = 20, seed: int = 42):
    """Загружает датасет openai/openai_humaneval и возвращает n случайных примеров.
    Каждый элемент — dict с ключами: task_id, prompt, test, entry_point, canonical_solution."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets") from None
    ds = load_dataset("openai/openai_humaneval", split="test")
    if n is not None and len(ds) > n:
        ds = ds.shuffle(seed=seed).select(range(n))
    return [
        {
            "task_id": row["task_id"],
            "prompt": row["prompt"],
            "test": row["test"],
            "entry_point": row["entry_point"],
            "canonical_solution": row.get("canonical_solution", ""),
        }
        for row in ds
    ]
