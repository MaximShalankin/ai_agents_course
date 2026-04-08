"""Сравнение квантованной (Q4_K_M) и полной (FP16) модели через Ollama.

Требуется: модели уже добавлены в Ollama (например: ollama create qwen-fp16 -f Modelfile.fp16,
ollama create qwen-q4 -f Modelfile.q4, где Modelfile указывает FROM путь к .gguf).
"""
import json
import sys
import time
from pathlib import Path

import ollama

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"

MODELS = [
    ("qwen-fp16", "FP16 (Обычная)"),
    ("qwen-q4", "Int4 (Квантованная)"),
]


def _ollama_model_names():
    """Список имён моделей, доступных в Ollama (name из list, без :tag при наличии)."""
    try:
        resp = ollama.list()
        names = set()
        for m in resp.get("models", []):
            name = m.get("name") or m.get("model", "")
            if ":" in name:
                name = name.split(":")[0]
            names.add(name)
        return names
    except Exception:
        return set()


def _check_models_exist():
    available = _ollama_model_names()
    missing = [model_id for model_id, _ in MODELS if model_id not in available]
    if missing:
        raise SystemExit(
            f"Ollama: модели не найдены: {missing}. Доступны: {sorted(available) or 'нет'}. "
            "Добавьте модели: ollama create qwen-fp16 -f Modelfile.fp16 и т.д."
        )


def compare_models(prompt: str, save_results: bool = True):
    print(f"PROMPT: {prompt}\n")
    print(f"{'Модель':<20} | {'Время (с)':<10} | {'Ток/сек':<10} | {'Ответ (первые 50 симв.)'}")
    print("-" * 85)

    rows = []
    for model_id, label in MODELS:
        start_time = time.time()
        response = ollama.chat(model=model_id, messages=[{"role": "user", "content": prompt}])
        end_time = time.time()
        duration = end_time - start_time

        tokens = response.get("eval_count", 0)
        eval_duration_ns = response.get("eval_duration", 1)
        tps = tokens / (eval_duration_ns / 1e9)

        content = response.get("message", {}).get("content", "")
        content_preview = (content[:50].replace("\n", " ") + "...") if content else ""

        print(f"{label:<20} | {duration:<10.2f} | {tps:<10.2f} | {content_preview}")
        rows.append({"model": label, "duration_sec": duration, "tokens_per_sec": tps, "preview": content_preview})

    if save_results and rows:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        path = ARTIFACTS_DIR / "compare_results.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {path}")
    return rows


def main():
    _check_models_exist()
    prompt_text = "Напиши функцию на Python для вычисления чисел Фибоначчи."
    compare_models(prompt_text)


if __name__ == "__main__":
    main()
