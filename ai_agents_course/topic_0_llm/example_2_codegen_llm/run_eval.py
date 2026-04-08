"""Оценка кодогенерации: промпт -> LLM -> извлечение кода -> проверка синтаксиса -> метрики."""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import ollama

from code_eval import extract_code_block, is_valid_python
from data import load_humaneval_samples

ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
MODEL_ID = "qwen-q4"
N_SAMPLES = 5


def _check_model_exists():
    try:
        resp = ollama.list()
        names = set()
        for m in resp.get("models", []):
            name = m.get("name") or m.get("model", "")
            if ":" in name:
                name = name.split(":")[0]
            names.add(name)
        if MODEL_ID not in names:
            raise SystemExit(
                f"Ollama: модель '{MODEL_ID}' не найдена. Доступны: {sorted(names) or 'нет'}. "
                "Добавьте модель в Ollama (например ollama create qwen-q4 -f Modelfile.q4)."
            )
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(f"Ollama: не удалось получить список моделей: {e}") from e


def main():
    _check_model_exists()
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    tasks = load_humaneval_samples(n=N_SAMPLES)
    print(f"Loaded {len(tasks)} samples. Model: {MODEL_ID}. Evaluating syntax validity.\n")

    rows = []
    for i, t in enumerate(tasks):
        task_id = t["task_id"]
        prompt = t["prompt"]
        response = ollama.chat(model=MODEL_ID, messages=[{"role": "user", "content": prompt}])
        content = response.get("message", {}).get("content", "")
        code = extract_code_block(content)
        syntax_valid = False
        if code:
            try:
                full_source = (prompt.strip() + "\n" + code).strip()
                syntax_valid = is_valid_python(full_source)
            except SyntaxError:
                pass
        rows.append({"task_id": task_id, "syntax_valid": int(syntax_valid)})
        print(f"  {i + 1}/{len(tasks)} {task_id} syntax_valid={syntax_valid}")

    valid_count = sum(r["syntax_valid"] for r in rows)
    rate = valid_count / len(rows) if rows else 0
    print(f"\nSyntax valid: {valid_count}/{len(rows)} ({rate:.2%})")

    path = ARTIFACTS_DIR / "codegen_metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Saved {path}")

    summary_path = ARTIFACTS_DIR / "codegen_summary.json"
    summary = {"syntax_valid_rate": rate, "syntax_valid_count": valid_count, "total": len(rows)}
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
