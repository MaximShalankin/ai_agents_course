"""Точка входа: загрузить 2–3 сэмпла HumanEval, запустить линейный workflow, вывести результат и сохранить в artifacts/."""
import json
import logging
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

from data import load_humaneval_samples
from workflow_lcel import run_linear_workflow, MODEL, NUM_STEPS

NUM_SAMPLES = 3
SEED = 42
ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
STEP_LABELS = ["analysis", "plan", "code"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _extract_code(raw: str) -> str:
    """Убирает обёртку ```python ... ``` из ответа LLM."""
    raw = raw.strip()
    m = re.search(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL)
    return m.group(1).strip() if m else raw


def _run_humaneval_check(prompt: str, solution_code: str, test_code: str, entry_point: str) -> tuple[bool, Optional[str]]:
    """Выполняет тесты HumanEval. Возвращает (passed, error_message)."""
    full_source = prompt.strip() + "\n" + solution_code
    namespace = {"__name__": "__main__"}
    try:
        exec(full_source, namespace)
    except Exception as e:
        return False, f"exec solution: {type(e).__name__}: {e}"
    fn = namespace.get(entry_point)
    if fn is None:
        return False, f"entry_point '{entry_point}' not found in namespace"
    test_ns = {"check": None}
    try:
        exec(test_code, test_ns)
    except Exception as e:
        return False, f"exec test: {type(e).__name__}: {e}"
    check = test_ns.get("check")
    if check is None:
        return False, "check() not defined in test"
    try:
        check(fn)
        return True, None
    except AssertionError as e:
        return False, f"assertion: {e}"
    except Exception as e:
        return False, f"check(candidate): {type(e).__name__}: {e}"


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    logger.info("Loading HumanEval samples: n=%s, seed=%s", NUM_SAMPLES, SEED)
    samples = load_humaneval_samples(n=NUM_SAMPLES, seed=SEED)
    logger.info("Loaded %s tasks", len(samples))
    results: list[bool] = []
    for i, sample in enumerate(samples):
        task_id = sample["task_id"]
        prompt = sample["prompt"]
        logger.info("Task %s/%s: %s", i + 1, len(samples), task_id)
        print(f"\n{'='*60}\nTask {i+1}/{len(samples)}: {task_id}\n{'='*60}")
        result = run_linear_workflow(prompt, verbose=True)
        logger.info("Workflow finished for %s", task_id)
        print("Final output (code):")
        print(result["final"][:800] + ("..." if len(result["final"]) > 800 else ""))
        code = _extract_code(result["final"])
        passed, err = _run_humaneval_check(
            prompt, code, sample["test"], sample["entry_point"]
        )
        logger.info("Task %s: passed=%s%s", task_id, passed, f" error={err}" if err else "")
        steps_with_labels = dict(zip(STEP_LABELS, result["steps"]))
        meta = {
            "task_id": task_id,
            "task_index": i + 1,
            "total_tasks": len(samples),
            "run_ts": run_ts,
            "model": MODEL,
            "num_steps": NUM_STEPS,
            "seed": SEED,
            "num_samples": NUM_SAMPLES,
            "correct": passed,
        }
        if err is not None:
            meta["error"] = err
        out = {
            "meta": meta,
            "input": {
                "prompt": prompt,
                "entry_point": sample["entry_point"],
                "test": sample["test"],
                "canonical_solution": sample.get("canonical_solution", ""),
            },
            "steps": steps_with_labels,
            "steps_raw": result["steps"],
            "final": result["final"],
        }
        out_path = ARTIFACTS_DIR / f"{task_id.replace('/', '_')}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        logger.info("Saved to %s (correct=%s)", out_path, passed)
        print(f"\nSaved to {out_path} [correct={passed}]")
        results.append(passed)

    total = len(results)
    passed_count = sum(results)
    score_str = f"{passed_count}/{total}"
    logger.info("Final score: %s", score_str)
    print(f"\n{'='*60}\nИтоговая оценка: {score_str} ({passed_count} из {total} задач пройдено)\n{'='*60}")


if __name__ == "__main__":
    main()
