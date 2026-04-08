"""Сбор метрик по обоим агентам на сэмплах HumanEval: executable, test_passed, n_steps. Графики сравнения."""
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from data_humaneval import load_humaneval_samples
from agent_custom_react import run_custom_react
from agent_langgraph import run_langgraph_agent
from tools import run_humaneval_test

import pandas as pd

ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
N_SAMPLES = 3


def _log(msg: str):
    ts = time.strftime("%H:%M:%S", time.localtime())
    print(f"[{ts}] {msg}")


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    _log("Start: loading HumanEval samples...")
    tasks = load_humaneval_samples(n=N_SAMPLES)
    _log(f"Loaded {len(tasks)} tasks. Artifacts dir: {ARTIFACTS_DIR}")
    _log("Running both agents (verbose=False).\n")

    rows_custom = []
    rows_langchain = []
    for i, t in enumerate(tasks):
        task_id = t["task_id"]
        prompt = t["prompt"]
        test = t["test"]
        entry_point = t["entry_point"]
        _log(f"--- Task {i + 1}/{len(tasks)} {task_id} ---")

        _log(f"  Custom ReAct: running...")
        t0 = time.perf_counter()
        out_c = run_custom_react(prompt, verbose=False)
        elapsed_c = time.perf_counter() - t0
        test_passed_c = run_humaneval_test(prompt, out_c["last_code"], test, entry_point) if out_c["last_code"] else False
        rows_custom.append({
            "task_id": task_id,
            "executable": int(out_c["executable"]),
            "test_passed": int(test_passed_c),
            "n_steps": out_c["n_steps"],
        })
        _log(f"  Custom ReAct: executable={out_c['executable']}, test_passed={test_passed_c}, n_steps={out_c['n_steps']}, {elapsed_c:.1f}s")

        _log(f"  LangGraph agent: running...")
        t0 = time.perf_counter()
        out_l = run_langgraph_agent(prompt, verbose=False)
        elapsed_l = time.perf_counter() - t0
        test_passed_l = run_humaneval_test(prompt, out_l["last_code"], test, entry_point) if out_l["last_code"] else False
        rows_langchain.append({
            "task_id": task_id,
            "executable": int(out_l["executable"]),
            "test_passed": int(test_passed_l),
            "n_steps": out_l["n_steps"],
        })
        _log(f"  LangGraph agent: executable={out_l['executable']}, test_passed={test_passed_l}, n_steps={out_l['n_steps']}, {elapsed_l:.1f}s\n")

    _log("Building DataFrames and saving JSON artifacts...")
    df_custom = pd.DataFrame(rows_custom)
    df_langchain = pd.DataFrame(rows_langchain)
    df_custom.to_json(ARTIFACTS_DIR / "metrics_custom.json", orient="records", indent=2)
    df_langchain.to_json(ARTIFACTS_DIR / "metrics_langchain.json", orient="records", indent=2)

    mean_c = df_custom[["executable", "test_passed", "n_steps"]].mean()
    mean_l = df_langchain[["executable", "test_passed", "n_steps"]].mean()
    comparison = pd.DataFrame({"custom": mean_c, "langchain": mean_l})
    comparison.to_json(ARTIFACTS_DIR / "metrics_comparison.json", orient="index", indent=2)
    _log("Means (custom / langchain):")
    _log(f"  executable:  {mean_c['executable']:.3f} / {mean_l['executable']:.3f}")
    _log(f"  test_passed: {mean_c['test_passed']:.3f} / {mean_l['test_passed']:.3f}")
    _log(f"  n_steps:     {mean_c['n_steps']:.2f} / {mean_l['n_steps']:.2f}")

    total = len(rows_custom)
    passed_c = sum(r["test_passed"] for r in rows_custom)
    passed_l = sum(r["test_passed"] for r in rows_langchain)
    _log("")
    _log(f"Итоговая оценка: Custom ReAct {passed_c}/{total} , LangGraph {passed_l}/{total}")
    print(f"\n{'='*60}\nИтоговая оценка: Custom ReAct {passed_c}/{total} , LangGraph {passed_l}/{total}\n{'='*60}")

    import matplotlib.pyplot as plt
    _log("Saving comparison plot...")
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, col in zip(axes, ["executable", "test_passed", "n_steps"]):
        comparison.loc[col].plot(kind="bar", ax=ax)
        ax.set_title(col.replace("_", " ").title())
        ax.set_ylabel("Score" if col != "n_steps" else "Count")
        ax.set_xticklabels(["Custom ReAct", "LangGraph"], rotation=25)
    fig.suptitle("Example_7: Custom ReAct vs LangGraph (HumanEval 3 samples)")
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "metrics_comparison.png", dpi=100)
    plt.close()
    _log(f"Done. Artifacts: metrics_custom.json, metrics_langchain.json, metrics_comparison.json, metrics_comparison.png")


if __name__ == "__main__":
    main()
