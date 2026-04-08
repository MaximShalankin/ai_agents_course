"""Демо LangGraph code-агента (StateGraph + ToolNode) на 3 сэмплах HumanEval."""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from data_humaneval import load_humaneval_samples
from agent_langgraph import run_langgraph_agent
from tools import run_humaneval_test

N_SAMPLES = 3


def main():
    tasks = load_humaneval_samples(n=N_SAMPLES)
    print(f"Loaded {len(tasks)} HumanEval samples. Running LangGraph agent.\n")
    results = []
    for i, t in enumerate(tasks):
        verbose = i == 0
        print(f"=== Task {i + 1}/{len(tasks)} {t['task_id']} ===\nPrompt (first 200 chars):\n{t['prompt'][:200]}...\n")
        out = run_langgraph_agent(t["prompt"], verbose=verbose)
        answer = out["final_answer"]
        print(f"--- Final answer (first 400 chars) ---\n{answer[:400] if answer else '(empty)'}{'...' if len(answer) > 400 else ''}\n")
        passed = run_humaneval_test(t["prompt"], out["last_code"] or "", t["test"], t["entry_point"])
        results.append(passed)
        print(f"[correct={passed}]\n")
    total = len(results)
    passed_count = sum(results)
    print(f"{'='*60}\nИтоговая оценка: {passed_count}/{total} ({passed_count} из {total} задач пройдено)\n{'='*60}")


if __name__ == "__main__":
    main()
