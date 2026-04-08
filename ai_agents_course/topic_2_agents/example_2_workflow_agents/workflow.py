"""Линейный LLM workflow: 3 шага (анализ -> план -> код) без инструментов. Ollama qwen2.5-coder:1.5b."""
import ollama

MODEL = "qwen2.5-coder:1.5b"
NUM_STEPS = 3

STEP_1_PROMPT = """Programming task:

{task}

Analyze this task briefly: what should the function do? List the requirements in a few bullet points."""

STEP_2_PROMPT = """Programming task:

{task}

Analysis from previous step:
{analysis}

Based on the task and analysis, write a short implementation plan (steps or pseudocode). Be concise."""

STEP_3_PROMPT = """Programming task:

{task}

Analysis:
{analysis}

Plan:
{plan}

Now write the Python code that implements this. Output only the code (function body or full snippet), no extra explanation."""


def run_linear_workflow(task: str, verbose: bool = False):
    """Цепочка из 3 вызовов LLM: анализ -> план -> код. Возвращает dict с ключами steps (list[str]) и final (str)."""
    steps = []
    analysis = ""
    plan = ""

    # Step 1: analyze
    prompt1 = STEP_1_PROMPT.format(task=task)
    r1 = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt1}])
    analysis = (r1.get("message") or {}).get("content") or ""
    steps.append(analysis)
    if verbose:
        print("[Step 1 - Analysis]\n", analysis[:400], "\n" if len(analysis) > 400 else "\n")

    # Step 2: plan
    prompt2 = STEP_2_PROMPT.format(task=task, analysis=analysis)
    r2 = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt2}])
    plan = (r2.get("message") or {}).get("content") or ""
    steps.append(plan)
    if verbose:
        print("[Step 2 - Plan]\n", plan[:400], "\n" if len(plan) > 400 else "\n")

    # Step 3: code
    prompt3 = STEP_3_PROMPT.format(task=task, analysis=analysis, plan=plan)
    r3 = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt3}])
    final = (r3.get("message") or {}).get("content") or ""
    steps.append(final)
    if verbose:
        print("[Step 3 - Code]\n", final[:500], "\n" if len(final) > 500 else "\n")

    return {"steps": steps, "final": final}
