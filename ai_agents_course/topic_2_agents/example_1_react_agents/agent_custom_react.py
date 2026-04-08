"""Самописный ReAct-агент: цикл Thought -> Action (run_python_code) -> Observation, ollama qwen2.5-coder:1.5b."""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from tools import run_python_code, extract_code_block

import ollama

MODEL = "qwen2.5-coder:1.5b"
MAX_STEPS = 10

SYSTEM_PROMPT = """You are a coding agent. For each task you must:
1. Think (Thought) about what code to write.
2. Use the only available action: run_python_code. You must output your Python code inside a markdown code block like this:
```python
# your code here
```
3. After seeing the Observation (OK or ERROR from running the code), either fix the code and output another code block, or give a final answer.

When you are done and the code runs successfully (or you give up), reply with "Final Answer:" followed by your summary. Do not output a code block after Final Answer."""


def _has_final_answer(text: str) -> bool:
    return "final answer:" in text.lower().strip() or "Final Answer:" in text


def run_custom_react(task: str, max_steps: int = MAX_STEPS, verbose: bool = True):
    """Запускает самописный ReAct на одной задаче.
    Возвращает dict: final_answer, steps, last_code, executable (bool), n_steps (число вызовов run_python)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Task:\n{task}"},
    ]
    steps = []
    final_answer = ""
    last_code = None
    last_observation = None

    for step in range(max_steps):
        response = ollama.chat(model=MODEL, messages=messages)
        content = (response.get("message") or {}).get("content") or ""
        steps.append({"step": step + 1, "response": content})

        if verbose:
            print(f"[Step {step + 1}]\n{content[:500]}{'...' if len(content) > 500 else ''}\n")

        code = extract_code_block(content)
        if code:
            last_code = code
            obs = run_python_code(code)
            last_observation = obs
            steps[-1]["observation"] = obs
            if verbose:
                print(f"Observation: {obs[:300]}{'...' if len(obs) > 300 else ''}\n")
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Observation: {obs}\n\nContinue or reply with Final Answer: ..."})
            continue

        if _has_final_answer(content):
            final_answer = content
            break

        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": "If done, reply with 'Final Answer: <summary>'. Otherwise output a Python code block to run."})
    else:
        final_answer = steps[-1]["response"] if steps else ""

    n_steps = sum(1 for s in steps if "observation" in s)
    executable = last_observation is not None and last_observation.startswith("OK:")
    return {
        "final_answer": final_answer,
        "steps": steps,
        "last_code": last_code or "",
        "executable": executable,
        "n_steps": n_steps,
    }
