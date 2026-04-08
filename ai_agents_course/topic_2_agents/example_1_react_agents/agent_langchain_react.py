"""ReAct-агент на LangChain: тот же сценарий, что и custom — код в ```python```, Final Answer. Только ChatOllama + tool.invoke()."""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from tools import run_python_code_tool, extract_code_block

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

MODEL = "qwen2.5-coder:1.5b"
MAX_STEPS = 10

# Тот же по смыслу системный промпт, что и у custom-агента
SYSTEM_PROMPT = """You are a coding agent. For each task you must:
1. Think (Thought) about what code to write.
2. Use the only available action: run_python_code. You must output your Python code inside a markdown code block like this:
```python
# your code here
```
3. After seeing the Observation (OK or ERROR from running the code), either fix the code and output another code block, or give a final answer.

When you are done and the code runs successfully (or you give up), reply with "Final Answer:" followed by your summary. Do not output a code block after Final Answer."""


def _has_final_answer(text: str) -> bool:
    return "final answer:" in (text or "").lower().strip() or "Final Answer:" in (text or "")


def run_langchain_react(task: str, verbose: bool = True):
    """Ручной ReAct-цикл: ChatOllama.invoke(messages) -> парсим блок кода -> tool.invoke(code) -> Observation в историю."""
    llm = ChatOllama(model=MODEL)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Task:\n{task}"),
    ]
    final_answer = ""
    last_code = ""
    last_observation = None
    n_steps = 0

    for step in range(MAX_STEPS):
        response = llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)
        content = (content or "").strip()

        if verbose:
            print(f"[Step {step + 1}]\n{content[:500]}{'...' if len(content) > 500 else ''}\n")

        code = extract_code_block(content)
        if code:
            last_code = code
            obs = run_python_code_tool.invoke({"code": code})
            last_observation = obs
            n_steps += 1
            if verbose:
                print(f"Observation: {obs[:300]}{'...' if len(obs) > 300 else ''}\n")
            messages.append(AIMessage(content=content))
            messages.append(HumanMessage(content=f"Observation: {obs}\n\nContinue or reply with Final Answer: ..."))
            continue

        if _has_final_answer(content):
            final_answer = content
            break

        messages.append(AIMessage(content=content))
        messages.append(HumanMessage(content="If done, reply with 'Final Answer: <summary>'. Otherwise output a Python code block to run."))
    else:
        final_answer = content if content else ""

    executable = last_observation is not None and last_observation.strip().startswith("OK:")
    return {
        "final_answer": final_answer,
        "steps": [],
        "last_code": last_code,
        "executable": executable,
        "n_steps": n_steps,
    }
