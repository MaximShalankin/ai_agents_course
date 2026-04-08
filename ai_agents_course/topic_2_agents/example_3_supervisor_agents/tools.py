"""Инструменты для coder и tester агентов в мультиагентном супервизоре."""
import os
import re
import subprocess
import tempfile

from langchain_core.tools import tool
from langchain_ollama import ChatOllama

CODING_MODEL = "qwen2.5-coder:1.5b"  # gpt-oss:20b
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOllama(model=CODING_MODEL, temperature=0)
    return _llm


def _extract_code(text: str) -> str:
    """Извлекает первый блок кода из ```python ... ``` или ``` ... ```."""
    text = (text or "").strip()
    for pattern in (r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return text


def _run_python_code(code: str, timeout_sec: int = 5) -> str:
    if not code or not code.strip():
        return "ERROR: empty code"
    fd, path = tempfile.mkstemp(suffix=".py", prefix="example_3_supervisor_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
        result = subprocess.run(
            ["python3", path],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            cwd=tempfile.gettempdir(),
        )
        if result.returncode == 0:
            return "OK: " + (result.stdout or "").strip()
        return "ERROR: " + (result.stderr or result.stdout or "non-zero exit").strip()
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    except Exception as e:
        return "ERROR: " + str(e)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


@tool
def write_python_code(task: str) -> str:
    """Генерирует Python-код для решения описанной задачи. Вход: описание задачи (task)."""
    try:
        llm = _get_llm()
        prompt = (
            f"Дай только Python-код (одну функцию или скрипт) для задачи: {task}. "
            "Без пояснений, только исполняемый код."
        )
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response)) or ""
        code = _extract_code(content)
        if not code.strip():
            return "ERROR: модель не вернула код"
        return code
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"


@tool
def run_tests(code: str) -> str:
    """Запускает переданный Python-код и возвращает результат (OK или ERROR). Вход: строка с кодом."""
    return _run_python_code(code)
