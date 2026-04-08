"""Общий инструмент: выполнение Python-кода в subprocess с таймаутом. OK/ERROR/TIMEOUT."""
from __future__ import annotations

import os
import re
import subprocess
import tempfile

from langchain_core.tools import tool


def extract_code_block(text: str) -> str | None:
    """Извлекает первый блок кода из ```python ... ``` или ``` ... ```. Общий парсер для обоих агентов."""
    for pattern in (r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


def run_python_code(code: str, timeout_sec: int = 5) -> str:
    """Выполняет код в subprocess. Возвращает 'OK: <stdout>' или 'ERROR: <stderr>' или 'TIMEOUT'."""
    if not code or not code.strip():
        return "ERROR: empty code"
    fd, path = tempfile.mkstemp(suffix=".py", prefix="example_1_react_")
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


def run_humaneval_test(prompt: str, completion: str, test: str, entry_point: str, timeout_sec: int = 5) -> bool:
    """Собирает prompt+completion+test, вызывает check(entry_point). True если exit 0."""
    if not test or not entry_point:
        return False
    completion = (completion or "").rstrip()
    if not completion:
        return False
    if completion.lstrip().startswith("def "):
        full = completion + "\n\n" + test.strip() + "\n\ncheck(" + entry_point + ")\n"
    else:
        full = prompt.rstrip() + "\n" + completion + "\n\n" + test.strip() + "\n\ncheck(" + entry_point + ")\n"
    out = run_python_code(full, timeout_sec=timeout_sec)
    return out.startswith("OK:")


@tool
def run_python_code_tool(code: str) -> str:
    """Run Python code in a subprocess and return the result.
    Use this to check if code is executable. Input: raw Python source code as a string.
    Returns 'OK: <stdout>' on success, 'ERROR: <stderr>' on failure, or 'TIMEOUT' if it runs too long."""
    return run_python_code(code)
