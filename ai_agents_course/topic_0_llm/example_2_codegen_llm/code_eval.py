"""Проверка синтаксиса сгенерированного Python-кода через ast.parse."""
from __future__ import annotations

import ast
import re


def is_valid_python(code: str) -> bool:
    """Возвращает True, если код синтаксически корректен. При SyntaxError пробрасывает исключение."""
    try:
        ast.parse(code)
        return True
    except SyntaxError as e:
        raise e


def extract_code_block(text: str):
    """Извлекает первый блок кода из ```python ... ``` или ``` ... ``` в ответе LLM.
    Если маркера нет — берёт текст до первого ``` (модель может выдать код без открывающего fence)."""
    if not text or not text.strip():
        return None
    for pattern in (r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    if "```" in text:
        before_fence = text.split("```")[0].strip()
        if before_fence:
            return before_fence
    return text.strip() or None
