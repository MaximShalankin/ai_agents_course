"""Запуск мультиагентного супервизора: два задания, шаги из истории сообщений, сохранение в artifacts/."""
import json
import logging
import re
from pathlib import Path

from agent_supervisor import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
TASKS = [
    (
        "Напиши функцию sum_range(n), которая возвращает сумму чисел от 1 до n. "
        "Проверь её: запусти код и убедись, что для n=10 результат 55."
    ),
    (
        "Напиши функцию greet(name), которая возвращает строку 'Hello, {name}!'. "
        "Проверь её: запусти код для name='World'."
    ),
]
STEP_SUMMARY_LEN = 200
RESULT_SUMMARY_LEN = 300


def _summary(text: str, max_len: int) -> str:
    s = (text or "").strip()
    if len(s) <= max_len:
        return s
    return s[:max_len] + "..."


def _steps_from_messages(messages: list) -> list[dict]:
    """Строит список шагов из истории сообщений: user, supervisor (с разбором tool calls), coder/tester по порядку."""
    steps = []
    tool_index = 0
    for m in messages:
        name = type(m).__name__
        content = getattr(m, "content", None) or ""
        content_str = str(content) if content else ""

        if "Human" in name:
            steps.append({"agent": "user", "summary": _summary(content_str, STEP_SUMMARY_LEN)})
        elif "AI" in name:
            if "transfer_to_coder" in content_str or "transfer_to_tester" in content_str:
                if "transfer_to_coder" in content_str:
                    steps.append({"agent": "supervisor", "summary": "transfer_to_coder(task=...)"})
                if "transfer_to_tester" in content_str:
                    steps.append({"agent": "supervisor", "summary": "transfer_to_tester(code=...)"})
            else:
                steps.append({"agent": "supervisor", "summary": _summary(content_str, STEP_SUMMARY_LEN)})
        elif "Tool" in name:
            agent = "coder" if tool_index % 2 == 0 else "tester"
            tool_index += 1
            steps.append({"agent": f"{agent}_result", "summary": _summary(content_str, STEP_SUMMARY_LEN)})
    return steps


def _extract_code_from_messages(messages: list) -> tuple[str | None, str | None]:
    """Извлекает сгенерированный код и результат тестов из истории (ToolMessage или из JSON в AIMessage)."""
    code_from_tool = None
    test_result = None
    tool_index = 0
    for m in messages:
        name = type(m).__name__
        content_str = str(getattr(m, "content", None) or "")
        if "Tool" in name:
            if tool_index == 0:
                code_from_tool = content_str
            else:
                test_result = content_str
            tool_index += 1
    if code_from_tool is not None or test_result is not None:
        return (code_from_tool, test_result)
    for m in messages:
        if "AI" not in type(m).__name__:
            continue
        content_str = str(getattr(m, "content", "") or "")
        if '"code"' not in content_str:
            continue
        match = re.search(r'"code"\s*:\s*"((?:[^"\\]|\\.)*)"', content_str)
        if match:
            code_str = match.group(1).encode().decode("unicode_escape")
            if "def " in code_str or "\n" in code_str or "return " in code_str:
                return (code_str, None)
    return (None, None)


def _raw_messages_preview(messages: list, max_per: int = 400) -> list[dict]:
    """Краткий дамп сообщений для артефакта: тип и превью content."""
    out = []
    for m in messages:
        name = type(m).__name__
        content_str = str(getattr(m, "content", None) or "")
        out.append({"type": name, "content": _summary(content_str, max_per)})
    return out


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("=== Мультиагентная система (Supervisor: coder + tester) ===")
    config = {"recursion_limit": 50}
    tasks_out = []

    for i, task in enumerate(TASKS):
        logger.info("--- Задание %s/%s ---", i + 1, len(TASKS))
        logger.info("Задача: %s", _summary(task, 120))
        result = app.invoke({"messages": [{"role": "user", "content": task}]}, config=config)
        msgs = result.get("messages") or []
        steps = _steps_from_messages(msgs)
        last_content = msgs[-1].content if msgs else ""
        result_summary = _summary(str(last_content), RESULT_SUMMARY_LEN)

        for s in steps:
            logger.info("[%s]: %s", s["agent"].upper(), s["summary"])

        generated_code, test_result = _extract_code_from_messages(msgs)
        raw_preview = _raw_messages_preview(msgs)
        task_out = {
            "task": task,
            "steps": steps,
            "result": {"summary": result_summary},
            "raw_messages": raw_preview,
        }
        if generated_code is not None:
            task_out["generated_code"] = generated_code
        if test_result is not None:
            task_out["test_result"] = test_result
        if generated_code is None and test_result is None:
            task_out["note"] = "Handoffs may not have run (no coder/tester output); supervisor response only."
        tasks_out.append(task_out)

    out = {"tasks": tasks_out}
    out_path = ARTIFACTS_DIR / "supervisor_demo.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    logger.info("Выполнено заданий: %s — %s", len(TASKS), out_path)


if __name__ == "__main__":
    main()
