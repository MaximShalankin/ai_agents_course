"""Guardrail-слой: запрет чувствительных имён/значений в аргументах MCP-tool."""
import asyncio
import concurrent.futures

SENSITIVE_KEYS = {"secret", "token", "api_key", "password"}


def run_async_tool(coro):
    """Вызов корутины: из CLI — asyncio.run(); из ipynb — в отдельном потоке."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return pool.submit(asyncio.run, coro).result()


def validate_tool_args(tool_name: str, args: dict) -> None:
    """Проверяет аргументы вызова тула; при нарушении выбрасывает ValueError."""
    for k, v in args.items():
        if any(sk in k.lower() for sk in SENSITIVE_KEYS):
            raise ValueError(
                f"Guardrail: param '{k}' запрещён для MCP tool '{tool_name}'"
            )
        if isinstance(v, str) and any(sk in v.lower() for sk in SENSITIVE_KEYS):
            raise ValueError(
                f"Guardrail: value of '{k}' выглядит как секрет для '{tool_name}'"
            )


def guarded_call(tool, name: str, args: dict):
    """Вызывает tool после validate_tool_args (MCP-тулы только ainvoke — запуск через run_async_tool)."""
    validate_tool_args(name, args)
    if hasattr(tool, "ainvoke"):
        return run_async_tool(tool.ainvoke(args))
    if hasattr(tool, "invoke"):
        return tool.invoke(args)
    return tool(args)
