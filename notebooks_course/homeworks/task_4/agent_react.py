"""
Task 4: ReAct-агент для конвертации валют

Реализует агента через LangGraph: StateGraph, MessagesState, ToolNode.
Инструменты: получение актуальных курсов валют из API.
LLM: OpenRouter
"""
import json
import os
import time
import urllib.request
from typing import Annotated, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict
import dotenv

dotenv.load_dotenv('../../.env')
dotenv.load_dotenv('/.env')

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")  # Requires tool use support
MAX_STEPS = 10

SYSTEM_PROMPT = """You are a currency conversion assistant. Your task is to convert amounts between different currencies.

For each request:
1. Identify the source currency, amount, and target currency
2. Use the get_exchange_rate tool to get the current exchange rate
3. Calculate the conversion
4. Provide the final answer clearly

Always use the tool to get real exchange rates. Do not guess or estimate rates."""


# ============ Инструменты ============


@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    """Get current exchange rate from one currency to another.

    Args:
        from_currency: Source currency code (e.g., "USD", "EUR", "RUB")
        to_currency: Target currency code (e.g., "USD", "EUR", "RUB")

    Returns:
        Exchange rate as float (how many to_currency units per 1 from_currency unit)
    """
    # Используем бесплатный API exchange rates
    url = f"https://api.exchangerate-api.com/v4/latest/{from_currency.upper()}"

    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                rate = data["rates"].get(to_currency.upper())
                if rate is None:
                    raise ValueError(f"Unknown currency: {to_currency}")
                return rate
    except Exception as e:
        raise ValueError(f"Failed to get exchange rate: {e}")


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another using current exchange rates.

    Args:
        amount: The amount to convert
        from_currency: Source currency code (e.g., "USD", "EUR", "RUB")
        to_currency: Target currency code (e.g., "USD", "EUR", "RUB")

    Returns:
        A string describing the conversion result
    """
    rate = get_exchange_rate.invoke({"from_currency": from_currency, "to_currency": to_currency})
    result = amount * rate
    return f"{amount} {from_currency.upper()} = {result:.2f} {to_currency.upper()} (rate: {rate:.4f})"


# ============ LangGraph Agent ============


class State(TypedDict):
    messages: Annotated[list, add_messages]


def _get_tool_call_args(tc: Any) -> dict:
    """Get tool call args: support .args, .arguments, and JSON string."""
    raw = None
    if isinstance(tc, dict):
        raw = tc.get("args") or tc.get("arguments")
    else:
        raw = getattr(tc, "args", None) or getattr(tc, "arguments", None)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _to_messages(raw):
    out = []
    for m in raw:
        if isinstance(m, (AIMessage, HumanMessage, SystemMessage, ToolMessage)):
            out.append(m)
        elif isinstance(m, dict):
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                out.append(HumanMessage(content=content))
            elif role == "assistant":
                out.append(AIMessage(content=content))
            else:
                out.append(HumanMessage(content=content))
        else:
            out.append(HumanMessage(content=str(m)))
    return out


tools = [get_exchange_rate, convert_currency]

# OpenRouter LLM
llm = ChatOpenAI(
    model=OPENROUTER_MODEL,
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)


def agent_node(state: State) -> dict:
    messages = _to_messages(state["messages"])
    full = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(full)
    return {"messages": [response]}


# Построение графа
builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": END})
builder.add_edge("tools", "agent")

graph = builder.compile()


def run_react_agent(question: str, verbose: bool = True) -> dict:
    """Run ReAct agent for currency conversion.

    Returns:
        dict with:
        - final_answer: str
        - tool_calls: int (number of tool invocations)
        - time_sec: float
        - success: bool (True if tool was used successfully)
    """
    config = {"recursion_limit": MAX_STEPS * 2 + 5}
    messages = [{"role": "user", "content": question}]

    tool_calls = 0
    last_tool_result = None
    final_answer = ""
    start_time = time.perf_counter()

    for event in graph.stream(
        {"messages": messages},
        config=config,
        stream_mode="values",
    ):
        state_msgs = event.get("messages") or []
        if not state_msgs:
            continue

        last_msg = state_msgs[-1]
        print(f"last_msg {last_msg}")

        if isinstance(last_msg, ToolMessage):
            tool_calls += 1
            last_tool_result = getattr(last_msg, "content", None) or str(last_msg)
            if verbose:
                result_preview = last_tool_result[:100] + "..." if len(str(last_tool_result)) > 100 else last_tool_result
                print(f"  [Tool] {result_preview}")

        elif isinstance(last_msg, AIMessage):
            tool_call_list = getattr(last_msg, "tool_calls", None) or []
            if tool_call_list and verbose:
                for tc in tool_call_list:
                    args = _get_tool_call_args(tc)
                    print(f"  [Agent] Calling tool: {tc.get('name', 'unknown')} with args: {args}")

            if last_msg.content and verbose:
                content_preview = last_msg.content[:200] + "..." if len(last_msg.content) > 200 else last_msg.content
                print(f"  [Agent] {content_preview}")

    elapsed = time.perf_counter() - start_time

    # Получаем финальный ответ
    if last_tool_result:
        final_answer = str(last_tool_result)

    # Проверяем успешность (был ли вызов инструмента и есть результат)
    success = tool_calls > 0 and last_tool_result is not None and "Error" not in str(last_tool_result)

    print(f"final_answer: {final_answer}")
    return {
        "final_answer": final_answer,
        "tool_calls": tool_calls,
        "time_sec": elapsed,
        "success": success,
    }


if __name__ == "__main__":
    # Тест агента
    test_question = "Сколько будет 100 долларов в рублях?"
    print(f"Question: {test_question}")
    result = run_react_agent(test_question)
    print(f"\nResult: {result}")
