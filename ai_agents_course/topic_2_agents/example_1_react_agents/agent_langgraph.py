"""Code agent on LangGraph 1.0.x: StateGraph + ToolNode + tools_condition. Same contract as agent_langchain_react."""
import json
import re
import sys
from pathlib import Path
from typing import Annotated, Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from tools import run_python_code_tool, extract_code_block

from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

MODEL = "qwen2.5-coder:1.5b"
MAX_STEPS = 10

SYSTEM_PROMPT = """You are a coding agent. For each task you must:
1. Think about what code to write.
2. Use the run_python_code_tool to execute your Python code. You must pass valid Python source code.
3. After seeing the Observation (OK or ERROR from running the code), either fix the code and call the tool again, or give a final answer.

When you are done and the code runs successfully (or you give up), reply with a final text answer. Do not call the tool after giving the final answer."""


class State(TypedDict):
    messages: Annotated[list, add_messages]


tools = [run_python_code_tool]
llm = ChatOllama(model=MODEL, temperature=0)
llm_with_tools = llm.bind_tools(tools)
tool_node = ToolNode(tools)


def _get_tool_call_args(tc: Any) -> dict:
    """Get tool call args: support .args, .arguments, and JSON string (e.g. Ollama)."""
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


def agent_node(state: State) -> dict:
    messages = _to_messages(state["messages"])
    full = [SystemMessage(content=SYSTEM_PROMPT)] + messages
    response = llm_with_tools.invoke(full)
    return {"messages": [response]}


builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition, {"tools": "tools", "__end__": END})
builder.add_edge("tools", "agent")

graph = builder.compile()


def run_langgraph_agent(task: str, verbose: bool = True) -> dict:
    """Run LangGraph code agent. Returns same shape as run_langchain_react."""
    config = {"recursion_limit": MAX_STEPS * 2 + 5}
    messages = [{"role": "user", "content": f"Task:\n{task}"}]
    n_steps = 0
    last_tool_result = None
    last_code = ""
    last_event = None

    for event in graph.stream(
        {"messages": messages},
        config=config,
        stream_mode="values",
    ):
        last_event = event
        state_msgs = event.get("messages") or []
        if not state_msgs:
            continue
        last_msg = state_msgs[-1]
        if isinstance(last_msg, ToolMessage):
            n_steps += 1
            last_tool_result = getattr(last_msg, "content", None) or str(last_msg)
            if verbose and last_tool_result:
                print(f"Observation: {last_tool_result[:300]}{'...' if len(last_tool_result) > 300 else ''}\n")
        elif isinstance(last_msg, AIMessage):
            tool_calls = getattr(last_msg, "tool_calls", None) or []
            if tool_calls:
                for tc in tool_calls:
                    args = _get_tool_call_args(tc)
                    if args.get("code"):
                        last_code = args["code"]
            if verbose and last_msg.content:
                print(f"[Step]\n{(last_msg.content or '')[:500]}{'...' if len(last_msg.content or '') > 500 else ''}\n")

    final_answer = ""
    if last_event:
        final_messages = last_event.get("messages") or []
        for m in reversed(final_messages):
            if isinstance(m, AIMessage) and (getattr(m, "content", None) or "").strip():
                final_answer = (m.content or "").strip()
                break

    if not last_code and final_answer:
        to_parse = final_answer.strip()
        if to_parse.startswith("```"):
            to_parse = re.sub(r"^```(?:json)?\s*\n?", "", to_parse)
            to_parse = re.sub(r"\n?```\s*$", "", to_parse)
        try:
            data = json.loads(to_parse)
            if isinstance(data, dict):
                args = data.get("arguments") or data.get("args") or {}
                if args.get("code"):
                    last_code = args["code"]
        except (json.JSONDecodeError, TypeError):
            pass
        if not last_code:
            extracted = extract_code_block(final_answer)
            if extracted:
                last_code = extracted

    executable = last_tool_result is not None and str(last_tool_result).strip().startswith("OK:")
    return {
        "final_answer": final_answer,
        "steps": [],
        "last_code": last_code,
        "executable": executable,
        "n_steps": n_steps,
    }
