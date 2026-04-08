"""Линейный LLM workflow на LCEL: 3 шага (анализ -> план -> код). ChatOllama + ChatPromptTemplate | StrOutputParser."""
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

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

llm = ChatOllama(model=MODEL, temperature=0)
parser = StrOutputParser()

analysis_prompt = ChatPromptTemplate.from_messages([("human", STEP_1_PROMPT)])
plan_prompt = ChatPromptTemplate.from_messages([("human", STEP_2_PROMPT)])
code_prompt = ChatPromptTemplate.from_messages([("human", STEP_3_PROMPT)])

analysis_chain = analysis_prompt | llm | parser
plan_chain = (
    RunnableLambda(lambda x: {"task": x["task"], "analysis": x["analysis"]})
    | plan_prompt
    | llm
    | parser
)
code_chain = (
    RunnableLambda(lambda x: {"task": x["task"], "analysis": x["analysis"], "plan": x["plan"]})
    | code_prompt
    | llm
    | parser
)


def run_linear_workflow(task: str, verbose: bool = False):
    """Цепочка из 3 вызовов LLM через LCEL: анализ -> план -> код. Возвращает dict с ключами steps и final."""
    steps = []
    analysis = analysis_chain.invoke({"task": task})
    steps.append(analysis)
    if verbose:
        print("[Step 1 - Analysis]\n", analysis[:400], "\n" if len(analysis) > 400 else "\n")

    plan = plan_chain.invoke({"task": task, "analysis": analysis})
    steps.append(plan)
    if verbose:
        print("[Step 2 - Plan]\n", plan[:400], "\n" if len(plan) > 400 else "\n")

    final = code_chain.invoke({"task": task, "analysis": analysis, "plan": plan})
    steps.append(final)
    if verbose:
        print("[Step 3 - Code]\n", final[:500], "\n" if len(final) > 500 else "\n")

    return {"steps": steps, "final": final}
