"""
Task 4: CoT-бейзлайн (Chain-of Thought)

LLM: OpenRouter
"""
import json
import os
import time
from pathlib import Path

from langchain_openai import ChatOpenAI

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"

OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")

# Промпт для CoT-бейзлайна
COT_PROMPT = """You are a currency conversion assistant.

Think through this step-by-step:

In your thinking process:
1. Identify the source currency and amount
2. Identify the target currency
3. Consider the exchange rate (note: you don't have access to real-time data, so estimate based on common rates)
4. Calculate the conversion
5. Provide the final answer

Use the Chain-of Thought approach. Note that you do NOT have access to real-time exchange rates - make your best estimate based on your knowledge.

Question: {question}

Let's think step by step:

Answer:"""


def is_valid_answer(answer: str) -> bool:
    """Проверяет, содержит ли ответ валидное число/валюту."""
    has_number = any(char.isdigit() for char in answer)
    answer_lower = answer.lower()
    has_currency = any(
        cur in answer_lower
        for cur in ["рубл", "rub", "₽", "доллар", "usd", "$", "евро", "eur", "йен", "jpy", "франк", "chf"]
    )
    return has_number and has_currency


def run_cot_baseline(question: str, verbose: bool = False) -> dict:
    """Запуск CoT бейзлайн через OpenRouter."""
    if verbose:
        print(f"Running CoT baseline for: {question[:50]}...")

    start = time.perf_counter()
    try:
        llm = ChatOpenAI(
            model=OPENROUTER_MODEL,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=0,
        )
        response = llm.invoke(COT_PROMPT.format(question=question))
        print(f"question {question}")
        answer = response.content if hasattr(response, "content") else str(response)
        print(f"answer {answer}")
    except Exception as e:
        return {"answer": f"Error: {e}", "time_sec": 0, "tool_calls": 0, "success": False}

    elapsed = time.perf_counter() - start
    success = is_valid_answer(answer)

    return {
        "answer": answer,
        "tool_calls": 0,
        "time_sec": elapsed,
        "success": success,
    }


if __name__ == "__main__":
    # Тест бейзлайна
    test_question = "Сколько будет 100 долларов в рублях?"
    print(f"Question: {test_question}")
    result = run_cot_baseline(test_question, verbose=True)
    print(f"\nResult: {result}")
