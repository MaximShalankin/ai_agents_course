"""
Task 4: Сравнение ReAct-агента и CoT-бейзлайн

Скрипт тестирования, который запускает вопросы и сравнивает метрики:
- tool_calls (количество обращений к инструментам)
- time-to-solution (время до ответа в секундах)
- success (успешность ответа)
- accuracy / MAPE (точность ответа относительно ReAct как ground truth)
"""
import json
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agent_react import run_react_agent
from baseline_cot import run_cot_baseline

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"

# Тестовые вопросы (конвертация валют) - расширенный набор
TEST_QUESTIONS = [
    # Простые однократные конвертации
    "Сколько будет 100 долларов в рублях?",
    "Конвертируй 50 евро в доллары",
    "Сколько 1000 рублей в евро?",
    "Переведи 200 японских йен в доллары",
    "Какова стоимость 500 швейцарских франков в рублях?",
    # Многошаговые конвертации
    "Конвертируй 500 евро в японские йены, затем результат в рублях",
    "Переведи 1000 долларов в фунты стерлингов, затем в рубли",
    # Запросы курсов
    "Каков курс евро к швейцарским франкам сегодня?",
    "Какой курс доллара к рублю?",
    "Какой курс фунта стерлингов к доллару?",
    # Экзотические валюты
    "Сколько будет 100 долларов в турецких лирах?",
    "Конвертируй 200 польских злотых в рубли",
    "Переведи 500 китайских юаней в евро",
    # Обратные конвертации
    "Сколько долларов будет в 10000 рублях?",
    "Какую сумму в евро можно получить за 5000 рублей?",
]


def extract_main_number(text: str) -> float | None:
    """Извлекает главное число из ответа (результат конвертации или курс).

    Приоритет:
    1. Число после "Final Answer:" / "Answer:"
    2. Формат ReAct: "100.0 USD = 8126.00 RUB"
    3. "примерно/приблизительно X" в последнем предложении
    4. Последнее число в тексте (fallback)
    """
    if not text:
        return None

    # Сначала проверяем, не является ли ответ просто числом
    text_clean = text.strip()
    if re.match(r"^[\d.]+$", text_clean):
        try:
            return float(text_clean)
        except ValueError:
            pass

    # Паттерн 1: ReAct формат "100.0 USD = 8126.00 RUB (rate: 81.2600)"
    conversion_pattern = r"[\d.]+\s*(?:USD|EUR|RUB|JPY|GBP|CHF|TRY|PLN|CNY)\s*=\s*([\d,.]+)\s*(?:USD|EUR|RUB|JPY|GBP|CHF|TRY|PLN|CNY)"
    match = re.search(conversion_pattern, text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1).replace(",", ""))
        except ValueError:
            pass

    # Паттерн 2: "Final Answer:" или "Answer:" - берём число ИЗ ЭТОЙ СТРОКИ
    # Ищем строку СОДЕРЖАЩУЮ Final Answer/Answer
    final_answer_line = None
    for line in text.split('\n'):
        line_lower = line.strip().lower()
        if 'final answer' in line_lower or (line_lower.startswith('answer') and ':' in line_lower):
            final_answer_line = line
            break

    if final_answer_line:
        # СНАЧАЛА ищем "approximately/примерно X" или "approximately $X"
        approx_pattern = r"(?:approximately|приблизительно|примерно)\s+(?:equal\s+to\s+)?(?:[\$€£])?([\d,\s]+)"
        match = re.search(approx_pattern, final_answer_line, re.IGNORECASE)
        if match:
            try:
                # Remove spaces from numbers like "50 000"
                num_str = match.group(1).replace(" ", "").replace(",", "")
                return float(num_str)
            except ValueError:
                pass

        # Ищем "$X" или "X dollars" или "X рублей" в Final Answer
        money_pattern = r"[\$€£]\s*([\d,.]+)|([\d,.]+)\s*(?:dollars?|euros?|pounds?|rubles?|rub|руб|евро|йен|yuan|юан)"
        match = re.search(money_pattern, final_answer_line, re.IGNORECASE)
        if match:
            try:
                val = match.group(1) if match.group(1) else match.group(2)
                return float(val.replace(",", ""))
            except (ValueError, TypeError):
                pass

        # Если "is" в строке - берём число ПОСЛЕ "is"
        if ' is ' in final_answer_line.lower():
            is_match = re.search(r'\bis\s+(?:approximately\s+)?(?:[\$€£])?([\d,.]+)', final_answer_line, re.IGNORECASE)
            if is_match:
                try:
                    return float(is_match.group(1).replace(",", ""))
                except ValueError:
                    pass

        # Число после двоеточия - берём ПОСЛЕДНЕЕ число (обычно это результат)
        if ':' in final_answer_line:
            after_colon = final_answer_line.split(':')[-1]
            # Ищем все числа
            numbers = re.findall(r"([\d,\s]+)", after_colon)
            # Фильтруем и берём последнее значимое
            valid_nums = []
            for num_str in numbers:
                try:
                    num = float(num_str.replace(" ", "").replace(",", ""))
                    if num > 0.01:
                        valid_nums.append(num)
                except ValueError:
                    continue
            if valid_nums:
                return valid_nums[-1]

    # Паттерн 3: Последнее предложение с "примерно/приблизительно/approximately"
    sentences = re.split(r'[.!?]', text)
    if sentences:
        last_sentence = sentences[-1].strip() if sentences[-1].strip() else sentences[-2].strip()
        approx_pattern = r"(?:approximately|приблизительно|примерно)\s+([\d,.]+)"
        match = re.search(approx_pattern, last_sentence, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                pass
        # Число с валютой в последнем предложении
        currency_pattern = r"([\d,.]+)\s*(?:рубл|rub|₽|доллар|dollar|usd|\$|евро|euro|eur|йен|yen|jpy|франк|franc|chf|лир|lir|try|злот|zloty|pln|юан|yuan|cny|фунт|pound|gbp)"
        match = re.search(currency_pattern, last_sentence, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except ValueError:
                pass

    # Fallback: последнее значимое число (только если других паттернов нет)
    numbers = re.findall(r"[\d,.]+", text)
    significant_numbers = []
    for num_str in numbers:
        try:
            num = float(num_str.replace(",", ""))
            if num > 0.01:
                significant_numbers.append(num)
        except ValueError:
            continue

    if significant_numbers:
        return significant_numbers[-1]

    return None


def calculate_mape(predicted: float | None, actual: float | None) -> float | None:
    """Вычисляет Mean Absolute Percentage Error."""
    if predicted is None or actual is None or actual == 0:
        return None
    return abs(predicted - actual) / abs(actual) * 100


def run_comparison(questions: list, save_path: Path = None):
    """Запуск сравнения и возвращает результаты."""
    results = []
    mape_scores = []  # Для хранения MAPE CoT относительно ReAct

    for i, q in enumerate(questions):
        print(f"\n[{i + 1}/{len(questions)}] {q[:50]}...")

        # ReAct агент (ground truth для точности)
        print("  Running ReAct agent...")
        r_react = run_react_agent(q, verbose=False)
        time.sleep(0.3)

        # CoT-бейзлайн
        print("  Running CoT baseline...")
        r_cot = run_cot_baseline(q, verbose=False)
        time.sleep(0.3)

        # Извлекаем числа для расчёта MAPE
        react_answer = r_react.get("final_answer", r_react.get("answer", ""))
        cot_answer = r_cot["answer"]

        react_value = extract_main_number(react_answer)
        cot_value = extract_main_number(cot_answer)
        mape = calculate_mape(cot_value, react_value)

        if mape is not None:
            mape_scores.append(mape)

        results.append(
            {
                "question": q,
                "react": {
                    "answer": react_answer,
                    "extracted_value": react_value,
                    "tool_calls": r_react["tool_calls"],
                    "time_sec": round(r_react["time_sec"], 2),
                    "success": r_react["success"],
                },
                "cot": {
                    "answer": cot_answer,
                    "extracted_value": cot_value,
                    "tool_calls": 0,
                    "time_sec": round(r_cot["time_sec"], 2),
                    "success": r_cot["success"],
                    "mape_pct": round(mape, 2) if mape is not None else None,
                },
            }
        )

        mape_str = f", MAPE={mape:.1f}%" if mape is not None else ""
        print(f"  ReAct: {r_react['tool_calls']} tools, {r_react['time_sec']:.2f}s, val={react_value}")
        print(f"  CoT: {r_cot['time_sec']:.2f}s, val={cot_value}{mape_str}")

    # Сводный DataFrame
    react_data = [
        {"architecture": "ReAct", "tool_calls": r["react"]["tool_calls"], "time_sec": r["react"]["time_sec"], "success": r["react"]["success"], "mape_pct": 0.0}
        for r in results
    ]
    cot_data = [
        {"architecture": "CoT", "tool_calls": 0, "time_sec": r["cot"]["time_sec"], "success": r["cot"]["success"], "mape_pct": r["cot"]["mape_pct"] or 0.0}
        for r in results
    ]

    df_all = pd.DataFrame(react_data + cot_data)
    df_summary = df_all.groupby("architecture").agg(
        avg_tool_calls=("tool_calls", "mean"),
        avg_time_sec=("time_sec", "mean"),
        success_rate=("success", "mean"),
        avg_mape_pct=("mape_pct", "mean"),
    ).reset_index()

    print("\n=== Summary ===")
    print(df_summary.to_string(index=False))

    # Средний MAPE только для CoT
    avg_mape = np.mean(mape_scores) if mape_scores else None
    print(f"\nCoT Average MAPE vs ReAct: {avg_mape:.2f}%" if avg_mape else "\nMAPE: N/A")

    # Сохранение результатов
    output = {
        "questions": questions,
        "comparison": results,
        "summary": df_summary.to_dict(orient="records"),
        "cot_avg_mape_pct": round(avg_mape, 2) if avg_mape else None,
    }

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path / "comparison.json", "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        print(f"\nSaved: {save_path / 'comparison.json'}")

        # График сравнения (4 метрики)
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        architectures = df_summary["architecture"].tolist()
        x = range(len(architectures))
        colors = ["steelblue", "coral"]

        # Tool calls
        ax = axes[0]
        ax.bar(x, df_summary["avg_tool_calls"], color=colors)
        ax.set_title("Avg Tool Calls")
        ax.set_ylabel("Count")
        ax.set_xticks(x)
        ax.set_xticklabels(architectures)

        # Time
        ax = axes[1]
        ax.bar(x, df_summary["avg_time_sec"], color=colors)
        ax.set_title("Avg Time (sec)")
        ax.set_ylabel("Seconds")
        ax.set_xticks(x)
        ax.set_xticklabels(architectures)

        # Success rate
        ax = axes[2]
        ax.bar(x, df_summary["success_rate"], color=colors)
        ax.set_title("Success Rate")
        ax.set_ylabel("Rate (0-1)")
        ax.set_xticks(x)
        ax.set_xticklabels(architectures)

        # MAPE (Accuracy) - чем ниже, тем лучше
        ax = axes[3]
        ax.bar(x, df_summary["avg_mape_pct"], color=colors)
        ax.set_title("Avg MAPE % (lower is better)")
        ax.set_ylabel("Percent")
        ax.set_xticks(x)
        ax.set_xticklabels(architectures)

        fig.tight_layout()
        fig.savefig(save_path / "comparison.png", dpi=100)
        plt.close()
        print(f"Saved: {save_path / 'comparison.png'}")

    return results, df_summary


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Task 4: ReAct vs CoT Comparison ===\n")
    run_comparison(TEST_QUESTIONS, ARTIFACTS_DIR)
    print("\n=== Task 4 Complete ===")


if __name__ == "__main__":
    main()
