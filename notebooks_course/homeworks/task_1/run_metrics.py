"""
Task 1: Измерение метрик prefill/decode (TTFT/TPOT/throughput)

- TTFT (Time To First Token): время обработки промпта (prefill) - из prompt_eval_duration
- TPOT (Time Per Output Token): среднее время на 1 выходной токен - eval_duration / eval_count
- Throughput: токенов в секунду

Метрики извлекаются из нативного API Ollama (не через стриминг chunks).

Улучшения v2:
- Используются нативные метрики Ollama: prompt_eval_duration, eval_duration, eval_count
- Добавлен warmup с keep_alive для устранения cold start
- Отключён KV Cache через независимые запросы
- Добавлены повторные измерения (N_RUNS) для статистики
- Расширен набор промптов
"""
import json
import statistics
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import ollama

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"

# Конфигурация
OLLAMA_MODEL = "qwen3.5:2b"
N_RUNS = 2  # Количество повторений для каждого промпта
KEEP_ALIVE = "30m"  # Удержание модели в памяти
MAX_TOKENS = 300  # Лимит генерации

# Промпты разной длины для тестирования TTFT (4 промпта)
PROMPTS_BY_LENGTH = [
    ("short", "Что такое AI?"),
    ("medium", "Объясни кратко, что такое машинное обучение и нейросети."),
    ("long", """Напиши краткий анализ применения ИИ в телекоммуникациях:
1. Прогнозирование оттока клиентов
2. Оптимизация сети
3. Обнаружение мошенничества

Опиши методы и результаты."""),
    ("very_long", """Проведи анализ современного состояния ИИ.

Включи:
1. ИСТОРИЯ - ключевые этапы развития
2. ТЕХНОЛОГИИ - трансформеры, LLM
3. ПРИМЕНЕНИЕ - телеком, финансы
4. БУДУЩЕЕ - перспективы AGI

Кратко по каждому пункту."""),
]


def warmup_model(model: str = OLLAMA_MODEL, keep_alive: str = KEEP_ALIVE) -> dict:
    """
    Прогрев модели: загружает веса в память и инициализирует Metal/MPS.
    Возвращает метрики загрузки.
    """
    print(f"[Warmup] Loading model {model} into memory...")
    start = time.perf_counter()

    # Используем generate API с keep_alive для явного удержания модели
    response = ollama.generate(
        model=model,
        prompt="Hi",
        keep_alive=keep_alive,
        options={"num_predict": 5},  # Минимальная генерация
    )

    elapsed = time.perf_counter() - start
    load_duration = response.get("load_duration", 0) / 1e9  # наносекунды -> секунды

    print(f"[Warmup] Model loaded in {load_duration:.2f}s (total: {elapsed:.2f}s)")
    print(f"[Warmup] Prompt eval: {response.get('prompt_eval_count', 0)} tokens")

    return {
        "load_duration_sec": load_duration,
        "total_warmup_time_sec": elapsed,
        "prompt_eval_count": response.get("prompt_eval_count"),
    }


def measure_native_metrics(prompt: str, model: str = OLLAMA_MODEL, keep_alive: str = KEEP_ALIVE) -> dict:
    """
    Измеряет метрики генерации используя нативные поля Ollama API.

    Использует stream=False для получения финального ответа с метаданными:
    - prompt_eval_count: количество токенов промпта
    - prompt_eval_duration: время обработки промпта (prefill) в наносекундах
    - eval_count: количество сгенерированных токенов
    - eval_duration: время генерации в наносекундах
    - load_duration: время загрузки модели (если не warmup)

    Returns:
        dict с нативными и вычисленными метриками
    """
    start_time = time.perf_counter()

    try:
        # Используем generate API без стриминга для получения точных метрик
        # Каждый запрос независим (нет контекста) - KV Cache не используется
        response = ollama.generate(
            model=model,
            prompt=prompt,
            keep_alive=keep_alive,
            options={"temperature": 0, "num_predict": MAX_TOKENS},
        )
    except Exception as e:
        return {
            "error": str(e),
            "prompt_length": len(prompt),
        }

    wall_time = time.perf_counter() - start_time

    # Извлекаем нативные метрики (время в наносекундах, конвертируем в секунды)
    prompt_eval_count = response.get("prompt_eval_count", 0)
    prompt_eval_duration_ns = response.get("prompt_eval_duration", 0)
    eval_count = response.get("eval_count", 0)
    eval_duration_ns = response.get("eval_duration", 0)
    load_duration_ns = response.get("load_duration", 0)

    # Конвертация в секунды
    prompt_eval_duration = prompt_eval_duration_ns / 1e9 if prompt_eval_duration_ns else 0
    eval_duration = eval_duration_ns / 1e9 if eval_duration_ns else 0
    load_duration = load_duration_ns / 1e9 if load_duration_ns else 0

    # Вычисляемые метрики
    # TTFT = время prefill (prompt_eval_duration) - чистое время обработки промпта
    ttft = prompt_eval_duration if prompt_eval_duration > 0 else None

    # TPOT = eval_duration / eval_count (время на один выходной токен)
    tpot = (eval_duration / eval_count) if eval_count > 0 else None

    # Throughput = токены / время генерации
    throughput = (eval_count / eval_duration) if eval_duration > 0 else None

    return {
        # Нативные метрики
        "prompt_eval_count": prompt_eval_count,
        "prompt_eval_duration_sec": round(prompt_eval_duration, 4),
        "eval_count": eval_count,
        "eval_duration_sec": round(eval_duration, 4),
        "load_duration_sec": round(load_duration, 4),

        # Вычисляемые метрики
        "ttft_sec": round(ttft, 4) if ttft else None,
        "tpot_sec": round(tpot, 6) if tpot else None,
        "throughput_tokens_per_sec": round(throughput, 2) if throughput else None,

        # Дополнительные данные
        "wall_time_sec": round(wall_time, 4),
        "prompt_length": len(prompt),
        "response_preview": response.get("response", "")[:200] + "..." if len(response.get("response", "")) > 200 else response.get("response", ""),
    }


def measure_with_repeats(prompt: str, prompt_type: str, n_runs: int = N_RUNS, model: str = OLLAMA_MODEL) -> dict:
    """
    Выполняет несколько измерений и вычисляет статистику.
    """
    results = []

    for run in range(n_runs):
        print(f"    Run {run + 1}/{n_runs}...", end=" ", flush=True)
        result = measure_native_metrics(prompt, model)
        result["run"] = run + 1
        results.append(result)

        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"TTFT={result['ttft_sec']:.4f}s, TPOT={result['tpot_sec']:.6f}s, tokens={result['eval_count']}")

        time.sleep(0.5)  # Небольшая пауза между запросами

    # Фильтруем успешные результаты
    valid_results = [r for r in results if "error" not in r]

    if not valid_results:
        return {
            "prompt_type": prompt_type,
            "prompt_length": len(prompt),
            "error": "All runs failed",
            "runs": results,
        }

    # Вычисляем статистику
    ttfts = [r["ttft_sec"] for r in valid_results if r["ttft_sec"]]
    tpots = [r["tpot_sec"] for r in valid_results if r["tpot_sec"]]
    throughputs = [r["throughput_tokens_per_sec"] for r in valid_results if r["throughput_tokens_per_sec"]]
    eval_counts = [r["eval_count"] for r in valid_results if r["eval_count"]]
    prompt_tokens = [r["prompt_eval_count"] for r in valid_results if r["prompt_eval_count"]]

    return {
        "prompt_type": prompt_type,
        "prompt_length": len(prompt),
        "n_runs": len(valid_results),

        # TTFT статистика
        "ttft_mean_sec": round(statistics.mean(ttfts), 4) if ttfts else None,
        "ttft_std_sec": round(statistics.stdev(ttfts), 4) if len(ttfts) > 1 else 0,
        "ttft_min_sec": round(min(ttfts), 4) if ttfts else None,
        "ttft_max_sec": round(max(ttfts), 4) if ttfts else None,

        # TPOT статистика
        "tpot_mean_sec": round(statistics.mean(tpots), 6) if tpots else None,
        "tpot_std_sec": round(statistics.stdev(tpots), 6) if len(tpots) > 1 else 0,

        # Throughput статистика
        "throughput_mean": round(statistics.mean(throughputs), 2) if throughputs else None,
        "throughput_std": round(statistics.stdev(throughputs), 2) if len(throughputs) > 1 else 0,

        # Токены
        "prompt_tokens_mean": round(statistics.mean(prompt_tokens), 1) if prompt_tokens else None,
        "eval_tokens_mean": round(statistics.mean(eval_counts), 1) if eval_counts else None,

        # Сырые данные
        "runs": results,
    }


def plot_ttft_vs_tokens(results: list, save_path: Path):
    """Строит график TTFT от количества токенов промпта."""
    prompt_tokens = [r["prompt_tokens_mean"] for r in results if r.get("ttft_mean_sec")]
    ttfts = [r["ttft_mean_sec"] for r in results if r.get("ttft_mean_sec")]
    ttft_stds = [r["ttft_std_sec"] for r in results if r.get("ttft_mean_sec")]
    labels = [r["prompt_type"] for r in results if r.get("ttft_mean_sec")]

    if not prompt_tokens:
        print("No data for TTFT plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # График с error bars
    ax.errorbar(prompt_tokens, ttfts, yerr=ttft_stds, fmt="bo-", markersize=8, capsize=5, capthick=2)

    ax.set_xlabel("Prompt Tokens (prompt_eval_count)")
    ax.set_ylabel("TTFT (seconds)")
    ax.set_title("Time To First Token vs Prompt Tokens\n(with std deviation)")
    ax.grid(True, alpha=0.3)

    # Аннотации
    for i, (pt, ttft, label) in enumerate(zip(prompt_tokens, ttfts, labels)):
        ax.annotate(f"{label}\n{ttft:.3f}s", (pt, ttft),
                   textcoords="offset points", xytext=(0, 15), ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close()
    print(f"Saved: {save_path}")


def plot_tpot_vs_output_tokens(results: list, save_path: Path):
    """Строит график TPOT от количества выходных токенов."""
    eval_tokens = [r["eval_tokens_mean"] for r in results if r.get("tpot_mean_sec")]
    tpots = [r["tpot_mean_sec"] for r in results if r.get("tpot_mean_sec")]
    tpot_stds = [r["tpot_std_sec"] for r in results if r.get("tpot_mean_sec")]

    if not eval_tokens:
        print("No data for TPOT plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.errorbar(eval_tokens, tpots, yerr=tpot_stds, fmt="go-", markersize=8, capsize=5, capthick=2)

    ax.set_xlabel("Output Tokens (eval_count)")
    ax.set_ylabel("TPOT (seconds/token)")
    ax.set_title("Time Per Output Token vs Output Tokens\n(with std deviation)")
    ax.grid(True, alpha=0.3)

    # Линия тренда
    if len(eval_tokens) > 1:
        z = np.polyfit(eval_tokens, tpots, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(eval_tokens), max(eval_tokens), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.5, label=f"Trend: {z[0]:.6f}x + {z[1]:.6f}")
        ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close()
    print(f"Saved: {save_path}")


def plot_throughput_comparison(results: list, save_path: Path):
    """Строит bar chart сравнения throughput."""
    labels = [r["prompt_type"] for r in results if r.get("throughput_mean")]
    throughputs = [r["throughput_mean"] for r in results if r.get("throughput_mean")]
    throughput_stds = [r["throughput_std"] for r in results if r.get("throughput_mean")]

    if not labels:
        print("No data for throughput plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(labels))
    bars = ax.bar(x, throughputs, yerr=throughput_stds, capsize=5, color="steelblue", alpha=0.7)

    ax.set_xlabel("Prompt Type")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Generation Throughput by Prompt Type")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    # Значения на барах
    for bar, val in zip(bars, throughputs):
        ax.annotate(f"{val:.1f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                   ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close()
    print(f"Saved: {save_path}")


def create_summary_table(results: list) -> list:
    """Создаёт сводную таблицу."""
    rows = []
    for r in results:
        if r.get("ttft_mean_sec"):
            rows.append({
                "prompt_type": r["prompt_type"],
                "prompt_length_chars": r["prompt_length"],
                "prompt_tokens": r["prompt_tokens_mean"],
                "eval_tokens": r["eval_tokens_mean"],
                "ttft_sec": f"{r['ttft_mean_sec']:.4f} ± {r['ttft_std_sec']:.4f}",
                "tpot_sec": f"{r['tpot_mean_sec']:.6f} ± {r['tpot_std_sec']:.6f}",
                "throughput": f"{r['throughput_mean']:.2f} ± {r['throughput_std']:.2f}",
            })
    return rows


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Task 1: TTFT/TPOT Metrics (v2 - Native Ollama Metrics)")
    print("=" * 60)
    print(f"Model: {OLLAMA_MODEL}")
    print(f"N_RUNS per prompt: {N_RUNS}")
    print(f"Keep-alive: {KEEP_ALIVE}")
    print(f"Artifacts: {ARTIFACTS_DIR}")
    print()

    # 0. Warmup - загрузка модели в память
    print("--- Warmup (loading model into memory) ---")
    warmup_result = warmup_model()
    print()

    # 1. Измерение метрик для промптов разной длины
    print("--- Measuring Metrics ---")
    results = []

    for prompt_type, prompt in PROMPTS_BY_LENGTH:
        print(f"\n[{prompt_type}] Prompt: {len(prompt)} chars")
        result = measure_with_repeats(prompt, prompt_type, N_RUNS)
        results.append(result)

        if "error" not in result:
            print(f"  => TTFT: {result['ttft_mean_sec']:.4f}s ± {result['ttft_std_sec']:.4f}s")
            print(f"  => TPOT: {result['tpot_mean_sec']:.6f}s ± {result['tpot_std_sec']:.6f}s")
            print(f"  => Throughput: {result['throughput_mean']:.2f} t/s")
        else:
            print(f"  => ERROR: {result.get('error')}")

    # 2. Графики
    print("\n--- Generating Plots ---")
    plot_ttft_vs_tokens(results, ARTIFACTS_DIR / "ttft_vs_prompt_tokens.png")
    plot_tpot_vs_output_tokens(results, ARTIFACTS_DIR / "tpot_vs_output_tokens.png")
    plot_throughput_comparison(results, ARTIFACTS_DIR / "throughput_comparison.png")

    # 3. Сводная таблица
    summary_table = create_summary_table(results)
    print("\n--- Summary Table ---")
    print(f"{'Type':<12} {'Chars':<7} {'PToks':<7} {'EToks':<7} {'TTFT':<20} {'TPOT':<22} {'Thru':<15}")
    print("-" * 90)
    for row in summary_table:
        print(f"{row['prompt_type']:<12} {row['prompt_length_chars']:<7} {row['prompt_tokens']:<7} "
              f"{row['eval_tokens']:<7} {row['ttft_sec']:<20} {row['tpot_sec']:<22} {row['throughput']:<15}")

    # 4. Сохранение результатов
    output = {
        "config": {
            "model": OLLAMA_MODEL,
            "n_runs": N_RUNS,
            "keep_alive": KEEP_ALIVE,
        },
        "warmup": warmup_result,
        "results": results,
        "summary_table": summary_table,
    }

    with open(ARTIFACTS_DIR / "metrics_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nSaved: {ARTIFACTS_DIR / 'metrics_results.json'}")
    print("=" * 60)
    print("Task 1 Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
