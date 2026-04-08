"""
Task 2: RAG vs RAG + Reranker (скорость/качество контекста)

Сравнение двух пайплайнов:
1. Basic RAG: FAISS + retriever (top-K)
2. RAG + Reranker: retriever (top-N) -> CrossEncoder reranker (top-K)

Корпус: data/video/video_2.txt
Эмбеддинги: bge-m3 через Ollama (ollama pull bge-m3)
Reranker: BAAI/bge-reranker-v2-m3 (локальный CrossEncoder)
LLM: OpenRouter (google/gemini-2.0-flash-001)

v2 улучшения:
- Исправлена модель LLM (была недоступна)
- Автоопределение device (mps/cpu) для reranker
- Добавлена оценка качества контекста (LLM-as-judge)
- Добавлены графики сравнения
"""
import json
import os
import platform
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
CORPUS_PATH = SCRIPT_DIR.parent.parent / "data" / "video" / "video_2.txt"

# Конфигурация
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-001")
EMBED_MODEL = "bge-m3"  # Через Ollama (ollama pull bge-m3)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

TOP_K = 5  # Финальное количество документов
TOP_N = 10  # Начальное количество для reranker
CHUNK_SIZE = 200  # Размер чанка (для ~40 чанков)
CHUNK_OVERLAP = 50  # Перекрытие чанков


def get_device() -> str:
    """Автоопределение устройства для inference."""
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return "mps"  # Apple Silicon
    return "cpu"


def get_llm():
    """Возвращает LLM через OpenRouter."""
    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
    )


def generate_questions_from_chunks(chunks: list, n_questions: int = 15, llm=None) -> list:
    """Генерирует вопросы из чанков корпуса с помощью LLM."""
    import random
    if llm is None:
        llm = get_llm()

    # Выбираем случайные чанки
    selected_chunks = random.sample(chunks, min(n_questions, len(chunks)))

    questions = []
    print(f"Generating {len(selected_chunks)} questions from corpus...")

    for i, chunk in enumerate(selected_chunks):
        prompt = f"""На основе следующего текста, составь 1 вопрос, на который можно ответить, используя этот текст.

Текст: {chunk[:500]}

Вопрос (только сам вопрос, без номера):"""

        try:
            response = llm.invoke(prompt)
            question = response.content.strip().strip('"')
            questions.append(question)
            print(f"  [{i+1}/{len(selected_chunks)}] {question[:60]}...")
        except Exception as e:
            print(f"  Error generating question {i+1}: {e}")
            continue

    return questions


# Вопросы генерируются из корпуса при запуске
TEST_QUESTIONS = []  # Генерируются динамически в main()


def load_corpus(path: Path) -> str:
    """Загружает корпус из файла."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def split_corpus(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list:
    """Разбивает текст на чанки с помощью RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def build_faiss_index(chunks: list, embed_model: str = EMBED_MODEL) -> FAISS:
    """Создаёт FAISS индекс из чанков через OllamaEmbeddings."""
    docs = [Document(page_content=chunk) for chunk in chunks]
    embeddings = OllamaEmbeddings(model=embed_model)
    return FAISS.from_documents(docs, embeddings)


class BasicRAG:
    """Базовый RAG пайплайн без reranker."""

    def __init__(self, vector_store: FAISS, top_k: int = TOP_K):
        self.retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
        self.top_k = top_k

    def retrieve(self, query: str) -> tuple:
        """Возвращает документы и время retrieval."""
        t0 = time.perf_counter()
        docs = self.retriever.invoke(query)
        t_retrieve = time.perf_counter() - t0
        return docs, {"retrieval": t_retrieve, "rerank": 0.0, "total": t_retrieve}


class RAGWithReranker:
    """RAG пайплайн с CrossEncoder reranker."""

    def __init__(self, vector_store: FAISS, top_n: int = TOP_N, top_k: int = TOP_K, reranker_model: str = RERANKER_MODEL):
        self.retriever = vector_store.as_retriever(search_kwargs={"k": top_n})
        self.top_n = top_n
        self.top_k = top_k

        device = get_device()
        print(f"Loading reranker model: {reranker_model} on {device}")
        self.reranker = HuggingFaceCrossEncoder(model_name=reranker_model, model_kwargs={"device": device})

    def retrieve(self, query: str) -> tuple:
        """Возвращает документы и времена retrieval + rerank."""
        # 1. Retrieval
        t0 = time.perf_counter()
        docs = self.retriever.invoke(query)
        t_retrieve = time.perf_counter() - t0

        # 2. Reranking
        t1 = time.perf_counter()
        scores_info = []
        if len(docs) > self.top_k:
            pairs = [[query, doc.page_content] for doc in docs]
            scores = self.reranker.score(pairs)
            scores_info = list(zip(range(len(docs)), scores))
            ranked_indices = np.argsort(scores)[::-1][: self.top_k]
            docs = [docs[i] for i in ranked_indices]

        t_rerank = time.perf_counter() - t1
        t_total = time.perf_counter() - t0

        return docs, {"retrieval": t_retrieve, "rerank": t_rerank, "total": t_total, "scores_info": scores_info}


def evaluate_context_quality(query: str, context: str, llm=None) -> dict:
    """
    Оценивает качество контекста через LLM-as-judge.
    Возвращает relevance_score (1-5) и explanation.
    """
    if llm is None:
        llm = get_llm()

    prompt = f"""Оцени, насколько приведённый контекст релевантен вопросу.

Вопрос: {query}

Контекст:
{context[:2000]}

Оцени по шкале от 1 до 5:
1 - Контекст полностью нерелевантен
2 - Контекст слабо связан с вопросом
3 - Контекст частично релевантен
4 - Контекст в основном релевантен
5 - Контекст полностью релевантен

Ответь в формате: "ОЦЕНКА: X" где X - число от 1 до 5.
Затем кратко объясни почему."""

    try:
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        # Извлекаем оценку
        match = re.search(r"ОЦЕНКА:\s*(\d)", answer)
        score = int(match.group(1)) if match else 3

        return {"relevance_score": score, "explanation": answer[:500]}
    except Exception as e:
        return {"relevance_score": None, "explanation": f"Error: {e}"}


def generate_answer(query: str, context: str, llm=None) -> tuple:
    """Генерирует ответ через OpenRouter и возвращает ответ + время."""
    if llm is None:
        llm = get_llm()

    prompt = f"""Контекст:
{context}

Вопрос: {query}

Ответь на вопрос, используя только информацию из контекста. Если в контексте нет ответа, скажи об этом."""

    t0 = time.perf_counter()
    try:
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        answer = f"Error: {e}"
    t_gen = time.perf_counter() - t0

    return answer, t_gen


def run_pipeline(pipeline, query: str, llm=None, evaluate_quality: bool = True) -> dict:
    """Запускает пайплайн и собирает метрики."""
    if llm is None:
        llm = get_llm()

    # Retrieval (+ rerank если есть)
    docs, times = pipeline.retrieve(query)
    context = "\n\n".join(doc.page_content for doc in docs)

    # Generation
    answer, t_gen = generate_answer(query, context, llm)

    # Quality evaluation - отключено для фокуса на времени
    quality = {}
    # if evaluate_quality:
    #     quality = evaluate_context_quality(query, context, llm)

    return {
        "query": query,
        "answer": answer,
        "context_length": len(context),
        "num_docs": len(docs),
        "times": {**{k: v for k, v in times.items() if k != "scores_info"}, "generation": t_gen},
        "total_time": times["total"] + t_gen,
        "quality": quality,
    }


def plot_comparison(comparison: dict, save_path: Path):
    """Строит графики сравнения."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    basic = comparison["basic_rag"]["aggregate"]
    rerank = comparison["rag_reranker"]["aggregate"]

    labels = ["Basic RAG", "RAG + Reranker"]
    colors = ["steelblue", "coral"]

    # 1. Retrieval Time
    ax = axes[0]
    values = [basic["avg_retrieval_time"], rerank["avg_retrieval_time"]]
    ax.bar(labels, values, color=colors)
    ax.set_title("Avg Retrieval Time")
    ax.set_ylabel("Seconds")
    for i, v in enumerate(values):
        ax.annotate(f"{v:.3f}s", (i, v), ha="center", va="bottom")

    # 2. Retrieval+Rerank Time
    ax = axes[1]
    values = [basic["avg_total_time"], rerank["avg_total_time"]]
    ax.bar(labels, values, color=colors)
    ax.set_title("Avg Retrieval+Rerank Time")
    ax.set_ylabel("Seconds")
    for i, v in enumerate(values):
        ax.annotate(f"{v:.3f}s", (i, v), ha="center", va="bottom")

    # 3. Generation Time
    ax = axes[2]
    values = [basic["avg_generation_time"], rerank["avg_generation_time"]]
    ax.bar(labels, values, color=colors)
    ax.set_title("Avg Generation Time")
    ax.set_ylabel("Seconds")
    for i, v in enumerate(values):
        ax.annotate(f"{v:.3f}s", (i, v), ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close()
    print(f"Saved: {save_path}")


def plot_time_breakdown(comparison: dict, save_path: Path):
    """Строит stacked bar chart с разбивкой по времени."""
    fig, ax = plt.subplots(figsize=(10, 6))

    basic = comparison["basic_rag"]["aggregate"]
    rerank = comparison["rag_reranker"]["aggregate"]

    labels = ["Basic RAG", "RAG + Reranker"]
    x = np.arange(len(labels))
    width = 0.5

    # Stacked bars
    retrieval = [basic["avg_retrieval_time"], rerank["avg_retrieval_time"]]
    rerank_time = [0, rerank["avg_rerank_time"]]
    generation = [basic["avg_generation_time"], rerank["avg_generation_time"]]

    bars1 = ax.bar(x, retrieval, width, label="Retrieval", color="steelblue")
    bars2 = ax.bar(x, rerank_time, width, bottom=retrieval, label="Rerank", color="coral")
    bars3 = ax.bar(x, generation, width, bottom=[r + rr for r, rr in zip(retrieval, rerank_time)], label="Generation", color="green")

    ax.set_ylabel("Time (seconds)")
    ax.set_title("Time Breakdown by Pipeline Stage")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Добавляем total labels
    for i, (r, rr, g) in enumerate(zip(retrieval, rerank_time, generation)):
        total = r + rr + g
        ax.annotate(f"Total: {total:.2f}s", (i, total), ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close()
    print(f"Saved: {save_path}")


def compare_pipelines(questions: list, basic_rag: BasicRAG, rag_rerank: RAGWithReranker) -> dict:
    """Сравнивает два пайплайна на списке вопросов."""
    results_basic = []
    results_rerank = []

    llm = get_llm()  # Общий LLM для всех запросов

    for i, question in enumerate(questions):
        print(f"\n[{i + 1}/{len(questions)}] {question[:50]}...")

        # Basic RAG
        print("  Running Basic RAG...")
        result_basic = run_pipeline(basic_rag, question, llm)
        results_basic.append(result_basic)
        print(f"    Time: {result_basic['total_time']:.3f}s, Retrieval: {result_basic['times']['retrieval']:.3f}s")

        # RAG + Reranker
        print("  Running RAG + Reranker...")
        result_rerank = run_pipeline(rag_rerank, question, llm)
        results_rerank.append(result_rerank)
        print(f"    Time: {result_rerank['total_time']:.3f}s, Retrieval: {result_rerank['times']['retrieval']:.3f}s, Rerank: {result_rerank['times']['rerank']:.3f}s")

    # Агрегация метрик
    def aggregate(results: list) -> dict:
        times = [r["times"] for r in results]
        qualities = [r["quality"].get("relevance_score") for r in results if r["quality"].get("relevance_score")]
        return {
            "avg_retrieval_time": sum(t["retrieval"] for t in times) / len(times),
            "avg_rerank_time": sum(t["rerank"] for t in times) / len(times),
            "avg_generation_time": sum(t["generation"] for t in times) / len(times),
            "avg_total_time": sum(r["total_time"] for r in results) / len(results),
            "avg_context_length": sum(r["context_length"] for r in results) / len(results),
            "avg_relevance_score": sum(qualities) / len(qualities) if qualities else None,
        }

    return {
        "basic_rag": {"aggregate": aggregate(results_basic), "details": results_basic},
        "rag_reranker": {"aggregate": aggregate(results_rerank), "details": results_rerank},
        "comparison": {
            "retrieval_speedup": aggregate(results_basic)["avg_retrieval_time"] / aggregate(results_rerank)["avg_retrieval_time"]
            if aggregate(results_rerank)["avg_retrieval_time"] > 0
            else 0,
            "total_time_diff": aggregate(results_basic)["avg_total_time"] - aggregate(results_rerank)["avg_total_time"],
            "relevance_improvement": (aggregate(results_rerank)["avg_relevance_score"] or 0)
            - (aggregate(results_basic)["avg_relevance_score"] or 0),
        },
    }


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== Task 2: RAG vs RAG + Reranker (v2) ===")
    print(f"Corpus: {CORPUS_PATH}")
    print(f"LLM: {OPENROUTER_MODEL}")
    print(f"Embeddings: {EMBED_MODEL}")
    print(f"Reranker: {RERANKER_MODEL} on {get_device()}")
    print()

    # 1. Загрузка и разделение корпуса
    print("--- Loading Corpus ---")
    corpus_text = load_corpus(CORPUS_PATH)
    print(f"Corpus length: {len(corpus_text)} chars")

    chunks = split_corpus(corpus_text)
    print(f"Chunks: {len(chunks)}")
    print(f"Avg chunk length: {sum(len(c) for c in chunks) / len(chunks):.0f} chars")

    # 2. Построение FAISS индекса
    print("\n--- Building FAISS Index ---")
    t0 = time.perf_counter()
    vector_store = build_faiss_index(chunks)
    t_index = time.perf_counter() - t0
    print(f"Index built in {t_index:.2f}s")

    # 3. Инициализация пайплайнов
    print("\n--- Initializing Pipelines ---")
    basic_rag = BasicRAG(vector_store, top_k=TOP_K)
    rag_rerank = RAGWithReranker(vector_store, top_n=TOP_N, top_k=TOP_K)

    # 4. Генерация вопросов из корпуса
    print("\n--- Generating Questions from Corpus ---")
    llm = get_llm()
    test_questions = generate_questions_from_chunks(chunks, n_questions=15, llm=llm)

    if not test_questions:
        print("ERROR: Failed to generate questions!")
        return

    # 5. Сравнение на сгенерированных вопросах
    print("\n--- Running Comparison ---")
    comparison = compare_pipelines(test_questions, basic_rag, rag_rerank)

    # 5. Вывод результатов
    print("\n=== Results ===")
    print("\nBasic RAG:")
    for k, v in comparison["basic_rag"]["aggregate"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nRAG + Reranker:")
    for k, v in comparison["rag_reranker"]["aggregate"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\nComparison:")
    for k, v in comparison["comparison"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # 6. Графики
    print("\n--- Generating Plots ---")
    plot_comparison(comparison, ARTIFACTS_DIR / "rag_comparison.png")
    plot_time_breakdown(comparison, ARTIFACTS_DIR / "time_breakdown.png")

    # 7. Сохранение результатов
    output = {
        "config": {
            "corpus_path": str(CORPUS_PATH),
            "corpus_length": len(corpus_text),
            "num_chunks": len(chunks),
            "embed_model": EMBED_MODEL,
            "reranker_model": RERANKER_MODEL,
            "device": get_device(),
            "llm_model": OPENROUTER_MODEL,
            "top_k": TOP_K,
            "top_n": TOP_N,
        },
        "comparison": comparison,
    }

    with open(ARTIFACTS_DIR / "rag_comparison.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nSaved: {ARTIFACTS_DIR / 'rag_comparison.json'}")
    print("=== Task 2 Complete ===")


if __name__ == "__main__":
    main()
