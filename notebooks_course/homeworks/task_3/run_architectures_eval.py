"""
Task 3: Сравнение 3 архитектур - Retrieval + Generation метрики

Сравниваем 3 архитектуры:
1. LLM без контекста (baseline)
2. Стандартный RAG (top-K чанков)
3. RAG + Reranker (top-N -> top-K)

Метрики:
- Retrieval: Recall@K, MRR@K, Hit Rate@K
- Generation: BERTScore, ROUGE-L, BLEU

Датасет: rajpurkar/squad
"""
import json
import os
import time
from pathlib import Path

import bert_score
import numpy as np
import pandas as pd
import sacrebleu
from datasets import load_dataset
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rouge_score import rouge_scorer

SCRIPT_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"

# Конфигурация
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemma-3-4b-it:free")
EMBED_MODEL = "bge-m3"  # Через Ollama (ollama pull bge-m3)
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

N_SAMPLES = 500  # Количество примеров для оценки
TOP_K = 5
TOP_N = 15


def get_llm():
    """Возвращает LLM через OpenRouter."""
    return ChatOpenAI(
        model=OPENROUTER_MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0,
    )


# ============ Retrieval метрики ============


def compute_retrieval_metrics(retrieved_docs_list: list, ground_truth_contexts: list, top_k: int = TOP_K) -> dict:
    """Вычисляет retrieval метрики.

    Args:
        retrieved_docs_list: список списков retrieved документов для каждого вопроса
        ground_truth_contexts: список эталонных контекстов (из SQuAD)
        top_k: для метрик @K

    Returns:
        dict с recall@k, mrr@k, hit_rate@k
    """
    recall_scores = []
    mrr_scores = []
    hit_scores = []

    for retrieved_docs, gt_context in zip(retrieved_docs_list, ground_truth_contexts):
        # Нормализуем ground truth для сравнения
        gt_normalized = gt_context.strip().lower()[:500]

        # Ищем релевантный документ среди retrieved
        found_rank = None
        relevant_count = 0

        for rank, doc in enumerate(retrieved_docs[:top_k], 1):
            doc_normalized = doc.page_content.strip().lower()[:500]

            # Проверяем, содержит ли документ часть ground truth (или наоборот)
            # Используем пересечение слов как proxy для релевантности
            gt_words = set(gt_normalized.split())
            doc_words = set(doc_normalized.split())
            overlap = len(gt_words & doc_words)
            overlap_ratio = overlap / max(len(gt_words), 1)

            if overlap_ratio > 0.3:  # 30% перекрытие слов = релевантный
                relevant_count += 1
                if found_rank is None:
                    found_rank = rank

        # Recall@K: доля релевантных в топ-K
        recall_scores.append(min(relevant_count, 1))  # бинарно: нашли хоть один?

        # MRR@K: обратный ранг первого релевантного
        if found_rank:
            mrr_scores.append(1.0 / found_rank)
        else:
            mrr_scores.append(0.0)

        # Hit Rate@K: 1 если нашли хоть один релевантный
        hit_scores.append(1 if relevant_count > 0 else 0)

    return {
        f"recall@{top_k}": np.mean(recall_scores),
        f"mrr@{top_k}": np.mean(mrr_scores),
        f"hit_rate@{top_k}": np.mean(hit_scores),
    }


# ============ Generation метрики ============


def compute_generation_metrics(predictions: list, references: list) -> dict:
    """Вычисляет метрики качества генерации.

    Args:
        predictions: список сгенерированных ответов
        references: список эталонных ответов

    Returns:
        dict с bertscore_f1, rouge_l, bleu
    """
    # Фильтруем пустые ответы
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p and r and not p.startswith("Error")]
    if not valid_pairs:
        return {"bertscore_f1": 0.0, "bertscore_precision": 0.0, "bertscore_recall": 0.0, "rouge_l": 0.0, "bleu": 0.0}

    preds, refs = zip(*valid_pairs)

    # BERTScore
    print("    Computing BERTScore...")
    P, R, F1 = bert_score.score(list(preds), list(refs), lang="en", verbose=False)
    bertscore_f1 = F1.mean().item()
    bertscore_precision = P.mean().item()
    bertscore_recall = R.mean().item()

    # ROUGE-L
    print("    Computing ROUGE-L...")
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, pred)['rougeL'].fmeasure for pred, ref in zip(preds, refs)]
    rouge_l = np.mean(rouge_scores)

    # BLEU (sacreBLEU)
    print("    Computing BLEU...")
    bleu_scores = []
    for pred, ref in zip(preds, refs):
        if pred.strip():
            try:
                bleu = sacrebleu.sentence_bleu(pred, [ref])
                bleu_scores.append(bleu.score)
            except Exception:
                bleu_scores.append(0.0)
        else:
            bleu_scores.append(0.0)
    bleu = np.mean(bleu_scores)

    return {
        "bertscore_f1": bertscore_f1,
        "bertscore_precision": bertscore_precision,
        "bertscore_recall": bertscore_recall,
        "rouge_l": rouge_l,
        "bleu": bleu,
    }


# ============ Архитектуры ============


class NoContextLLM:
    """Baseline: LLM без контекста через OpenRouter."""

    def __init__(self):
        self.llm = get_llm()

    def run(self, question: str) -> dict:
        prompt = f"Question: {question}\n\nAnswer concisely and accurately:"

        t0 = time.perf_counter()
        try:
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            answer = f"Error: {e}"
        latency = time.perf_counter() - t0

        return {"answer": answer, "latency": latency, "retrieved_docs": []}


class RAGPipeline:
    """Стандартный RAG: retriever -> LLM через OpenRouter."""

    def __init__(self, vector_store: FAISS, top_k: int = TOP_K):
        self.vector_store = vector_store
        self.top_k = top_k
        self.llm = get_llm()

    def run(self, question: str) -> dict:
        # Retrieval
        t0 = time.perf_counter()
        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_k})
        docs = retriever.invoke(question)
        t_retrieve = time.perf_counter() - t0

        context = "\n\n".join(doc.page_content for doc in docs)

        # Generation
        prompt = f"""Context:
{context}

Question: {question}

Answer the question using only the context above. Be concise and accurate."""

        t1 = time.perf_counter()
        try:
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            answer = f"Error: {e}"
        t_gen = time.perf_counter() - t1

        return {
            "answer": answer,
            "latency": t_retrieve + t_gen,
            "retrieval_time": t_retrieve,
            "generation_time": t_gen,
            "retrieved_docs": docs,
            "num_docs": len(docs),
        }


class RAGWithReranker:
    """RAG + Reranker: retriever (top-N) -> reranker (top-K) -> LLM через OpenRouter."""

    def __init__(
        self,
        vector_store: FAISS,
        top_n: int = TOP_N,
        top_k: int = TOP_K,
        reranker_model: str = RERANKER_MODEL,
    ):
        self.vector_store = vector_store
        self.top_n = top_n
        self.top_k = top_k
        self.llm = get_llm()
        print(f"Loading reranker: {reranker_model}")
        self.reranker = HuggingFaceCrossEncoder(model_name=reranker_model, model_kwargs={"device": "cpu"})

    def run(self, question: str) -> dict:
        # Retrieval (с запасом)
        t0 = time.perf_counter()
        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.top_n})
        docs = retriever.invoke(question)
        t_retrieve = time.perf_counter() - t0

        # Reranking
        t1 = time.perf_counter()
        if len(docs) > self.top_k:
            pairs = [[question, doc.page_content] for doc in docs]
            scores = self.reranker.score(pairs)
            ranked_indices = np.argsort(scores)[::-1][: self.top_k]
            docs = [docs[i] for i in ranked_indices]
        t_rerank = time.perf_counter() - t1

        context = "\n\n".join(doc.page_content for doc in docs)

        # Generation
        prompt = f"""Context:
{context}

Question: {question}

Answer the question using only the context above. Be concise and accurate."""

        t2 = time.perf_counter()
        try:
            response = self.llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            answer = f"Error: {e}"
        t_gen = time.perf_counter() - t2

        return {
            "answer": answer,
            "latency": t_retrieve + t_rerank + t_gen,
            "retrieval_time": t_retrieve,
            "rerank_time": t_rerank,
            "generation_time": t_gen,
            "retrieved_docs": docs,
            "num_docs": len(docs),
        }


def build_full_corpus_index(n_samples: int = N_SAMPLES) -> tuple:
    """Строит FAISS индекс из ПОЛНОГО датасета SQuAD.

    Возвращает:
        vector_store: FAISS индекс со всеми контекстами
        questions: список вопросов для оценки
        ground_truth_contexts: список эталонных контекстов (для retrieval метрик)
        references: список эталонных ответов
    """
    # Загружаем полный validation split
    print("Loading full SQuAD validation dataset...")
    full_dataset = load_dataset("rajpurkar/squad", split="validation")

    # Извлекаем все уникальные контексты
    all_contexts = list(set(sample["context"] for sample in full_dataset))
    print(f"Found {len(all_contexts)} unique contexts in full dataset")

    # Разбиваем на чанки
    all_text = "\n\n".join(all_contexts)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_text(all_text)

    docs = [Document(page_content=chunk) for chunk in chunks]
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    print(f"Building FAISS index from {len(chunks)} chunks...")
    vector_store = FAISS.from_documents(docs, embeddings)

    # Выбираем samples для оценки
    samples = full_dataset.select(range(min(n_samples, len(full_dataset))))
    questions = [s["question"] for s in samples]
    ground_truth_contexts = [s["context"] for s in samples]
    references = [s["answers"]["text"][0] if s["answers"]["text"] else "" for s in samples]

    return vector_store, questions, ground_truth_contexts, references


def evaluate_architecture(pipeline, name: str, questions: list, ground_truth_contexts: list, references: list) -> dict:
    """Оценивает архитектуру и возвращает результаты."""
    print(f"\n--- Evaluating: {name} ---")

    predictions = []
    latencies = []
    all_retrieved_docs = []

    for i, question in enumerate(questions):
        print(f"  [{i + 1}/{len(questions)}] Processing...")
        result = pipeline.run(question)
        predictions.append(result["answer"])
        latencies.append(result["latency"])
        all_retrieved_docs.append(result.get("retrieved_docs", []))

    # Retrieval метрики
    print("  Computing retrieval metrics...")
    retrieval_metrics = compute_retrieval_metrics(all_retrieved_docs, ground_truth_contexts, TOP_K)

    # Generation метрики
    print("  Computing generation metrics...")
    gen_metrics = compute_generation_metrics(predictions, references)

    # Итоговые метрики
    metrics = {
        "architecture": name,
        # Retrieval
        "recall@k": retrieval_metrics[f"recall@{TOP_K}"],
        "mrr@k": retrieval_metrics[f"mrr@{TOP_K}"],
        "hit_rate@k": retrieval_metrics[f"hit_rate@{TOP_K}"],
        # Generation
        "bertscore_f1": gen_metrics["bertscore_f1"],
        "bertscore_precision": gen_metrics["bertscore_precision"],
        "bertscore_recall": gen_metrics["bertscore_recall"],
        "rouge_l": gen_metrics["rouge_l"],
        "bleu": gen_metrics["bleu"],
        # Latency
        "avg_latency": np.mean(latencies),
    }

    print(f"  Retrieval: Recall@{TOP_K}={metrics['recall@k']:.4f}, MRR@{TOP_K}={metrics['mrr@k']:.4f}, "
          f"Hit Rate@{TOP_K}={metrics['hit_rate@k']:.4f}")
    print(f"  Generation: BERTScore={metrics['bertscore_f1']:.4f}, ROUGE-L={metrics['rouge_l']:.4f}, "
          f"BLEU={metrics['bleu']:.4f}")
    print(f"  Latency: {metrics['avg_latency']:.2f}s")

    return {"metrics": metrics, "predictions": predictions, "latencies": latencies}


def main():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"=== Task 3: Architecture Comparison (Retrieval + Generation Metrics) ===")
    print(f"LLM: {OPENROUTER_MODEL}")
    print(f"Embeddings: {EMBED_MODEL}")
    print(f"Samples: {N_SAMPLES}")
    print(f"Top-K: {TOP_K}, Top-N: {TOP_N}")
    print(f"Metrics: Recall@K, MRR@K, Hit Rate@K, BERTScore, ROUGE-L, BLEU")
    print()

    # 1. Построение индекса из ПОЛНОГО датасета SQuAD и загрузка samples
    vector_store, questions, ground_truth_contexts, references = build_full_corpus_index(N_SAMPLES)
    print(f"Loaded {len(questions)} questions for evaluation")

    # 2. Инициализация архитектур
    no_context = NoContextLLM()
    rag = RAGPipeline(vector_store, top_k=TOP_K)
    rag_rerank = RAGWithReranker(vector_store, top_n=TOP_N, top_k=TOP_K)

    # 3. Оценка каждой архитектуры
    results = {}

    results["no_context"] = evaluate_architecture(no_context, "LLM (no context)", questions, ground_truth_contexts, references)
    results["rag"] = evaluate_architecture(rag, "RAG", questions, ground_truth_contexts, references)
    results["rag_rerank"] = evaluate_architecture(rag_rerank, "RAG + Reranker", questions, ground_truth_contexts, references)

    # 5. Создание сводного DataFrame
    df = pd.DataFrame([r["metrics"] for r in results.values()])
    df = df.set_index("architecture")
    print("\n=== Summary DataFrame ===")
    print(df.to_string())

    # 6. Сохранение результатов
    df.to_json(ARTIFACTS_DIR / "architecture_comparison.json", orient="index", indent=2)

    # Сохраняем детальные результаты
    detailed = {
        "config": {
            "n_samples": N_SAMPLES,
            "llm_model": OPENROUTER_MODEL,
            "embed_model": EMBED_MODEL,
            "reranker_model": RERANKER_MODEL,
            "top_k": TOP_K,
            "top_n": TOP_N,
        },
        "questions": questions,
        "ground_truth_contexts": ground_truth_contexts,
        "references": references,
        "results": {k: {"predictions": v["predictions"], "latencies": v["latencies"]} for k, v in results.items()},
    }

    with open(ARTIFACTS_DIR / "detailed_results.json", "w", encoding="utf-8") as f:
        json.dump(detailed, f, ensure_ascii=False, indent=2, default=str)

    print(f"\nSaved: {ARTIFACTS_DIR / 'architecture_comparison.json'}")
    print(f"Saved: {ARTIFACTS_DIR / 'detailed_results.json'}")
    print("=== Task 3 Complete ===")


if __name__ == "__main__":
    main()
