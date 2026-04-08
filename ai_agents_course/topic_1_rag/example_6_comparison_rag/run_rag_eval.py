"""
Оценка двух RAG-пайплайнов на mirage-bench/ru (5 случайных примеров):
1) Custom RAG (example_3_custom_rag): ollama embed bge-m3, cosine top-k, ollama generate gemma3:1b
2) LangChain RAG (example_5_langchain_rag): FAISS + OllamaEmbeddings + LCEL (retriever | prompt | llm) + ChatOllama
Метрики: ROUGE, BLEU, BERTScore (эталон = context_ref из датасета).
"""
import logging
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent))
from data_ru import load_ru_dev_small, build_corpus

import numpy as np
import ollama
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import sacrebleu
from bert_score import score as bert_score_fun

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

ARTIFACTS_DIR = SCRIPT_DIR / "artifacts"
LOG_FILE = ARTIFACTS_DIR / "run_rag_eval.log"
N_SAMPLES = 35
TOP_K = 5
BATCH_SIZE = 50
MAX_RETRIES = 3
EMBED_MODEL = "bge-m3"
LLM_MODEL = "gemma3:1b"
REF_MAX_LEN = 500


def setup_logging():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return logging.getLogger(__name__)


def get_embeddings(texts, model=EMBED_MODEL, batch_size=BATCH_SIZE):
    if isinstance(texts, str):
        texts = [texts]
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = ollama.embed(model=model, input=batch)
        out.append(np.array(resp["embeddings"], dtype=np.float32))
    return np.vstack(out)


def run_custom_rag(df, corpus_ids, corpus_texts, log):
    """Custom pipeline: embed corpus, для каждого вопроса retrieve top-k, ollama generate."""
    log.info("Custom RAG: индексация корпуса (%s пассажей)...", len(corpus_texts))
    chunk_emb = get_embeddings(corpus_texts)
    questions = df["question"].tolist()
    query_emb = get_embeddings(questions)
    answers = []
    for i, q in enumerate(questions):
        sims = cosine_similarity(query_emb[i : i + 1], chunk_emb).squeeze()
        top_idx = np.argsort(-sims)[:TOP_K]
        context = "\n\n".join(corpus_texts[j] for j in top_idx)[:3500]
        prompt = f"Контекст:\n{context}\n\nВопрос: {q}\n\nОтветь кратко по контексту."
        answer = ""
        for attempt in range(MAX_RETRIES):
            try:
                r = ollama.generate(model=LLM_MODEL, prompt=prompt)
                answer = (r.get("response") or "").strip()
            except Exception as e:
                log.warning("Custom query %s attempt %s: %s", i + 1, attempt + 1, e)
            if answer:
                break
        answers.append(answer)
    return answers


def _format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def run_langchain_rag(df, corpus_texts, log):
    """LangChain pipeline: FAISS + retriever + LCEL (retriever | prompt | llm)."""
    log.info("LangChain RAG: построение FAISS (%s документов)...", len(corpus_texts))
    docs = [Document(page_content=t) for t in corpus_texts]
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_store = FAISS.from_documents(docs, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    llm = ChatOllama(model=LLM_MODEL)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Отвечай кратко только на основе контекста. Контекст:\n\n{context}"),
        ("human", "{input}"),
    ])
    def get_context(x):
        docs = retriever.invoke(x["input"])
        return _format_docs(docs)

    chain = (
        {"context": RunnableLambda(get_context), "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    answers = []
    for i, q in enumerate(df["question"]):
        try:
            answer = (chain.invoke({"input": q}) or "").strip()
        except Exception as e:
            log.warning("LangChain query %s: %s", i + 1, e)
            answer = ""
        answers.append(answer)
    return answers


def compute_metrics(answers, refs, log, prefix=""):
    refs_short = [r[:REF_MAX_LEN] for r in refs]
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    rows = []
    for j, (ans, ref) in enumerate(zip(answers, refs_short)):
        r1 = rouge.score(ref, ans)["rouge1"].fmeasure
        rL = rouge.score(ref, ans)["rougeL"].fmeasure
        bleu = sacrebleu.sentence_bleu(ans, [ref], smooth_method="exp").score / 100.0
        rows.append({"query_id": j, "rouge1_f": r1, "rougeL_f": rL, "bleu": bleu})
    df_metrics = pd.DataFrame(rows)
    P, R, F1 = bert_score_fun(cands=answers, refs=refs_short, lang="ru", verbose=False)
    df_metrics["bertscore_f1"] = F1.numpy()
    means = df_metrics.drop(columns=["query_id"]).mean()
    log.info("%s mean: rouge1=%.4f rougeL=%.4f bleu=%.4f bertscore_f1=%.4f",
             prefix, means["rouge1_f"], means["rougeL_f"], means["bleu"], means["bertscore_f1"])
    return df_metrics, means


def main():
    log = setup_logging()
    log.info("=== RAG eval mirage-bench/ru (n=%s) ===", N_SAMPLES)

    df = load_ru_dev_small(n=N_SAMPLES)
    log.info("Загружено примеров: %s", len(df))
    corpus_ids, corpus_texts = build_corpus(df)
    log.info("Корпус: %s пассажей", len(corpus_texts))
    refs = df["context_ref"].tolist()

    answers_custom = run_custom_rag(df, corpus_ids, corpus_texts, log)
    df_custom, means_custom = compute_metrics(answers_custom, refs, log, "Custom RAG")

    answers_lc = run_langchain_rag(df, corpus_texts, log)
    df_lc, means_lc = compute_metrics(answers_lc, refs, log, "LangChain RAG")

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df_custom.to_json(ARTIFACTS_DIR / "rag_metrics_custom.json", orient="records", indent=2)
    df_lc.to_json(ARTIFACTS_DIR / "rag_metrics_langchain.json", orient="records", indent=2)

    comparison = pd.DataFrame({
        "custom": means_custom,
        "langchain": means_lc,
    })
    comparison.to_json(ARTIFACTS_DIR / "rag_metrics_comparison.json", orient="index", indent=2)
    log.info("Saved rag_metrics_custom.json, rag_metrics_langchain.json, rag_metrics_comparison.json")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    x = range(len(means_custom))
    w = 0.35
    ax.bar([i - w / 2 for i in x], means_custom.values, width=w, label="Custom RAG")
    ax.bar([i + w / 2 for i in x], means_lc.values, width=w, label="LangChain RAG")
    ax.set_xticks(x)
    ax.set_xticklabels(means_custom.index, rotation=25)
    ax.set_ylabel("Score")
    ax.set_title("RAG evaluation mirage-bench/ru (ROUGE, BLEU, BERTScore)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "rag_metrics_comparison.png", dpi=100)
    plt.close()
    log.info("Saved rag_metrics_comparison.png | Log: %s", LOG_FILE)


if __name__ == "__main__":
    main()
