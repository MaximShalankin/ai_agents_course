"""Simple RAG: index chunks with bge-m3, retrieve top-k, prompt Ollama (gemma3:1b), evaluate with ROUGE, BLEU, BERTScore."""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.common import extract_text_from_rtf, chunk_by_points, DATA_RTF_PATH

import numpy as np
import ollama
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import sacrebleu
from bert_score import score as bert_score_fun

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
LOG_FILE = ARTIFACTS_DIR / "run_rag_eval.log"
N_QUERIES = 5
TOP_K = 5
BATCH_SIZE = 100
MAX_RETRIES = 3


def setup_logging():
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.DEBUG,
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


def get_embeddings(texts, model="bge-m3", batch_size=BATCH_SIZE):
    if isinstance(texts, str):
        texts = [texts]
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = ollama.embed(model=model, input=batch)
        out.append(np.array(resp["embeddings"], dtype=np.float32))
    return np.vstack(out)


def main():
    log = setup_logging()
    log.info("=== run_rag_eval start ===")
    log.info("DATA_RTF_PATH=%s", DATA_RTF_PATH)

    text = extract_text_from_rtf(DATA_RTF_PATH)
    if not text:
        log.error("No text loaded from RTF")
        return
    chunks = chunk_by_points(text)
    log.info("Chunks loaded: %s", len(chunks))

    rng = np.random.default_rng(42)
    indices = rng.choice(len(chunks), size=min(N_QUERIES, len(chunks)), replace=False)
    queries_gt = [(f"О чём этот пункт нормативов? {chunks[i][:60]}...", chunks[i]) for i in indices]
    log.info("Query indices: %s", indices.tolist())

    log.info("Indexing chunks (embeddings)...")
    chunk_emb = get_embeddings(chunks)
    log.info("Embedding queries...")
    query_texts = [q for q, _ in queries_gt]
    query_emb = get_embeddings(query_texts)

    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    answers = []
    refs_short = []
    for i, (question, ground_truth) in enumerate(queries_gt):
        log.debug("--- Query %s --- question: %.80s...", i + 1, question)
        sims = cosine_similarity(query_emb[i : i + 1], chunk_emb).squeeze()
        top_idx = np.argsort(-sims)[:TOP_K]
        context = "\n\n".join(chunks[j] for j in top_idx)[:3500]
        prompt = f"Контекст из нормативов:\n{context}\n\nВопрос: {question}\n\nОтветь кратко по контексту."
        log.debug("Prompt length: %s chars, context chunks: %s", len(prompt), top_idx.tolist())
        answer = ""
        for attempt in range(MAX_RETRIES):
            try:
                r = ollama.generate(model="gemma3:1b", prompt=prompt)
                answer = r.get("response", "").strip()
                if not answer:
                    log.warning(
                        "Query %s attempt %s: empty response | keys=%s done_reason=%s response_len=%s",
                        i + 1, attempt + 1,
                        list(r.keys()) if isinstance(r, dict) else type(r).__name__,
                        r.get("done_reason"), len(r.get("response", "")),
                    )
            except Exception as e:
                log.warning("Query %s attempt %s: %s", i + 1, attempt + 1, e, exc_info=True)
            if answer:
                log.debug("Query %s answer length=%s, head: %.100s...", i + 1, len(answer), answer)
                break
            if attempt < MAX_RETRIES - 1:
                log.warning("Query %s: empty response, retry %s/%s", i + 1, attempt + 2, MAX_RETRIES)
        if not answer:
            log.warning("Query %s: still empty after %s retries", i + 1, MAX_RETRIES)
        answers.append(answer)
        refs_short.append(ground_truth[:500])

    rows = []
    for j, (ans, ref) in enumerate(zip(answers, refs_short)):
        r1 = rouge.score(ref, ans)["rouge1"].fmeasure
        rL = rouge.score(ref, ans)["rougeL"].fmeasure
        bleu = sacrebleu.sentence_bleu(ans, [ref], smooth_method="exp").score / 100.0
        rows.append({"query_id": j, "rouge1_f": r1, "rougeL_f": rL, "bleu": bleu})
        log.debug("Query %s metrics: rouge1=%.4f rougeL=%.4f bleu=%.4f", j + 1, r1, rL, bleu)

    df = pd.DataFrame(rows)
    P, R, F1 = bert_score_fun(cands=answers, refs=refs_short, lang="ru", verbose=False)
    df["bertscore_f1"] = F1.numpy()
    for j, f1 in enumerate(F1.numpy()):
        log.debug("Query %s bertscore_f1=%.4f", j + 1, f1)

    means = df.drop(columns=["query_id"]).mean()
    log.info("RAG eval metrics (mean): rouge1_f=%.4f rougeL_f=%.4f bleu=%.4f bertscore_f1=%.4f",
             means["rouge1_f"], means["rougeL_f"], means["bleu"], means["bertscore_f1"])
    empty_count = sum(1 for a in answers if not a)
    log.info("Empty answers: %s / %s", empty_count, len(answers))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_json(ARTIFACTS_DIR / "rag_metrics.json", orient="records", indent=2)
    means.to_json(ARTIFACTS_DIR / "rag_metrics_summary.json", orient="index", indent=2)
    log.info("Saved rag_metrics.json, rag_metrics_summary.json")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    means.plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_title("RAG evaluation (ROUGE, BLEU, BERTScore)")
    plt.xticks(rotation=25)
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "rag_metrics.png", dpi=100)
    plt.close()
    log.info("Saved rag_metrics.png | Log file: %s", LOG_FILE)


if __name__ == "__main__":
    main()
