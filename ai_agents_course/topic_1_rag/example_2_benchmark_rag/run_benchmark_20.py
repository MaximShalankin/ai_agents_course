"""Benchmark: 5 QA samples (demo), retrieval (recall@k, MRR) and optional generation metrics (ROUGE, BLEU, BERTScore)."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.common import extract_text_from_rtf, chunk_by_points, DATA_RTF_PATH

import numpy as np
import ollama
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
SAMPLES_PATH = Path(__file__).resolve().parent / "samples_20.json"
N_SAMPLES = 5
TOP_K = 10
SEED = 42
GEN_TOP_K = 3
GEN_N_SAMPLES = 5


def get_embeddings(texts, model="bge-m3", batch_size=100):
    if isinstance(texts, str):
        texts = [texts]
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = ollama.embed(model=model, input=batch)
        out.append(np.array(resp["embeddings"], dtype=np.float32))
    return np.vstack(out)


def _parse_qa(llm_output):
    """Из вывода LLM извлекаем вопрос и ответ (ищем ВОПРОС:/ОТВЕТ:, или по разделителю)."""
    raw = llm_output.strip()
    question, answer = "", ""
    upper = raw.upper()
    for marker in ("ВОПРОС:", "ВОПРОС "):
        if marker in upper:
            idx = upper.find(marker)
            end = upper.find("ОТВЕТ:", idx + 1)
            if end == -1:
                end = len(raw)
            question = raw[idx + len(marker) : end].strip()
            if end < len(raw):
                answer = raw[end + 6 :].strip()
            break
    if not question and "\n" in raw:
        parts = raw.split("\n", 1)
        if parts[0].strip().endswith("?"):
            question = parts[0].strip()
            answer = parts[1].strip() if len(parts) > 1 else ""
    if not question:
        question = raw[:300].strip() or "О чём этот фрагмент?"
    if not answer:
        answer = raw[-500:].strip() if len(raw) > 500 else raw
    return question, answer


def build_samples_with_llm(chunks, model="qwen-q4"):
    """Генерируем N_SAMPLES пар (вопрос, эталонный ответ) по чанкам через LLM."""
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(chunks), size=min(N_SAMPLES, len(chunks)), replace=False)
    samples = []
    prompt_tpl = """Ниже фрагмент из нормативного документа. Придумай ОДИН конкретный вопрос по нему (на который этот текст отвечает) и краткий эталонный ответ (1-3 предложения).

Текст:
{text}

Формат ответа — только две строки, без лишнего текста:
ВОПРОС: твой вопрос здесь
ОТВЕТ: краткий ответ здесь"""
    for i, idx in enumerate(indices):
        ch = chunks[int(idx)]
        text = ch[:2500] if len(ch) > 2500 else ch
        prompt = prompt_tpl.format(text=text)
        try:
            r = ollama.generate(model=model, prompt=prompt)
            raw = r.get("response", "").strip()
        except Exception:
            raw = ""
        question, ground_truth = _parse_qa(raw)
        if not question:
            question = "О чём этот фрагмент нормативов?"
        if not ground_truth:
            ground_truth = ch[:500]
        samples.append({
            "question": question,
            "ground_truth": ground_truth,
            "chunk_idx": int(idx),
        })
        print(f"  Sample {i+1}/{N_SAMPLES}: question length {len(question)}, answer length {len(ground_truth)}")
    return samples


def load_or_build_samples(chunks, rebuild=False):
    if SAMPLES_PATH.exists() and not rebuild:
        with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"Generating {N_SAMPLES} QA pairs with qwen-q4 from chunks...")
    samples = build_samples_with_llm(chunks)
    SAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SAMPLES_PATH, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=1)
    print(f"Saved {SAMPLES_PATH}")
    return samples


def retrieval_metrics(chunk_embeddings, query_emb, relevant_idx):
    sims = cosine_similarity(query_emb, chunk_embeddings).squeeze()
    order = np.argsort(-sims)
    rank = np.where(order == relevant_idx)[0]
    rank = rank[0] + 1 if len(rank) else len(order) + 1
    recall_at_k = {}
    for k in [1, 3, 5, 10]:
        recall_at_k[f"recall@{k}"] = 1.0 if relevant_idx in order[:k] else 0.0
    mrr = 1.0 / rank
    return recall_at_k, mrr


def run_generation_metrics(samples, chunk_embeddings, query_embeddings, chunks):
    from rouge_score import rouge_scorer
    import sacrebleu
    from bert_score import score as bert_score_fun

    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    subset = samples[:GEN_N_SAMPLES]
    cands, refs = [], []
    for i, s in enumerate(subset):
        sims = cosine_similarity(query_embeddings[i : i + 1], chunk_embeddings).squeeze()
        top_idx = np.argsort(-sims)[:GEN_TOP_K]
        context = "\n\n".join(chunks[j] for j in top_idx)
        prompt = f"Контекст:\n{context[:3000]}\n\nВопрос: {s['question']}\n\nОтветь кратко по контексту."
        try:
            r = ollama.generate(model="qwen-q4", prompt=prompt)
            ans = r.get("response", "").strip()
        except Exception:
            ans = ""
        cands.append(ans)
        refs.append(s["ground_truth"][:500])
    rouge1_f = [rouge.score(ref, hyp)["rouge1"].fmeasure for ref, hyp in zip(refs, cands)]
    bleu_s = [sacrebleu.sentence_bleu(hyp, [ref], smooth_method="exp").score / 100.0 for ref, hyp in zip(refs, cands)]
    P, R, F1 = bert_score_fun(cands=cands, refs=refs, lang="ru", verbose=False)
    return {"rouge1_f": np.mean(rouge1_f), "bleu": np.mean(bleu_s), "bertscore_f1": float(F1.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-generation", action="store_true", help="Run generation + ROUGE/BLEU/BERTScore on 5 samples")
    ap.add_argument("--rebuild-samples", action="store_true", help="Regenerate 5 QA pairs with LLM (qwen-q4), overwrite samples_20.json")
    args = ap.parse_args()

    text = extract_text_from_rtf(DATA_RTF_PATH)
    if not text:
        return
    chunks = chunk_by_points(text)
    samples = load_or_build_samples(chunks, rebuild=args.rebuild_samples)
    if len(samples) > N_SAMPLES:
        samples = samples[:N_SAMPLES]

    print("Computing chunk embeddings...")
    chunk_embeddings = get_embeddings(chunks)
    print("Computing query embeddings...")
    query_texts = [s["question"] for s in samples]
    query_embeddings = get_embeddings(query_texts)

    rows = []
    for i, s in enumerate(samples):
        rec, mrr = retrieval_metrics(chunk_embeddings, query_embeddings[i : i + 1], s["chunk_idx"])
        rows.append({"sample_id": i, **rec, "mrr": mrr})

    df = pd.DataFrame(rows)
    means = df.drop(columns=["sample_id"]).mean().to_dict()
    print("Retrieval metrics (mean):")
    print(means)

    if args.with_generation:
        print("Running generation metrics (ROUGE, BLEU, BERTScore) on 5 samples...")
        gen_means = run_generation_metrics(samples, chunk_embeddings, query_embeddings, chunks)
        means.update(gen_means)
        print("Generation metrics:", gen_means)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_json(ARTIFACTS_DIR / "metrics_per_sample.json", orient="records", indent=2)
    pd.Series(means).to_json(ARTIFACTS_DIR / "metrics_summary.json", orient="index", indent=2)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    pd.Series(means).plot(kind="bar", ax=ax)
    ax.set_ylabel("Score")
    ax.set_title("Benchmark metrics (5 samples)")
    plt.xticks(rotation=25)
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "metrics_summary.png", dpi=100)
    plt.close()
    print(f"Saved {ARTIFACTS_DIR / 'metrics_summary.png'}")


if __name__ == "__main__":
    main()
