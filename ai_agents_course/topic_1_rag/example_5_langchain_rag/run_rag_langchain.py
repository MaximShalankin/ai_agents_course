"""
RAG на LangChain: загрузка данных из RTF, чанки -> эмбеддинги (Ollama bge-m3),
FAISS-индекс, retriever, LCEL-цепочка с ChatOllama (gemma3:1b). Наглядный запуск с запросами.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.common import extract_text_from_rtf, chunk_by_points, DATA_RTF_PATH

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
INDEX_DIR = ARTIFACTS_DIR / "faiss_index"
EMBED_MODEL = "bge-m3"
LLM_MODEL = "gemma3:1b"
TOP_K = 5

# Запросы для наглядного примера
SAMPLE_QUERIES = [
    "Какие требования к освещению?",
    "О чём говорится в пункте про вентиляцию?",
    "Что сказано про пожарную безопасность?",
]


def _format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def build_rag():
    text = extract_text_from_rtf(DATA_RTF_PATH)
    if not text:
        raise RuntimeError("Не удалось загрузить текст из RTF")
    chunks = chunk_by_points(text)
    docs = [Document(page_content=c) for c in chunks]
    print(f"Загружено чанков: {len(docs)}")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    if INDEX_DIR.exists():
        vector_store = FAISS.load_local(
            str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
        )
        print(f"Индекс загружен из {INDEX_DIR}")
    else:
        vector_store = FAISS.from_documents(docs, embeddings)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(INDEX_DIR))
        print(f"Индекс построен и сохранён в {INDEX_DIR}")

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
    return chain, retriever


def main():
    print("=== RAG на LangChain (Ollama) ===\n")
    chain, retriever = build_rag()

    for q in SAMPLE_QUERIES:
        print(f"Вопрос: {q}")
        context_docs = retriever.invoke(q)
        print("Топ чанки:")
        for i, doc in enumerate(context_docs, 1):
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            text = content[:400] + ("..." if len(content) > 400 else "")
            print(f"  [{i}] {text}")
        print()
        answer = (chain.invoke({"input": q}) or "").strip()
        print(f"Ответ: {answer}\n")


if __name__ == "__main__":
    main()
