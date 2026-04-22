import os
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Импорты для Reranker (современный подход LangChain)
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 1. Настройка LLM (OpenRouter)
os.environ["OPENROUTER_API_KEY"] = "ваш_ключ"
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    model="meta-llama/llama-3.3-70b-instruct"
)

# 2. Загрузка данных и создание векторной базы
loader = TextLoader("my_russian_data.txt", encoding="utf-8")
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# 3. Настройка базового ретривера (берем с запасом - 15 фрагментов)
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 15})

# 4. НАСТРОЙКА RERANKER ДЛЯ РУССКОГО ЯЗЫКА
# Используем мультиязычную модель bge-reranker-v2-m3
cross_encoder_model = HuggingFaceCrossEncoder(
    model_name="BAAI/bge-reranker-v2-m3",
    model_kwargs={"device": "cpu"} # Замените на "cuda", если есть видеокарта
)

# Указываем, что после переранжирования нам нужны только топ-3 лучших документа
reranker = CrossEncoderReranker(model=cross_encoder_model, top_n=3)

# Оборачиваем базовый ретривер в ретривер с компрессией (переранжированием)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker, 
    base_retriever=base_retriever
)

# 5. Промпт и функция форматирования
prompt = ChatPromptTemplate.from_messages([
    ("system", "Ты — ИИ-помощник. Ответь на вопрос на русском языке, используя только предоставленный контекст.\n\nКонтекст:\n{context}"),
    ("human", "{question}"),
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 6. Сборка RAG-цепочки (теперь используем compression_retriever)
rag_chain = (
    {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Запуск
question = "Какие ключевые метрики упоминаются в отчете?"
result = rag_chain.invoke(question)

print(result)
