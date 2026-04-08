# RAG: от данных до оценки

**example_1_chunking_rag** — Подготовка данных: RTF/PDF, чанкинг по нумерованным пунктам (regex по границам). Артефакт: распределение длин чанков.

**example_2_benchmark_rag** — Бенчмарк на 5 сэмплах: пары (вопрос, эталон). Метрики: retrieval (recall@k, MRR). Артефакты: JSON.

**example_3_custom_rag** — RAG без фреймворков: эмбеддинги → top-k по косинусу → промпт с контекстом → LLM. Оценка: ROUGE, BLEU, BERTScore. Артефакты: JSON.

**example_4_video_rag** — Мультимодальный RAG по видео (data/video/video.mp4): Whisper, поиск по запросу с timecode.

**example_5_langchain_rag** — RAG на LangChain (LCEL): RTF → чанки → FAISS, retriever | prompt | ChatOllama.

**example_6_comparison_rag** — Сравнение custom RAG и LangChain RAG (LCEL) на mirage-bench/ru (5 сэмплов). Данные: data/benchmarks/mirage-bench. Артефакты: JSON.
