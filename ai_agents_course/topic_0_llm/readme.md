# LLM: квантование и оценка кодогенерации

**example_1_compare_llm** — Загрузка GGUF-моделей (Hugging Face), сравнение FP16 и Q4_K_M через Ollama: время, ток/сек. Скрипты: `download_models.py`, `run_compare.py`.

**example_2_codegen_llm** — Оценка кодогенерации на HumanEval (5 сэмплов): промпт → Ollama → извлечение кода → проверка синтаксиса (ast.parse). Артефакты: JSON.