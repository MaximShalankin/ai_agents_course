## Репозитории команд

| Team | Repo | Key features |
|---|---|---|
| team_01 | [KStepura/agents](https://github.com/KStepura/agents) | Dual-mode Builder (LLM critic / deterministic); stacking и pseudo-labels; сильные guardrails; архитектура без внешнего agent-фреймворка |
| team_02 | [kre1ses/Agentic_System](https://github.com/kre1ses/Agentic_System) | Fail-fast validator до планирования; rule-based режим без LLM; multi-provider с fallback; classifier+regressor под zero-inflated target; tag-aware RAG; experiment-to-knowledge loop |
| team_03 | [timamz/data-science-agent](https://github.com/timamz/data-science-agent) | Фабрика LLM (Ollama/HF); guardrails (injection, allowlist гиперпараметров, rate limits); RAG по ML best practices; бенчмаркинг агентов; experiment memory в JSON |
| team_04 | [Fakhretdinov-A/ds_agent](https://github.com/Fakhretdinov-A/ds_agent) | Полная codegen пайплайна; auto-feedback по метрикам; сжатие контекста; сильные результаты на валидации; RAG по docstrings библиотек |
| team_05 | [annalzrv/rentals_downtime](https://github.com/annalzrv/rentals_downtime) | Repair loop до 5 попыток; изолированное выполнение в subprocess; artifact contracts; generic solver; graceful degradation при сбоях web/LLM |
| team_06 | [AlexxanderrSid/Multi-agent-system-for-regression](https://github.com/AlexxanderrSid/Multi-agent-system-for-regression)  | Две модели (7B coder + 72B reasoning) через Ollama; Docker Compose; единый State как контракт команды; приоритет guardrails над маршрутизацией; тесты с mock для CI |
| team_07 | [sodeniZzz/watafa-agentic-ml](https://github.com/sodeniZzz/watafa-agentic-ml) | WATAFA: pipeline из 10 узлов с LLM-judge на этапах; AST guardrails; автоопределение regression/classification; Voting + Stacking; защита от CSV injection |
| team_09 | —  | Модульные блоки вместо монолита; лимиты debug с fallback; meta-контракт данных; RAG-debug (база ошибок); независимый Validator; сильный ExtraTrees |
| team_10 | [sacr1ficerq/mts-red-balls](https://github.com/sacr1ficerq/mts-red-balls) | Web UI FastAPI + Alpine.js + SSE; 260+ тестов; богатый набор Kaggle tools; sandbox и защита от path/command injection; semantic RAG с 3-уровневым fallback; autosave сессий |
| team_11 | [mdkrs/mws-ai-agents](https://github.com/mdkrs/mws-ai-agents) | Critic–Executor loop; полная работа без LLM (fallback); персистентная Agent Memory; adversarial robustness тесты; PipelineMonitor JSON |
| team_12 | [Bochkov1/mws-ai-agents-2026](https://github.com/Bochkov1/mws-ai-agents-2026) | Docker sandbox без сети; verification funnel (compile → AST → import → schema → smoke); AST anti-leakage; role-scoped записи в FS; сильный набор тестов |
| team_13 | [AmKovylyaev/ai_agents_course](https://github.com/AmKovylyaev/ai_agents_course) | Параллельные candidate-ветки и Judge; мини-loop Planner–Coder–Verifier; гибридный RAG BM25 + FAISS + RRF; детерминированный fallback на шагах; полная трассировка артефактов |
| team_14 | — | Минимальный каркас (ipynb + README); 7 агентов; до 60 итераций улучшений; multi-candidate генерация; plateau detection |
| team_15 | [While-true-codeanything/AIAgents_FinalProject_KaggleSolver](https://github.com/While-true-codeanything/AIAgents_FinalProject_KaggleSolver) | Разные модели под роли (thinking vs coder); двухуровневый debug; авто-submit на Kaggle; фиксированный holdout; RouterAI; ветка на AutoGen |
| team_16 | [ArthurGaleev/kaggle-mas](https://github.com/ArthurGaleev/kaggle-mas) | SafetyGuard (15 классов угроз); adaptive feedback с порогом улучшения; target encoding строго в CV; multi-provider LLM + backoff; мониторинг и дашборд; 9 групп feature engineering |
| team_17 | [sergak0/mws-ai-agents](https://github.com/sergak0/mws-ai-agents) | Production-уровень логирования и guardrails; thread-safe LLM client + GPU; планы через LLM, исполнение детерминированное; adaptive early stop; 43 pytest с mock LLM; сильные итоги кросс-валидации |
| team_18 | [IoplachkinI/mts-ai-agents (GitLab)](https://gitlab.com/IoplachkinI/mts-ai-agents) | Детерминированный planner без LLM; feedback с blacklist моделей; tools вместо codegen; hybrid RAG + RRF; принудительный OSS-only; Langflow-компоненты; подробное JSON-логирование |
| team_19 | [ProgiFrogi/Agentic-Modeling-Operational-Engineering-Beta-Application](https://github.com/ProgiFrogi/Agentic-Modeling-Operational-Engineering-Beta-Application) | Tool-driven исполнение; sanitizer против prompt injection в RAG; полный fallback без LLM; offline-метрики по solution.csv; W&B; обширная документация и трассировка ранов |
| team_20 | [Durakavalyanie/MTS_AI_AGENTS](https://github.com/Durakavalyanie/MTS_AI_AGENTS) | Многоуровневые ретраи и fingerprint ошибок; guardrails до LLM-verifier; ground truth из FS в промпте verifier; TUI на Textual; epoch-директории; жёсткий Docker; опциональный RAG по ноутбукам |
| team_21 | — | Вложенные LangGraph на агента; ChromaDB + CrossEncoder rerank; защита RAG от injection (DeBERTa); vLLM для вспомогательных задач; изолированные сессии; авто-submit; известный баг подстановки чанков в промпт |
| team_22 | — | AutoGen + кастомный текстовый протокол tool calls; orchestrator-workers и вложенные code loops; AST code policy; бесплатные модели OpenRouter; поэтапный polling Kaggle |

## Лучшие практики

Сводный образ сильных решений: оркестрация на **LangGraph** (или зрелый pure-Python аналог) с **типизированным state** и режимом **Plan & Execute**, где тяжёлая работа идёт через **детерминированные tools**, а опциональная **code generation** всегда прикрыта **AST-, policy- и sandbox-проверками**. **LLM** подключают через **единый клиент** с **несколькими провайдерами, retry и fallback**, плюс **режим без API** (правила, локальные модели), чтобы пайплайн не останавливался. **RAG** там, где нужен: **гибрид lexical + vector** с **RRF** или rerank, отдельно — **память ошибок**, **trajectory reuse**; критично **smoke-тестировать**, что retrieved-текст реально попадает в промпт. На стороне ML доминируют **GBDT-ансамбли** с **stacking/blending**, **Optuna** и **target encoding только внутри CV**; цикл улучшений замыкает **orchestrator** с решениями вроде **ACCEPT/IMPROVE**, **ретраями** и при необходимости **epoch/restart** и **жёсткими контрактами артефактов** между шагами. **Безопасность** — это **input/output/safety** guardrails, **Docker** с отключённой сетью там, где исполняется чужой код, и **верификация** до дорогих LLM-вызовов.

## Визуализации

### Поток по архитектурным измерениям

<img src="imgs/viz_01_alluvial.png" alt="Аллювиальная диаграмма: архитектура и tier" width="1000">

Поток команд по цепочке (фреймворк → LLM → безопасность → tier): толщина ленты — сколько команд на пути, цвет — исходная категория.

### Поток по технологическому стеку

<img src="imgs/viz_12_tech_alluvial.png" alt="Аллювиальная диаграмма: технологический стек" width="1000">

Поток по слоям стека (фреймворк → RAG → vector DB → стратегия LLM); про связки решений, не про баллы.

### Сеть неочевидных зависимостей (PMI)

<img src="imgs/viz_17_nonobvious_dependencies_network.png" alt="Граф PMI между технологиями" width="1000">

Узлы — технологии (с группировкой по смыслу), рёбра — PMI; размер узла — частота. Неочевидные созависимости, не просто популярные пары.

### Тепловая карта пересечений агентных сигналов

<img src="imgs/viz_18_agent_dependencies_heatmap.png" alt="Тепловая карта Jaccard по агентным токенам" width="1000">

Jaccard между токенами (RAG/agents/tooling/runtime/safety)
