# Агенты и workflow

**example_1_react_agents** — Custom ReAct и LangGraph (StateGraph + ToolNode). Инструмент: run_python_code. Оценка на 5 сэмплах HumanEval: executable, test_passed, n_steps. Артефакты: JSON + график.

**example_2_workflow_agents** — Workflow на LCEL: три шага (анализ → план → код), ChatOllama + StrOutputParser. Запуск: run_workflow.py, артефакты: JSON по сэмплам HumanEval.

**example_3_supervisor_agents** — Мультиагентный Supervisor (langgraph-supervisor): coder + tester под управлением супервизора. Запуск: run_demo.py, артефакт: JSON.