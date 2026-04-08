"""Мультиагентная система с паттерном Supervisor: coder + tester под управлением супервизора (langgraph-supervisor)."""
from tools import write_python_code, run_tests
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor

MODEL = "qwen2.5-coder:1.5b"
llm = ChatOllama(model=MODEL, temperature=0)

coder_agent = create_react_agent(
    llm,
    tools=[write_python_code],
    name="coder",
    prompt="Ты Python-разработчик. Пиши короткий и рабочий код по задаче.",
)
tester_agent = create_react_agent(
    llm,
    tools=[run_tests],
    name="tester",
    prompt="Ты QA. Запускай тесты для переданного кода через run_tests и сообщай результат.",
)

workflow = create_supervisor(
    [coder_agent, tester_agent],
    model=llm,
    prompt=(
        "Ты техлид. Есть специалисты: coder (пишет код) и tester (проверяет код). "
        "Доступные инструменты: transfer_to_coder(task), transfer_to_tester(code). "
        "За один ответ вызывай только один инструмент. "
        "Шаг 1: вызови transfer_to_coder и передай описание задачи (task). Дождись ответа с кодом. "
        "Шаг 2: вызови transfer_to_tester и передай в code именно тот исходный код (текст), который вернул coder. "
        "Когда тесты пройдут — заверши работу."
    ),
    include_agent_name="inline",
)
app = workflow.compile()
