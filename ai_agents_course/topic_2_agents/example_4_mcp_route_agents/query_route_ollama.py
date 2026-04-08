"""
Текстовый запрос → Ollama извлекает параметры → вызов сервиса маршрутов (REST /route).
Сервер route planner (server.py) должен быть запущен: uvicorn server:app --port 8000.
"""
import json
import re
import sys
import urllib.request

import ollama

ROUTE_URL = "http://127.0.0.1:8000/route"
OLLAMA_MODEL = "qwen2.5-coder:1.5b"  # или llama3.2, mistral и т.д.

SYSTEM_PROMPT = """You are a route planning assistant. From the user's message extract:
- origin: start point (place or address)
- destination: end point (place or address)
- mode: one of "car", "walk", "bike" (default "car" if not specified)

Reply with ONLY a single JSON object, no other text. Example:
{"origin": "Moscow, Red Square", "destination": "Sheremetyevo Airport", "mode": "car"}"""


def extract_route_params(text: str) -> dict:
    """Извлекает origin, destination, mode из текста через Ollama."""
    r = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    content = (r.get("message") or {}).get("content") or ""
    raw = content.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    start = raw.find("{")
    if start < 0:
        raise ValueError(f"В ответе модели нет JSON: {content[:300]}")
    depth = 0
    for i, c in enumerate(raw[start:], start):
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                data = json.loads(raw[start : i + 1])
                if "origin" in data and "destination" in data:
                    break
                data = None
    else:
        data = None
    if not data:
        raise ValueError(f"Не удалось извлечь JSON с origin/destination: {content[:300]}")
    origin = data.get("origin") or ""
    destination = data.get("destination") or ""
    mode = (data.get("mode") or "car").lower()
    if mode not in ("car", "walk", "bike"):
        mode = "car"
    return {"origin": origin, "destination": destination, "mode": mode}


def call_route_service(origin: str, destination: str, mode: str) -> dict:
    """Вызов REST /route сервиса маршрутов."""
    body = json.dumps({"origin": origin, "destination": destination, "mode": mode}).encode("utf-8")
    req = urllib.request.Request(
        ROUTE_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Запрос (маршрут из A в B): ").strip()
    if not query:
        print("Задайте текстовый запрос.", file=sys.stderr)
        sys.exit(1)

    params = extract_route_params(query)
    print("Параметры:", params, flush=True)
    result = call_route_service(params["origin"], params["destination"], params["mode"])
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
