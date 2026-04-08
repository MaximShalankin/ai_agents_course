from datetime import datetime, timedelta
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from fastapi_mcp import FastApiMCP

# ---------- Модели маршрута ----------


class RouteRequest(BaseModel):
    origin: str = Field(..., description="Стартовая точка, например 'Moscow, Red Square'")
    destination: str = Field(..., description="Конечная точка, например 'Moscow, Sheremetyevo Airport'")
    mode: Literal["car", "walk", "bike"] = Field(
        "car",
        description="Способ передвижения: car, walk или bike",
    )


class RouteLeg(BaseModel):
    step: int
    instruction: str
    distance_km: float
    duration_min: int


class RouteResponse(BaseModel):
    origin: str
    destination: str
    mode: str
    distance_km: float
    duration_min: int
    eta: datetime
    legs: list[RouteLeg]


# ---------- Бизнес-логика ----------


def fake_route_planner(req: RouteRequest) -> RouteResponse:
    """
    Примитивный мок-сервис прогноза движения:
    - расстояние — фейковая эвристика по длине строк
    - скорость зависит от режима
    """
    base_distance = (len(req.origin) + len(req.destination)) * 0.5
    if req.mode == "car":
        speed = 40
    elif req.mode == "bike":
        speed = 15
    else:
        speed = 5

    duration_hours = base_distance / speed
    duration_min = int(duration_hours * 60)
    eta = datetime.utcnow() + timedelta(minutes=duration_min)

    legs = [
        RouteLeg(
            step=1,
            instruction=f"Start at {req.origin}",
            distance_km=round(base_distance * 0.3, 2),
            duration_min=int(duration_min * 0.3),
        ),
        RouteLeg(
            step=2,
            instruction=f"Continue towards {req.destination}",
            distance_km=round(base_distance * 0.5, 2),
            duration_min=int(duration_min * 0.5),
        ),
        RouteLeg(
            step=3,
            instruction=f"Arrive at {req.destination}",
            distance_km=round(base_distance * 0.2, 2),
            duration_min=int(duration_min * 0.2),
        ),
    ]

    return RouteResponse(
        origin=req.origin,
        destination=req.destination,
        mode=req.mode,
        distance_km=round(base_distance, 2),
        duration_min=duration_min,
        eta=eta,
        legs=legs,
    )


# ---------- FastAPI ----------

app = FastAPI(title="Route Planning API with MCP")


@app.post("/route", response_model=RouteResponse, tags=["route"], operation_id="plan_route")
async def plan_route_endpoint(request: RouteRequest):
    """
    Построить маршрут из точки A в точку B и получить прогноз времени в пути.
    """
    return fake_route_planner(request)


# ---------- MCP ----------

mcp = FastApiMCP(
    app,
    name="Route Planner MCP",
    description="MCP server that exposes a route planning tool based on a FastAPI service.",
)
mcp.mount_http(mount_path="/mcp")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
