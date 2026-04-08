"""Честный MCP-сервер: маршрут по origin/destination, без утечек."""
from fastapi import FastAPI
from pydantic import BaseModel, Field

from fastapi_mcp import FastApiMCP

HOST = "0.0.0.0"
PORT = 8000


class RouteRequest(BaseModel):
    origin: str = Field(..., description="Точка отправления")
    destination: str = Field(..., description="Точка назначения")


class RouteResponse(BaseModel):
    distance_km: float = Field(..., description="Расстояние в км")
    eta_minutes: int = Field(..., description="Оценка времени в минутах")


app = FastAPI(title="honest-route-mcp")


@app.post("/route", response_model=RouteResponse, tags=["route"], operation_id="plan_route")
async def plan_route(body: RouteRequest):
    """Построить маршрут между двумя точками и вернуть оценку расстояния и времени."""
    return RouteResponse(distance_km=10.0, eta_minutes=20)


mcp = FastApiMCP(
    app,
    name="honest-route-mcp",
    description="MCP server: построить маршрут между двумя точками (origin, destination).",
)
mcp.mount_http(mount_path="/mcp")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
