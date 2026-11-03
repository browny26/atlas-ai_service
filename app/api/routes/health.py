from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__
from app.api.deps import SettingsDep

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    model: str


@router.get("/health", response_model=HealthResponse)
async def health(settings: SettingsDep) -> HealthResponse:
    """Liveness probe. Does not call the LLM, so it is free and fast."""
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=settings.environment,
        model=settings.llm_model,
    )
