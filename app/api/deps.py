"""FastAPI dependencies: how routes get their collaborators.

Routes ask for an ItineraryService; tests override get_llm_client with a fake.
"""

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import Settings, get_settings
from app.llm.base import LLMClient
from app.services.itinerary_service import ItineraryService

SettingsDep = Annotated[Settings, Depends(get_settings)]

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(settings: SettingsDep, api_key: str | None = Security(_api_key_header)) -> None:
    """Service-to-service authentication. Disabled when SERVICE_API_KEY is empty."""
    expected = settings.service_api_key.get_secret_value()
    if not expected:
        return
    if api_key is None or not secrets.compare_digest(api_key, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key")


def get_llm_client(request: Request) -> LLMClient:
    # Created once at startup (see app.main.lifespan) and reused: one HTTP connection pool.
    client: LLMClient = request.app.state.llm_client
    return client


def get_itinerary_service(
    settings: SettingsDep, llm: Annotated[LLMClient, Depends(get_llm_client)]
) -> ItineraryService:
    return ItineraryService(
        llm,
        max_attempts=settings.generation_max_attempts,
        enable_fallback=settings.enable_fallback,
    )
