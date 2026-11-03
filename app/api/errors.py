"""Maps domain exceptions to HTTP responses, so routes contain no try/except."""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.services.itinerary_service import ItineraryGenerationError

logger = logging.getLogger(__name__)


async def _generation_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Itinerary generation failed: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_502_BAD_GATEWAY,
        content={"detail": "The AI provider could not generate a valid itinerary."},
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(ItineraryGenerationError, _generation_error_handler)
