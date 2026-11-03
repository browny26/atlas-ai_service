from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.deps import get_itinerary_service, require_api_key
from app.schemas.itinerary import ItineraryRequest, ItineraryResponse
from app.services.itinerary_service import ItineraryService

router = APIRouter(
    prefix="/v1/itineraries", tags=["itineraries"], dependencies=[Depends(require_api_key)]
)


@router.post(
    "",
    response_model=ItineraryResponse,
    summary="Generate a travel itinerary",
    responses={
        401: {"description": "Invalid or missing API key"},
        502: {"description": "The LLM could not produce a valid itinerary (fallback disabled)"},
    },
)
async def generate_itinerary(
    request: ItineraryRequest,
    service: Annotated[ItineraryService, Depends(get_itinerary_service)],
) -> ItineraryResponse:
    return await service.generate(request)
