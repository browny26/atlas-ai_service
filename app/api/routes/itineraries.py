import json
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.deps import get_itinerary_service, require_api_key
from app.schemas.itinerary import ItineraryRequest, ItineraryResponse, RegenerateDayRequest
from app.services.itinerary_service import ItineraryService

router = APIRouter(
    prefix="/v1/itineraries", tags=["itineraries"], dependencies=[Depends(require_api_key)]
)

ServiceDep = Annotated[ItineraryService, Depends(get_itinerary_service)]

_ERRORS: dict[int | str, dict[str, str]] = {
    401: {"description": "Invalid or missing API key"},
    502: {"description": "The LLM could not produce a valid itinerary (fallback disabled)"},
}


@router.post(
    "", response_model=ItineraryResponse, summary="Generate a travel itinerary", responses=_ERRORS
)
async def generate_itinerary(request: ItineraryRequest, service: ServiceDep) -> ItineraryResponse:
    return await service.generate(request)


@router.post(
    "/stream",
    summary="Generate a travel itinerary, streamed as Server-Sent Events",
    response_class=StreamingResponse,
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": (
                "Events: `day` (one validated day, as soon as it is written), `status` "
                "(progress message), `result` (the final ItineraryResponse, authoritative), "
                "`error`."
            ),
        },
        401: _ERRORS[401],
    },
)
async def stream_itinerary(request: ItineraryRequest, service: ServiceDep) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        async for event in service.generate_stream(request):
            yield f"event: {event.event}\ndata: {json.dumps(event.data, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # Disable buffering in proxies, otherwise events arrive all together at the end.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post(
    "/regenerate-day",
    response_model=ItineraryResponse,
    summary="Rewrite one day of an itinerary, optionally following user feedback",
    responses=_ERRORS,
)
async def regenerate_day(request: RegenerateDayRequest, service: ServiceDep) -> ItineraryResponse:
    return await service.regenerate_day(request)
