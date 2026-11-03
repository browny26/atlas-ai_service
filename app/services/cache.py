"""Response cache: identical requests reuse the previous answer instead of paying the LLM again.

In memory (per process) to keep the service dependency-free; the ItineraryCache Protocol
allows a shared Redis implementation when running several instances.
"""

import hashlib
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Protocol

from app.schemas.itinerary import ItineraryRequest, ItineraryResponse


class ItineraryCache(Protocol):
    def get(self, key: str) -> ItineraryResponse | None: ...

    def set(self, key: str, value: ItineraryResponse) -> None: ...


def cache_key(request: ItineraryRequest, model: str, prompt_version: str) -> str:
    # Model and prompt version are part of the key: changing either invalidates old answers.
    raw = f"{model}|{prompt_version}|{request.model_dump_json()}"
    return hashlib.sha256(raw.encode()).hexdigest()


class TTLCache:
    """LRU cache with a time-to-live per entry."""

    def __init__(
        self,
        ttl_seconds: float,
        max_entries: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._entries: OrderedDict[str, tuple[float, ItineraryResponse]] = OrderedDict()

    def get(self, key: str) -> ItineraryResponse | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if self._clock() >= expires_at:
            del self._entries[key]
            return None
        self._entries.move_to_end(key)  # most recently used
        return value

    def set(self, key: str, value: ItineraryResponse) -> None:
        self._entries[key] = (self._clock() + self._ttl, value)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)  # evict the least recently used

    def __len__(self) -> int:
        return len(self._entries)
