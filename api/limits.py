"""Small in-memory request limiter for the single-instance public demo."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from collections.abc import Callable


class SlidingWindowRateLimiter:
    def __init__(
        self,
        max_requests: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clock = clock
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str) -> tuple[bool, int]:
        now = self._clock()
        cutoff = now - self.window_seconds

        async with self._lock:
            bucket = self._requests[key]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()

            if len(bucket) >= self.max_requests:
                retry_after = max(1, int(bucket[0] + self.window_seconds - now) + 1)
                return False, retry_after

            bucket.append(now)
            return True, 0
