from __future__ import annotations

import asyncio

from api.limits import SlidingWindowRateLimiter


def test_sliding_window_rate_limiter_recovers_after_window() -> None:
    async def scenario() -> None:
        times = iter([0, 1, 2, 61])
        limiter = SlidingWindowRateLimiter(
            max_requests=2,
            window_seconds=60,
            clock=lambda: next(times),
        )
        assert await limiter.check("client") == (True, 0)
        assert await limiter.check("client") == (True, 0)
        allowed, retry_after = await limiter.check("client")
        assert allowed is False
        assert retry_after == 59
        assert await limiter.check("client") == (True, 0)

    asyncio.run(scenario())
