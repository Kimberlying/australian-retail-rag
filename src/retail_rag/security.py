"""API-key authentication and per-client rate limiting for the HTTP API.

Both are deliberately in-process and dependency-free. That is correct for a
single replica; with several replicas behind a load balancer, move the rate
limiter's state to Redis (or enforce limits at the gateway) so a client cannot
multiply its budget by the replica count.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


def key_fingerprint(key: str) -> str:
    """A short, non-reversible id for logs and rate-limit buckets (never log the key)."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


class APIKeyAuth:
    def __init__(self, keys: list[str]):
        # Store digests, and compare digests in constant time so response timing
        # reveals nothing about how much of a guessed key was right.
        self._digests = [hashlib.sha256(key.encode("utf-8")).digest() for key in keys]

    @property
    def enabled(self) -> bool:
        return bool(self._digests)

    def verify(self, presented: str | None) -> bool:
        if not self.enabled:
            return True
        if not presented:
            return False
        digest = hashlib.sha256(presented.encode("utf-8")).digest()
        # Check every key (no early exit) so timing does not depend on which key matched.
        matches = [hmac.compare_digest(digest, known) for known in self._digests]
        return any(matches)


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    """Token bucket per client: ``burst`` requests at once, refilled at ``per_minute``.

    ``per_minute=0`` disables limiting.
    """

    max_clients = 10_000

    def __init__(
        self, per_minute: int, burst: int, *, monotonic: Callable[[], float] = time.monotonic
    ):
        self.rate = per_minute / 60.0
        self.burst = burst
        self._monotonic = monotonic
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        """Forget clients whose bucket has refilled: they are indistinguishable from new ones."""
        full_after = self.burst / self.rate
        self._buckets = {
            client: bucket
            for client, bucket in self._buckets.items()
            if now - bucket.updated < full_after
        }

    @property
    def enabled(self) -> bool:
        return self.rate > 0

    def acquire(self, client: str) -> float:
        """Take one token. Returns 0 if allowed, else the seconds until a token is available."""
        if not self.enabled:
            return 0.0
        now = self._monotonic()
        with self._lock:
            bucket = self._buckets.get(client)
            if bucket is None:
                if len(self._buckets) >= self.max_clients:
                    self._prune(now)
                bucket = self._buckets[client] = _Bucket(tokens=float(self.burst), updated=now)
            bucket.tokens = min(self.burst, bucket.tokens + (now - bucket.updated) * self.rate)
            bucket.updated = now
            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return 0.0
            return (1.0 - bucket.tokens) / self.rate

    @staticmethod
    def retry_after_header(wait_seconds: float) -> str:
        return str(max(1, math.ceil(wait_seconds)))
