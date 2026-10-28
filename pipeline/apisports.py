"""Tiny API-Sports client with on-disk caching and a daily request budget.

Raw responses are cached under pipeline/cache/ keyed by endpoint+params, so refits
don't refetch history and we stay within the plan's request budget.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from config import CACHE_DIR, DAILY_REQUEST_BUDGET, REQUEST_THROTTLE_SEC


class BudgetExceeded(RuntimeError):
    pass


class APISports:
    def __init__(self, key: str, cache_dir: Path = CACHE_DIR, budget: int = DAILY_REQUEST_BUDGET,
                 throttle: float = REQUEST_THROTTLE_SEC):
        self.key = key
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.budget = budget
        self.throttle = throttle
        self.requests_made = 0

    def _fetch(self, req, attempts: int = 6) -> dict:
        """Fetch with backoff on HTTP 429 (the Free plan's per-minute limit)."""
        delay = 8.0
        for i in range(attempts):
            try:
                with urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode())
            except HTTPError as exc:
                if exc.code == 429 and i < attempts - 1:
                    print(f"    429 rate-limited — waiting {delay:.0f}s (attempt {i + 1})")
                    time.sleep(delay)
                    delay = min(delay * 1.5, 30.0)
                    continue
                raise

    def _cache_path(self, base: str, path: str, params: dict) -> Path:
        raw = f"{base}/{path}?{urlencode(sorted(params.items()))}"
        digest = hashlib.sha1(raw.encode()).hexdigest()[:16]
        safe = path.strip("/").replace("/", "_")
        return self.cache_dir / f"{safe}_{digest}.json"

    def get(self, base: str, path: str, params: dict | None = None, ttl_hours: float = 12.0) -> dict:
        params = params or {}
        cache_path = self._cache_path(base, path, params)

        # Serve fresh cache.
        if cache_path.exists():
            age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
            if age_hours < ttl_hours:
                return json.loads(cache_path.read_text())

        if not self.key:
            # No key: fall back to any stale cache, else empty response.
            if cache_path.exists():
                return json.loads(cache_path.read_text())
            return {"response": []}

        if self.requests_made >= self.budget:
            if cache_path.exists():
                return json.loads(cache_path.read_text())
            raise BudgetExceeded(f"Daily request budget of {self.budget} reached")

        url = f"{base}/{path}"
        if params:
            url += "?" + urlencode(params)
        req = Request(url, headers={"x-apisports-key": self.key})
        data = self._fetch(req)
        self.requests_made += 1
        cache_path.write_text(json.dumps(data))
        # Be gentle with rate limits (Free plan ~10/min → set throttle higher).
        time.sleep(self.throttle)
        return data
