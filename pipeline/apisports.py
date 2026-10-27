"""Tiny API-Sports client with on-disk caching and a daily request budget.

Raw responses are cached under pipeline/cache/ keyed by endpoint+params, so refits
don't refetch history and we stay within the plan's request budget.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from config import CACHE_DIR, DAILY_REQUEST_BUDGET


class BudgetExceeded(RuntimeError):
    pass


class APISports:
    def __init__(self, key: str, cache_dir: Path = CACHE_DIR, budget: int = DAILY_REQUEST_BUDGET):
        self.key = key
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.budget = budget
        self.requests_made = 0

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
        with urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        self.requests_made += 1
        cache_path.write_text(json.dumps(data))
        # Be gentle with rate limits.
        time.sleep(0.5)
        return data
