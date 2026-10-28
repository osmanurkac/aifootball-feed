"""Pipeline configuration: leagues, API-Sports ids, seasons, budgets, paths.

Keys are read from the environment (GitHub Actions secret APISPORTS_KEY). No
secrets are stored in the repo.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / "cache"
OUT_DIR = ROOT / "out"

# API-Sports (api-sports.io) endpoints.
FOOTBALL_BASE = "https://v3.football.api-sports.io"
BASKETBALL_BASE = "https://v1.basketball.api-sports.io"

APISPORTS_KEY = os.environ.get("APISPORTS_KEY", "")

# Daily request budget (free plans are ~100/day). The client stops before this.
DAILY_REQUEST_BUDGET = int(os.environ.get("APISPORTS_BUDGET", "90"))

# TEST MODE. When the current season is gated (e.g. the Free plan only exposes
# 2022–2024), fall back to the most recent ACCESSIBLE finished season so the whole
# app can be exercised with real (if not current) data. Leave this UNSET in
# production — then only the live current season is ever shown, and gated leagues
# are hidden. Enabled via the "Build feed" workflow's allow_finished_season input.
FEED_ALLOW_FINISHED = os.environ.get("FEED_ALLOW_FINISHED", "").strip().lower() in ("1", "true", "yes")
# How many seasons back to probe when falling back in test mode.
SEASON_LOOKBACK = int(os.environ.get("FEED_SEASON_LOOKBACK", "4"))

# Time-decay half-life (days) for Dixon-Coles weighting; ~3 seasons of history.
DECAY_HALF_LIFE_DAYS = 365.0
HISTORY_SEASONS = 3


@dataclass(frozen=True)
class League:
    slug: str          # feed filename slug (football/<slug>.json)
    name: str          # display name, must match the app feed's league strings
    sport: str         # "football" | "basketball"
    api_id: int        # API-Sports league id
    country: str
    # How this league's seasons are identified in the API-Sports *basketball* API.
    # NBA uses a "YYYY-YYYY" range (e.g. "2024-2025"); EuroLeague uses the single
    # starting year (e.g. "2024"). Ignored for football (always a start year).
    season_style: str = "range"   # "range" | "year"


# API-Sports league ids. Football ids are stable on v3; basketball ids can vary
# by plan/season — verify in the API-Sports dashboard before the first live run.
LEAGUES: list[League] = [
    League("premier_league", "Premier League", "football", 39, "England"),
    League("laliga", "LaLiga", "football", 140, "Spain"),
    League("serie_a", "Serie A", "football", 135, "Italy"),
    League("bundesliga", "Bundesliga", "football", 78, "Germany"),
    League("ligue_1", "Ligue 1", "football", 61, "France"),
    League("champions_league", "Champions League", "football", 2, "Europe"),
    League("nba", "NBA", "basketball", 12, "USA", season_style="range"),
    League("euroleague", "EuroLeague", "basketball", 120, "Europe", season_style="year"),
]


def current_season() -> int:
    """API-Football season is the starting year of the campaign."""
    import datetime
    now = datetime.date.today()
    return now.year if now.month >= 7 else now.year - 1
