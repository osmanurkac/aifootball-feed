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


# API-Sports league ids. Football ids are stable on v3; basketball ids can vary
# by plan/season — verify in the API-Sports dashboard before the first live run.
LEAGUES: list[League] = [
    League("premier_league", "Premier League", "football", 39, "England"),
    League("laliga", "LaLiga", "football", 140, "Spain"),
    League("serie_a", "Serie A", "football", 135, "Italy"),
    League("bundesliga", "Bundesliga", "football", 78, "Germany"),
    League("ligue_1", "Ligue 1", "football", 61, "France"),
    League("champions_league", "Champions League", "football", 2, "Europe"),
    League("nba", "NBA", "basketball", 12, "USA"),
    League("euroleague", "EuroLeague", "basketball", 120, "Europe"),
]


def current_season() -> int:
    """API-Football season is the starting year of the campaign."""
    import datetime
    now = datetime.date.today()
    return now.year if now.month >= 7 else now.year - 1
