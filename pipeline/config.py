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

# Per-run safety cap on API calls. Football is Pro (7500/day) and the per-fixture
# statistics calls add up, so this is well above a run's needs; basketball (Free)
# is rate-limited by the 6.5s throttle and makes few calls. Lower via APISPORTS_BUDGET
# if a sport is still on Free and you want a hard stop.
DAILY_REQUEST_BUDGET = int(os.environ.get("APISPORTS_BUDGET", "2000"))

# TEST MODE. When the current season is gated (e.g. the Free plan only exposes
# 2022–2024), fall back to the most recent ACCESSIBLE finished season so the whole
# app can be exercised with real (if not current) data. Leave this UNSET in
# production — then only the live current season is ever shown, and gated leagues
# are hidden. Enabled via the "Build feed" workflow's allow_finished_season input.
FEED_ALLOW_FINISHED = os.environ.get("FEED_ALLOW_FINISHED", "").strip().lower() in ("1", "true", "yes")
# How many seasons back to probe when falling back in test mode.
SEASON_LOOKBACK = int(os.environ.get("FEED_SEASON_LOOKBACK", "4"))

# Seconds to wait between API requests. The Free plan is ~10 req/min, so test
# runs set this higher (via the workflow) to stay under the per-minute cap; Pro
# allows the fast default.
REQUEST_THROTTLE_SEC = float(os.environ.get("APISPORTS_THROTTLE", "0.5"))

# Time-decay half-life (days) for Dixon-Coles weighting; ~3 seasons of history.
# 540 (not 365) minimised out-of-sample 1X2 log-loss on ~30k historical matches
# (see research/ backtest). Longer memory ⇒ steadier ratings.
DECAY_HALF_LIFE_DAYS = 540.0
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
    # Full enrichment (real players + injuries + corner/card statistics) costs a lot
    # of API calls. Keep it on for the headline leagues; off for the long tail, which
    # still get the model, standings, fixtures and a projection — just cheaper.
    enrich: bool = True
    # "league" gets a table + title/relegation projection; "cup" is knockout, so we
    # publish its fixtures + teams (simulatable) but no standings/projection.
    kind: str = "league"          # "league" | "cup"
    # Priority orders leagues in the app (lower = more prominent in "today").
    priority: int = 100


# API-Sports league ids (API-Football v3). The app shows every league that has
# current-season data; a gated/empty/unknown id is simply skipped at build time.
# Headline leagues are fully enriched; the wider set is cheaper (no players/injuries)
# so the daily quota stays well under Pro's budget. Cups add fixtures + teams only.
LEAGUES: list[League] = [
    # ── Headline leagues (fully enriched) ───────────────────────────────
    League("premier_league", "Premier League", "football", 39, "England", enrich=True, priority=1),
    League("laliga", "LaLiga", "football", 140, "Spain", enrich=True, priority=2),
    League("serie_a", "Serie A", "football", 135, "Italy", enrich=True, priority=3),
    League("bundesliga", "Bundesliga", "football", 78, "Germany", enrich=True, priority=4),
    League("ligue_1", "Ligue 1", "football", 61, "France", enrich=True, priority=5),
    League("super_lig", "Süper Lig", "football", 203, "Turkey", enrich=True, priority=6),
    # ── UEFA club competitions ──────────────────────────────────────────
    League("champions_league", "Champions League", "football", 2, "Europe", enrich=True, priority=7),
    League("europa_league", "Europa League", "football", 3, "Europe", enrich=False, priority=8),
    League("conference_league", "Conference League", "football", 848, "Europe", enrich=False, priority=9),
    # ── Most-followed leagues worldwide (cheaper, still simulated) ───────
    League("eredivisie", "Eredivisie", "football", 88, "Netherlands", enrich=False, priority=20),
    League("primeira_liga", "Primeira Liga", "football", 94, "Portugal", enrich=False, priority=21),
    League("championship", "Championship", "football", 40, "England", enrich=False, priority=22),
    League("brasileirao", "Brasileirão", "football", 71, "Brazil", enrich=False, priority=23),
    League("liga_profesional", "Liga Profesional", "football", 128, "Argentina", enrich=False, priority=24),
    League("liga_mx", "Liga MX", "football", 262, "Mexico", enrich=False, priority=25),
    League("mls", "MLS", "football", 253, "USA", enrich=False, priority=26),
    League("saudi_pro_league", "Saudi Pro League", "football", 307, "Saudi Arabia", enrich=False, priority=27),
    League("efl_1lig", "TFF 1. Lig", "football", 204, "Turkey", enrich=False, priority=28),
    League("jupiler", "Pro League", "football", 144, "Belgium", enrich=False, priority=29),
    League("eredivisie_scot", "Scottish Premiership", "football", 179, "Scotland", enrich=False, priority=30),
    League("super_league_gr", "Super League", "football", 197, "Greece", enrich=False, priority=31),
    League("j1_league", "J1 League", "football", 98, "Japan", enrich=False, priority=32),
    # ── Domestic cups (fixtures + teams only, no table) ─────────────────
    League("turkiye_kupasi", "Türkiye Kupası", "football", 206, "Turkey", enrich=False, kind="cup", priority=40),
    League("fa_cup", "FA Cup", "football", 45, "England", enrich=False, kind="cup", priority=41),
    League("efl_cup", "EFL Cup", "football", 48, "England", enrich=False, kind="cup", priority=42),
    League("copa_del_rey", "Copa del Rey", "football", 143, "Spain", enrich=False, kind="cup", priority=43),
    League("coppa_italia", "Coppa Italia", "football", 137, "Italy", enrich=False, kind="cup", priority=44),
    League("dfb_pokal", "DFB-Pokal", "football", 81, "Germany", enrich=False, kind="cup", priority=45),
    # ── Basketball ──────────────────────────────────────────────────────
    League("nba", "NBA", "basketball", 12, "USA", season_style="range", priority=10),
    League("euroleague", "EuroLeague", "basketball", 120, "Europe", season_style="year", priority=11),
]


def current_season() -> int:
    """API-Football season is the starting year of the campaign."""
    import datetime
    now = datetime.date.today()
    return now.year if now.month >= 7 else now.year - 1
