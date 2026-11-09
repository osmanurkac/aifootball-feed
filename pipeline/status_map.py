"""Map API-Sports fixture/game status to the app feed's status + score + clock.

Shared by build_feed.py (so the daily build publishes today's live/finished games
with real scores) and live_update.py (the lightweight in-match refresh). The app
reads `status` ("scheduled"|"live"|"finished"), `finalScore {home,away}` (the
current score while live, the final once finished) and `liveClock` (e.g. "67'").
"""
from __future__ import annotations

import datetime

# API-Football fixture status short codes.
# https://www.api-football.com/documentation-v3#tag/Fixtures (status.short)
LIVE_STATUSES = {"1H", "HT", "2H", "ET", "BT", "P", "LIVE", "INT", "SUSP"}
FINISHED_STATUSES = {"FT", "AET", "PEN"}

# API-Basketball status short codes.
BB_LIVE_STATUSES = {"Q1", "Q2", "Q3", "Q4", "OT", "BT", "HT"}
BB_FINISHED_STATUSES = {"FT", "AOT"}

# A finished match keeps showing in the feed (so "today" includes this morning's
# results) for this many hours after kick-off, then drops off.
FINISHED_FEED_WINDOW_H = 36.0


def feed_status(short: str | None, *, basketball: bool = False) -> str:
    """API status.short -> "scheduled" | "live" | "finished"."""
    s = short or "NS"
    live = BB_LIVE_STATUSES if basketball else LIVE_STATUSES
    done = BB_FINISHED_STATUSES if basketball else FINISHED_STATUSES
    if s in live:
        return "live"
    if s in done:
        return "finished"
    return "scheduled"


def hours_since(iso_date: str | None) -> float:
    try:
        dt = datetime.datetime.fromisoformat((iso_date or "").replace("Z", "+00:00"))
        return (datetime.datetime.now(datetime.timezone.utc) - dt).total_seconds() / 3600.0
    except Exception:
        return 1e9


def live_fields(status: str, home_goals, away_goals, elapsed=None) -> dict:
    """The feed fields for a given mapped status + score. `finalScore` is attached
    for live and finished; `liveClock` ("67'") only while live and when known."""
    out: dict = {"status": status}
    if status in ("live", "finished"):
        out["finalScore"] = {"home": int(home_goals or 0), "away": int(away_goals or 0)}
    if status == "live" and elapsed is not None:
        out["liveClock"] = f"{int(elapsed)}'"
    return out
