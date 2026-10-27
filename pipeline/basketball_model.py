"""Fit basketball model parameters from game history.

Pace (possessions/game), offensive/defensive ratings (points per 100 possessions)
and a league average rating. Player per-minute rates come from box-score data when
available, else a generic rotation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from config import DECAY_HALF_LIFE_DAYS
from team_colors import team_mark


@dataclass
class Game:
    days_ago: float
    home: str
    away: str
    home_points: int
    away_points: int
    possessions: float | None = None   # estimated possessions for the game


def _decay(days_ago: float) -> float:
    return math.pow(0.5, days_ago / DECAY_HALF_LIFE_DAYS)


def fit_league(name: str, team_ids: list[str], team_names: dict[str, str],
               games: list[Game]) -> dict:
    if not games:
        raise ValueError(f"No games to fit {name}")

    w_sum = sum(_decay(g.days_ago) for g in games)
    # League average points -> rating baseline (points per 100 possessions).
    avg_points = sum((g.home_points + g.away_points) / 2 * _decay(g.days_ago) for g in games) / w_sum
    avg_poss = sum((g.possessions or 99.0) * _decay(g.days_ago) for g in games) / w_sum
    league_rating = round(avg_points / avg_poss * 100, 1) if avg_poss else 114.0

    home_pts = sum(g.home_points * _decay(g.days_ago) for g in games) / w_sum
    away_pts = sum(g.away_points * _decay(g.days_ago) for g in games) / w_sum
    home_adv = round(max(0.0, home_pts - away_pts), 2) or 2.5

    teams = []
    for tid in team_ids:
        scored, allowed, poss, wt = 0.0, 0.0, 0.0, 0.0
        for g in games:
            w = _decay(g.days_ago)
            p = g.possessions or 99.0
            if g.home == tid:
                scored += g.home_points * w; allowed += g.away_points * w; poss += p * w; wt += w
            elif g.away == tid:
                scored += g.away_points * w; allowed += g.home_points * w; poss += p * w; wt += w
        if wt == 0:
            pace, off, deff = 99.0, league_rating, league_rating
        else:
            pace = round(poss / wt, 1)
            off = round(scored / wt / pace * 100, 1)
            deff = round(allowed / wt / pace * 100, 1)
        teams.append({
            "teamId": tid, "pace": pace, "offRating": off, "defRating": deff,
            "players": _generic_players(tid, team_names.get(tid, tid)),
        })

    return {
        "schemaVersion": 1,
        "league": name,
        "leagueAverageRating": league_rating,
        "homeAdvantage": home_adv,
        "teams": teams,
    }


def _generic_players(tid: str, name: str) -> list[dict]:
    code = team_mark(name)[0]
    specs = [("g1", f"{code} Guard", 35, 0.70, 0.12, 0.22, 0.10, "G"),
             ("f1", f"{code} Forward", 34, 0.66, 0.22, 0.13, 0.08, "F"),
             ("c1", f"{code} Center", 31, 0.52, 0.34, 0.09, 0.02, "C"),
             ("g2", f"{code} Guard 2", 26, 0.45, 0.10, 0.16, 0.09, "G"),
             ("f2", f"{code} Forward 2", 24, 0.40, 0.20, 0.08, 0.05, "F")]
    return [{"id": f"{tid}_{s[0]}", "name": s[1], "minutes": s[2], "ptsPerMin": s[3],
             "rebPerMin": s[4], "astPerMin": s[5], "threePerMin": s[6], "position": s[7]}
            for s in specs]
