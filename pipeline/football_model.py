"""Fit football model parameters from match history.

Attack/defence are time-decayed strengths relative to the league average (the
Dixon-Coles low-score correction `rho` and the simulation itself live in the app's
engine). Corners and cards are fit by method of moments (negative binomial).
Player goal shares come from scorer data when available, else a generic split.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from config import DECAY_HALF_LIFE_DAYS
from team_colors import team_mark


@dataclass
class Match:
    days_ago: float
    home: str            # team id
    away: str
    home_goals: int
    away_goals: int
    home_corners: float | None = None
    away_corners: float | None = None
    home_cards: float | None = None
    away_cards: float | None = None
    first_half_goals: int | None = None   # both teams, first half


def _decay(days_ago: float) -> float:
    return math.pow(0.5, days_ago / DECAY_HALF_LIFE_DAYS)


def _nb_size(mean: float, var: float) -> float:
    """Negative-binomial dispersion from mean/variance (large when ~Poisson)."""
    if var <= mean or mean <= 0:
        return 8.0
    return max(1.0, (mean * mean) / (var - mean))


def fit_league(name: str, team_ids: list[str], team_names: dict[str, str],
               matches: list[Match]) -> dict:
    if not matches:
        raise ValueError(f"No matches to fit {name}")

    # League averages (weighted).
    w_sum = sum(_decay(m.days_ago) for m in matches)
    total_goals = sum((m.home_goals + m.away_goals) * _decay(m.days_ago) for m in matches)
    league_avg = (total_goals / w_sum) / 2 if w_sum else 1.4

    home_goals_avg = sum(m.home_goals * _decay(m.days_ago) for m in matches) / w_sum
    away_goals_avg = sum(m.away_goals * _decay(m.days_ago) for m in matches) / w_sum
    home_adv = max(1.0, (home_goals_avg / away_goals_avg) ** 0.5) if away_goals_avg else 1.15

    # First-half share.
    fh = [m for m in matches if m.first_half_goals is not None]
    if fh:
        fh_goals = sum(m.first_half_goals * _decay(m.days_ago) for m in fh)
        fh_total = sum((m.home_goals + m.away_goals) * _decay(m.days_ago) for m in fh)
        first_half_share = round(fh_goals / fh_total, 3) if fh_total else 0.45
    else:
        first_half_share = 0.45

    teams = []
    for tid in team_ids:
        scored, conceded, wt = 0.0, 0.0, 0.0
        corners, cards = [], []
        for m in matches:
            w = _decay(m.days_ago)
            if m.home == tid:
                scored += m.home_goals * w; conceded += m.away_goals * w; wt += w
                if m.home_corners is not None: corners.append(m.home_corners)
                if m.home_cards is not None: cards.append(m.home_cards)
            elif m.away == tid:
                scored += m.away_goals * w; conceded += m.home_goals * w; wt += w
                if m.away_corners is not None: corners.append(m.away_corners)
                if m.away_cards is not None: cards.append(m.away_cards)
        if wt == 0:
            attack = defence = 1.0
        else:
            attack = max(0.4, (scored / wt) / league_avg)
            defence = max(0.4, (conceded / wt) / league_avg)

        c_mean = (sum(corners) / len(corners)) if corners else 5.5
        c_var = _variance(corners, c_mean) if len(corners) > 1 else c_mean * 1.4
        k_mean = (sum(cards) / len(cards)) if cards else 2.0
        k_var = _variance(cards, k_mean) if len(cards) > 1 else k_mean * 1.3

        teams.append({
            "teamId": tid,
            "attack": round(attack, 3),
            "defence": round(defence, 3),
            "cornersMean": round(c_mean, 2),
            "cornersSize": round(_nb_size(c_mean, c_var), 2),
            "cardsMean": round(k_mean, 2),
            "cardsSize": round(_nb_size(k_mean, k_var), 2),
            "players": _generic_players(tid, team_names.get(tid, tid)),
        })

    return {
        "schemaVersion": 1,
        "league": name,
        "leagueAverageGoals": round(league_avg, 3),
        "homeAdvantage": round(home_adv, 3),
        "rho": -0.08,
        "firstHalfGoalShare": first_half_share,
        "teams": teams,
    }


def _variance(values: list[float], mean: float) -> float:
    return sum((v - mean) ** 2 for v in values) / (len(values) - 1)


def _generic_players(tid: str, name: str) -> list[dict]:
    code = team_mark(name)[0]
    roles = [("fw", f"{code} Forward", 0.25, 82, 0.1),
             ("am", f"{code} Playmaker", 0.17, 85, 0.15),
             ("wg", f"{code} Winger", 0.14, 78, 0.12),
             ("df", f"{code} Defender", 0.06, 90, 0.2)]
    return [{"id": f"{tid}_{r[0]}", "name": r[1], "goalShare": r[2],
             "expectedMinutes": r[3], "cardRate": r[4]} for r in roles]
