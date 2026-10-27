"""Backtest the football model on recent finished matches.

Computes the Brier score (lower is better) and the favourite hit rate from the
model's pre-match win/draw/loss probabilities, and writes track_record.json for
the onboarding proof screen.
"""
from __future__ import annotations

import datetime
import math

MAX_GOALS = 10


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))


def _dc_tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    if x == 0 and y == 0: return 1 - lam * mu * rho
    if x == 0 and y == 1: return 1 + lam * rho
    if x == 1 and y == 0: return 1 + mu * rho
    if x == 1 and y == 1: return 1 - rho
    return 1.0


def win_draw_loss(lam: float, mu: float, rho: float) -> tuple[float, float, float]:
    hp = [_poisson_pmf(i, lam) for i in range(MAX_GOALS + 1)]
    ap = [_poisson_pmf(i, mu) for i in range(MAX_GOALS + 1)]
    home = draw = away = total = 0.0
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            p = max(0.0, _dc_tau(h, a, lam, mu, rho)) * hp[h] * ap[a]
            total += p
            if h > a: home += p
            elif h == a: draw += p
            else: away += p
    if total > 0:
        home, draw, away = home / total, draw / total, away / total
    return home, draw, away


def run(model: dict, matches: list) -> dict:
    """matches: list of football_model.Match (finished), scored by the model."""
    attack = {t["teamId"]: t["attack"] for t in model["teams"]}
    defence = {t["teamId"]: t["defence"] for t in model["teams"]}
    lam_avg, hadv, rho = model["leagueAverageGoals"], model["homeAdvantage"], model["rho"]

    brier_sum = 0.0
    hits = 0
    n = 0
    for m in matches:
        if m.home not in attack or m.away not in attack:
            continue
        lh = lam_avg * attack[m.home] * defence[m.away] * hadv
        la = lam_avg * attack[m.away] * defence[m.home]
        ph, pd, pa = win_draw_loss(lh, la, rho)
        # Actual outcome one-hot.
        oh = 1.0 if m.home_goals > m.away_goals else 0.0
        od = 1.0 if m.home_goals == m.away_goals else 0.0
        oa = 1.0 if m.home_goals < m.away_goals else 0.0
        brier_sum += (ph - oh) ** 2 + (pd - od) ** 2 + (pa - oa) ** 2
        predicted = max((ph, "h"), (pd, "d"), (pa, "a"))[1]
        actual = "h" if oh else ("d" if od else "a")
        if predicted == actual:
            hits += 1
        n += 1

    if n == 0:
        return {"schemaVersion": 1, "hitRate": 0.0, "sampleSize": 0,
                "brierScore": 0.0, "lastUpdated": _now()}
    return {
        "schemaVersion": 1,
        "hitRate": round(hits / n, 3),
        "sampleSize": n,
        "brierScore": round(brier_sum / n, 3),
        "lastUpdated": _now(),
    }


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
