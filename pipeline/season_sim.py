"""Precompute season simulations (10,000 runs of the remaining schedule).

Football: 3pts/win, 1/draw; title = 1st, topFinish = top 4, relegation = bottom 3.
Basketball: win = positive simulated margin; title = most wins, topFinish = top 8.
Keeps last week's title chance (from a previous output) for the ▲▼ arrows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

N_RUNS = 10_000


def _load_prev(out_dir: Path, slug: str) -> dict[str, float]:
    path = out_dir / "seasons" / f"{slug}.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
        return {e["teamId"]: e["title"] for e in data.get("entries", [])}
    except Exception:
        return {}


def simulate_football(league_name: str, slug: str, model: dict,
                      standings: dict[str, int], remaining: list[tuple[str, str]],
                      out_dir: Path) -> dict:
    teams = list(standings.keys())
    T = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    attack = {t["teamId"]: t["attack"] for t in model["teams"]}
    defence = {t["teamId"]: t["defence"] for t in model["teams"]}
    lam, hadv = model["leagueAverageGoals"], model["homeAdvantage"]

    base = np.array([standings[t] for t in teams], dtype=float)
    pts = np.tile(base, (N_RUNS, 1))
    rng = np.random.default_rng(42)

    # Rating uncertainty: our attack/defence are estimates, not the truth. Give
    # each team a per-run strength draw (held across all its games in that run)
    # so the simulated table reflects "if this team is really this good". Without
    # it the leader drifts to a near-certain title and the field looks too settled.
    SIGMA = 0.13
    atk_s = {t: attack.get(t, 1.0) * np.exp(rng.normal(0, SIGMA, N_RUNS)) for t in teams}
    def_s = {t: defence.get(t, 1.0) * np.exp(rng.normal(0, SIGMA, N_RUNS)) for t in teams}

    for home, away in remaining:
        if home not in idx or away not in idx:
            continue
        lh = lam * atk_s[home] * def_s[away] * hadv
        la = lam * atk_s[away] * def_s[home]
        hg = rng.poisson(lh)
        ag = rng.poisson(la)
        pts[hg > ag, idx[home]] += 3
        draw = hg == ag
        pts[draw, idx[home]] += 1
        pts[draw, idx[away]] += 1
        pts[hg < ag, idx[away]] += 3

    order = np.argsort(-pts, axis=1)
    title = np.bincount(order[:, 0], minlength=T) / N_RUNS
    top = np.bincount(order[:, :4].ravel(), minlength=T) / N_RUNS
    releg = np.bincount(order[:, -3:].ravel(), minlength=T) / N_RUNS

    prev = _load_prev(out_dir, slug)
    entries = [{
        "teamId": t,
        "title": round(float(title[idx[t]]), 3),
        "lastWeek": round(prev.get(t, float(title[idx[t]])), 3),
        "topFinish": round(float(top[idx[t]]), 3),
        "relegation": round(float(releg[idx[t]]), 3),
        "record": str(int(standings[t])),       # current points
    } for t in teams]
    entries.sort(key=lambda e: -e["title"])
    return {"schemaVersion": 1, "league": league_name, "sport": "football",
            "topFinishLabel": "Top 4", "recordLabel": "Pts", "entries": entries}


def simulate_basketball(league_name: str, slug: str, model: dict,
                        wins: dict[str, int], remaining: list[tuple[str, str]],
                        out_dir: Path) -> dict:
    teams = list(wins.keys())
    T = len(teams)
    idx = {t: i for i, t in enumerate(teams)}
    off = {t["teamId"]: t["offRating"] for t in model["teams"]}
    deff = {t["teamId"]: t["defRating"] for t in model["teams"]}
    pace = {t["teamId"]: t["pace"] for t in model["teams"]}
    lr, hadv = model["leagueAverageRating"], model["homeAdvantage"]

    base = np.array([wins[t] for t in teams], dtype=float)
    total = np.tile(base, (N_RUNS, 1))
    rng = np.random.default_rng(42)

    for home, away in remaining:
        if home not in idx or away not in idx:
            continue
        poss = (pace.get(home, 99) + pace.get(away, 99)) / 2
        hm = poss * (off.get(home, lr) * deff.get(away, lr) / lr) / 100 + hadv
        am = poss * (off.get(away, lr) * deff.get(home, lr) / lr) / 100
        margin = rng.normal(hm - am, 11.0, N_RUNS)
        total[margin > 0, idx[home]] += 1
        total[margin <= 0, idx[away]] += 1

    order = np.argsort(-total, axis=1)
    title = np.bincount(order[:, 0], minlength=T) / N_RUNS
    top_n = min(8, T)
    top = np.bincount(order[:, :top_n].ravel(), minlength=T) / N_RUNS

    prev = _load_prev(out_dir, slug)
    entries = [{
        "teamId": t,
        "title": round(float(title[idx[t]]), 3),
        "lastWeek": round(prev.get(t, float(title[idx[t]])), 3),
        "topFinish": round(float(top[idx[t]]), 3),
        "relegation": None,
    } for t in teams]
    entries.sort(key=lambda e: -e["title"])
    return {"schemaVersion": 1, "league": league_name, "sport": "basketball",
            "topFinishLabel": "Playoffs", "entries": entries}
