"""Fit football model parameters from match history.

Attack/defence are time-decayed strengths relative to the league average (the
Dixon-Coles low-score correction `rho` and the simulation itself live in the app's
engine). Corners and cards are fit by method of moments (negative binomial).
Player goal shares come from scorer data when available, else a generic split.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from config import DECAY_HALF_LIFE_DAYS
from team_colors import team_mark

# L2 shrinkage on attack/defence in the MLE fit (toward the league average).
GLM_RIDGE = 8.0


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


def _fit_glm(team_ids: list[str], matches: list[Match]):
    """Time-weighted ridge Poisson GLM (Dixon-Coles MLE) for attack / defence /
    home advantage. log λ = μ + home + attack[scorer] + defence[conceder].

    Beats the old scored/conceded-ratio heuristic out-of-sample (1X2 log-loss
    0.991 → 0.988 on ~28k historical matches; see the feed's research/). Returns
    (leagueAverageGoals, homeAdvantage, {team: attack}, {team: defence}); each
    exp()'d so the app engine's multiplicative λ = avg·atk·def·homeAdv is unchanged.
    """
    present = [t for t in team_ids if any(m.home == t or m.away == t for m in matches)]
    index = {t: i for i, t in enumerate(present)}
    n_teams = len(present)
    if n_teams < 2:
        return 1.35, 1.15, {t: 1.0 for t in team_ids}, {t: 1.0 for t in team_ids}

    ncol = 2 + 2 * n_teams          # [intercept, home, attack…, defence…]
    rows, ys, ws = [], [], []
    for m in matches:
        if m.home not in index or m.away not in index:
            continue
        w = _decay(m.days_ago)
        hi, ai = index[m.home], index[m.away]
        r_home = [0.0] * ncol; r_home[0] = 1; r_home[1] = 1; r_home[2 + hi] = 1; r_home[2 + n_teams + ai] = 1
        rows.append(r_home); ys.append(m.home_goals); ws.append(w)
        r_away = [0.0] * ncol; r_away[0] = 1; r_away[2 + ai] = 1; r_away[2 + n_teams + hi] = 1
        rows.append(r_away); ys.append(m.away_goals); ws.append(w)

    X = np.asarray(rows); y = np.asarray(ys, dtype=float); w = np.asarray(ws)
    ridge = np.ones(ncol) * GLM_RIDGE; ridge[0] = 0.0; ridge[1] = 0.0
    R = np.diag(ridge)
    beta = np.zeros(ncol)
    for _ in range(15):                 # IRLS
        eta = np.clip(X @ beta, -5.0, 5.0)
        mu = np.exp(eta)
        wd = w * mu
        A = (X.T * wd) @ X + R
        z = eta + (y - mu) / np.maximum(mu, 1e-6)
        b = (X.T * wd) @ z
        try:
            beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            beta = np.linalg.lstsq(A, b, rcond=None)[0]

    league_avg = float(math.exp(min(1.5, beta[0])))
    home_adv = float(max(1.0, math.exp(beta[1])))
    attack, defence = {}, {}
    for t in team_ids:
        if t in index:
            attack[t] = round(min(3.0, max(0.3, math.exp(beta[2 + index[t]]))), 3)
            defence[t] = round(min(3.0, max(0.3, math.exp(beta[2 + n_teams + index[t]]))), 3)
        else:
            attack[t] = defence[t] = 1.0
    return round(league_avg, 3), round(home_adv, 3), attack, defence


# Empirical-Bayes regression constants.
REG_K = 25.0        # pseudo-matches pulling a team toward the league mean
PROMO_ATK = 0.14    # up to −14% attack for a no-history (promoted) side
PROMO_DEF = 0.20    # up to +20% goals conceded for a no-history side


def _regress_strengths(team_ids: list[str], atk: dict[str, float],
                       dfc: dict[str, float], counts: dict[str, int]):
    """Shrink attack/defence toward the mean by sample size, and bias low-data
    (newly promoted) teams toward the weaker end. A team with lots of history is
    unchanged; one with a handful of games is pulled to ~average and then down."""
    out_a, out_d = {}, {}
    for tid in team_ids:
        n = counts.get(tid, 0)
        rel = n / (n + REG_K)             # 0 (no data) … → 1 (lots of data)
        promo = 1.0 - rel                 # weakness bias, strongest with little data
        a = (1.0 + (atk.get(tid, 1.0) - 1.0) * rel) * (1.0 - PROMO_ATK * promo)
        d = (1.0 + (dfc.get(tid, 1.0) - 1.0) * rel) * (1.0 + PROMO_DEF * promo)
        out_a[tid] = round(min(3.0, max(0.3, a)), 3)
        out_d[tid] = round(min(3.0, max(0.3, d)), 3)
    return out_a, out_d


def fit_league(name: str, team_ids: list[str], team_names: dict[str, str],
               matches: list[Match]) -> dict:
    if not matches:
        raise ValueError(f"No matches to fit {name}")

    # Attack / defence / league average / home advantage via the MLE GLM.
    league_avg, home_adv, atk, dfc = _fit_glm(team_ids, matches)

    # Empirical-Bayes regression (538/Opta-style): a team's strength is only as
    # trustworthy as the matches behind it. Shrink each team toward the league
    # mean by how much data informs it, and bias low-data sides (newly promoted,
    # little/no top-flight history) toward the weaker end — promoted teams are
    # usually relegation-zone, not mid-table, so a hot 8-game start shouldn't make
    # them look like title/top-4 contenders.
    counts = {t: 0 for t in team_ids}
    for m in matches:
        if m.home in counts: counts[m.home] += 1
        if m.away in counts: counts[m.away] += 1
    atk, dfc = _regress_strengths(team_ids, atk, dfc, counts)

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
        corners, cards = [], []
        for m in matches:
            if m.home == tid:
                if m.home_corners is not None: corners.append(m.home_corners)
                if m.home_cards is not None: cards.append(m.home_cards)
            elif m.away == tid:
                if m.away_corners is not None: corners.append(m.away_corners)
                if m.away_cards is not None: cards.append(m.away_cards)
        attack, defence = atk[tid], dfc[tid]

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
        # -0.11 (not -0.08) fit best out-of-sample on ~30k historical matches.
        "rho": -0.11,
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
