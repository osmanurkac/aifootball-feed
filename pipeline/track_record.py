"""Model track record — the credibility layer for serious users.

Two parts, both honest about what they are:

  • LIVE sicil (the real track record). Every upcoming fixture's prediction is
    LOCKED the first time we see it (1X2, Over/Under 2.5, both-teams-to-score and
    the most likely score, from that day's fitted model) and stored in a prediction
    log. When the match later finishes, the locked prediction is graded against the
    real result. This is genuinely out-of-sample — the match hadn't been played
    when the prediction was made — so it is the number a bettor can trust. It starts
    empty and grows every day.

  • VALIDATION (historical model test). The model scored over all finished matches
    it was fit on. This is in-sample, so it slightly overstates accuracy; it is
    shown only as a labelled baseline so day one isn't blank while the live sicil
    accumulates.

The log round-trips through the published feed: at build start we fetch the
previous run's `predictions_state.json` from FEED_BASE_URL, update it, and
republish it into out/ alongside the feed. The gh-pages branch is rebuilt fresh
each run (force_orphan), so the live URL is the only durable store — no extra
state branch or bot commits.

track_record.json (schemaVersion 2) is consumed by the app's "Model Sicili" panel.
"""
from __future__ import annotations

import datetime
import json
import urllib.request
from pathlib import Path

from backtest import _poisson_pmf, _dc_tau, _now
from config import FEED_BASE_URL

STATE_FILE = "predictions_state.json"
MAX_GOALS = 10

# Keep the log bounded: drop graded predictions older than this, and never store
# more than this many entries (newest kept). Plenty for a rolling sicil.
PRUNE_DAYS = 240
MAX_ENTRIES = 6000

# A market's pick is "confident" above 50%; calibration buckets the top-pick
# probability so we can show predicted vs. actual frequency.
CALIB_EDGES = [0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 1.01]

# Below this many graded live predictions, the app headlines the validation number
# instead (clearly labelled) so the panel is never empty on a fresh install.
LIVE_MIN_SAMPLE = 20


# ---------------------------------------------------------------------------
# Model → market probabilities (Dixon-Coles scoreline grid)
# ---------------------------------------------------------------------------

def _scoreline_grid(lh: float, la: float, rho: float) -> list[list[float]]:
    hp = [_poisson_pmf(i, lh) for i in range(MAX_GOALS + 1)]
    ap = [_poisson_pmf(i, la) for i in range(MAX_GOALS + 1)]
    grid, total = [], 0.0
    for h in range(MAX_GOALS + 1):
        row = []
        for a in range(MAX_GOALS + 1):
            p = max(0.0, _dc_tau(h, a, lh, la, rho)) * hp[h] * ap[a]
            row.append(p)
            total += p
        grid.append(row)
    if total > 0:
        grid = [[p / total for p in row] for row in grid]
    return grid


def _markets(grid: list[list[float]]) -> tuple[dict, list[int]]:
    """Derive 1X2, Over/Under 2.5 and BTTS probabilities + the most likely score."""
    home = draw = away = over = btts = 0.0
    best_p, best = -1.0, [0, 0]
    for h, row in enumerate(grid):
        for a, p in enumerate(row):
            if h > a: home += p
            elif h == a: draw += p
            else: away += p
            if h + a >= 3: over += p
            if h >= 1 and a >= 1: btts += p
            if p > best_p:
                best_p, best = p, [h, a]
    return ({"h": round(home, 4), "d": round(draw, 4), "a": round(away, 4),
             "o25": round(over, 4), "btts": round(btts, 4)}, best)


def predict(model: dict, home_id: str, away_id: str):
    """Locked pre-match prediction for a fixture, or None if a team isn't modelled."""
    atk = {t["teamId"]: t["attack"] for t in model["teams"]}
    dfc = {t["teamId"]: t["defence"] for t in model["teams"]}
    if home_id not in atk or away_id not in atk:
        return None
    lam, hadv, rho = model["leagueAverageGoals"], model["homeAdvantage"], model["rho"]
    lh = lam * atk[home_id] * dfc[away_id] * hadv
    la = lam * atk[away_id] * dfc[home_id]
    probs, top = _markets(_scoreline_grid(lh, la, rho))
    return probs, top


# ---------------------------------------------------------------------------
# Grading a single prediction against a real result
# ---------------------------------------------------------------------------

def _pick_1x2(probs: dict) -> str:
    return max(("1", probs["h"]), ("X", probs["d"]), ("2", probs["a"]), key=lambda t: t[1])[0]


def _actual_1x2(hg: int, ag: int) -> str:
    return "1" if hg > ag else ("2" if ag > hg else "X")


def _grade(probs: dict, hg: int, ag: int) -> dict:
    """Per-market hit/miss + 1X2 Brier for one finished prediction."""
    pick = _pick_1x2(probs)
    actual = _actual_1x2(hg, ag)
    oh, od, oa = (1.0 if actual == "1" else 0.0), (1.0 if actual == "X" else 0.0), (1.0 if actual == "2" else 0.0)
    brier = (probs["h"] - oh) ** 2 + (probs["d"] - od) ** 2 + (probs["a"] - oa) ** 2
    over_actual = (hg + ag) >= 3
    btts_actual = hg >= 1 and ag >= 1
    conf = max(probs["h"], probs["d"], probs["a"])
    return {
        "x2_hit": pick == actual, "x2_conf": conf, "brier": brier,
        "ou_hit": (probs["o25"] >= 0.5) == over_actual,
        "btts_hit": (probs["btts"] >= 0.5) == btts_actual,
    }


# ---------------------------------------------------------------------------
# Aggregating many graded predictions into the record shape
# ---------------------------------------------------------------------------

def _evaluate(graded: list[dict]) -> dict:
    """graded: list of {probs, home, away} with actual score. Returns the rich block."""
    n = len(graded)
    if n == 0:
        return {"sampleSize": 0, "hitRate": 0.0, "brierScore": 0.0, "markets": [], "calibration": []}

    x2 = ou = btts = 0
    brier_sum = 0.0
    buckets = [{"lo": CALIB_EDGES[i], "hi": CALIB_EDGES[i + 1], "conf": 0.0, "hit": 0, "n": 0}
               for i in range(len(CALIB_EDGES) - 1)]
    for g in graded:
        r = _grade(g["probs"], g["actual"]["home"], g["actual"]["away"])
        x2 += r["x2_hit"]; ou += r["ou_hit"]; btts += r["btts_hit"]; brier_sum += r["brier"]
        for b in buckets:
            if b["lo"] <= r["x2_conf"] < b["hi"]:
                b["conf"] += r["x2_conf"]; b["hit"] += r["x2_hit"]; b["n"] += 1
                break

    calibration = [{
        "label": f"{int(round(b['lo'] * 100))}-{int(round(b['hi'] * 100))}%",
        "predicted": round(b["conf"] / b["n"], 3),
        "actual": round(b["hit"] / b["n"], 3),
        "sampleSize": b["n"],
    } for b in buckets if b["n"] > 0]

    return {
        "sampleSize": n,
        "hitRate": round(x2 / n, 3),
        "brierScore": round(brier_sum / n, 3),
        "markets": [
            {"key": "1x2", "label": "Maç sonucu", "hitRate": round(x2 / n, 3), "sampleSize": n},
            {"key": "ou25", "label": "2.5 Alt/Üst", "hitRate": round(ou / n, 3), "sampleSize": n},
            {"key": "btts", "label": "Karşılıklı gol", "hitRate": round(btts / n, 3), "sampleSize": n},
        ],
        "calibration": calibration,
    }


def _recent(graded: list[dict], limit: int = 20) -> list[dict]:
    """Newest-first graded picks for the 'son tahminler' list (teams by id; the app
    resolves names from teams.json)."""
    rows = sorted(graded, key=lambda g: g.get("date") or "", reverse=True)[:limit]
    out = []
    for g in rows:
        hg, ag = g["actual"]["home"], g["actual"]["away"]
        pick = _pick_1x2(g["probs"])
        out.append({
            "home": g["home"], "away": g["away"], "league": g.get("league"),
            "pick": pick, "actualHome": hg, "actualAway": ag,
            "hit": pick == _actual_1x2(hg, ag), "date": g.get("date"),
        })
    return out


# ---------------------------------------------------------------------------
# State round-trip (previous run's log ← live feed URL)
# ---------------------------------------------------------------------------

def _load_state() -> dict:
    url = f"{FEED_BASE_URL}/{STATE_FILE}"
    try:
        req = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        preds = data.get("predictions", [])
        print(f"  track record: loaded {len(preds)} prior predictions from live feed")
        return {p["fixtureId"]: p for p in preds if "fixtureId" in p}
    except Exception as exc:  # noqa: BLE001 — first run / 404 / network: start empty
        print(f"  track record: no prior log ({exc}); starting fresh")
        return {}


def _prune(index: dict) -> list[dict]:
    cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=PRUNE_DAYS))
    kept = []
    for p in index.values():
        if p.get("graded"):
            try:
                ko = datetime.datetime.fromisoformat((p.get("kickoff") or "").replace("Z", "+00:00"))
                if ko < cutoff:
                    continue
            except Exception:
                pass
        kept.append(p)
    kept.sort(key=lambda p: p.get("kickoff") or "", reverse=True)
    return kept[:MAX_ENTRIES]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_record(out_dir: Path, all_fixtures: list[dict], finished_by_id: dict,
                 models_by_slug: dict, backtest_matches: list, slug_by_name: dict) -> dict:
    """Update the forward log, grade finished fixtures, (re)publish the log, and
    return the full track_record.json object."""
    index = _load_state()

    # 1) Lock a prediction for every upcoming football fixture we haven't seen.
    recorded = 0
    for fx in all_fixtures:
        if fx.get("sport") != "football" or fx.get("status") != "scheduled":
            continue
        fid = fx["id"]
        if fid in index:
            continue
        slug = slug_by_name.get(fx.get("league"))
        model = models_by_slug.get(slug)
        if not model:
            continue
        pred = predict(model, fx["homeTeamId"], fx["awayTeamId"])
        if pred is None:
            continue
        probs, top = pred
        index[fid] = {
            "fixtureId": fid, "league": fx.get("league"), "kickoff": fx.get("kickoff"),
            "home": fx["homeTeamId"], "away": fx["awayTeamId"],
            "probs": probs, "topScore": top, "recordedAt": _now(),
            "graded": False, "actual": None,
        }
        recorded += 1

    # 2) Grade any locked prediction whose match has now finished.
    graded_now = 0
    for fid, res in finished_by_id.items():
        p = index.get(fid)
        if p and not p.get("graded"):
            p["actual"] = {"home": res["home"], "away": res["away"]}
            p["graded"] = True
            p["date"] = res.get("date")
            graded_now += 1

    predictions = _prune(index)
    print(f"  track record: +{recorded} locked, +{graded_now} graded, {len(predictions)} total in log")

    # 3) Republish the log so the next run continues it.
    (out_dir / STATE_FILE).write_text(json.dumps(
        {"schemaVersion": 1, "predictions": predictions}, ensure_ascii=False, indent=2))

    # 4) LIVE block — graded forward predictions (the real sicil).
    live_graded = [p for p in predictions if p.get("graded") and p.get("actual")]
    live = _evaluate(live_graded)

    # 5) VALIDATION block — in-sample backtest across every modelled league. Score
    #    each match with its own league's model (team ids are globally unique).
    val_graded = []
    for model in models_by_slug.values():
        ids = {t["teamId"] for t in model["teams"]}
        for m in backtest_matches:
            if m.home in ids and m.away in ids:
                pred = predict(model, m.home, m.away)
                if pred is None:
                    continue
                probs, _ = pred
                days = getattr(m, "days_ago", 0.0) or 0.0
                date = (datetime.datetime.now(datetime.timezone.utc)
                        - datetime.timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
                val_graded.append({"probs": probs, "home": m.home, "away": m.away,
                                   "league": model.get("league"), "date": date,
                                   "actual": {"home": m.home_goals, "away": m.away_goals}})
    validation = _evaluate(val_graded)

    # 6) The panel headlines live once it has a real sample; until then, validation.
    use_live = live["sampleSize"] >= LIVE_MIN_SAMPLE
    recent = _recent(live_graded if use_live else val_graded)

    return {
        "schemaVersion": 2,
        "lastUpdated": _now(),
        "headline": "live" if use_live else "validation",
        # Legacy top-level fields (schemaVersion-1 readers) point at the headline block.
        "hitRate": (live if use_live else validation)["hitRate"],
        "sampleSize": (live if use_live else validation)["sampleSize"],
        "brierScore": (live if use_live else validation)["brierScore"],
        "live": live,
        "validation": validation,
        "recent": recent,
    }
