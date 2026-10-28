"""Build the static JSON feed consumed by the app.

Usage:
  python build_feed.py --source demo   # deterministic demo feed (no API key)
  python build_feed.py --source api    # fetch API-Sports, fit, write (needs key)

Output goes to pipeline/out/, matching the app's remote layout:
  fixtures.json · teams.json · track_record.json
  football/<slug>.json · basketball/<slug>.json · seasons/<slug>.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import random
import shutil
from pathlib import Path

import basketball_model as bb
import football_model as fb
import backtest
import season_sim
from config import LEAGUES, OUT_DIR, APISPORTS_KEY, FEED_ALLOW_FINISHED, SEASON_LOOKBACK
from team_colors import team_mark


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    print(f"  wrote {path.relative_to(OUT_DIR.parent)}")


def iso(days_from_now: float) -> str:
    dt = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=days_from_now)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Demo feed (no API key) — exercises the real fitting/sim/backtest code.
# ---------------------------------------------------------------------------

DEMO_FOOTBALL = {
    "premier_league": ["Arsenal", "Chelsea", "Liverpool", "Manchester City", "Newcastle", "Aston Villa"],
    "champions_league": ["Arsenal", "Barcelona", "Fenerbahçe", "Bayern München"],
    "laliga": ["Real Madrid", "Barcelona", "Atlético Madrid"],
    "bundesliga": ["Bayern München", "Borussia Dortmund", "Borussia Mönchengladbach"],
}
DEMO_BASKETBALL = {
    "nba": ["Los Angeles Lakers", "Boston Celtics", "Golden State Warriors", "Miami Heat"],
    "euroleague": ["Real Madrid", "Olympiacos", "Panathinaikos"],
}


def _team_id(name: str, sport: str) -> str:
    # Namespace by sport so clubs that exist in both a football league and a
    # basketball league (Barcelona, Real Madrid, ...) don't collide on one id.
    prefix = "fb" if sport == "football" else "bb"
    return f"{prefix}-{team_mark(name)[0].lower()}"


def build_demo() -> None:
    rng = random.Random(7)
    teams_out: dict[str, dict] = {}
    all_fixtures: list[dict] = []
    all_backtest_matches: list = []

    def register(name: str, league: str, sport: str) -> str:
        tid = _team_id(name, sport)
        if tid not in teams_out:
            code, c1, c2, c3 = team_mark(name)
            teams_out[tid] = {"id": tid, "name": name, "shortCode": code,
                              "primaryColor": c1, "secondaryColor": c2, "textColor": c3,
                              "league": league, "sport": sport}
        return tid

    # Football leagues.
    for slug, names in DEMO_FOOTBALL.items():
        league = next(l for l in LEAGUES if l.slug == slug)
        ids = [register(n, league.name, "football") for n in names]
        name_map = {register(n, league.name, "football"): n for n in names}
        history, standings, remaining = _football_season(rng, ids)
        model = fb.fit_league(league.name, ids, name_map, history)
        write_json(OUT_DIR / "football" / f"{slug}.json", model)
        season = season_sim.simulate_football(league.name, slug, model, standings, remaining, OUT_DIR)
        write_json(OUT_DIR / "seasons" / f"{slug}.json", season)
        all_backtest_matches.extend(history)
        _emit_fixtures(all_fixtures, rng, league, ids, "football")

    # Basketball leagues.
    for slug, names in DEMO_BASKETBALL.items():
        league = next(l for l in LEAGUES if l.slug == slug)
        ids = [register(n, league.name, "basketball") for n in names]
        name_map = {register(n, league.name, "basketball"): n for n in names}
        games, wins, remaining = _basketball_season(rng, ids)
        model = bb.fit_league(league.name, ids, name_map, games)
        write_json(OUT_DIR / "basketball" / f"{slug}.json", model)
        season = season_sim.simulate_basketball(league.name, slug, model, wins, remaining, OUT_DIR)
        write_json(OUT_DIR / "seasons" / f"{slug}.json", season)
        _emit_fixtures(all_fixtures, rng, league, ids, "basketball")

    # Backtest (football) -> track record.
    # Fit a combined model just for backtest scoring reuse: use the PL model.
    pl_model = json.loads((OUT_DIR / "football" / "premier_league.json").read_text())
    record = backtest.run(pl_model, [m for m in all_backtest_matches if m.home in
                                     {t["teamId"] for t in pl_model["teams"]}])
    write_json(OUT_DIR / "track_record.json", record)

    write_json(OUT_DIR / "teams.json", {"schemaVersion": 1, "teams": list(teams_out.values())})
    write_json(OUT_DIR / "fixtures.json", {"schemaVersion": 1, "fixtures": all_fixtures})
    demo_slugs = set(DEMO_FOOTBALL) | set(DEMO_BASKETBALL)
    available = [{"slug": l.slug, "name": l.name, "sport": l.sport}
                 for l in LEAGUES if l.slug in demo_slugs]
    write_json(OUT_DIR / "leagues.json", {"schemaVersion": 1, "leagues": available})


def _football_season(rng, ids):
    history, standings = [], {t: 0 for t in ids}
    remaining = []
    for i, h in enumerate(ids):
        for j, a in enumerate(ids):
            if h == a:
                continue
            if (i + j) % 2 == 0:  # first half of the season: played
                hg, ag = rng.randint(0, 4), rng.randint(0, 3)
                history.append(fb.Match(days_ago=rng.uniform(5, 300), home=h, away=a,
                                        home_goals=hg, away_goals=ag,
                                        home_corners=rng.randint(2, 11), away_corners=rng.randint(2, 10),
                                        home_cards=rng.randint(0, 4), away_cards=rng.randint(0, 4),
                                        first_half_goals=min(hg + ag, rng.randint(0, hg + ag) if hg + ag else 0)))
                if hg > ag: standings[h] += 3
                elif hg < ag: standings[a] += 3
                else: standings[h] += 1; standings[a] += 1
            else:
                remaining.append((h, a))
    return history, standings, remaining


def _basketball_season(rng, ids):
    games, wins, remaining = [], {t: 0 for t in ids}, []
    for i, h in enumerate(ids):
        for j, a in enumerate(ids):
            if h == a:
                continue
            if (i + j) % 2 == 0:
                hp, ap = rng.randint(95, 125), rng.randint(95, 125)
                games.append(bb.Game(days_ago=rng.uniform(5, 300), home=h, away=a,
                                     home_points=hp, away_points=ap, possessions=rng.uniform(96, 103)))
                wins[h if hp > ap else a] += 1
            else:
                remaining.append((h, a))
    return games, wins, remaining


def _emit_fixtures(out, rng, league, ids, sport):
    # A couple of scheduled, one finished, maybe one live per league.
    pairs = [(ids[0], ids[1])]
    if len(ids) >= 4:
        pairs.append((ids[2], ids[3]))
    for k, (h, a) in enumerate(pairs):
        out.append({
            "id": f"{league.slug}-{h}-{a}-sched",
            "sport": sport, "league": league.name, "round": "MD4",
            "kickoff": iso(1 + k), "homeTeamId": h, "awayTeamId": a,
            "status": "scheduled", "featured": (league.slug in ("premier_league", "nba") and k == 0),
            "upset": False,
        })
    # Finished example.
    hg, ag = (rng.randint(0, 3), rng.randint(0, 3)) if sport == "football" else (rng.randint(95, 120), rng.randint(95, 120))
    out.append({
        "id": f"{league.slug}-{ids[0]}-{ids[1]}-fin",
        "sport": sport, "league": league.name, "round": "MD3",
        "kickoff": iso(-4), "homeTeamId": ids[1], "awayTeamId": ids[0],
        "status": "finished", "finalScore": {"home": hg, "away": ag},
        "featured": False, "upset": hg < ag,
    })


# ---------------------------------------------------------------------------
# API feed — fetch, fit, write (needs APISPORTS_KEY; validate live).
# ---------------------------------------------------------------------------

class LeagueUnavailable(Exception):
    """Raised when a league has no accessible CURRENT-season data (e.g. the plan
    gates the current season). The league is then HIDDEN from the feed — we never
    fall back to an old season or to synthesised / projected data."""


def _plan_blocked(data) -> bool:
    """True when the API refused the request (e.g. a Free-plan season gate)."""
    errs = data.get("errors")
    if isinstance(errs, dict):
        return bool(errs)
    if isinstance(errs, list):
        return len(errs) > 0
    return False


def _current_basketball_season(style: str, year: int, month: int) -> str:
    """The in-progress basketball campaign. Campaigns tip off around October."""
    start = year if month >= 9 else year - 1
    return str(start) if style == "year" else f"{start}-{start + 1}"


def build_from_api() -> None:
    """Fetch API-Sports CURRENT-season data, fit, and write.

    A league is included only if its current season is accessible on the plan and
    has real fixtures/results. Leagues without current-season data are HIDDEN
    (omitted from the feed, teams.json and leagues.json) — never backfilled with
    an old season or synthesised/projected data. The app shows exactly the
    leagues listed in leagues.json.
    """
    from apisports import APISports
    from config import FOOTBALL_BASE, BASKETBALL_BASE, current_season, HISTORY_SEASONS

    if not APISPORTS_KEY:
        print("No APISPORTS_KEY set — producing the demo feed instead.")
        build_demo()
        return

    client = APISports(APISPORTS_KEY)
    season = current_season()
    teams_out: dict[str, dict] = {}
    all_fixtures: list[dict] = []
    backtest_matches: list = []
    available: list[dict] = []

    def register(name, league, sport):
        tid = _team_id(name, sport)
        if tid not in teams_out:
            code, c1, c2, c3 = team_mark(name)
            teams_out[tid] = {"id": tid, "name": name, "shortCode": code,
                              "primaryColor": c1, "secondaryColor": c2, "textColor": c3,
                              "league": league, "sport": sport}
        return tid

    for league in LEAGUES:
        try:
            if league.sport == "football":
                _api_football_league(client, FOOTBALL_BASE, league, season, HISTORY_SEASONS,
                                     register, all_fixtures, backtest_matches)
            else:
                _api_basketball_league(client, BASKETBALL_BASE, league, HISTORY_SEASONS,
                                       register, all_fixtures)
        except LeagueUnavailable as exc:
            print(f"  - {league.name}: HIDDEN — {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 — resilient per-league
            print(f"  ! {league.name}: HIDDEN — error: {exc}")
            continue
        available.append({"slug": league.slug, "name": league.name, "sport": league.sport})

    # Backtest from the Premier League model, only if it's actually available.
    pl = OUT_DIR / "football" / "premier_league.json"
    if any(a["slug"] == "premier_league" for a in available) and pl.exists():
        model = json.loads(pl.read_text())
        ids = {t["teamId"] for t in model["teams"]}
        record = backtest.run(model, [m for m in backtest_matches if m.home in ids and m.away in ids])
    else:
        record = {"schemaVersion": 1, "hitRate": 0.0, "sampleSize": 0, "brierScore": 0.0,
                  "lastUpdated": backtest._now()}
    write_json(OUT_DIR / "track_record.json", record)
    write_json(OUT_DIR / "teams.json", {"schemaVersion": 1, "teams": list(teams_out.values())})
    write_json(OUT_DIR / "fixtures.json", {"schemaVersion": 1, "fixtures": all_fixtures})
    write_json(OUT_DIR / "leagues.json", {"schemaVersion": 1, "leagues": available})
    print(f"Available leagues: {[a['slug'] for a in available] or 'NONE — feed is empty'}")
    print(f"API requests made: {client.requests_made}")


def _days_ago(iso_date: str) -> float:
    try:
        dt = datetime.datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        return max(0.0, (datetime.datetime.now(datetime.timezone.utc) - dt).days)
    except Exception:
        return 180.0


def _test_pairs(team_ids, standings=None, n=4):
    """A handful of real-team pairings for synthesised upcoming fixtures (test
    mode only), top teams first when standings are known."""
    order = sorted(team_ids, key=lambda t: (standings or {}).get(t, 0), reverse=True) \
        if standings else list(team_ids)
    pairs = []
    k = 0
    while k + 1 < len(order) and len(pairs) < n:
        pairs.append((order[k], order[k + 1]))
        k += 2
    return pairs


def _resolve_football_season(client, base, league, season):
    """Return (effective_season, fixtures_data) — normally the current season.
    In FEED_ALLOW_FINISHED test mode, if the current season is plan-gated or empty,
    step back to the most recent accessible season that has fixtures/results."""
    candidates = [season]
    if FEED_ALLOW_FINISHED:
        candidates += [season - i for i in range(1, SEASON_LOOKBACK + 1)]
    last = f"current season {season} not available on this plan"
    for yr in candidates:
        data = client.get(base, "fixtures", {"league": league.api_id, "season": yr})
        if _plan_blocked(data):
            last = f"season {yr} not available on this plan"
            continue
        items = data.get("response", []) or []
        statuses = {i["fixture"]["status"]["short"] for i in items}
        if statuses & {"FT", "AET", "PEN", "NS"}:
            if yr != season:
                print(f"  - {league.name}: TEST MODE — current {season} gated, using finished season {yr}")
            return yr, data
        last = f"season {yr} has no fixtures/results"
    raise LeagueUnavailable(last)


def _resolve_basketball_season(client, base, league):
    now = datetime.datetime.now(datetime.timezone.utc)
    cur = _current_basketball_season(league.season_style, now.year, now.month)
    start_year = int(cur) if league.season_style == "year" else int(cur.split("-")[0])
    fmt = (lambda s: str(s)) if league.season_style == "year" else (lambda s: f"{s}-{s + 1}")
    candidates = [cur]
    if FEED_ALLOW_FINISHED:
        candidates += [fmt(start_year - i) for i in range(1, SEASON_LOOKBACK + 1)]
    last = f"current season {cur} not available on this plan"
    for sv in candidates:
        data = client.get(base, "games", {"league": league.api_id, "season": sv})
        if _plan_blocked(data):
            last = f"season {sv} not available on this plan"
            continue
        games = data.get("response", []) or []
        statuses = {g.get("status", {}).get("short") for g in games}
        if statuses & {"FT", "AOT", "NS"}:
            if sv != cur:
                print(f"  - {league.name}: TEST MODE — current {cur} gated, using finished season {sv}")
            return sv, data
        last = f"season {sv} has no games"
    raise LeagueUnavailable(last)


def _api_football_league(client, base, league, season, history_seasons,
                         register, all_fixtures, backtest_matches):
    # Gate: the CURRENT season must be accessible (not plan-gated) and have real
    # fixtures. Otherwise the league is hidden — unless FEED_ALLOW_FINISHED test
    # mode lets us fall back to the most recent accessible finished season.
    season, cur = _resolve_football_season(client, base, league, season)

    name_map, team_ids, history = {}, [], []
    # Fit on accessible seasons up to and including the current one (the current
    # season's results are already in `cur`); older seasons add rating stability.
    for yr in range(season - history_seasons + 1, season + 1):
        data = cur if yr == season else client.get(base, "fixtures", {"league": league.api_id, "season": yr})
        if _plan_blocked(data):
            continue
        for item in data.get("response", []):
            fx, goals, teams = item["fixture"], item["goals"], item["teams"]
            if fx.get("status", {}).get("short") not in ("FT", "AET", "PEN"):
                continue
            hid = register(teams["home"]["name"], league.name, "football")
            aid = register(teams["away"]["name"], league.name, "football")
            name_map[hid] = teams["home"]["name"]; name_map[aid] = teams["away"]["name"]
            if hid not in team_ids: team_ids.append(hid)
            if aid not in team_ids: team_ids.append(aid)
            history.append(fb.Match(days_ago=_days_ago(fx["date"]), home=hid, away=aid,
                                    home_goals=goals["home"] or 0, away_goals=goals["away"] or 0))
    if not history:
        raise LeagueUnavailable("no finished fixtures to fit a model")
    backtest_matches.extend(history)
    model = fb.fit_league(league.name, team_ids, name_map, history)
    write_json(OUT_DIR / "football" / f"{league.slug}.json", model)

    # Current-season standings -> points.
    standings = {t: 0 for t in team_ids}
    st = client.get(base, "standings", {"league": league.api_id, "season": season})
    for grp in st.get("response", []):
        for table in grp.get("league", {}).get("standings", []):
            for row in table:
                tid = _team_id(row["team"]["name"], "football")
                if tid in standings:
                    standings[tid] = row.get("points", 0)

    # Upcoming current-season fixtures -> remaining + app fixtures.
    remaining = []
    up = client.get(base, "fixtures", {"league": league.api_id, "season": season, "next": "40"})
    for k, item in enumerate(up.get("response", [])):
        fx, teams = item["fixture"], item["teams"]
        hid = register(teams["home"]["name"], league.name, "football")
        aid = register(teams["away"]["name"], league.name, "football")
        remaining.append((hid, aid))
        if k < 6:
            all_fixtures.append({"id": f"{league.slug}-{fx['id']}", "sport": "football",
                                 "league": league.name, "round": None, "kickoff": fx["date"],
                                 "homeTeamId": hid, "awayTeamId": aid, "status": "scheduled",
                                 "featured": league.slug == "premier_league" and k == 0, "upset": False})
    # Test mode: a finished season has no upcoming fixtures. Synthesise a few from
    # real teams (future kickoffs) so Home/Picker have matches to simulate; leave
    # `remaining` empty so the Explore race shows the real final table.
    if not up.get("response") and FEED_ALLOW_FINISHED:
        for k, (hid, aid) in enumerate(_test_pairs(team_ids, standings)):
            all_fixtures.append({"id": f"{league.slug}-test-{k}", "sport": "football",
                                 "league": league.name, "round": None, "kickoff": iso(1 + k),
                                 "homeTeamId": hid, "awayTeamId": aid, "status": "scheduled",
                                 "featured": league.slug == "premier_league" and k == 0, "upset": False})
    season_json = season_sim.simulate_football(league.name, league.slug, model, standings, remaining, OUT_DIR)
    write_json(OUT_DIR / "seasons" / f"{league.slug}.json", season_json)


def _api_basketball_league(client, base, league, history_seasons, register, all_fixtures):
    # Gate on the current campaign — or, in test mode, the most recent accessible
    # finished campaign.
    cur_season, cur = _resolve_basketball_season(client, base, league)
    cur_games = cur.get("response", []) or []
    cur_finished = [g for g in cur_games if g.get("status", {}).get("short") in ("FT", "AOT")]
    cur_upcoming = [g for g in cur_games if g.get("status", {}).get("short") == "NS"]

    # Fit window: current campaign + older accessible seasons for stability.
    if league.season_style == "year":
        start = int(cur_season)
        window = [str(start - i) for i in range(history_seasons)]
    else:
        start = int(cur_season.split("-")[0])
        window = [f"{start - i}-{start - i + 1}" for i in range(history_seasons)]

    name_map, team_ids, games_fit, wins = {}, [], [], {}
    for sv in window:
        data = cur if sv == cur_season else client.get(base, "games", {"league": league.api_id, "season": sv})
        if _plan_blocked(data):
            continue
        for g in data.get("response", []):
            if g.get("status", {}).get("short") not in ("FT", "AOT"):
                continue
            teams, scores = g["teams"], g["scores"]
            hid = register(teams["home"]["name"], league.name, "basketball")
            aid = register(teams["away"]["name"], league.name, "basketball")
            name_map[hid] = teams["home"]["name"]; name_map[aid] = teams["away"]["name"]
            for t in (hid, aid):
                if t not in team_ids: team_ids.append(t); wins[t] = 0
            hp, ap = scores["home"]["total"] or 0, scores["away"]["total"] or 0
            games_fit.append(bb.Game(days_ago=_days_ago(g["date"]), home=hid, away=aid,
                                     home_points=hp, away_points=ap))
            if sv == cur_season:  # standings reflect the CURRENT campaign only
                wins[hid if hp > ap else aid] += 1
    if not games_fit:
        raise LeagueUnavailable("no finished games to fit a model")
    model = bb.fit_league(league.name, team_ids, name_map, games_fit)
    write_json(OUT_DIR / "basketball" / f"{league.slug}.json", model)

    # Upcoming current-season games -> remaining + app fixtures. No projection:
    # if there are none, the season race simply has no games left to simulate.
    remaining = []
    for k, g in enumerate(cur_upcoming):
        teams = g["teams"]
        hid = register(teams["home"]["name"], league.name, "basketball")
        aid = register(teams["away"]["name"], league.name, "basketball")
        remaining.append((hid, aid))
        if len([f for f in all_fixtures if f["league"] == league.name]) < 6:
            all_fixtures.append({"id": f"{league.slug}-{g['id']}", "sport": "basketball",
                                 "league": league.name, "round": None, "kickoff": g["date"],
                                 "homeTeamId": hid, "awayTeamId": aid, "status": "scheduled",
                                 "featured": league.slug == "nba" and k == 0, "upset": False})
    # Test mode: synthesise a few upcoming games from real teams so the simulate
    # flow can be exercised; the Explore race keeps the real final standings.
    if not cur_upcoming and FEED_ALLOW_FINISHED:
        for k, (hid, aid) in enumerate(_test_pairs(team_ids, wins)):
            all_fixtures.append({"id": f"{league.slug}-test-{k}", "sport": "basketball",
                                 "league": league.name, "round": None, "kickoff": iso(1 + k),
                                 "homeTeamId": hid, "awayTeamId": aid, "status": "scheduled",
                                 "featured": league.slug == "nba" and k == 0, "upset": False})
    season_json = season_sim.simulate_basketball(league.name, league.slug, model, wins, remaining, OUT_DIR)
    write_json(OUT_DIR / "seasons" / f"{league.slug}.json", season_json)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["demo", "api"], default="demo")
    args = parser.parse_args()
    # Clean rebuild: a league hidden this run must not leave a stale file behind.
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building feed ({args.source}) -> {OUT_DIR}")
    if args.source == "demo":
        build_demo()
    else:
        build_from_api()
    print("Done.")


if __name__ == "__main__":
    main()
