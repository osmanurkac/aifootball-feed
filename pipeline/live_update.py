"""Lightweight in-match refresh of the published feed.

The daily build (build_feed.py) is heavy — it refits models, fetches stats and
runs the AI analysis — so it runs once a day. This script does the ONE thing that
has to stay fresh during match hours: patch `status` / `finalScore` / `liveClock`
of fixtures that are currently live or have just finished, straight into the
already-published fixtures.json. No models, no AI, a couple of cheap API calls.

Run by .github/workflows/live.yml every few minutes during match windows:
    python live_update.py --fixtures <path-to-current-fixtures.json>
The workflow fetches the current gh-pages fixtures.json into that path, runs this,
then republishes it with keep_files (so nothing else in the feed is touched).
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

from apisports import APISports
from config import FOOTBALL_BASE, LEAGUES, APISPORTS_KEY
import status_map

# API league id -> our feed slug, football only (basketball is not live-tracked
# yet; it is disabled in the app).
_SLUG_BY_API_ID = {l.api_id: l.slug for l in LEAGUES if l.sport == "football"}


def _today_dates() -> list[str]:
    """Today and yesterday (UTC) — covers late-night games that finished after
    midnight UTC without missing their result."""
    today = datetime.datetime.now(datetime.timezone.utc).date()
    return [(today - datetime.timedelta(days=d)).isoformat() for d in (0, 1)]


def _collect_patches(client: APISports) -> dict[str, dict]:
    """feed fixture id -> {status, finalScore?, liveClock?} for every live or
    recently-finished fixture in one of our football leagues today."""
    patches: dict[str, dict] = {}
    for date in _today_dates():
        # ttl_hours=0 → always refetch; this is the whole point of the live job.
        data = client.get(FOOTBALL_BASE, "fixtures", {"date": date}, ttl_hours=0)
        for item in data.get("response", []) or []:
            fx = item.get("fixture") or {}
            lg = item.get("league") or {}
            slug = _SLUG_BY_API_ID.get(lg.get("id"))
            if not slug:
                continue
            st = status_map.feed_status((fx.get("status") or {}).get("short"))
            if st == "scheduled":
                continue
            if st == "finished" and status_map.hours_since(fx.get("date")) > status_map.FINISHED_FEED_WINDOW_H:
                continue
            goals = item.get("goals") or {}
            patches[f"{slug}-{fx.get('id')}"] = status_map.live_fields(
                st, goals.get("home"), goals.get("away"), (fx.get("status") or {}).get("elapsed"))
    return patches


def _apply(fixtures: list[dict], patches: dict[str, dict]) -> int:
    changed = 0
    for f in fixtures:
        p = patches.get(f.get("id"))
        if not p:
            continue
        before = (f.get("status"), f.get("finalScore"), f.get("liveClock"))
        f.pop("liveClock", None)           # cleared on finish; reset while live
        f.update(p)
        if (f.get("status"), f.get("finalScore"), f.get("liveClock")) != before:
            changed += 1
    return changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", required=True, help="path to the current fixtures.json")
    args = ap.parse_args()

    path = Path(args.fixtures)
    if not path.exists():
        print(f"live_update: {path} not found — nothing to patch.")
        return 0
    if not APISPORTS_KEY:
        print("live_update: no APISPORTS_KEY — skipping.")
        return 0

    doc = json.loads(path.read_text())
    fixtures = doc.get("fixtures", []) if isinstance(doc, dict) else doc

    client = APISports(APISPORTS_KEY)
    client.throttle = 0.4
    patches = _collect_patches(client)
    if not patches:
        print("live_update: no live/finished fixtures right now — feed unchanged.")
        return 0

    changed = _apply(fixtures, patches)
    if not changed:
        print(f"live_update: {len(patches)} live/finished found, feed already up to date.")
        return 0

    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2))
    print(f"live_update: patched {changed} fixture(s) from {len(patches)} live/finished; {client.requests_made} API call(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
