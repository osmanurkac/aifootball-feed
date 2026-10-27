# Data pipeline

Tiny Python pipeline that fetches football/basketball data, fits the model
parameters the app's engine needs, and publishes a static JSON feed to GitHub
Pages. No database, no server. Run daily by GitHub Actions (`.github/workflows/feed.yml`).

## Files

- `config.py` — leagues, API-Sports ids, seasons, request budget, paths.
- `apisports.py` — API-Sports client with on-disk caching + a daily request budget.
- `team_colors.py` — our own team marks (colours + 3-letter code). **Never** uses
  official crests/logos — only team names + our palette.
- `football_model.py` — time-decayed attack/defence, negative-binomial corners/cards.
- `basketball_model.py` — pace + offensive/defensive ratings, player rates.
- `season_sim.py` — 10,000-run season simulations (title / top-4 / relegation; title / playoffs).
- `backtest.py` — Brier score + hit rate → `track_record.json`.
- `build_feed.py` — orchestrator. `--source demo` (no key) or `--source api`.

Output goes to `pipeline/out/` and mirrors the app's remote layout:

```
fixtures.json  teams.json  track_record.json
football/<league>.json  basketball/<league>.json  seasons/<league>.json
```

## Run locally

```bash
pip install -r pipeline/requirements.txt
cd pipeline
python build_feed.py --source demo    # complete feed, no API key
python build_feed.py --source api     # real data (needs APISPORTS_KEY); per-league demo fallback
```

The `demo` feed exercises the real fitting / simulation / backtest code, so it's a
faithful stand-in while the API key is being set up.

## What you need to set up (first time)

### 1. API-Sports account (football + basketball)
1. Go to **https://dashboard.api-sports.io** and create an account (or sign in via
   RapidAPI). The direct API-Sports plan is simplest here.
2. Subscribe to **API-Football** and **API-Basketball** (the Free plan works to
   start — ~100 requests/day; the pipeline caches and respects a budget).
3. Open **Dashboard → Account/Profile** and copy your **API key** (one key works
   for both APIs on api-sports.io).
4. In GitHub: **repo → Settings → Secrets and variables → Actions → New repository
   secret**. Name it **`APISPORTS_KEY`**, paste the key, save.
   (Optional: add `APISPORTS_BUDGET` to change the daily request cap, default 90.)

### 2. Enable GitHub Pages
1. Push this repo to GitHub.
2. Run the workflow once: **Actions → "Build feed" → Run workflow** (this creates
   the `gh-pages` branch).
3. **repo → Settings → Pages →** Source: **Deploy from a branch**, Branch:
   **`gh-pages`** / **`/ (root)`**, Save.
4. Your feed base URL is **`https://<your-username>.github.io/<your-repo>`**.

### 3. Point the app at the feed
- Edit **`Config/Shared.xcconfig`** → set `FEED_BASE_URL` to your Pages URL from
  step 2.4 (keep the `/$()/` slash escape). The app reads it from Info.plist via
  `AppConfig.feedBaseURL`, caches the last feed to disk, and falls back to the
  bundled sample when offline.

## API parsing — verify on first live run

`build_from_api` follows the API-Sports **v3 football** (`/fixtures`, `/standings`)
and **v1 basketball** (`/games`) response shapes. The first live run should be
checked against the dashboard's API tester — confirm field names (`response[].fixture`,
`goals`, `teams`, `scores.*.total`, `status.short`) and the basketball **league ids
and season format** in `config.py` (basketball ids/seasons vary by plan). Any league
that returns no usable data automatically falls back to demo synthesis, so the feed
is always complete.
