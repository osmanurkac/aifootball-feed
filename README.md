# aifootball-feed

Static JSON feed for the **Match Simulator** iOS app. The app reads this feed to
list fixtures, draw team marks, and seed its on-device Monte-Carlo simulations.

- Built by [`pipeline/`](pipeline/) (numpy only — fits the models, runs season
  simulations, scores a backtest track record).
- Published daily to the **`gh-pages`** branch by [GitHub Actions](.github/workflows/feed.yml)
  and served via **GitHub Pages**:

  ```
  https://osmanurkac.github.io/aifootball-feed/
  ```

## Layout

```
fixtures.json            upcoming / live / finished fixtures
teams.json               team marks (3-letter code + colors) — no club crests
track_record.json        model hit-rate / Brier backtest
football/<slug>.json     fitted league model (Dixon-Coles)
basketball/<slug>.json   fitted league model
seasons/<slug>.json      season title-race simulation
```

## Data source

Real data comes from [API-Sports](https://api-sports.io). The daily Action runs
`build_feed.py --source api`; when no key is present it falls back to a
deterministic **demo** feed, so the feed is always complete.

To enable real data, add a repository secret here:

- **Settings → Secrets and variables → Actions → `APISPORTS_KEY`**

Then run the **Build feed** workflow (Actions tab → Run workflow).

## Notes

Not betting advice. No odds, no tips — this is a simulation toy. Team marks use
our own 3-letter codes and colors; no official club crests or player photos.
