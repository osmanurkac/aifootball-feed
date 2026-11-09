"""
LLM match-preview layer (the "haber analizi" card).

For each enriched upcoming fixture we assemble structured facts — standings &
stakes (title / top-4 / relegation race, points), recent form, FIXTURE
CONGESTION across all competitions (so a side with a Champions/Europa game the
same week is flagged), and injuries/suspensions — then have Claude, WITH web
search for current team news, write a short Turkish preview as a real analyst
would (not just "player X is out"): where each side stands, what's at stake, the
midweek European load, momentum, derby/motivation, home/away.

Crucially the model stays the source of truth. The LLM does NOT invent
probabilities; it proposes a BOUNDED (±0.15) attack/defence multiplier per side
with reasons, which the app applies on-device before its Monte-Carlo. So the
final result is "news + model", not "news instead of model".

Degrades gracefully: no ANTHROPIC_API_KEY, no SDK, or a failed/garbled call ->
the fixture simply gets no aiAnalysis and the app shows the base model result.
"""

import concurrent.futures
import datetime
import json
import os
import urllib.request

from config import FEED_BASE_URL

# Haiku is cheap and, fed rich structured facts, produces grounded previews; bump
# to a stronger model via NEWS_MODEL if preview quality isn't enough.
MODEL = os.environ.get("NEWS_MODEL", "claude-haiku-5-5")
# Basic web-search variant (the dynamic-filtering _20260209 one needs Opus/Sonnet).
# Web search is billed separately (~$10 / 1000 searches) and its results inflate
# input tokens, so it dominates cost — one search per match is plenty.
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search",
                   "max_uses": int(os.environ.get("NEWS_WEB_SEARCH_USES", "1"))}

ADJ_CAP = 0.15                      # the model's calibration is protected by this
ADJ_KEYS = ("homeAttack", "homeDefense", "awayAttack", "awayDefense")
CONGESTION_DAYS = 4                 # another competitive game within ±4 days = load
# Per-build budget of NEW/refreshed analyses. Because previous analyses are reused
# from the live feed (see _load_prior_analyses), this caps only what we PAY for
# each run — featured + soonest matches first; the long tail gets no AI layer and
# simulates on the base model (the user can still run those matches themselves).
MAX_ANALYSES = int(os.environ.get("NEWS_MAX_ANALYSES", "40"))
# Reuse last run's analysis until a match nears kick-off; then refresh once (so
# lineups / late news are reflected) rather than re-paying every single build.
REFRESH_HOURS = float(os.environ.get("NEWS_REFRESH_HOURS", "48"))
FIXTURES_FILE = "fixtures.json"
# Web search is the dominant cost, so grant it to only the top priority matches
# per build; everyone else gets a grounded preview from the structured facts
# (standings, form, congestion, REAL injuries) for free. ~$0.01/search, so 12 ≈
# $0.12/day cap. Set to 0 to turn web search off entirely.
WEB_SEARCH_MATCHES = int(os.environ.get("NEWS_WEB_SEARCH_MATCHES", "12"))
# League priority: the leagues a (Turkish) audience is most likely to open get
# first claim on both the per-build analysis budget and the web-search slots —
# Süper Lig, then the Champions League and the big five. Any other enriched league
# falls to the back. Fixtures carry the league NAME, mapped to slug via slug_by_name.
LEAGUE_PRIORITY = ("super_lig", "champions_league", "premier_league",
                   "laliga", "serie_a", "bundesliga", "ligue_1")
# Analyst calls are independent network requests, so run them concurrently to keep
# the daily build short (the Anthropic client is thread-safe). Tune per rate limit.
WORKERS = int(os.environ.get("NEWS_WORKERS", "8"))

# Only (re)analyse matches that kick off within this many hours — "~1 day ahead",
# not 10 days out (lineups/news a week away are worthless and the far-future spend
# is wasted). The daily cron runs once, so 48h covers the rest of today + all of
# tomorrow; each day's run then picks up the next day's fixtures. Stored analyses
# are still REUSED for free regardless of horizon; this gate only bounds what we
# PAY to newly analyse. 0 disables the gate.
HORIZON_HOURS = float(os.environ.get("NEWS_HORIZON_HOURS", "48"))

# Two-tier model routing. Headline leagues (LEAGUE_PRIORITY) get the Claude MODEL
# above (+ web search for the top few) — best quality. EVERY OTHER league gets a
# cheaper Gemini Flash-Lite preview from the same structured facts (no web search),
# so no in-window match is left without an analysis. Needs GEMINI_API_KEY; without
# it the non-headline tier is skipped (headline still runs on Claude).
FLASH_MODEL = os.environ.get("NEWS_FLASH_MODEL", "gemini-2.5-flash-lite")
# Safety cap on the cheap tier per build (the horizon window already bounds it; a
# normal day is well under this). Flash-Lite is ~10× cheaper than Haiku on text.
FLASH_MAX = int(os.environ.get("NEWS_FLASH_MAX", "200"))

# Most-recent-first W/D/L per team id, filled by build_feed from finished results.
FORM_INDEX: dict[str, list[tuple[str, str]]] = {}

SYSTEM = (
    "Sen deneyimli, tarafsız bir futbol analistisin. Sana bir maçın yapısal "
    "verileri verilecek: iki takımın lig sıralaması/puanı ve sezon hedefleri "
    "(şampiyonluk / ilk 4 / küme düşme yüzdeleri), son 5 maç formu, bu hafta "
    "oynayacakları DİĞER maçlar (lig/kupa/Avrupa — fikstür yükü), ve sakat/cezalı "
    "oyuncular. ÖNCE web_search aracıyla maça dair GÜNCEL haberleri araştır: takım "
    "haberleri, son sakatlık/ceza gelişmeleri, teknik direktör açıklamaları, "
    "muhtemel 11, moral/motivasyon, Avrupa veya kupa yorgunluğu, derbi havası. "
    "SONRA maçı bütüncül analiz et — sadece 'şu oyuncu yok' deme; sıralama ve "
    "hedefleri, haftanın yükünü (bu hafta Avrupa/kupa maçı varsa rotasyon ve "
    "yorgunluk), ev sahibi avantajını, momentumu, rekabeti ve motivasyonu birlikte "
    "tart. Türkçe, akıcı ve maça özel yaz; genel klişe cümleler kurma.\n\n"
    "Analizi önce Türkçe yaz, SONRA aynı başlık/özet/maddelerin doğal ve akıcı bir "
    "İNGİLİZCE karşılığını da ver (çeviri gibi değil, İngilizce bir analist nasıl "
    "yazarsa öyle). İngilizce alanlar boş kalmasın.\n\n"
    "ÇIKTI: başka hiçbir metin olmadan, YALNIZCA tek bir JSON nesnesi döndür:\n"
    "{\n"
    '  "headline": "KISA, çarpıcı başlık (Türkçe) — en fazla 70 karakter, tek satır",\n'
    '  "summary": "2-3 cümlelik analiz (Türkçe)",\n'
    '  "factors": ["kısa madde", "kısa madde", "kısa madde"],\n'
    '  "headline_en": "SHORT, punchy English headline — max 70 chars, one line",\n'
    '  "summary_en": "2-3 sentence English analysis",\n'
    '  "factors_en": ["short point", "short point", "short point"],\n'
    '  "adjustments": {"homeAttack": 0.0, "homeDefense": 0.0, "awayAttack": 0.0, "awayDefense": 0.0},\n'
    '  "confidence": "low | medium | high"\n'
    "}\n"
    "adjustments: her biri -0.15 ile +0.15 arası bir ÇARPAN düzeltmesidir. Pozitif "
    "= o takımın o yönü (hücum/savunma) güçlenir; negatif = zayıflar. Bir yönü "
    "habere/veriye dayandıramıyorsan 0 bırak. Düzeltmeler küçük ve gerekçeli "
    "olmalı; modelin kalibrasyonunu ezme. Örn. kilit golcü sakat -> ilgili takımın "
    "homeAttack/awayAttack'ına küçük negatif; yoğun Avrupa haftası -> hafif negatif."
)

# Web-search-free variant for the cheap (Gemini Flash-Lite) tier and any Claude
# structured-only call: same analyst role and output schema, but it works ONLY
# from the structured facts it's given (no "search the web" instruction, since
# this tier has no search tool). Built from SYSTEM with the search sentence
# swapped for a "use only the given data" one, so the schema never drifts apart.
SYSTEM_STRUCTURED = SYSTEM.replace(
    "ÖNCE web_search aracıyla maça dair GÜNCEL haberleri araştır: takım "
    "haberleri, son sakatlık/ceza gelişmeleri, teknik direktör açıklamaları, "
    "muhtemel 11, moral/motivasyon, Avrupa veya kupa yorgunluğu, derbi havası. "
    "SONRA maçı bütüncül analiz et",
    "Yalnızca sana verilen yapısal verilere (sıralama, hedefler, form, fikstür "
    "yükü, sakat/cezalılar) dayan — web araması yapma, veri dışında bilgi uydurma. "
    "Maçı bütüncül analiz et",
)


def _clamp(x) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v != v:  # NaN
        return 0.0
    return max(-ADJ_CAP, min(ADJ_CAP, round(v, 3)))


def _iso(d):
    if not d:
        return None
    try:
        return datetime.datetime.fromisoformat(d.replace("Z", "+00:00"))
    except Exception:
        return None


def _form(tid: str) -> list[str]:
    rows = sorted(FORM_INDEX.get(tid, []), key=lambda r: r[0] or "", reverse=True)
    return [r[1] for r in rows[:5]]


def _congestion(tid, this_fixture, all_fixtures, kdt, teams_out):
    """Other competitive matches for `tid` within ±CONGESTION_DAYS (fixture load,
    e.g. a midweek European game)."""
    if kdt is None:
        return []
    out = []
    for fx in all_fixtures:
        if fx is this_fixture or fx.get("sport") != "football":
            continue
        if tid not in (fx.get("homeTeamId"), fx.get("awayTeamId")):
            continue
        odt = _iso(fx.get("kickoff"))
        if odt is None or abs((odt - kdt).days) > CONGESTION_DAYS:
            continue
        opp = fx["awayTeamId"] if fx.get("homeTeamId") == tid else fx["homeTeamId"]
        out.append({
            "competition": fx.get("league"),
            "date": fx.get("kickoff"),
            "opponent": teams_out.get(opp, {}).get("name", opp),
            "venue": "home" if fx.get("homeTeamId") == tid else "away",
        })
    out.sort(key=lambda o: o.get("date") or "")
    return out[:3]


def _team_facts(tid, fixture, all_fixtures, teams_out, season_by_team, kdt):
    s = season_by_team.get(tid, {})
    unavailable = [u.get("name") for u in (fixture.get("unavailable") or [])
                   if u.get("name") and str(u.get("id", "")).startswith(f"{tid}_")]
    return {
        "name": teams_out.get(tid, {}).get("name", tid),
        "leaguePoints": s.get("record"),
        "titlePct": s.get("title"),
        "top4Pct": s.get("topFinish"),
        "relegationPct": s.get("relegation"),
        "recentForm": _form(tid),
        "unavailable": unavailable,
        "otherMatchesThisWeek": _congestion(tid, fixture, all_fixtures, kdt, teams_out),
    }


def _facts_for(fixture, all_fixtures, teams_out, season_by_team):
    kdt = _iso(fixture.get("kickoff"))
    hid, aid = fixture.get("homeTeamId"), fixture.get("awayTeamId")
    return {
        "league": fixture.get("league"),
        "kickoff": fixture.get("kickoff"),
        "home": _team_facts(hid, fixture, all_fixtures, teams_out, season_by_team, kdt),
        "away": _team_facts(aid, fixture, all_fixtures, teams_out, season_by_team, kdt),
    }


def _extract_json(text: str):
    """Pull the last well-formed {...} object out of the model's text."""
    if not text:
        return None
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            else:
                if c == '"':
                    in_str = True
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1])
                        except Exception:
                            break
        start = text.find("{", start + 1)
    return None


def _trim(s: str, n: int) -> str:
    """Trim to <= n chars at a word boundary (no mid-word cut), add an ellipsis."""
    s = s.strip()
    if len(s) <= n:
        return s
    cut = s[:n].rsplit(" ", 1)[0].rstrip(" ,.;:—-")
    return (cut or s[:n]) + "…"


def _normalize(data: dict, model: str = MODEL) -> dict:
    adj = data.get("adjustments") or {}
    conf = data.get("confidence")
    en_headline = _trim(str(data.get("headline_en", "")), 90)
    en_summary = str(data.get("summary_en", "")).strip()[:600]
    en_factors = [str(x).strip()[:120] for x in (data.get("factors_en") or []) if str(x).strip()][:5]
    en = ({"headline": en_headline, "summary": en_summary, "factors": en_factors}
          if (en_headline or en_summary or en_factors) else None)
    return {
        "schemaVersion": 2,
        "headline": _trim(str(data.get("headline", "")), 90),
        "summary": str(data.get("summary", "")).strip()[:600],
        "factors": [str(x).strip()[:120] for x in (data.get("factors") or []) if str(x).strip()][:5],
        # English variant of headline/summary/factors (the app picks by locale;
        # adjustments/confidence are language-neutral and stay top-level).
        "en": en,
        "adjustments": {k: _clamp(adj.get(k, 0.0)) for k in ADJ_KEYS},
        "confidence": conf if conf in ("low", "medium", "high") else "medium",
        "model": model,
        # When this analysis was written — used next build to decide whether to
        # reuse it or refresh it as kick-off nears. The app ignores unknown keys.
        "generatedAt": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def analyze(facts: dict, client, use_search: bool = True) -> dict | None:
    prompt = (
        "Aşağıdaki maçı analiz et. Yapısal veriler (JSON):\n\n"
        + json.dumps(facts, ensure_ascii=False, indent=2)
        + ("\n\nÖnce web'de bu maça dair güncel haberleri araştır, sonra yalnızca "
           "istenen JSON nesnesini döndür."
           if use_search else
           "\n\nVerilen yapısal verilere dayanarak analiz et ve yalnızca istenen "
           "JSON nesnesini döndür.")
    )
    # Web search only for the priority matches; otherwise structured-only (free).
    # With search on, fall back to a no-tools pass if the search call fails.
    system = SYSTEM if use_search else SYSTEM_STRUCTURED
    tool_options = ([WEB_SEARCH_TOOL], []) if use_search else ([],)
    for tools in tool_options:
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=2400,
                system=system,
                tools=tools,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(getattr(b, "text", "") for b in resp.content
                           if getattr(b, "type", "") == "text")
            data = _extract_json(text)
            if data:
                return _normalize(data, MODEL)
        except Exception as exc:  # noqa: BLE001 — resilient per-fixture
            print(f"    ai: attempt failed ({type(exc).__name__}: {exc})")
            continue
    return None


def analyze_gemini(facts: dict, gclient) -> dict | None:
    """Cheap-tier preview for a non-headline match: Gemini Flash-Lite over the same
    structured facts, no web search. Same JSON schema as `analyze`, so the app and
    `_normalize` treat it identically; tagged with the Flash model name."""
    prompt = (
        "Aşağıdaki maçı analiz et. Yapısal veriler (JSON):\n\n"
        + json.dumps(facts, ensure_ascii=False, indent=2)
        + "\n\nVerilen yapısal verilere dayanarak analiz et ve yalnızca istenen "
          "JSON nesnesini döndür."
    )
    try:
        from google.genai import types
        resp = gclient.models.generate_content(
            model=FLASH_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_STRUCTURED,
                temperature=0.7,
                max_output_tokens=2400,
                response_mime_type="application/json",
            ),
        )
        data = _extract_json(getattr(resp, "text", "") or "")
        if data:
            return _normalize(data, FLASH_MODEL)
    except Exception as exc:  # noqa: BLE001 — resilient per-fixture
        print(f"    gemini: attempt failed ({type(exc).__name__}: {exc})")
    return None


def _load_seasons(out_dir, slug_by_name):
    """teamId -> {title, topFinish, relegation, record} from seasons/<slug>.json."""
    by_team = {}
    seasons_dir = out_dir / "seasons"
    for slug in set(slug_by_name.values()):
        path = seasons_dir / f"{slug}.json"
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        for e in data.get("entries", []):
            by_team[e["teamId"]] = e
    return by_team


def _load_prior_analyses() -> dict:
    """fixtureId -> aiAnalysis from the LAST published feed.

    gh-pages is rebuilt fresh every run (force_orphan), so the live feed is the
    only carrier of state. Reusing yesterday's analysis for matches that haven't
    changed is the main cost saver — otherwise every daily build re-pays to
    analyse the same upcoming games. Best-effort: empty on first run / 404 / net
    error (then everything is analysed fresh, within the budget)."""
    url = f"{FEED_BASE_URL}/{FIXTURES_FILE}"
    try:
        req = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        prior = {fx["id"]: fx["aiAnalysis"] for fx in data.get("fixtures", [])
                 if fx.get("id") and fx.get("aiAnalysis")}
        print(f"news_analyst: loaded {len(prior)} prior analyses from live feed")
        return prior
    except Exception as exc:  # noqa: BLE001 — first run / 404 / network
        print(f"news_analyst: no prior analyses ({exc}); analysing fresh")
        return {}


def _priority_rank(fx, slug_by_name) -> int:
    """Lower = higher priority. Ranks a fixture by its league (LEAGUE_PRIORITY),
    so Süper Lig / big-league matches get the budget and web-search slots first."""
    slug = slug_by_name.get(fx.get("league"))
    try:
        return LEAGUE_PRIORITY.index(slug)
    except ValueError:
        return len(LEAGUE_PRIORITY)


def _needs_analysis(fx, prior) -> bool:
    """True when we must (re)analyse this fixture rather than reuse the stored one:
    it has no stored analysis, or it's within REFRESH_HOURS of kick-off and the
    stored analysis was written while it was still outside that window (so we
    refresh exactly once as the match nears)."""
    stored = prior.get(fx.get("id"))
    if not stored:
        return True
    if not stored.get("en"):
        return True   # TR-only (pre-English) analysis — regenerate to add the en block
    kdt = _iso(fx.get("kickoff"))
    if kdt is None:
        return False
    now = datetime.datetime.now(datetime.timezone.utc)
    if (kdt - now).total_seconds() / 3600.0 > REFRESH_HOURS:
        return False  # far out — keep the stored preview
    gen = _iso(stored.get("generatedAt"))
    if gen is None:
        return True   # legacy stored analysis (no stamp) — refresh once in-window
    return (kdt - gen).total_seconds() / 3600.0 > REFRESH_HOURS


def attach_analyses(all_fixtures, teams_out, out_dir, slug_by_name):
    """Attach `aiAnalysis` to enriched upcoming football fixtures, in place.

    Reuses the previous build's analyses from the live feed (free) and only pays to
    (re)analyse matches kicking off within HORIZON_HOURS (~1 day ahead). Two-tier:
    headline leagues go to Claude (+ web search for the top few), every other league
    to Gemini Flash-Lite — so no in-window match is left without a preview."""
    prior = _load_prior_analyses()

    # Enriched upcoming football fixtures (these carry injuries + resolvable season
    # data — covers all configured leagues, not just the headline ones).
    targets = [fx for fx in all_fixtures
               if fx.get("sport") == "football" and fx.get("unavailable") is not None]

    # 1) Reuse last run's analysis wherever a refresh isn't due (the main saver).
    #    Reuse is free, so it's NOT gated by the horizon.
    for fx in targets:
        stored = prior.get(fx.get("id"))
        if stored and not _needs_analysis(fx, prior):
            fx["aiAnalysis"] = stored
    reused = sum(1 for fx in targets if fx.get("aiAnalysis"))

    # 2) What still needs a fresh/refreshed analysis AND kicks off within the
    #    horizon (~1 day ahead) — we never pay to analyse games a week+ out.
    now = datetime.datetime.now(datetime.timezone.utc)

    def _in_horizon(fx) -> bool:
        kdt = _iso(fx.get("kickoff"))
        if kdt is None or kdt < now:
            return False
        if HORIZON_HOURS <= 0:
            return True
        return (kdt - now).total_seconds() / 3600.0 <= HORIZON_HOURS

    todo = [fx for fx in targets if _needs_analysis(fx, prior) and _in_horizon(fx)]

    # Route by tier: headline leagues -> Claude (best quality, web search for the
    # top few); EVERY OTHER league -> Gemini Flash-Lite (cheap) so no in-window
    # match is left without an analysis.
    headline = [fx for fx in todo if _priority_rank(fx, slug_by_name) < len(LEAGUE_PRIORITY)]
    other = [fx for fx in todo if _priority_rank(fx, slug_by_name) >= len(LEAGUE_PRIORITY)]
    headline.sort(key=lambda f: (_priority_rank(f, slug_by_name), f.get("kickoff") or ""))
    other.sort(key=lambda f: f.get("kickoff") or "")
    headline = headline[:MAX_ANALYSES]
    other = other[:FLASH_MAX]

    season_by_team = _load_seasons(out_dir, slug_by_name)
    done = searched = flash_done = 0

    # --- Headline tier: Claude (+ web search for the top few) ---
    akey = os.environ.get("ANTHROPIC_API_KEY")
    if headline and akey:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=akey)
            search_ids = {fx.get("id") for fx in headline[:WEB_SEARCH_MATCHES]}

            def cwork(fx):
                facts = _facts_for(fx, all_fixtures, teams_out, season_by_team)
                return fx, analyze(facts, client, use_search=fx.get("id") in search_ids)

            with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
                for fx, res in pool.map(cwork, headline):
                    if res:
                        fx["aiAnalysis"] = res
                        done += 1
                        if fx.get("id") in search_ids:
                            searched += 1
                        h = teams_out.get(fx.get("homeTeamId"), {}).get("name", "?")
                        a = teams_out.get(fx.get("awayTeamId"), {}).get("name", "?")
                        print(f"    ai ✓ {h} - {a} ({res['confidence']})")
        except Exception as exc:  # noqa: BLE001
            print(f"news_analyst: Claude tier unavailable ({type(exc).__name__}: {exc})")
    elif headline:
        print(f"news_analyst: no ANTHROPIC_API_KEY — skipping {len(headline)} headline analyses")

    # --- Cheap tier: Gemini Flash-Lite for every other league (structured-only) ---
    gkey = os.environ.get("GEMINI_API_KEY")
    if other and gkey:
        try:
            from google import genai
            gclient = genai.Client(api_key=gkey)

            def gwork(fx):
                facts = _facts_for(fx, all_fixtures, teams_out, season_by_team)
                return fx, analyze_gemini(facts, gclient)

            with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
                for fx, res in pool.map(gwork, other):
                    if res:
                        fx["aiAnalysis"] = res
                        flash_done += 1
                        h = teams_out.get(fx.get("homeTeamId"), {}).get("name", "?")
                        a = teams_out.get(fx.get("awayTeamId"), {}).get("name", "?")
                        print(f"    flash ✓ {h} - {a} ({res['confidence']})")
        except Exception as exc:  # noqa: BLE001
            print(f"news_analyst: Gemini tier unavailable ({type(exc).__name__}: {exc})")
    elif other:
        print(f"news_analyst: no GEMINI_API_KEY — skipping {len(other)} Flash-Lite analyses")

    # A refresh that failed shouldn't drop the existing preview — fall back to it.
    for fx in headline + other:
        if not fx.get("aiAnalysis") and prior.get(fx.get("id")):
            fx["aiAnalysis"] = prior[fx["id"]]

    print(f"news_analyst: {done} Claude ({searched} w/ web search) + {flash_done} "
          f"Flash-Lite + {reused} reused "
          f"(models: {MODEL} / {FLASH_MODEL}, horizon={HORIZON_HOURS}h)")
