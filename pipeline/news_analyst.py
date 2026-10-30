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

import datetime
import json
import os

# Haiku is cheap and, fed rich structured facts, produces grounded previews; bump
# to a stronger model via NEWS_MODEL if preview quality isn't enough.
MODEL = os.environ.get("NEWS_MODEL", "claude-haiku-5-5")
# Basic web-search variant (the dynamic-filtering _20260209 one needs Opus/Sonnet).
WEB_SEARCH_TOOL = {"type": "web_search_20250305", "name": "web_search", "max_uses": 4}

ADJ_CAP = 0.15                      # the model's calibration is protected by this
ADJ_KEYS = ("homeAttack", "homeDefense", "awayAttack", "awayDefense")
CONGESTION_DAYS = 4                 # another competitive game within ±4 days = load
MAX_ANALYSES = int(os.environ.get("NEWS_MAX_ANALYSES", "150"))

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
    "ÇIKTI: başka hiçbir metin olmadan, YALNIZCA tek bir JSON nesnesi döndür:\n"
    "{\n"
    '  "headline": "tek cümlelik, maça özel başlık",\n'
    '  "summary": "2-3 cümlelik analiz",\n'
    '  "factors": ["kısa madde", "kısa madde", "kısa madde"],\n'
    '  "adjustments": {"homeAttack": 0.0, "homeDefense": 0.0, "awayAttack": 0.0, "awayDefense": 0.0},\n'
    '  "confidence": "low | medium | high"\n'
    "}\n"
    "adjustments: her biri -0.15 ile +0.15 arası bir ÇARPAN düzeltmesidir. Pozitif "
    "= o takımın o yönü (hücum/savunma) güçlenir; negatif = zayıflar. Bir yönü "
    "habere/veriye dayandıramıyorsan 0 bırak. Düzeltmeler küçük ve gerekçeli "
    "olmalı; modelin kalibrasyonunu ezme. Örn. kilit golcü sakat -> ilgili takımın "
    "homeAttack/awayAttack'ına küçük negatif; yoğun Avrupa haftası -> hafif negatif."
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


def _normalize(data: dict) -> dict:
    adj = data.get("adjustments") or {}
    conf = data.get("confidence")
    return {
        "schemaVersion": 1,
        "headline": str(data.get("headline", "")).strip()[:120],
        "summary": str(data.get("summary", "")).strip()[:600],
        "factors": [str(x).strip()[:120] for x in (data.get("factors") or []) if str(x).strip()][:5],
        "adjustments": {k: _clamp(adj.get(k, 0.0)) for k in ADJ_KEYS},
        "confidence": conf if conf in ("low", "medium", "high") else "medium",
        "model": MODEL,
    }


def analyze(facts: dict, client) -> dict | None:
    prompt = (
        "Aşağıdaki maçı analiz et. Yapısal veriler (JSON):\n\n"
        + json.dumps(facts, ensure_ascii=False, indent=2)
        + "\n\nÖnce web'de bu maça dair güncel haberleri araştır, sonra yalnızca "
          "istenen JSON nesnesini döndür."
    )
    # Try with web search; if that fails for any reason, fall back to a
    # structured-only pass so we still get a grounded preview.
    for tools in ([WEB_SEARCH_TOOL], []):
        try:
            resp = client.messages.create(
                model=MODEL,
                max_tokens=1600,
                system=SYSTEM,
                tools=tools,
                messages=[{"role": "user", "content": prompt}],
            )
            text = "".join(getattr(b, "text", "") for b in resp.content
                           if getattr(b, "type", "") == "text")
            data = _extract_json(text)
            if data:
                return _normalize(data)
        except Exception as exc:  # noqa: BLE001 — resilient per-fixture
            print(f"    ai: attempt failed ({type(exc).__name__}: {exc})")
            continue
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


def attach_analyses(all_fixtures, teams_out, out_dir, slug_by_name):
    """Attach `aiAnalysis` to each enriched upcoming football fixture, in place."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("news_analyst: no ANTHROPIC_API_KEY — skipping AI analysis")
        return
    try:
        import anthropic
    except Exception:
        print("news_analyst: anthropic SDK not installed — skipping AI analysis")
        return

    client = anthropic.Anthropic(api_key=key)
    season_by_team = _load_seasons(out_dir, slug_by_name)

    # Enriched upcoming fixtures only (these carry injuries) — the headline leagues.
    targets = [fx for fx in all_fixtures
               if fx.get("sport") == "football" and fx.get("unavailable") is not None]
    targets.sort(key=lambda f: f.get("kickoff") or "")

    done = 0
    for fx in targets:
        if done >= MAX_ANALYSES:
            print(f"news_analyst: hit MAX_ANALYSES ({MAX_ANALYSES}) — stopping")
            break
        facts = _facts_for(fx, all_fixtures, teams_out, season_by_team)
        res = analyze(facts, client)
        if res:
            fx["aiAnalysis"] = res
            done += 1
            h = facts["home"]["name"]
            a = facts["away"]["name"]
            print(f"    ai ✓ {h} - {a} ({res['confidence']})")
    print(f"news_analyst: attached {done} analyses (model={MODEL})")
