"""Walk-forward backtest + calibration for the AIFootball model.

Trains/evaluates the SAME model the app uses (time-decayed Dixon-Coles attack/
defence, home advantage, rho) strictly out-of-sample: every match is predicted
from only the matches that happened BEFORE it. Measures 1X2, Over/Under 2.5,
BTTS, corners and cards — accuracy, log-loss, Brier and calibration — and
benchmarks 1X2 and O/U 2.5 against the bookmaker's own (de-vigged) odds.

Data: free football-data.co.uk mirror (results + corners + cards + odds).
Incremental decayed sums make it O(N): decayed_avg at time t uses weights
2^(t_i/H); the 2^(t/H) factor cancels in the ratio, so we keep running sums.

Run: python3 -I backtest_research.py
"""
from __future__ import annotations

import csv, math, sys, datetime
from collections import defaultdict

DATA = "data/Matches.csv"
# App's leagues (+ Turkish Süper Lig for BJK–FB style checks).
DIVISIONS = {"E0": "Premier League", "SP1": "LaLiga", "I1": "Serie A",
             "D1": "Bundesliga", "F1": "Ligue 1", "T1": "Süper Lig"}
MAX_GOALS = 10
REF = datetime.date(2000, 1, 1)

# ---- Model parameters (the knobs we tune) --------------------------------
PARAMS = {
    "half_life_days": 365.0,   # time-decay half-life
    "rho": -0.08,              # Dixon-Coles low-score correction
    "shrink": 4.0,             # pseudo-matches toward league average (regularization)
    "min_history": 8,          # per-team prior matches before we predict
    "max_mult": 3.0,           # clamp attack/defence multiplier
}


def parse_date(s: str):
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        try: return datetime.datetime.strptime(s, fmt).date()
        except ValueError: pass
    return None


def fnum(s):
    try: return float(s)
    except (ValueError, TypeError): return None


def load():
    rows = defaultdict(list)
    with open(DATA, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            div = r["Division"]
            if div not in DIVISIONS: continue
            d = parse_date(r.get("MatchDate", ""))
            fh, fa = fnum(r.get("FTHome")), fnum(r.get("FTAway"))
            if d is None or fh is None or fa is None: continue
            rows[div].append({
                "date": d, "home": r["HomeTeam"], "away": r["AwayTeam"],
                "fh": int(fh), "fa": int(fa),
                "hc": fnum(r.get("HomeCorners")), "ac": fnum(r.get("AwayCorners")),
                "hk": (fnum(r.get("HomeYellow")) or 0) + (fnum(r.get("HomeRed")) or 0),
                "ak": (fnum(r.get("AwayYellow")) or 0) + (fnum(r.get("AwayRed")) or 0),
                "has_ck": r.get("HomeCorners") not in (None, "", "NA"),
                "oh": fnum(r.get("OddHome")), "od": fnum(r.get("OddDraw")), "oa": fnum(r.get("OddAway")),
                "o25": fnum(r.get("Over25")), "u25": fnum(r.get("Under25")),
            })
    for div in rows: rows[div].sort(key=lambda m: m["date"])
    return rows


# ---- Dixon-Coles score grid ---------------------------------------------

def _pois(lam):
    out = [0.0] * (MAX_GOALS + 1)
    for k in range(MAX_GOALS + 1):
        out[k] = math.exp(k * math.log(lam) - lam - math.lgamma(k + 1)) if lam > 0 else (1.0 if k == 0 else 0.0)
    return out


def dc_grid(lh, la, rho):
    hp, ap = _pois(lh), _pois(la)
    g = [[0.0] * (MAX_GOALS + 1) for _ in range(MAX_GOALS + 1)]
    tot = 0.0
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            if (h, a) == (0, 0): tau = 1 - lh * la * rho
            elif (h, a) == (0, 1): tau = 1 + lh * rho
            elif (h, a) == (1, 0): tau = 1 + la * rho
            elif (h, a) == (1, 1): tau = 1 - rho
            else: tau = 1.0
            p = max(0.0, tau) * hp[h] * ap[a]
            g[h][a] = p; tot += p
    if tot > 0:
        for h in range(MAX_GOALS + 1):
            for a in range(MAX_GOALS + 1): g[h][a] /= tot
    return g


def grid_markets(g):
    home = draw = away = over25 = btts = 0.0
    for h in range(MAX_GOALS + 1):
        for a in range(MAX_GOALS + 1):
            p = g[h][a]
            if h > a: home += p
            elif h == a: draw += p
            else: away += p
            if h + a >= 3: over25 += p
            if h >= 1 and a >= 1: btts += p
    return home, draw, away, over25, btts


# ---- Metrics -------------------------------------------------------------

class Metric:
    """Accumulates log-loss, Brier, accuracy and a 10-bin reliability curve."""
    def __init__(self):
        self.ll = self.brier = 0.0
        self.hit = self.n = 0
        self.bins = [[0.0, 0, 0] for _ in range(10)]  # [sum_p, n, n_pos] for binary

    def add_binary(self, p, outcome):
        p = min(max(p, 1e-9), 1 - 1e-9)
        self.ll += -(outcome * math.log(p) + (1 - outcome) * math.log(1 - p))
        self.brier += (p - outcome) ** 2
        self.hit += 1 if (p >= 0.5) == (outcome == 1) else 0
        self.n += 1
        b = self.bins[min(9, int(p * 10))]
        b[0] += p; b[1] += 1; b[2] += outcome

    def add_multi(self, probs, idx):
        probs = [min(max(x, 1e-9), 1) for x in probs]
        s = sum(probs); probs = [x / s for x in probs]
        self.ll += -math.log(probs[idx])
        self.brier += sum((p - (1 if i == idx else 0)) ** 2 for i, p in enumerate(probs))
        self.hit += 1 if max(range(len(probs)), key=lambda i: probs[i]) == idx else 0
        self.n += 1

    def row(self, name):
        if self.n == 0: return f"  {name:<16} (no samples)"
        return f"  {name:<16} n={self.n:<6} logloss={self.ll/self.n:.4f}  brier={self.brier/self.n:.4f}  acc={self.hit/self.n:.3f}"

    def calibration(self, name):
        out = [f"  {name} reliability (pred% -> actual%):"]
        for i, (sp, n, pos) in enumerate(self.bins):
            if n == 0: continue
            out.append(f"    [{i*10:>2}-{i*10+10:>3}%] pred={100*sp/n:5.1f}  actual={100*pos/n:5.1f}  n={n}")
        return "\n".join(out)


def devig3(oh, od, oa):
    if not (oh and od and oa): return None
    ih, idr, ia = 1/oh, 1/od, 1/oa
    s = ih + idr + ia
    return ih/s, idr/s, ia/s


# ---- Walk-forward --------------------------------------------------------

def run(params, divisions=None, start_year=2012, verbose=True):
    H = params["half_life_days"]; rho = params["rho"]
    shrink = params["shrink"]; min_h = params["min_history"]; mx = params["max_mult"]
    data = run.cache
    divs = divisions or list(DIVISIONS)

    M = {k: Metric() for k in ["1x2", "ou25", "btts", "corners", "cards"]}
    MK = {k: Metric() for k in ["1x2", "ou25"]}  # bookmaker benchmark on same matches

    def w(d):  # decay weight relative to REF (ratio-stable)
        return 2.0 ** ((d - REF).days / H)

    for div in divs:
        # Per-team running decayed sums.
        s_sc = defaultdict(float); s_cc = defaultdict(float); s_w = defaultdict(float)
        s_corn = defaultdict(float); s_cards = defaultdict(float); s_ck = defaultdict(float)
        cnt = defaultdict(int)
        g_goals = g_w = g_home = g_away = 0.0

        for m in data[div]:
            h, a = m["home"], m["away"]
            predict = (m["date"].year >= start_year and cnt[h] >= min_h and cnt[a] >= min_h
                       and g_w > 0 and s_w[h] > 0 and s_w[a] > 0)
            if predict:
                lavg = (g_goals / g_w) / 2
                hadv = max(1.0, math.sqrt((g_home / g_w) / (g_away / g_w))) if g_away > 0 else 1.15

                def mult(num, wt):
                    # shrink toward league average with `shrink` pseudo-matches
                    val = (num + shrink * lavg) / (wt + shrink)
                    return min(mx, max(0.4, val / lavg)) if lavg > 0 else 1.0
                atk_h = mult(s_sc[h], s_w[h]); def_h = mult(s_cc[h], s_w[h])
                atk_a = mult(s_sc[a], s_w[a]); def_a = mult(s_cc[a], s_w[a])
                lh = max(0.05, lavg * atk_h * def_a * hadv)
                la = max(0.05, lavg * atk_a * def_h)

                g = dc_grid(lh, la, rho)
                ph, pd, pa, pov, pbtts = grid_markets(g)

                # Actual outcomes
                res = 0 if m["fh"] > m["fa"] else (1 if m["fh"] == m["fa"] else 2)
                M["1x2"].add_multi([ph, pd, pa], res)
                M["ou25"].add_binary(pov, 1 if m["fh"] + m["fa"] >= 3 else 0)
                M["btts"].add_binary(pbtts, 1 if m["fh"] >= 1 and m["fa"] >= 1 else 0)

                mk = devig3(m["oh"], m["od"], m["oa"])
                if mk: MK["1x2"].add_multi(list(mk), res)
                if m["o25"] and m["u25"]:
                    io, iu = 1/m["o25"], 1/m["u25"]
                    MK["ou25"].add_binary(io/(io+iu), 1 if m["fh"]+m["fa"] >= 3 else 0)

                # Corners / cards (team means -> Poisson total). Line 9.5 / 3.5.
                if s_ck[h] > 0 and s_ck[a] > 0:
                    cmean = s_corn[h]/s_ck[h] + s_corn[a]/s_ck[a]
                    kmean = s_cards[h]/s_ck[h] + s_cards[a]/s_ck[a]
                    if m["has_ck"] and m["hc"] is not None and m["ac"] is not None:
                        pc = sum(_pois_total(cmean, k) for k in range(10, 60))
                        M["corners"].add_binary(pc, 1 if (m["hc"] + m["ac"]) >= 10 else 0)
                        pk = sum(_pois_total(kmean, k) for k in range(4, 40))
                        M["cards"].add_binary(pk, 1 if (m["hk"] + m["ak"]) >= 4 else 0)

            # Update running sums with this match.
            wt = w(m["date"])
            s_sc[h] += m["fh"] * wt; s_cc[h] += m["fa"] * wt; s_w[h] += wt
            s_sc[a] += m["fa"] * wt; s_cc[a] += m["fh"] * wt; s_w[a] += wt
            if m["has_ck"] and m["hc"] is not None:
                s_corn[h] += m["hc"] * wt; s_corn[a] += m["ac"] * wt
                s_cards[h] += m["hk"] * wt; s_cards[a] += m["ak"] * wt
                s_ck[h] += wt; s_ck[a] += wt
            g_goals += (m["fh"] + m["fa"]) * wt; g_home += m["fh"] * wt; g_away += m["fa"] * wt; g_w += wt
            cnt[h] += 1; cnt[a] += 1

    if verbose:
        print(f"\n=== Params: {params} | start_year={start_year} ===")
        for k in ["1x2", "ou25", "btts", "corners", "cards"]:
            print(M[k].row(k))
        print("  --- bookmaker benchmark (same matches) ---")
        print(MK["1x2"].row("1x2 (market)"))
        print(MK["ou25"].row("ou25 (market)"))
    return M, MK


_pois_cache = {}
def _pois_total(mean, k):
    if mean <= 0: return 1.0 if k == 0 else 0.0
    return math.exp(k * math.log(mean) - mean - math.lgamma(k + 1))


if __name__ == "__main__":
    print("loading…", file=sys.stderr)
    run.cache = load()
    print("loaded divisions:", {d: len(run.cache[d]) for d in run.cache}, file=sys.stderr)
    M, MK = run(PARAMS)
    print()
    print(M["1x2"].calibration("1X2 home"))  # note: multi; shows home-bin proxy off, so:
    print(M["ou25"].calibration("Over 2.5"))
    print(M["btts"].calibration("BTTS"))
