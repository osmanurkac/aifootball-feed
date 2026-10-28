"""Tune hyperparameters + fit probability calibration, out-of-sample.

Reuses the model from backtest_research. Produces, from the free historical data:
  1. a small grid search over (half_life, shrink, rho) by out-of-sample 1X2 +
     O/U 2.5 log-loss,
  2. post-hoc calibration (temperature for 1X2, temperature+bias for the binary
     markets) fit on a TRAIN split (pre-2021) and validated on TEST (2021+),
     so the displayed % matches the real hit rate.

Run: python3 -I tune_calibrate.py
"""
from __future__ import annotations
import math, sys, itertools
import backtest_research as B


def predictions(params, start_year=2012):
    """Walk-forward; return per-market samples: lists of (prob(s), outcome, year)."""
    H = params["half_life_days"]; rho = params["rho"]
    shrink = params["shrink"]; min_h = params["min_history"]; mx = params["max_mult"]
    from collections import defaultdict
    out = {k: [] for k in ["1x2", "ou25", "btts", "corners", "cards", "mk1x2", "mkou"]}

    def w(d): return 2.0 ** ((d - B.REF).days / H)

    for div in B.DIVISIONS:
        s_sc = defaultdict(float); s_cc = defaultdict(float); s_w = defaultdict(float)
        s_corn = defaultdict(float); s_cards = defaultdict(float); s_ck = defaultdict(float)
        cnt = defaultdict(int); g_goals = g_w = g_home = g_away = 0.0
        for m in B.run.cache[div]:
            h, a = m["home"], m["away"]
            if (m["date"].year >= start_year and cnt[h] >= min_h and cnt[a] >= min_h
                    and g_w > 0 and s_w[h] > 0 and s_w[a] > 0):
                lavg = (g_goals / g_w) / 2
                hadv = max(1.0, math.sqrt((g_home/g_w)/(g_away/g_w))) if g_away > 0 else 1.15
                def mult(num, wt):
                    val = (num + shrink*lavg)/(wt+shrink)
                    return min(mx, max(0.4, val/lavg)) if lavg > 0 else 1.0
                lh = max(0.05, lavg*mult(s_sc[h],s_w[h])*mult(s_cc[a],s_w[a])*hadv)
                la = max(0.05, lavg*mult(s_sc[a],s_w[a])*mult(s_cc[h],s_w[h]))
                g = B.dc_grid(lh, la, rho)
                ph, pd, pa, pov, pbtts = B.grid_markets(g)
                yr = m["date"].year
                res = 0 if m["fh"] > m["fa"] else (1 if m["fh"] == m["fa"] else 2)
                out["1x2"].append(((ph, pd, pa), res, yr))
                out["ou25"].append((pov, 1 if m["fh"]+m["fa"] >= 3 else 0, yr))
                out["btts"].append((pbtts, 1 if m["fh"] >= 1 and m["fa"] >= 1 else 0, yr))
                mk = B.devig3(m["oh"], m["od"], m["oa"])
                if mk: out["mk1x2"].append((mk, res, yr))
                if m["o25"] and m["u25"]:
                    io, iu = 1/m["o25"], 1/m["u25"]
                    out["mkou"].append((io/(io+iu), 1 if m["fh"]+m["fa"] >= 3 else 0, yr))
                if s_ck[h] > 0 and s_ck[a] > 0 and m["has_ck"] and m["hc"] is not None:
                    cmean = s_corn[h]/s_ck[h] + s_corn[a]/s_ck[a]
                    kmean = s_cards[h]/s_ck[h] + s_cards[a]/s_ck[a]
                    pc = sum(B._pois_total(cmean, k) for k in range(10, 60))
                    pk = sum(B._pois_total(kmean, k) for k in range(4, 40))
                    out["corners"].append((pc, 1 if (m["hc"]+m["ac"]) >= 10 else 0, yr))
                    out["cards"].append((pk, 1 if (m["hk"]+m["ak"]) >= 4 else 0, yr))
            wt = w(m["date"])
            s_sc[h]+=m["fh"]*wt; s_cc[h]+=m["fa"]*wt; s_w[h]+=wt
            s_sc[a]+=m["fa"]*wt; s_cc[a]+=m["fh"]*wt; s_w[a]+=wt
            if m["has_ck"] and m["hc"] is not None:
                s_corn[h]+=m["hc"]*wt; s_corn[a]+=m["ac"]*wt
                s_cards[h]+=m["hk"]*wt; s_cards[a]+=m["ak"]*wt; s_ck[h]+=wt; s_ck[a]+=wt
            g_goals+=(m["fh"]+m["fa"])*wt; g_home+=m["fh"]*wt; g_away+=m["fa"]*wt; g_w+=wt
            cnt[h]+=1; cnt[a]+=1
    return out


# ---- scoring helpers ----
def ll_multi(samples):
    s = 0.0
    for probs, idx, _ in samples:
        p = [min(max(x, 1e-9), 1) for x in probs]; z = sum(p)
        s += -math.log(p[idx]/z)
    return s/len(samples) if samples else 0

def ll_bin(samples):
    s = 0.0
    for p, y, _ in samples:
        p = min(max(p, 1e-9), 1-1e-9); s += -(y*math.log(p)+(1-y)*math.log(1-p))
    return s/len(samples) if samples else 0

def acc_multi(samples):
    return sum(1 for pr, i, _ in samples if max(range(3), key=lambda k: pr[k]) == i)/len(samples)

def sig(x): return 1/(1+math.exp(-max(-30, min(30, x))))
def logit(p): p = min(max(p, 1e-6), 1-1e-6); return math.log(p/(1-p))


def fit_temp_multi(samples):
    best = (1.0, 1e9)
    for T in [x/100 for x in range(70, 251, 5)]:
        s = 0.0
        for probs, idx, _ in samples:
            p = [max(x, 1e-9)**(1/T) for x in probs]; z = sum(p)
            s += -math.log(p[idx]/z)
        if s/len(samples) < best[1]: best = (T, s/len(samples))
    return best[0]

def apply_temp_multi(probs, T):
    p = [max(x, 1e-9)**(1/T) for x in probs]; z = sum(p); return [x/z for x in p]

def fit_platt(samples):
    """p_cal = sigmoid(logit(p)/T + b); grid over T,b minimizing logloss."""
    best = (1.0, 0.0, 1e9)
    for T in [x/100 for x in range(70, 261, 10)]:
        for b in [x/100 for x in range(-40, 41, 5)]:
            s = 0.0
            for p, y, _ in samples:
                pc = sig(logit(p)/T + b); pc = min(max(pc, 1e-9), 1-1e-9)
                s += -(y*math.log(pc)+(1-y)*math.log(1-pc))
            m = s/len(samples)
            if m < best[2]: best = (T, b, m)
    return best[0], best[1]

def apply_platt(p, T, b): return sig(logit(p)/T + b)

def reliability_bin(samples, transform=None):
    bins = [[0.0, 0, 0] for _ in range(10)]
    for p, y, _ in samples:
        pc = transform(p) if transform else p
        bk = bins[min(9, int(pc*10))]; bk[0]+=pc; bk[1]+=1; bk[2]+=y
    rows = []
    for i, (sp, n, pos) in enumerate(bins):
        if n: rows.append(f"    [{i*10:>2}-{i*10+10:>3}%] pred={100*sp/n:5.1f} actual={100*pos/n:5.1f} n={n}")
    return "\n".join(rows)


if __name__ == "__main__":
    print("loading…", file=sys.stderr)
    B.run.cache = B.load()

    # 1) Grid search on raw out-of-sample log-loss.
    print("\n### GRID SEARCH (out-of-sample, 2012+) ###")
    grid = list(itertools.product([180, 270, 365, 540], [2.0, 4.0, 8.0], [-0.06, -0.08, -0.11]))
    results = []
    for hl, sh, rho in grid:
        p = dict(B.PARAMS, half_life_days=hl, shrink=sh, rho=rho)
        s = predictions(p)
        l1, lo = ll_multi(s["1x2"]), ll_bin(s["ou25"])
        results.append((l1+lo, l1, lo, hl, sh, rho))
    results.sort()
    print("  best (by 1x2+ou25 logloss):  half_life / shrink / rho")
    for tot, l1, lo, hl, sh, rho in results[:6]:
        print(f"    hl={hl:<4} shrink={sh:<4} rho={rho:<6}  1x2_ll={l1:.4f}  ou25_ll={lo:.4f}")
    best = results[0]
    bestp = dict(B.PARAMS, half_life_days=best[3], shrink=best[4], rho=best[5])

    # 2) Calibration with the best params (TRAIN pre-2021, TEST 2021+).
    print(f"\n### CALIBRATION (best params {best[3]}/{best[4]}/{best[5]}) ###")
    s = predictions(bestp)
    def split(key):
        tr = [x for x in s[key] if x[2] < 2021]; te = [x for x in s[key] if x[2] >= 2021]
        return tr, te

    tr, te = split("1x2"); T = fit_temp_multi(tr)
    te_cal = [(apply_temp_multi(pr, T), i, y) for pr, i, y in te]
    print(f"\n1X2  temp T={T}")
    print(f"  TEST logloss raw={ll_multi(te):.4f}  calibrated={ll_multi(te_cal):.4f}  acc={acc_multi(te):.3f}")
    mk = [x for x in s["mk1x2"] if x[2] >= 2021]
    print(f"  bookmaker logloss={ll_multi(mk):.4f}  acc={acc_multi(mk):.3f}")

    for key, line, actual_label in [("ou25", "Over 2.5", "O2.5"), ("btts", "BTTS", "BTTS"),
                                    ("corners", "Corners 9.5+", "C10+"), ("cards", "Cards 3.5+", "K4+")]:
        tr, te = split(key)
        if not tr or not te: continue
        T, b = fit_platt(tr)
        trans = lambda p, T=T, b=b: apply_platt(p, T, b)
        raw, cal = ll_bin(te), ll_bin([(trans(p), y, yr) for p, y, yr in te])
        print(f"\n{line}  platt T={T} b={b}")
        print(f"  TEST logloss raw={raw:.4f}  calibrated={cal:.4f}")
        print("  raw reliability:");  print(reliability_bin(te))
        print("  calibrated reliability:"); print(reliability_bin(te, trans))
