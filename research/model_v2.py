"""Model v2 research: proper Dixon-Coles MLE vs the current heuristic vs market.

Walk-forward, out-of-sample, on the free historical mirror. For each match we
predict with THREE models and compare log-loss / accuracy:
  - heuristic : the current app fit (time-decayed scored/conceded ratios)
  - mle       : time-weighted ridge Poisson GLM (attack/defence/home) — the DC MLE
  - market    : the bookmaker's de-vigged odds (the ceiling)
Optionally blends MLE with the pre-match Elo in the data.

Run: python3 model_v2.py
"""
from __future__ import annotations
import csv, math, sys, datetime
from collections import defaultdict
import numpy as np

DATA = "data/Matches.csv"
DIVISIONS = ["E0", "SP1", "I1", "D1", "F1", "T1"]
REF = datetime.date(2000, 1, 1)
MAXG = 10
RHO = -0.11

# fit knobs
HALF_LIFE = 540.0
RIDGE = 8.0            # L2 on attack/defence (shrinkage; larger = toward league avg)
WINDOW_DAYS = 760      # trailing fit window
REFIT_EVERY = 25       # matches between refits
MIN_HISTORY = 8
START_YEAR = 2013
ELO_BLEND = 0.0        # 0 = pure MLE; >0 blends Elo-implied 1X2


def pdate(s):
    for f in ("%Y-%m-%d", "%d/%m/%Y", "%d/%m/%y"):
        try: return datetime.datetime.strptime(s, f).date()
        except ValueError: pass
    return None

def fnum(s):
    try: return float(s)
    except (ValueError, TypeError): return None

def load():
    rows = defaultdict(list)
    with open(DATA, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if r["Division"] not in DIVISIONS: continue
            d = pdate(r.get("MatchDate", "")); fh = fnum(r.get("FTHome")); fa = fnum(r.get("FTAway"))
            if d is None or fh is None or fa is None: continue
            rows[r["Division"]].append({
                "date": d, "home": r["HomeTeam"], "away": r["AwayTeam"],
                "fh": int(fh), "fa": int(fa),
                "eloH": fnum(r.get("HomeElo")), "eloA": fnum(r.get("AwayElo")),
                "oh": fnum(r.get("OddHome")), "od": fnum(r.get("OddDraw")), "oa": fnum(r.get("OddAway")),
            })
    for d in rows: rows[d].sort(key=lambda m: m["date"])
    return rows

# ---- Dixon-Coles grid ----
def _pois(lam):
    return [math.exp(k*math.log(lam)-lam-math.lgamma(k+1)) if lam > 0 else (1.0 if k == 0 else 0.0)
            for k in range(MAXG+1)]

def dc_probs(lh, la, rho=RHO):
    hp, ap = _pois(lh), _pois(la)
    home=draw=away=over=btts=tot=0.0
    for h in range(MAXG+1):
        for a in range(MAXG+1):
            if (h,a)==(0,0): tau=1-lh*la*rho
            elif (h,a)==(0,1): tau=1+lh*rho
            elif (h,a)==(1,0): tau=1+la*rho
            elif (h,a)==(1,1): tau=1-rho
            else: tau=1.0
            p=max(0.0,tau)*hp[h]*ap[a]; tot+=p
            if h>a: home+=p
            elif h==a: draw+=p
            else: away+=p
            if h+a>=3: over+=p
            if h>=1 and a>=1: btts+=p
    if tot>0: home,draw,away,over,btts=home/tot,draw/tot,away/tot,over/tot,btts/tot
    return home,draw,away,over,btts

# ---- MLE: time-weighted ridge Poisson GLM ----
def fit_mle(matches, as_of, half_life, ridge, max_mult=None):
    """matches: list within window. Returns (teams idx, mu, home_coef, atk[], def[])."""
    teams = sorted({m["home"] for m in matches} | {m["away"] for m in matches})
    idx = {t:i for i,t in enumerate(teams)}
    T = len(teams)
    if T < 2: return None
    # columns: [intercept, home, atk_0..atk_{T-1}, def_0..def_{T-1}]
    ncol = 2 + 2*T
    rows=[]; ys=[]; ws=[]
    for m in matches:
        w = 2.0 ** (-(as_of - m["date"]).days / half_life)
        hi, ai = idx[m["home"]], idx[m["away"]]
        # home scoring row
        r1=np.zeros(ncol); r1[0]=1; r1[1]=1; r1[2+hi]=1; r1[2+T+ai]=1
        rows.append(r1); ys.append(m["fh"]); ws.append(w)
        # away scoring row
        r2=np.zeros(ncol); r2[0]=1; r2[2+ai]=1; r2[2+T+hi]=1
        rows.append(r2); ys.append(m["fa"]); ws.append(w)
    X=np.array(rows); y=np.array(ys,dtype=float); w=np.array(ws)
    # ridge: 0 on intercept+home, `ridge` on atk/def
    R=np.ones(ncol)*ridge; R[0]=0; R[1]=0; Rm=np.diag(R)
    beta=np.zeros(ncol)
    for _ in range(12):
        eta=X@beta; eta=np.clip(eta,-5,5); mu=np.exp(eta)
        Wd=w*mu
        XtW=X.T*Wd
        A=XtW@X + Rm
        z=eta + (y-mu)/np.maximum(mu,1e-6)
        b=XtW@z
        try: beta=np.linalg.solve(A,b)
        except np.linalg.LinAlgError: beta=np.linalg.lstsq(A,b,rcond=None)[0]
    return idx, beta, T

def mle_lambdas(fit, home, away):
    idx, beta, T = fit
    mu, hc = beta[0], beta[1]
    def atk(t): return beta[2+idx[t]] if t in idx else 0.0
    def dfc(t): return beta[2+T+idx[t]] if t in idx else 0.0
    lh=math.exp(min(2.5, mu+hc+atk(home)+dfc(away)))
    la=math.exp(min(2.5, mu+atk(away)+dfc(home)))
    return max(0.05,lh), max(0.05,la)

# ---- Elo -> 1X2 (simple, for blending) ----
def elo_probs(eh, ea):
    # expected home "score" in [0,1]; split into W/D/L with a draw band.
    e = 1/(1+10**(-((eh+60)-ea)/400))   # +60 ~ home field in Elo pts
    draw = 0.28 - 0.20*abs(e-0.5)*2      # more draws when even
    draw = max(0.18, min(0.30, draw))
    home = e*(1-draw); away=(1-e)*(1-draw)
    s=home+draw+away
    return home/s, draw/s, away/s

# ---- metrics ----
def ll_multi(p, i):
    p=[min(max(x,1e-9),1) for x in p]; z=sum(p); return -math.log(p[i]/z)
def ll_bin(p,y):
    p=min(max(p,1e-9),1-1e-9); return -(y*math.log(p)+(1-y)*math.log(1-p))
def devig(oh,od,oa):
    if not(oh and od and oa): return None
    a,b,c=1/oh,1/od,1/oa; s=a+b+c; return a/s,b/s,c/s


def run(elo_blend=ELO_BLEND, start_year=START_YEAR):
    data=run.cache
    agg={m:{"ll":0.0,"n":0,"hit":0} for m in ["heur","mle","mkt","mle_ou","mle_btts"]}
    for div in DIVISIONS:
        ms=data[div]
        # heuristic incremental decayed sums
        s_sc=defaultdict(float); s_cc=defaultdict(float); s_w=defaultdict(float); cnt=defaultdict(int)
        gg=gw=gh=ga=0.0
        fit=None; since_fit=10**9; wstart=0
        for i,m in enumerate(ms):
            h,a=m["home"],m["away"]
            ready=(m["date"].year>=start_year and cnt[h]>=MIN_HISTORY and cnt[a]>=MIN_HISTORY and gw>0)
            if ready:
                # refit MLE periodically
                if since_fit>=REFIT_EVERY or fit is None:
                    while ms[wstart]["date"] < m["date"]-datetime.timedelta(days=WINDOW_DAYS): wstart+=1
                    window=[x for x in ms[wstart:i]]
                    fit=fit_mle(window, m["date"], HALF_LIFE, RIDGE) if len(window)>=40 else None
                    since_fit=0
                res = 0 if m["fh"]>m["fa"] else (1 if m["fh"]==m["fa"] else 2)
                # heuristic
                lavg=(gg/gw)/2; hadv=max(1.0,math.sqrt((gh/gw)/(ga/gw))) if ga>0 else 1.15
                def mult(num,wt):
                    v=(num+8.0*lavg)/(wt+8.0); return min(3.0,max(0.4,v/lavg)) if lavg>0 else 1.0
                lh=max(0.05,lavg*mult(s_sc[h],s_w[h])*mult(s_cc[a],s_w[a])*hadv)
                la=max(0.05,lavg*mult(s_sc[a],s_w[a])*mult(s_cc[h],s_w[h]))
                ph,pd,pa,_,_=dc_probs(lh,la)
                agg["heur"]["ll"]+=ll_multi([ph,pd,pa],res); agg["heur"]["n"]+=1
                agg["heur"]["hit"]+= (max(range(3),key=lambda k:[ph,pd,pa][k])==res)
                # MLE
                if fit:
                    lh2,la2=mle_lambdas(fit,h,a)
                    mh,md,ma,mov,mbt=dc_probs(lh2,la2)
                    if elo_blend>0 and m["eloH"] and m["eloA"]:
                        eh,ed,ea=elo_probs(m["eloH"],m["eloA"])
                        mh=(1-elo_blend)*mh+elo_blend*eh; md=(1-elo_blend)*md+elo_blend*ed; ma=(1-elo_blend)*ma+elo_blend*ea
                    agg["mle"]["ll"]+=ll_multi([mh,md,ma],res); agg["mle"]["n"]+=1
                    agg["mle"]["hit"]+=(max(range(3),key=lambda k:[mh,md,ma][k])==res)
                    agg["mle_ou"]["ll"]+=ll_bin(mov,1 if m["fh"]+m["fa"]>=3 else 0); agg["mle_ou"]["n"]+=1
                    agg["mle_btts"]["ll"]+=ll_bin(mbt,1 if m["fh"]>=1 and m["fa"]>=1 else 0); agg["mle_btts"]["n"]+=1
                # market
                mk=devig(m["oh"],m["od"],m["oa"])
                if mk:
                    agg["mkt"]["ll"]+=ll_multi(list(mk),res); agg["mkt"]["n"]+=1
                    agg["mkt"]["hit"]+=(max(range(3),key=lambda k:mk[k])==res)
                since_fit+=1
            # update heuristic sums
            wt=2.0**((m["date"]-REF).days/HALF_LIFE)
            s_sc[h]+=m["fh"]*wt; s_cc[h]+=m["fa"]*wt; s_w[h]+=wt
            s_sc[a]+=m["fa"]*wt; s_cc[a]+=m["fh"]*wt; s_w[a]+=wt
            gg+=(m["fh"]+m["fa"])*wt; gh+=m["fh"]*wt; ga+=m["fa"]*wt; gw+=wt
            cnt[h]+=1; cnt[a]+=1
    def rpt(k,label):
        x=agg[k];
        if x["n"]==0: return f"  {label}: n=0"
        acc=f" acc={x['hit']/x['n']:.3f}" if x.get('hit') else ""
        return f"  {label:<16} n={x['n']:<6} logloss={x['ll']/x['n']:.4f}{acc}"
    print(f"\n=== v2 (ridge={RIDGE} hl={HALF_LIFE} window={WINDOW_DAYS}d refit/{REFIT_EVERY} elo_blend={elo_blend}) ===")
    print(rpt("heur","1X2 heuristic")); print(rpt("mle","1X2 MLE")); print(rpt("mkt","1X2 market"))
    print(rpt("mle_ou","OU2.5 MLE")); print(rpt("mle_btts","BTTS MLE"))


if __name__=="__main__":
    print("loading…",file=sys.stderr); run.cache=load()
    print("loaded:",{d:len(run.cache[d]) for d in run.cache},file=sys.stderr)
    run(elo_blend=0.0)
