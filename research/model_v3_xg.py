"""Does training on shots / shots-on-target (xG) beat goals-only? Out-of-sample.

Same GLM, but the Poisson target is a blend of actual goals and an xG proxy
(shots-on-target x league conversion). Goals are noisy; xG is steadier, so xG
ratings often predict future goals better. We measure 1X2 + O/U2.5 log-loss for
several blends and compare to the bookmaker.
"""
from __future__ import annotations
import csv, math, datetime, sys
from collections import defaultdict
import numpy as np
import model_v2 as M

DIVS = ["E0","SP1","I1","D1","F1","T1"]
HL = 540.0; RIDGE = 8.0; WINDOW = 760; REFIT = 25; MINH = 8; START = 2013

def load():
    rows=defaultdict(list)
    with open("data/Matches.csv",newline="",encoding="utf-8",errors="replace") as f:
        for r in csv.DictReader(f):
            if r["Division"] not in DIVS: continue
            d=M.pdate(r.get("MatchDate","")); fh=M.fnum(r.get("FTHome")); fa=M.fnum(r.get("FTAway"))
            if d is None or fh is None or fa is None: continue
            rows[r["Division"]].append({"date":d,"home":r["HomeTeam"],"away":r["AwayTeam"],
                "fh":int(fh),"fa":int(fa),
                "sh":M.fnum(r.get("HomeTarget")),"sa":M.fnum(r.get("AwayTarget")),
                "oh":M.fnum(r.get("OddHome")),"od":M.fnum(r.get("OddDraw")),"oa":M.fnum(r.get("OddAway"))})
    for d in rows: rows[d].sort(key=lambda m:m["date"])
    return rows

def fit(matches, as_of, blend):
    """blend: weight on actual goals (1=goals only, 0=xG only). Uses per-window
    SoT->goals conversion for the xG proxy."""
    teams=sorted({m["home"] for m in matches}|{m["away"] for m in matches})
    idx={t:i for i,t in enumerate(teams)}; T=len(teams)
    if T<2: return None
    # conversion
    sg=sum(m["fh"]+m["fa"] for m in matches)
    ss=sum((m["sh"] or 0)+(m["sa"] or 0) for m in matches)
    conv=(sg/ss) if ss>0 else 0.1
    ncol=2+2*T; rows=[]; ys=[]; ws=[]
    for m in matches:
        w=2.0**(-(as_of-m["date"]).days/HL)
        hi,ai=idx[m["home"]],idx[m["away"]]
        def tgt(goals, sot):
            xg=(sot if sot is not None else goals/conv)*conv
            return blend*goals+(1-blend)*xg
        r1=[0.0]*ncol; r1[0]=1;r1[1]=1;r1[2+hi]=1;r1[2+T+ai]=1
        rows.append(r1); ys.append(tgt(m["fh"],m["sh"])); ws.append(w)
        r2=[0.0]*ncol; r2[0]=1;r2[2+ai]=1;r2[2+T+hi]=1
        rows.append(r2); ys.append(tgt(m["fa"],m["sa"])); ws.append(w)
    X=np.asarray(rows);y=np.asarray(ys,float);w=np.asarray(ws)
    R=np.ones(ncol)*RIDGE;R[0]=0;R[1]=0;Rm=np.diag(R);beta=np.zeros(ncol)
    for _ in range(12):
        eta=np.clip(X@beta,-5,5);mu=np.exp(eta);wd=w*mu
        A=(X.T*wd)@X+Rm; z=eta+(y-mu)/np.maximum(mu,1e-6); b=(X.T*wd)@z
        try: beta=np.linalg.solve(A,b)
        except np.linalg.LinAlgError: beta=np.linalg.lstsq(A,b,rcond=None)[0]
    return idx,beta,T

def lambdas(fitres,h,a):
    idx,beta,T=fitres; mu,hc=beta[0],beta[1]
    at=lambda t: beta[2+idx[t]] if t in idx else 0.0
    df=lambda t: beta[2+T+idx[t]] if t in idx else 0.0
    return max(0.05,math.exp(min(2.5,mu+hc+at(h)+df(a)))), max(0.05,math.exp(min(2.5,mu+at(a)+df(h))))

def llmulti(p,i): p=[min(max(x,1e-9),1) for x in p];z=sum(p);return -math.log(p[i]/z)
def llbin(p,y): p=min(max(p,1e-9),1-1e-9);return -(y*math.log(p)+(1-y)*math.log(1-p))

def run(blend):
    data=run.cache
    agg={"1x2":[0.0,0,0],"ou":[0.0,0],"mkt":[0.0,0,0]}
    for div in DIVS:
        ms=data[div]; cnt=defaultdict(int); fitres=None; since=10**9; ws=0
        for i,m in enumerate(ms):
            h,a=m["home"],m["away"]
            if m["date"].year>=START and cnt[h]>=MINH and cnt[a]>=MINH:
                if since>=REFIT or fitres is None:
                    while ms[ws]["date"]<m["date"]-datetime.timedelta(days=WINDOW): ws+=1
                    win=ms[ws:i]; fitres=fit(win,m["date"],blend) if len(win)>=40 else None; since=0
                if fitres:
                    lh,la=lambdas(fitres,h,a); ph,pd,pa,pov,_=M.dc_probs(lh,la)
                    res=0 if m["fh"]>m["fa"] else (1 if m["fh"]==m["fa"] else 2)
                    agg["1x2"][0]+=llmulti([ph,pd,pa],res);agg["1x2"][1]+=1
                    agg["1x2"][2]+=(max(range(3),key=lambda k:[ph,pd,pa][k])==res)
                    agg["ou"][0]+=llbin(pov,1 if m["fh"]+m["fa"]>=3 else 0);agg["ou"][1]+=1
                    mk=M.devig(m["oh"],m["od"],m["oa"])
                    if mk: agg["mkt"][0]+=llmulti(list(mk),res);agg["mkt"][1]+=1;agg["mkt"][2]+=(max(range(3),key=lambda k:mk[k])==res)
                since+=1
            cnt[h]+=1;cnt[a]+=1
    n=agg["1x2"][1]
    print(f"  blend(goals={blend:.2f}): 1X2 ll={agg['1x2'][0]/n:.4f} acc={agg['1x2'][2]/n:.3f}  OU ll={agg['ou'][0]/agg['ou'][1]:.4f}  (n={n})")
    if blend==1.0:
        print(f"  bookmaker        : 1X2 ll={agg['mkt'][0]/agg['mkt'][1]:.4f} acc={agg['mkt'][2]/agg['mkt'][1]:.3f}")

if __name__=="__main__":
    print("loading…",file=sys.stderr); run.cache=load()
    print("### xG blend test (1.0=goals only, 0.0=pure shots-on-target) ###")
    for b in [1.0,0.7,0.5,0.3,0.0]:
        run(b)
