"""Backtest every Beşiktaş 2025/26 Süper Lig match, out-of-sample, per market.

For each match: fit the model on all Süper Lig matches BEFORE it (time-decayed),
predict each market with calibration, compare to the real result. Reports per-
market hit rates + the model's single "best call" hit rate and calibration, and a
per-match table so we can see where the model is strong.
"""
from __future__ import annotations
import csv, math, datetime, sys
from collections import defaultdict
import model_v2 as M

TEAM = "Besiktas"; DIV = "T1"; HL = 540.0
SEASON_START = datetime.date(2025, 7, 15)
SEASON_END   = datetime.date(2026, 6, 30)
CAL = {"ou":(1.5,-0.05),"btts":(2.0,0.0),"cards":(1.1,0.1),"1x2":(1.0,0.0),"corners":(2.6,0.1)}
def sig(x): return 1/(1+math.exp(-max(-30,min(30,x))))
def logit(p): p=min(max(p,1e-6),1-1e-6); return math.log(p/(1-p))
def cal(p,k): T,b=CAL[k]; return sig(logit(p)/T+b)

def load():
    rows=[]
    with open("data/Matches.csv",newline="",encoding="utf-8",errors="replace") as f:
        for r in csv.DictReader(f):
            if r["Division"]!=DIV: continue
            d=M.pdate(r.get("MatchDate","")); fh=M.fnum(r.get("FTHome")); fa=M.fnum(r.get("FTAway"))
            if d is None or fh is None or fa is None: continue
            rows.append({"date":d,"home":r["HomeTeam"],"away":r["AwayTeam"],"fh":int(fh),"fa":int(fa),
                "hc":M.fnum(r.get("HomeCorners")),"ac":M.fnum(r.get("AwayCorners")),
                "hk":(M.fnum(r.get("HomeYellow")) or 0)+(M.fnum(r.get("HomeRed")) or 0),
                "ak":(M.fnum(r.get("AwayYellow")) or 0)+(M.fnum(r.get("AwayRed")) or 0)})
    rows.sort(key=lambda m:m["date"]); return rows

def team_means(before, team, as_of):
    cf=kf=sw=0.0
    for m in before:
        w=2.0**(-(as_of-m["date"]).days/HL)
        if m["home"]==team and m["hc"] is not None: cf+=m["hc"]*w; kf+=m["hk"]*w; sw+=w
        elif m["away"]==team and m["ac"] is not None: cf+=m["ac"]*w; kf+=m["ak"]*w; sw+=w
    return (cf/sw if sw else 5.0),(kf/sw if sw else 2.0)

def pois_over(mean, line):
    # P(X > line) for integer line.5 => P(X >= line+1)
    c=0.0
    for k in range(0,int(line)+1): c+=math.exp(k*math.log(mean)-mean-math.lgamma(k+1)) if mean>0 else (1 if k==0 else 0)
    return 1-c

def main():
    rows=load()
    season=[m for m in rows if (m["home"]==TEAM or m["away"]==TEAM) and SEASON_START<=m["date"]<=SEASON_END]
    print(f"Beşiktaş 2025/26: {len(season)} Süper Lig maçı\n")
    hits=defaultdict(int); tot=defaultdict(int)
    best_hit=0; best_n=0; best_conf=0.0
    print(f"{'Tarih':<11}{'Maç':<26}{'Skor':<6}{'Model 1X2':<11}{'En güv. tahmin':<22}{'✓?'}")
    for m in season:
        before=[x for x in rows if x["date"]<m["date"]]
        fit=M.fit_mle(before, m["date"], HL, M.RIDGE)
        if not fit: continue
        home,away=m["home"],m["away"]
        lh,la=M.mle_lambdas(fit,home,away)
        ph,pd,pa,pov,pbtts=M.dc_probs(lh,la)
        ch,kh=team_means(before,home,m["date"]); ca,ka=team_means(before,away,m["date"])
        corn_mean=ch+ca; card_mean=kh+ka
        pc=pois_over(corn_mean,9.5); pk=pois_over(card_mean,3.5)
        # actuals
        res=0 if m["fh"]>m["fa"] else (1 if m["fh"]==m["fa"] else 2)
        aov=1 if m["fh"]+m["fa"]>=3 else 0
        abt=1 if m["fh"]>=1 and m["fa"]>=1 else 0
        has_ck=m["hc"] is not None
        acorn=1 if has_ck and (m["hc"]+m["ac"])>=10 else 0
        acard=1 if has_ck and (m["hk"]+m["ak"])>=4 else 0
        # picks & hits
        pick1x2=max(range(3),key=lambda k:[ph,pd,pa][k]); tot["1X2"]+=1; hits["1X2"]+=(pick1x2==res)
        tot["O/U2.5"]+=1; hits["O/U2.5"]+=((pov>=0.5)==(aov==1))
        tot["KG"]+=1; hits["KG"]+=((pbtts>=0.5)==(abt==1))
        if has_ck:
            tot["Korner9.5"]+=1; hits["Korner9.5"]+=((pc>=0.5)==(acorn==1))
            tot["Kart3.5"]+=1; hits["Kart3.5"]+=((pk>=0.5)==(acard==1))
        # double chance: Beşiktaş not-lose
        bes_home = home==TEAM
        p_bes_nl = (ph+pd) if bes_home else (pd+pa)
        bes_not_lose = (res==1) or (res==0 and bes_home) or (res==2 and not bes_home)
        tot["BJK kaybetmez"]+=1; hits["BJK kaybetmez"]+=((p_bes_nl>=0.5)==bes_not_lose)
        # best calibrated non-trivial call
        cands=[]
        lbl1={0:home,1:"Beraberlik",2:away}[pick1x2]
        cands.append((cal(max(ph,pd,pa),'1x2'), f"{lbl1[:10]} kazanır", pick1x2==res))
        cands.append((cal(p_bes_nl,'1x2'), "BJK kaybetmez", bes_not_lose))
        cands.append((cal(pov,'ou'),"2.5 Üst",aov==1) if pov>=0.5 else (cal(1-pov,'ou'),"2.5 Alt",aov==0))
        cands.append((cal(pbtts,'btts'),"KG Var",abt==1) if pbtts>=0.5 else (cal(1-pbtts,'btts'),"KG Yok",abt==0))
        cands.append((cal(pois_over(corn_mean,8.5),'corners') if False else 0,"",False))  # corners excluded (weak)
        meaningful=[c for c in cands if 0<c[0]<=0.92]
        bc=max(meaningful,key=lambda c:c[0]) if meaningful else max(cands,key=lambda c:c[0])
        best_n+=1; best_hit+=1 if bc[2] else 0; best_conf+=bc[0]
        label={0:home[:3],1:"X",2:away[:3]}[pick1x2]
        mark="✓" if bc[2] else "✗"
        matchname=f"{home[:11]}-{away[:11]}"
        print(f"{str(m['date']):<11}{matchname:<26}{m['fh']}-{m['fa']:<4}{label+' %'+str(round(100*max(ph,pd,pa))):<11}{(bc[1]+' %'+str(round(100*bc[0]))):<22}{mark}")

    print("\n=== MARKET BAŞARI ORANLARI (2025/26 Beşiktaş) ===")
    for k in ["1X2","BJK kaybetmez","O/U2.5","KG","Korner9.5","Kart3.5"]:
        if tot[k]: print(f"  {k:<16} {hits[k]}/{tot[k]}  = %{100*hits[k]/tot[k]:.0f}")
    print(f"\n  ★ 'En güvenilir tahmin' isabeti: {best_hit}/{best_n} = %{100*best_hit/best_n:.0f}"
          f"   (ortalama güven %{100*best_conf/best_n:.0f} → kalibrasyon kontrolü)")

if __name__=="__main__":
    main()
