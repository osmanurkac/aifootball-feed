"""Simulate one real fixture with our model, 50,000 Monte-Carlo runs.

Uses the same pipeline the app would: fit the ridge Poisson GLM (Dixon-Coles MLE)
on the league's history (time-decayed to the match date), then Monte-Carlo the
match exactly like the app engine (DC score sampling + negative-binomial corners
and cards). Prints the full market book with calibrated, trustworthy percentages
and the model's single most reliable call.
"""
from __future__ import annotations
import math, datetime, random
from collections import defaultdict
import sys
import model_v2 as M

DIV = "T1"
# argv: home away YYYY-MM-DD  (defaults: Beşiktaş–Başakşehir 26/10/2026)
HOME = sys.argv[1] if len(sys.argv) > 1 else "Besiktas"
AWAY = sys.argv[2] if len(sys.argv) > 2 else "Buyuksehyr"
MATCH_DATE = (datetime.date.fromisoformat(sys.argv[3]) if len(sys.argv) > 3
              else datetime.date(2026, 10, 26))
LABELS = {"Besiktas": "Beşiktaş", "Buyuksehyr": "Başakşehir", "Trabzonspor": "Trabzonspor",
          "Fenerbahce": "Fenerbahçe", "Galatasaray": "Galatasaray"}
def lab(t): return LABELS.get(t, t)
N = 50_000
HALF_LIFE = 540.0
RNG = random.Random(20261026)

# calibration temperatures from research (p_cal = sigmoid(logit(p)/T + b))
CAL = {"ou": (1.5, -0.05), "btts": (2.0, 0.0), "cards": (1.1, 0.1),
       "1x2": (1.0, 0.0), "corners": (2.6, 0.1)}

def sig(x): return 1/(1+math.exp(-max(-30, min(30, x))))
def logit(p): p=min(max(p,1e-6),1-1e-6); return math.log(p/(1-p))
def cal(p, key):
    T,b=CAL[key]; return sig(logit(p)/T + b)

def nb_size(mean, var):
    if var<=mean or mean<=0: return 8.0
    return max(1.0,(mean*mean)/(var-mean))

def sample_nb(mean, size):
    if mean<=0: return 0
    shape=size; scale=mean/size
    lam=RNG.gammavariate(shape, scale)
    # Poisson(lam)
    L=math.exp(-lam); k=0; p=1.0
    while True:
        k+=1; p*=RNG.random()
        if p<=L: return k-1

def team_rate(matches, team, as_of, hl):
    """decayed corners/cards means for/against for a team."""
    cf=cc=kf=sw=0.0
    for m in matches:
        if m["date"]>=as_of: continue
        w=2.0**(-(as_of-m["date"]).days/hl)
        if m["home"]==team and m.get("hc") is not None:
            cf+=m["hc"]*w; kf+=m["hk"]*w; sw+=w
        elif m["away"]==team and m.get("ac") is not None:
            cf+=m["ac"]*w; kf+=m["ak"]*w; sw+=w
    return (cf/sw if sw else 5.0), (kf/sw if sw else 2.0), sw

def main():
    data=M.load()           # note: model_v2.load keeps only goals+elo+odds
    # richer load for corners/cards:
    import csv
    rows=[]
    with open("data/Matches.csv", newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if r["Division"]!=DIV: continue
            d=M.pdate(r.get("MatchDate","")); fh=M.fnum(r.get("FTHome")); fa=M.fnum(r.get("FTAway"))
            if d is None or fh is None or fa is None: continue
            rows.append({"date":d,"home":r["HomeTeam"],"away":r["AwayTeam"],
                         "fh":int(fh),"fa":int(fa),
                         "hc":M.fnum(r.get("HomeCorners")),"ac":M.fnum(r.get("AwayCorners")),
                         "hk":(M.fnum(r.get("HomeYellow")) or 0)+(M.fnum(r.get("HomeRed")) or 0),
                         "ak":(M.fnum(r.get("AwayYellow")) or 0)+(M.fnum(r.get("AwayRed")) or 0)})
    rows.sort(key=lambda m:m["date"])
    before=[m for m in rows if m["date"]<MATCH_DATE]
    actual=next((m for m in rows if m["date"]==MATCH_DATE and m["home"]==HOME and m["away"]==AWAY), None)
    # recent form (last 5 before the match) for context
    def last5(team):
        res=[]
        for m in reversed(before):
            if m["home"]==team: res.append(f"{m['fh']}-{m['fa']} (ev)")
            elif m["away"]==team: res.append(f"{m['fa']}-{m['fh']} (dep)")
            if len(res)==5: break
        return ", ".join(res)
    # convert to model_v2.Match-like for fit (needs .home/.away/.fh/.fa/.date)
    fit=M.fit_mle(before, MATCH_DATE, HALF_LIFE, M.RIDGE)
    lh,la=M.mle_lambdas(fit, HOME, AWAY)
    print(f"Fit on {len(before)} Süper Lig matches up to {before[-1]['date']} (decayed to {MATCH_DATE}).")
    print(f"Son 5 — {lab(HOME)}: {last5(HOME)}")
    print(f"Son 5 — {lab(AWAY)}: {last5(AWAY)}")
    print(f"Beklenen gol (λ):  {lab(HOME)} {lh:.2f}   {lab(AWAY)} {la:.2f}\n")

    # corners/cards means
    ch,kh,_=team_rate(rows,HOME,MATCH_DATE,HALF_LIFE)
    ca,ka,_=team_rate(rows,AWAY,MATCH_DATE,HALF_LIFE)
    corn_mean=ch+ca; card_mean=kh+ka
    corn_size=nb_size(corn_mean, corn_mean*1.3); card_size=nb_size(card_mean, card_mean*1.25)

    # DC score matrix for sampling
    MAXG=M.MAXG
    hp=M._pois(lh); ap=M._pois(la); flat=[]; cells=[]
    for h in range(MAXG+1):
        for a in range(MAXG+1):
            if (h,a)==(0,0): tau=1-lh*la*M.RHO
            elif (h,a)==(0,1): tau=1+lh*M.RHO
            elif (h,a)==(1,0): tau=1+la*M.RHO
            elif (h,a)==(1,1): tau=1-M.RHO
            else: tau=1.0
            flat.append(max(0.0,tau)*hp[h]*ap[a]); cells.append((h,a))
    s=sum(flat); flat=[x/s for x in flat]
    cum=[]; r=0.0
    for x in flat: r+=x; cum.append(r)

    # ---- 50k Monte-Carlo ----
    import bisect
    hw=dw=aw=0; sc=defaultdict(int); tg=defaultdict(int)
    corn=defaultdict(int); card=defaultdict(int); btts=0; fh_goals=defaultdict(int)
    for _ in range(N):
        u=RNG.random(); h,a=cells[bisect.bisect_left(cum,u)]
        if h>a: hw+=1
        elif h==a: dw+=1
        else: aw+=1
        sc[(h,a)]+=1; tg[h+a]+=1
        if h>=1 and a>=1: btts+=1
        corn[sample_nb(corn_mean,corn_size)]+=1
        card[sample_nb(card_mean,card_size)]+=1
    nd=float(N)
    def over(d, line): return sum(v for k,v in d.items() if k>line)/nd
    ph,pd,pa=hw/nd,dw/nd,aw/nd
    pbtts=btts/nd
    mls=max(sc,key=sc.get); mlp=sc[mls]/nd
    egt=sum(k*v for k,v in tg.items())/nd
    ecorn=sum(k*v for k,v in corn.items())/nd
    ecard=sum(k*v for k,v in card.items())/nd

    print(f"=== 50,000 simülasyon · {lab(HOME)} (ev) – {lab(AWAY)} ===\n")
    print(f"MAÇ SONUCU (1X2):   {lab(HOME)} %{100*cal(ph,'1x2'):.0f}   Beraberlik %{100*cal(pd,'1x2'):.0f}   {lab(AWAY)} %{100*cal(pa,'1x2'):.0f}")
    print(f"En olası skor:      {mls[0]}-{mls[1]}  (%{100*mlp:.1f})   | beklenen toplam gol {egt:.2f}")
    print(f"Çifte şans:         1X %{100*cal(ph+pd,'1x2'):.0f}   12 %{100*cal(ph+pa,'1x2'):.0f}   X2 %{100*cal(pd+pa,'1x2'):.0f}")
    print(f"KG (karşılıklı gol):Var %{100*cal(pbtts,'btts'):.0f}   Yok %{100*cal(1-pbtts,'btts'):.0f}")
    print(f"\nGOL ALT/ÜST (kalibre %):")
    for line in [1.5,2.5,3.5]:
        po=over(tg,int(line)); print(f"   {line} Üst %{100*cal(po,'ou'):.0f}   /   {line} Alt %{100*cal(1-po,'ou'):.0f}")
    print(f"\nKORNER (beklenen {ecorn:.1f}) — ⚠ model korner sinyali zayıf:")
    for line in [8.5,9.5,10.5]:
        po=over(corn,int(line)); print(f"   {line} Üst %{100*cal(po,'corners'):.0f}")
    print(f"\nKART (beklenen {ecard:.1f}):")
    for line in [3.5,4.5,5.5]:
        po=over(card,int(line)); print(f"   {line} Üst %{100*cal(po,'cards'):.0f}")

    # ---- best (most reliable, non-trivial) call ----
    cand=[]
    if ph>=max(pd,pa): cand.append((cal(ph,'1x2'), f"{lab(HOME)} kazanır"))
    elif pa>=max(ph,pd): cand.append((cal(pa,'1x2'), f"{lab(AWAY)} kazanır"))
    else: cand.append((cal(pd,'1x2'), "Beraberlik"))
    cand.append((cal(ph+pd,'1x2'), f"{lab(HOME)} kaybetmez (1X)"))
    cand.append((cal(pd+pa,'1x2'), f"{lab(AWAY)} kaybetmez (X2)"))
    cand.append((cal(pbtts,'btts'), "KG Var") if pbtts>=0.5 else (cal(1-pbtts,'btts'),"KG Yok"))
    for line in [1.5,2.5,3.5]:
        po=over(tg,int(line))
        cand.append((cal(po,'ou'), f"{line} Üst") if po>=0.5 else (cal(1-po,'ou'), f"{line} Alt"))
    for line in [3.5,4.5]:
        po=over(card,int(line)); cand.append((cal(po,'cards'), f"{line} karttan fazla") if po>=0.5 else (cal(1-po,'cards'), f"{line} karttan az"))
    meaningful=[c for c in cand if c[0]<=0.92]
    best=max(meaningful or cand, key=lambda c:c[0])
    print(f"\n★ MODELİN EN GÜVENİLİR TAHMİNİ:  {best[1]}  —  %{100*best[0]:.0f}")

    if actual:
        rfh, rfa = actual["fh"], actual["fa"]
        who = f"{lab(HOME)} kazandı" if rfh>rfa else ("Beraberlik" if rfh==rfa else f"{lab(AWAY)} kazandı")
        print(f"\n— GERÇEK SONUÇ (verimizde):  {lab(HOME)} {rfh}-{rfa} {lab(AWAY)}  → {who}")

if __name__=="__main__":
    main()
