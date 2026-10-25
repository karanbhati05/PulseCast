"""Forecast daily audience, rank content, and allocate a constrained schedule."""
import argparse, csv, sqlite3
from pathlib import Path
import numpy as np
from common import save_report, table, lines

def features(day, audience):
    return [1, day/365, np.sin(2*np.pi*day/7), np.cos(2*np.pi*day/7), audience[day-1], audience[day-7]]

def forecast(y, train_end):
    # All lagged features use observations available before the prediction day.
    x=np.array([features(d,y) for d in range(7,train_end)])
    coef=np.linalg.lstsq(x,y[7:train_end],rcond=None)[0]
    return np.array([features(d,y) for d in range(train_end,len(y))]) @ coef

def recommend(events, user, k=5):
    # Implicit item-item cosine similarity, with popularity cold-start fallback.
    norms=np.linalg.norm(events,axis=0); similarity=events.T@events/np.maximum(norms[:,None]*norms[None,:],1e-9)
    np.fill_diagonal(similarity,0)
    scores=events[user]@similarity if events[user].sum() else events.sum(axis=0)
    scores=np.where(events[user]>0,-np.inf,scores)
    return np.argsort(-scores)[:k]

def schedule(values, durations, budget):
    # Exact 0/1 knapsack; at most one use of each content asset.
    dp=np.zeros((len(values)+1,budget+1))
    for i,(v,w) in enumerate(zip(values,durations),1):
        for b in range(budget+1):
            dp[i,b]=dp[i-1,b]
            if w<=b: dp[i,b]=max(dp[i,b],dp[i-1,b-w]+v)
    selected=[]; b=budget
    for i in range(len(values),0,-1):
        if dp[i,b]>dp[i-1,b]: selected.append(i-1); b-=durations[i-1]
    return selected

def run(out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); rng=np.random.default_rng(42)
    days=np.arange(420); y=1000+1.8*days+220*np.sin(2*np.pi*days/7)+rng.normal(0,65,len(days))
    with (out/'audience.csv').open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['day','audience']); w.writerows(zip(days,y.round(2)))
    pred=forecast(y,350); baseline=y[343:413]; actual=y[350:]
    # Three rolling-origin validation windows before the untouched test period.
    validation=[]
    for end in [266,294,322]:
        vp=forecast(y[:end+28],end); validation.append(float(np.mean(abs(vp-y[end:end+28]))))
    # Chronological per-user holdout: train on first events, evaluate later events.
    preferences=rng.integers(0,4,180); genres=np.arange(32)%4
    train=np.zeros((180,32)); held=[]; rows=[]
    for u,g in enumerate(preferences):
        p=np.where(genres==g,8.,1.); p/=p.sum(); items=rng.choice(32,size=10,replace=False,p=p)
        train[u,items[:8]]=1; held.append(set(items[8:])); rows.extend((u,int(it),t) for t,it in enumerate(items))
    db=sqlite3.connect(out/'analytics.sqlite'); db.execute('DROP TABLE IF EXISTS events'); db.execute('CREATE TABLE events(user_id INT, item_id INT, event_order INT)'); db.executemany('INSERT INTO events VALUES (?,?,?)',rows); db.commit()
    popular=np.argsort(-train.sum(axis=0)); recalls=[]; pop_recalls=[]
    for u in range(len(train)):
        recalls.append(len(set(recommend(train,u))&held[u])/2)
        pop=[int(i) for i in popular if not train[u,i]][:5]; pop_recalls.append(len(set(pop)&held[u])/2)
    counts=db.execute('SELECT item_id, COUNT(*) views FROM events WHERE event_order < 8 GROUP BY item_id ORDER BY views DESC').fetchall(); db.close()
    durations=rng.integers(10,31,32); values=train.sum(axis=0); chosen=schedule(values,durations,120)
    metrics={'Forecast MAE':round(float(np.mean(abs(pred-actual))),2),'Seasonal baseline MAE':round(float(np.mean(abs(baseline-actual))),2),'Rolling validation MAE':round(float(np.mean(validation)),2),'Recall@5':round(float(np.mean(recalls)),3),'Popularity Recall@5':round(float(np.mean(pop_recalls)),3),'Scheduled minutes':int(sum(durations[chosen]))}
    save_report(out,'Audience Intelligence',metrics,[('Held-out daily forecast',lines({'actual':actual,'forecast':pred,'seasonal baseline':baseline})),('Recommended content for user 0',table(['item','genre'],[(int(i),int(genres[i])) for i in recommend(train,0)])),('120-minute schedule',table(['item','minutes','training views'],[(i,int(durations[i]),int(values[i])) for i in chosen])),('SQL content demand',table(['item','training views'],counts[:10])),('Evaluation design','<p>Final 70 days: one-day-ahead forecasts using observed prior-day lags, not a 70-day recursive forecast. Recommendations hold out the last two simulated events per user. No test interactions enter similarity scores. Schedule value is a popularity proxy, not causal audience lift.</p>')])
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default='results'); print(run(p.parse_args().out))
