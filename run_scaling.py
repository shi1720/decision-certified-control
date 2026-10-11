from pathlib import Path
import csv,time
import numpy as np
from benchmarks import query_state
from src.certified import Archive,DecisionCache,filter_model
OUT=Path(__file__).parent/'results';records=[]
for n in [2,4,8,12]:
 for N in [128,512,2048,8192]:
  for seed in range(6000,6003):
   rng=np.random.default_rng(seed+n);m=2;A=.65*np.eye(n);B=rng.normal(size=(n,m));B*=.8/np.max(np.sum(abs(B),axis=1));center=np.c_[A,B];radius=.12/(n+m);eta=.015;theta=center+rng.uniform(-.7,.7,center.shape)*radius
   Z=[];Y=[];x=rng.uniform(-.3,.3,n)
   for t in range(N):
    u=rng.uniform(-.8,.8,m)
    while np.max(abs(center@np.r_[x,u])+radius*np.sum(abs(np.r_[x,u])))+eta>1:u*=.5
    z=np.r_[x,u];y=theta@z+rng.uniform(-eta,eta,n);Z.append(z);Y.append(y);x=y
   a=Archive(center,radius,eta,np.asarray(Z),np.asarray(Y));a.umax=.8;cache=DecisionCache(a,1e-4)
   for q in range(16):
    x,v=query_state(a,rng,q);started=time.perf_counter();uf,jf,_,_=filter_model(a,x,v);fulltime=time.perf_counter()-started
    u,r=cache.query(x,v)
    records.append(dict(n=n,m=m,N=N,seed=seed,query=q,full_time=fulltime,cache_time=r['elapsed'],gap=r['gap'],actual_gap=r['upper']-jf,archive_calls=r['archive_calls'],scalar_constraints=r['scalar_constraints'],raw_transitions=r['raw_transitions'],models=r['model_witnesses']))
   print(n,N,seed,'complete',flush=True)
with (OUT/'scaling.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
