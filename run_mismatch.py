"""Declared sensitivity extension: fixed radial model error, two certificates."""
import sys,csv
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'src'))
from safe_dwell import rotation
ROOT=Path(__file__).parent
# Written before running: a=.98, delta=.005, eta=.001, P=8 and no rotation;
# initial norm Uniform[.8,.95], gain Uniform[.6,1.4], 50 seeds, 240 steps.
# Compare nominal eta to eta+delta R, R=1, RCD rule for both. Disturbances
# point outward relative to the next disturbance-free state. No tuning.
rows=[]
for period in [8,0]:
 for seed in range(2000,2050):
  rng=np.random.default_rng(seed); theta=rng.uniform(.6,1.4)
  x0=rng.normal(size=2);x0*=rng.uniform(.8,.95)/np.linalg.norm(x0)
  for adjusted in [False,True]:
   p=rotation(.98,2*np.pi/period if period else 0,eta=.006 if adjusted else .001)
   actual=p.A/.98*.985
   prior=np.array([.5,1.5]); x=x0.copy(); t=0; failures=0; peak=np.linalg.norm(x);empty=0
   while t<240:
    u,h,b=p.choose(x,prior,'resolution_constrained',remaining=240-t)
    start=x.copy()
    for j in range(h):
     nx=actual@x+theta*p.c*u
     w=.001*nx/max(np.linalg.norm(nx),1e-15); x=nx+w
     peak=max(peak,np.linalg.norm(x))
    t+=h
    try: prior=p.posterior(start,x,u,h,prior)
    except ValueError:empty=1;break
    failures+=int(not(prior[0]-1e-9<=theta<=prior[1]+1e-9))
   rows.append(dict(period=period,seed=seed,adjusted=adjusted,peak=peak,containment_failures=failures,empty=empty,steps=t))
with open(ROOT/'results/mismatch.csv','w') as f:
 wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
for period in [8,0]:
 for adjusted in [False,True]:
  r=[x for x in rows if x['period']==period and x['adjusted']==adjusted]
  print(period,adjusted,'violated',sum(x['peak']>1+1e-9 for x in r),'containment episodes',sum(x['containment_failures']>0 for x in r),'empty',sum(x['empty'] for x in r),'max peak',max(x['peak'] for x in r))
