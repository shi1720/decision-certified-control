"""Append-only reuse and stale-witness falsification; all query gaps audited."""
from pathlib import Path
import csv,json
import numpy as np
from benchmarks import history,query_state
from src.certified import Archive,DecisionCache,append_history,filter_model
from src.portable import export_cache,evaluate,payload_bytes
OUT=Path(__file__).parent/'results';records=[];portable=[]
for name in ['thermal4','mechanical4','nonnormal6','dense8']:
 for seed in range(4000,4003):
  full,theta,W=history(name,seed,'uniform',1024);cache=None;rng=np.random.default_rng(seed+77)
  for N in [64,128,256,512,1024]:
   a=Archive(full.center,full.radius,full.eta,full.Z[:N],full.Y[:N]);a.umax=full.umax
   removed=0
   if cache is None:cache=DecisionCache(a,1e-4)
   else:removed=append_history(cache,a)
   for t in range(12):
    x,v=query_state(a,rng,t);u,r=cache.query(x,v);uf,jf,_,_=filter_model(a,x,v)
    payload=export_cache(cache);up,rp=evaluate(payload,x,v)
    records.append(dict(plant=name,seed=seed,N=N,query=t,invalidated=removed if t==0 else 0,lower=r['lower'],full=jf,upper=r['upper'],gap=r['gap'],action_error=np.linalg.norm(u-uf),portable_error=np.linalg.norm(up-u),portable_gap=rp['gap'],payload_json_bytes=payload_bytes(payload),raw_float64_bytes=8*N*(a.d+a.n),scalar_constraints=r['scalar_constraints'],models=r['model_witnesses'],archive_calls=r['archive_calls']))
    if t==11 and N==1024:portable.append(dict(plant=name,seed=seed,x=x.tolist(),v=v.tolist(),certificate=payload))
  print(name,seed,'complete',flush=True)
# An explicit stale-witness counterexample, not a random failure search.
old=Archive(np.array([[0.,1.]]),np.array([[0.,.5]]),.1,np.array([[0.,.1]]),np.array([[.1]]));old.umax=.8
new=Archive(old.center,old.radius,old.eta,np.array([[0.,.1],[0.,1.]]),np.array([[.1],[.5]]));new.umax=.8
cache=DecisionCache(old);cache.models[0]=[np.array([0.,1.5])]
x=np.array([0.]);v=np.array([1.]);u,j,l,s=filter_model(old,x,v);ov,jv,lv,used=cache.optimistic(x,v)
un,jnew,_,_=filter_model(new,x,v)
removed=append_history(cache,new)
example=dict(stale_upper=j,stale_lower=lv,claimed_stale_gap=j-lv,actual_new_optimum=jnew,actual_regret=j-jnew,invalidated=removed)
with (OUT/'streaming.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
(OUT/'portable_examples.json').write_text(json.dumps(portable,separators=(',',':')))
(OUT/'stale_witness.json').write_text(json.dumps(example,indent=2))
print(example,flush=True)
