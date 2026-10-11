"""Declared reuse/compaction ablations on independent histories and queries."""
from pathlib import Path
import csv
import numpy as np
from benchmarks import history,query_state
from src.certified import DecisionCache,filter_model
OUT=Path(__file__).parent/'results';records=[]
for name in ['thermal4','mechanical4','nonnormal6','dense8']:
 for seed in [5000,5001]:
  a,theta,w=history(name,seed,'uniform',512);rng=np.random.default_rng(seed+55)
  caches={'reuse':DecisionCache(a,1e-4),'no_compaction':DecisionCache(a,1e-4,compact=False)}
  for q in range(32):
   x,v=query_state(a,rng,q);uf,jf,_,_=filter_model(a,x,v)
   caches['no_reuse']=DecisionCache(a,1e-4)
   for method,cache in caches.items():
    u,r=cache.query(x,v)
    # Removing all lower witnesses leaves a valid but weak universal bound.
    lower_box=.5*np.sum((np.clip(v,-a.umax,a.umax)-v)**2)
    records.append(dict(plant=name,seed=seed,query=q,method=method,gap=r['gap'],actual_gap=r['upper']-jf,upper_only_gap=r['upper']-lower_box,archive_calls=r['archive_calls'],scalar_constraints=r['scalar_constraints'],models=r['model_witnesses'],elapsed=r['elapsed']))
   # An inner-model-only action is a relaxation, hence needs no safety claim.
   u,j,l,used=caches['reuse'].optimistic(x,v);z=np.r_[x,u]
   violation=max(a.support(i,sign*z).value+a.eta[i]-1 for i in range(a.n) for sign in (1.,-1.))
   records[-1]['inner_only_violation']=violation
  print(name,seed,'complete',flush=True)
with (OUT/'ablation.csv').open('w') as f:
 keys=list(records[0])+['inner_only_violation'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(records)
