import numpy as np
from benchmarks import history,query_state
from src.certified import DecisionCache,filter_model
from time import perf_counter
for name in ['thermal4','mechanical4','nonnormal6','dense8']:
 a,theta,W=history(name,1000,'uniform',256)
 cache=DecisionCache(a,epsilon=1e-5);rng=np.random.default_rng(440)
 started=perf_counter()
 for t in range(8):
  x,v=query_state(a,rng,t)
  u,r=cache.query(x,v);uf,jf,lf,_=filter_model(a,x,v)
  assert r['lower']<=jf+1e-6 and jf<=r['upper']+1e-6
  assert np.linalg.norm(u-uf)<=np.sqrt(2*r['gap'])+2e-4
  print(name,t,'gap',round(r['gap'],7),'oracle',r['archive_calls'],'data',r['scalar_constraints'],'models',r['model_witnesses'],'truegap',r['upper']-jf,flush=True)
 print(name,'elapsed',perf_counter()-started,flush=True)
