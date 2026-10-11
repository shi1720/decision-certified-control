import csv,json,time
from pathlib import Path
import numpy as np
from benchmarks import history,subset_constraints,leverage_ids
from src.certified import DecisionCache,filter_model
ROOT=Path(__file__).parent;OUT=ROOT/'results';METHODS=['full','DCC_1e-5','DCC_1e-3','random','logdet','rolling','prior']
records=[];trajectories={};failures=[]
for name in ['thermal4','mechanical4','nonnormal6','dense8']:
 for regime in ['uniform','boundary']:
  for seed in range(3000,3003):
   a,theta,Wtrain=history(name,seed,regime,1024);k=min(len(a.Z),2*a.n*a.d);rng=np.random.default_rng(seed+991)
   subsets={'random':subset_constraints(a,rng.choice(len(a.Z),k,replace=False)),
     'logdet':subset_constraints(a,leverage_ids(a.Z,k)),
     'rolling':subset_constraints(a,np.arange(len(a.Z)-k,len(a.Z))),
     'prior':[np.arange(a.prior_count) for _ in range(a.n)]}
   phases=rng.choice([-1.,1.],size=(6,a.m));refs=np.repeat(phases,16,axis=0)*1.6
   noise=rng.uniform(-a.eta,a.eta,size=(96,a.n)) if regime=='uniform' else rng.choice([-1.,1.],size=(96,a.n))*a.eta
   xinitial=rng.uniform(-.3,.3,a.n)
   for method in METHODS:
    x=xinitial.copy();states=[x.copy()];inputs=[];certs=[];status='ok';started=time.perf_counter()
    cache=DecisionCache(a,float(method.split('_')[1])) if method.startswith('DCC') else None
    for t in range(96):
     try:
      if cache:u,r=cache.query(x,refs[t]);gap=r['gap'];calls=r['archive_calls']
      else:u,j,l,s=filter_model(a,x,refs[t],subsets.get(method));gap=np.nan;calls=0
      xn=theta@np.r_[x,u]+noise[t]
      states.append(xn.copy());inputs.append(u.copy());certs.append([gap,calls]);x=xn
     except Exception as exc:
      status=str(exc);failures.append(dict(plant=name,regime=regime,seed=seed,method=method,time=t,error=str(exc)));break
    key=f'{name}_{regime}_{seed}_{method}';states=np.asarray(states);inputs=np.asarray(inputs);certs=np.asarray(certs)
    trajectories[key+'_states']=states;trajectories[key+'_inputs']=inputs;trajectories[key+'_noise']=noise[:len(inputs)];trajectories[key+'_theta']=theta;trajectories[key+'_references']=refs[:len(inputs)]
    records.append(dict(plant=name,regime=regime,seed=seed,method=method,steps=len(inputs),max_state=np.max(abs(states)),input_peak=np.max(abs(inputs),initial=0.),intervention=np.sum((inputs-refs[:len(inputs)])**2)/2,archive_calls=np.sum(certs[:,1]) if len(certs) else 0,elapsed=time.perf_counter()-started,status=status))
   print(name,regime,seed,'complete',len(failures),'failures',flush=True)
with (OUT/'closed_loop.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=records[0].keys());writer.writeheader();writer.writerows(records)
np.savez_compressed(OUT/'closed_loop_trajectories.npz',**trajectories)
(OUT/'closed_loop_failures.json').write_text(json.dumps(failures,indent=2))
