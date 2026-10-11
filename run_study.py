"""Frozen main query benchmark. Every method and failed solve is retained."""
import csv,json,platform,sys,time,traceback
from pathlib import Path
import numpy as np
from benchmarks import history,query_state,leverage_ids,subset_constraints
from src.certified import DecisionCache,filter_model
ROOT=Path(__file__).parent
METHODS=['full','DCC_1e-5','DCC_1e-3','random','logdet','rolling','prior']

def main(seeds=12,queries=96,N=1024,names=None,outname='results'):
 out=ROOT/outname;out.mkdir(exist_ok=True);(out/'archives').mkdir(exist_ok=True)
 records=[];raw_queries=[];raw_actions=[];failures=[];started=time.perf_counter()
 fields=['archive','plant','regime','seed','query','method','objective','full_objective','objective_gap','action_error','certificate_gap','certificate_lower','certificate_upper','scalar_constraints','raw_transitions','model_witnesses','archive_calls','refinement_rounds','elapsed','archive_seconds','max_true_next','max_robust_next','input_peak','status']
 with (out/'queries.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
  for name in (names or ['thermal4','mechanical4','nonnormal6','dense8']):
   for regime in ['uniform','boundary']:
    for seed in range(2000,2000+seeds):
     a,theta,W=history(name,seed,regime,N);key=f'{name}_{regime}_{seed}'
     np.savez_compressed(out/'archives'/f'{key}.npz',center=a.center,radius=a.radius,eta=a.eta,umax=a.umax,backup=a.backup,Z=a.Z,Y=a.Y,W=W,theta=theta)
     budget=min(N,2*a.n*a.d);rng=np.random.default_rng(seed+90000)
     fixed={'random':subset_constraints(a,rng.choice(N,budget,replace=False)),
            'logdet':subset_constraints(a,leverage_ids(a.Z,budget)),
            'rolling':subset_constraints(a,np.arange(N-budget,N)),
            'prior':[np.arange(a.prior_count) for _ in range(a.n)]}
     caches={s:DecisionCache(a,float(s.split('_')[1])) for s in METHODS if s.startswith('DCC')}
     q_rng=np.random.default_rng(seed+70000)
     for q in range(queries):
      x,v=query_state(a,q_rng,q);before=time.perf_counter()
      uf,jf,lf,sf=filter_model(a,x,v);full_elapsed=time.perf_counter()-before
      raw_queries.append(np.r_[x,np.full(8-a.n,np.nan),v,np.full(3-a.m,np.nan)])
      actions=[]
      for method in METHODS:
       r={k:0 for k in ['archive_calls','refinement_rounds','archive_seconds','model_witnesses']}
       try:
        if method=='full':
         u,j,supports=uf,jf,sf;r.update(elapsed=full_elapsed,scalar_constraints=2*N*a.n,raw_transitions=N)
        elif method.startswith('DCC'):
         u,r=caches[method].query(x,v);j=r['upper'];supports=None
        else:
         before=time.perf_counter();u,j,low,supports=filter_model(a,x,v,fixed[method]);r.update(elapsed=time.perf_counter()-before,scalar_constraints=0 if method=='prior' else 2*budget*a.n,raw_transitions=0 if method=='prior' else budget)
        z=np.r_[x,u]
        # Selected-set duals give an upper bound also valid for the full
        # model. The full objective is independently recomputed above.
        robust=r['model_upper'] if supports is None else max(s.value+a.eta[i] for i,sign,s in supports)
        row=dict(archive=key,plant=name,regime=regime,seed=seed,query=q,method=method,
                 objective=j,full_objective=jf,objective_gap=j-jf,action_error=np.linalg.norm(u-uf),
                 certificate_gap=r.get('gap',np.nan),certificate_lower=r.get('lower',np.nan),certificate_upper=r.get('upper',np.nan),
                 max_true_next=np.max(abs(theta@z))+np.max(a.eta),max_robust_next=robust,input_peak=np.max(abs(u)),status='ok',
                 **{k:r[k] for k in ['scalar_constraints','raw_transitions','model_witnesses','archive_calls','refinement_rounds','elapsed','archive_seconds']})
        actions.append(np.r_[u,np.full(3-a.m,np.nan)])
       except Exception as exc:
        row={k:np.nan for k in fields};row.update(archive=key,plant=name,regime=regime,seed=seed,query=q,method=method,status=str(exc));actions.append(np.full(3,np.nan))
        failures.append(dict(archive=key,query=q,method=method,error=str(exc),traceback=traceback.format_exc()))
       writer.writerow(row);records.append(row)
      raw_actions.append(actions)
     f.flush();print(key,'complete; total failures',len(failures),flush=True)
 np.savez_compressed(out/'query_arrays.npz',queries=np.asarray(raw_queries),actions=np.asarray(raw_actions))
 (out/'failures.json').write_text(json.dumps(failures,indent=2))
 (out/'environment.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),seeds=seeds,queries=queries,N=N,methods=METHODS,elapsed=time.perf_counter()-started),indent=2))
 print('DONE',len(records),'rows',len(failures),'failures',flush=True)
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--plant');parser.add_argument('--out',default='results');args=parser.parse_args()
 main(names=[args.plant] if args.plant else None,outname=args.out)
