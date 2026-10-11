"""Recompute all dynamics, state/noise bounds, and endpoint certificates."""
import csv,gzip,json,sys
from pathlib import Path
import numpy as np
from run_experiments import plants
ROOT=Path(__file__).parent
runs=list(csv.DictReader(open(ROOT/'results/runs.csv')));data=np.load(ROOT/'results/physical_trajectories.npz')
states=data['states'];inputs=data['inputs'];disturbances=data['disturbances']
ps=plants();dynmax=0.;peakmax=0.;noisemax=0.;endpoints=0;certificate_gap=-1.;containment=0
for k,r in enumerate(runs):
 p=ps[r['plant']];n=len(p.c);xs=states[k,:,:n];us=inputs[k];ws=disturbances[k,:,:n]
 err=xs[1:]-(xs[:-1]@p.A.T+float(r['theta'])*us[:,None]*p.c+ws)
 dynmax=max(dynmax,np.max(np.abs(err)));peakmax=max(peakmax,np.linalg.norm(xs,axis=1).max());noisemax=max(noisemax,np.linalg.norm(ws,axis=1).max())
 assert abs(np.linalg.norm(xs,axis=1).max()-float(r['peak']))<1e-12
 assert np.max(np.abs(us))<=p.umax+1e-10
with gzip.open(ROOT/'results/endpoints.csv.gz','rt') as f:
 oldtimes={};oldbounds={}
 for e in csv.DictReader(f):
  k=int(e['run_index']);t=int(e['time']);h=int(e['duration']);r=runs[k];p=ps[r['plant']];old=oldtimes.get(k,0)
  assert t-old==h;oldtimes[k]=t;lo,hi=float(e['lower']),float(e['upper']);theta=float(r['theta'])
  containment+=int(theta<lo-1e-9 or theta>hi+1e-9)
  assert lo>=oldbounds.get(k,(.5,1.5))[0]-1e-9 and hi<=oldbounds.get(k,(.5,1.5))[1]+1e-9
  oldbounds[k]=(lo,hi)
  # Independent endpoint residual and separate certificate value.
  q=sum((np.linalg.matrix_power(p.A,j)@p.c for j in range(h)),np.zeros(len(p.c)))
  rho=p.eta*sum(p.a**j for j in range(h));u=float(e['input'])
  z=states[k,t,:len(p.c)]-np.linalg.matrix_power(p.A,h)@states[k,old,:len(p.c)]
  for th in [lo,hi]:assert np.linalg.norm(z-th*u*q)<=rho+1e-9
  certificate_gap=max(certificate_gap,hi-lo-float(e['diameter_bound']));endpoints+=1
report=dict(runs=len(runs),physical_transitions=len(runs)*240,endpoint_observations=endpoints,max_dynamics_residual=float(dynmax),max_state_norm=float(peakmax),max_disturbance_norm=float(noisemax),max_diameter_excess=float(certificate_gap),containment_failures=containment)
assert dynmax<1e-12 and peakmax<=1+1e-10 and noisemax<=.001+1e-12 and certificate_gap<1e-8 and containment==0
(ROOT/'results/saved_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
