"""Run the frozen protocol; no external services or model calls."""
import os,sys,json,csv,time,platform,gzip
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent/'src'))
from safe_dwell import Plant,rotation
ROOT=Path(__file__).parent
METHODS=['longest','fixed','one_step','max_resolution','gaussian_proxy','resolution_constrained']
def plants():
    out={'rotation8':rotation(.98,2*np.pi/8,eta=.001),'rotation24':rotation(.995,2*np.pi/24,eta=.001),
         'nonoscillatory':rotation(.94,0,eta=.001)}
    Q=np.eye(3); phi=2*np.pi/12
    Q[:2,:2]=[[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]]
    c=np.array([1.,0.,.3]); c/=np.linalg.norm(c)
    out['mixed3d']=Plant(.98*Q,c,.001)
    return out

def run(p,seed,method,regime):
    rng=np.random.default_rng(seed); n=len(p.c); theta=rng.uniform(.6,1.4)
    x=rng.normal(size=n); x=x/np.linalg.norm(x)*.2*rng.random()**(1/n)
    tape=rng.normal(size=(240,n)); tape/=np.linalg.norm(tape,axis=1)[:,None]
    tape*=p.eta*rng.random(240)[:,None]**(1/n)
    prior=np.array([.5,1.5]); t=0; obs=0; peak=float(np.linalg.norm(x)); energy=0.; resolved=0; first=241
    containment=0; certfails=0; trace=[]; states=[x.copy()]; inputs=[]; noises=[]; clock=time.perf_counter()
    while t<240:
        u,h,diam=p.choose(x,prior,method,remaining=240-t)
        start=x.copy(); old=prior.copy()
        if old[1]-old[0]<=.08+1e-10:
            resolved+=h
            if first==241: first=t
        direction=(theta-np.mean(prior))*u*p.q[h]
        mag=np.linalg.norm(direction)
        if mag>1e-15:
            direction/=mag
        else: direction=np.zeros(n)
        fraction=min(1.,mag/p.rho[h])
        Q=p.A/p.a
        for j in range(h):
            if regime=='random': w=tape[t+j]
            else:
                w=-p.eta*fraction*np.linalg.matrix_power(Q.T,h-1-j)@direction
            assert np.linalg.norm(w)<=p.eta+1e-12
            x=p.A@x+theta*p.c*u+w
            states.append(x.copy()); inputs.append(u); noises.append(w.copy())
            peak=max(peak,float(np.linalg.norm(x)))
            energy+=u*u
        prior=p.posterior(start,x,u,h,prior)
        if not (prior[0]-1e-9<=theta<=prior[1]+1e-9): containment+=1
        if prior[1]-prior[0]>diam+1e-8: certfails+=1
        t+=h; obs+=1
        trace.append([t,prior[0],prior[1],h,u,float(np.linalg.norm(x)),diam])
        if first==241 and prior[1]-prior[0]<=.08+1e-10: first=t
    return dict(seed=seed,method=method,regime=regime,theta=theta,diameter=float(prior[1]-prior[0]),observations=obs,
                peak=peak,energy=energy,resolved_fraction=resolved/240,first_resolved=first,containment_failures=containment,
                certificate_failures=certfails,seconds=time.perf_counter()-clock),trace,(states,inputs,noises)

def main():
    results=ROOT/'results'; results.mkdir(exist_ok=True)
    rows=[]; traces={}; allstates=[]; allinputs=[]; allnoises=[]; endpoints=[]; start=time.time()
    for name,p in plants().items():
        for regime in ['random','adversarial']:
            for seed in range(1000,1050):
                for method in METHODS:
                    row,tr,path=run(p,seed,method,regime); row['plant']=name; rows.append(row)
                    states,inputs,noises=path
                    pad=np.full((241,3),np.nan); pad[:,:len(p.c)]=states; allstates.append(pad)
                    pad=np.full((240,3),np.nan); pad[:,:len(p.c)]=noises; allnoises.append(pad); allinputs.append(inputs)
                    endpoints.extend([[len(rows)-1]+r for r in tr])
                    if seed==1000: traces[f'{name}/{regime}/{method}']=tr
            print(name,regime,'done',flush=True)
    with open(results/'runs.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(results/'physical_trajectories.npz',states=np.array(allstates),inputs=np.array(allinputs),disturbances=np.array(allnoises))
    with gzip.open(results/'endpoints.csv.gz','wt') as f:
        wr=csv.writer(f); wr.writerow(['run_index','time','lower','upper','duration','input','endpoint_norm','diameter_bound']);wr.writerows(endpoints)
    (results/'traces.json').write_text(json.dumps(traces,indent=2))
    summary=[]
    for name in plants():
        for regime in ['random','adversarial']:
            for method in METHODS:
                rs=[r for r in rows if (r['plant'],r['regime'],r['method'])==(name,regime,method)]
                s=dict(plant=name,regime=regime,method=method)
                for field in ['diameter','observations','peak','energy','resolved_fraction','first_resolved','seconds']:
                    vals=[r[field] for r in rs]; s[field+'_mean']=np.mean(vals);s[field+'_sd']=np.std(vals,ddof=1)
                summary.append(s)
    with open(results/'summary.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    (results/'environment.json').write_text(json.dumps(dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),elapsed=time.time()-start,seeds=[1000,1049],runs=len(rows)),indent=2))
    print('runs',len(rows),'seconds',time.time()-start,'max peak',max(r['peak'] for r in rows),'failures',sum(r['containment_failures']+r['certificate_failures'] for r in rows))
if __name__=='__main__':main()
