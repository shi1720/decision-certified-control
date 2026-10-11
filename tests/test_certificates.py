"""Independent geometry and optimization checks; no benchmark seed is reused."""
import sys,unittest,itertools
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import linprog
from src.certified import Archive,DecisionCache,filter_model,project_qp
from benchmarks import history,query_state

class CertificateChecks(unittest.TestCase):
 def test_support_against_vertex_enumeration(self):
  rng=np.random.default_rng(551)
  for _ in range(80):
   Z=rng.normal(size=(12,2));theta=np.array([[.2,-.3]])
   Y=Z@theta.T+rng.uniform(-.1,.1,(12,1))
   a=Archive(np.zeros((1,2)),1.,.1,Z,Y)
   vertices=[]
   for pair in itertools.combinations(range(len(a.C)),2):
    C=a.C[list(pair)]
    if abs(np.linalg.det(C))<1e-10:continue
    t=np.linalg.solve(C,a.b[0,list(pair)])
    if np.max(a.C@t-a.b[0])<=1e-9:vertices.append(t)
   v=rng.normal(size=2);s=a.support(0,v)
   self.assertAlmostEqual(s.value,max(v@t for t in vertices),places=8)
   self.assertLessEqual(len(s.ids),2)

 def test_one_input_projection_against_interval(self):
  rng=np.random.default_rng(552)
  for _ in range(100):
   H=rng.normal(size=(20,1));b=rng.uniform(.01,1.,20);v=rng.normal(size=1)
   lo,hi=-.8,.8
   for h,bi in zip(H[:,0],b):
    if h>0:hi=min(hi,bi/h)
    else:lo=max(lo,bi/h)
   u,j,l,used=project_qp(v,H,b,.8)
   self.assertAlmostEqual(u[0],np.clip(v[0],lo,hi),places=7)
   self.assertLessEqual(l,j+1e-9);self.assertLessEqual(len(used),1)

 def test_decision_sandwich_and_compaction(self):
  for name in ['thermal4','nonnormal6','dense8']:
   a,theta,w=history(name,777,'uniform',128);cache=DecisionCache(a,1e-4)
   rng=np.random.default_rng(553)
   for t in range(15):
    x,v=query_state(a,rng,t);u,r=cache.query(x,v)
    uf,jf,_,_=filter_model(a,x,v)
    self.assertLessEqual(r['lower'],jf+1e-7);self.assertLessEqual(jf,r['upper']+1e-7)
    self.assertLessEqual(np.linalg.norm(u-uf),np.sqrt(2*r['gap'])+2e-5)
    self.assertLessEqual(r['scalar_constraints'],2*a.n*a.d)
    self.assertLessEqual(r['model_witnesses'],a.m)
    u2,r2=cache.query(x,v)
    self.assertLess(np.linalg.norm(u-u2),2e-5)
    self.assertEqual(r2['archive_calls'],0)

 def test_witness_screening(self):
  a,theta,w=history('thermal4',778,'uniform',32);cache=DecisionCache(a)
  cache.models[0]=[theta[0].copy(),theta[0]+10]
  removed=cache.screen_witnesses(a.Z,a.Y)
  self.assertEqual(removed,1);self.assertEqual(len(cache.models[0]),1)

 def test_irredundant_polygon(self):
  # Repeated independent one-step resets with y=0 and true A=B=0.
  for N in [4,8,16,32]:
   angles=np.arange(N)*np.pi/N;Z=np.c_[np.cos(angles),np.sin(angles)]
   a=Archive(np.zeros((1,2)),3.,1.,Z,np.zeros((N,1)))
   for j in range(N):
    ids=np.delete(np.arange(len(a.C)),[a.prior_count+j,a.prior_count+N+j])
    original=a.support(0,Z[j]).value;reduced=a.support(0,Z[j],ids).value
    self.assertGreater(reduced-original,1e-4)

 def test_prior_backup_for_all_box_models(self):
  for name in ['thermal4','mechanical4','nonnormal6','dense8']:
   a,theta,w=history(name,779,'boundary',128)
   F=a.center[:,:a.n]+a.center[:,a.n:]@a.backup
   bound=np.sum(abs(F),axis=1)+a.radius[:,0]*(a.n+np.sum(abs(a.backup)))+a.eta
   self.assertLess(bound.max(),1.)
   self.assertLessEqual(np.sum(abs(a.backup),axis=1).max(),a.umax)

if __name__=='__main__':unittest.main()
