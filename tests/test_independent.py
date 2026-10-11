"""Independent numerical and algebraic checks. Run with unittest."""
import sys,unittest
from pathlib import Path
import numpy as np
from scipy.optimize import brentq,linprog
sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
from safe_dwell import rotation

class ExactChecks(unittest.TestCase):
 def test_safe_intervals_against_scalar_convex_root(self):
  rng=np.random.default_rng(721)
  worst=0.
  for _ in range(200):
   p=rotation(rng.uniform(.8,.995),rng.uniform(0,np.pi),eta=.001)
   x=rng.normal(size=2);x*=rng.uniform(0,.999)/np.linalg.norm(x)
   prior=sorted(rng.uniform(.1,2,size=2)); h=int(rng.integers(1,25))
   def violation(u):
    vals=[]
    for j in range(1,h+1):
     P=np.linalg.matrix_power(p.A,j)
     q=sum((np.linalg.matrix_power(p.A,k)@p.c for k in range(j)),np.zeros(2))
     rho=.001*sum(p.a**k for k in range(j))
     vals += [np.linalg.norm(P@x+th*q*u)+rho-1 for th in prior]
    return max(vals)
   lo=-p.umax if violation(-p.umax)<=0 else brentq(violation,-p.umax,0,xtol=1e-13)
   hi=p.umax if violation(p.umax)<=0 else brentq(violation,0,p.umax,xtol=1e-13)
   actual=p.safe_intervals(x,prior)[h-1]
   worst=max(worst,float(np.max(np.abs(actual-[lo,hi]))))
   np.testing.assert_allclose(actual,[lo,hi],atol=2e-10)
  print('200 independent safe-interval checks, max error',worst)
 def test_posterior_against_independent_inversion(self):
  rng=np.random.default_rng(722)
  for _ in range(400):
   p=rotation(rng.uniform(.8,.995),rng.uniform(0,np.pi),eta=.001)
   x=rng.normal(size=2)*.1; th=rng.uniform(.5,1.5); h=int(rng.integers(1,25));u=rng.uniform(-.25,.25)
   y=x.copy()
   for j in range(h):
    w=rng.normal(size=2);w*=p.eta*rng.random()/np.linalg.norm(w)
    y=p.A@y+th*p.c*u+w
   post=p.posterior(x,y,u,h,[.5,1.5])
   def loss(t):
    pred=x.copy()
    for j in range(h):pred=p.A@pred+t*p.c*u
    return np.linalg.norm(y-pred)-.001*sum(p.a**k for k in range(h))
   lo=.5 if loss(.5)<=0 else brentq(loss,.5,th,xtol=1e-13)
   hi=1.5 if loss(1.5)<=0 else brentq(loss,th,1.5,xtol=1e-13)
   np.testing.assert_allclose(post,[lo,hi],atol=1e-8)
   self.assertLessEqual(post[1]-post[0],p.resolution(u,h,1)+1e-8)
 def test_attained_minimax_diameter(self):
  for a in [.8,.98,.995,1.]:
   for period in [4,8,12,24]:
    p=rotation(a,2*np.pi/period,eta=.001)
    for h in range(1,25):
     y=p.q[h]*.25 # midpoint gain 1
     post=p.posterior(np.zeros(2),y,.25,h,[.5,1.5])
     self.assertAlmostEqual(post[1]-post[0],p.resolution(.25,h,1),places=9)
 def test_damping_formula(self):
  for period in [4,8,12,24]:
   for a in [.8,.95,.98,.995,.9999]:
    p=rotation(a,2*np.pi/period,eta=.001)
    direct=np.linalg.norm(p.q[period])/sum(a**k for k in range(period))
    formula=(1-a)/abs(1-a*np.exp(2j*np.pi/period))
    self.assertAlmostEqual(direct,formula,places=12)
 def test_aliasing_and_no_noise_learning(self):
  p=rotation(1.,2*np.pi/12,eta=0.,hmax=12)
  self.assertLess(np.linalg.norm(p.q[12]),1e-12)
  self.assertAlmostEqual(p.resolution(.2,12,1.),1.)
  post=p.posterior(np.zeros(2),.87*.2*p.q[11],.2,11,[.5,1.5])
  np.testing.assert_allclose(post,[.87,.87],atol=1e-10)
 def test_resolution_constrained_global_choice(self):
  rng=np.random.default_rng(723)
  for _ in range(200):
   p=rotation(.98,rng.uniform(.1,2.5),eta=.001)
   x=rng.normal(size=2);x*=rng.uniform(0,.99)/np.linalg.norm(x)
   u,h,d=p.choose(x,[.5,1.5],'resolution_constrained')
   # Dense search is a separate lower-resolution design comparison.
   feasible_h=[]
   for j,(lo,hi) in enumerate(p.safe_intervals(x,[.5,1.5]),1):
    if lo<=hi and min(p.resolution(v,j,1) for v in np.linspace(lo,hi,101))<=.08+1e-10:
     feasible_h.append(j)
   if feasible_h:self.assertEqual(h,max(feasible_h))
 def test_general_zonotope_gauge(self):
  # Verify gamma_(sum A^j W)(sum A^j c) <= gamma_W(c)
  rng=np.random.default_rng(724)
  for _ in range(100):
   A=rng.normal(size=(3,3))*.4;c=rng.normal(size=3);h=int(rng.integers(1,12))
   G=np.hstack([np.linalg.matrix_power(A,k) for k in range(h)]);q=sum((np.linalg.matrix_power(A,k)@c for k in range(h)),np.zeros(3))
   m=G.shape[1];obj=np.r_[np.zeros(m),1.]
   inequality=np.vstack([np.c_[np.eye(m),-np.ones(m)],np.c_[-np.eye(m),-np.ones(m)]])
   r=linprog(obj,A_ub=inequality,b_ub=np.zeros(2*m),A_eq=np.c_[G,np.zeros(3)],b_eq=q,bounds=[(None,None)]*m+[(0,None)],method='highs')
   self.assertTrue(r.success);self.assertLessEqual(r.fun,np.max(np.abs(c))+1e-8)

if __name__=='__main__':unittest.main(verbosity=2)
