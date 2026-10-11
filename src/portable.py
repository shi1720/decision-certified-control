"""Portable controller-side certificate. It contains no raw-history archive."""
import json
import numpy as np
from scipy.optimize import linprog
from src.certified import Support,filter_model,project_qp

class PortableModel:
 def __init__(self,payload):
  self.payload=payload;self.n=payload['n'];self.m=payload['m'];self.d=self.n+self.m
  self.eta=np.array(payload['eta']);self.umax=payload['umax']
  self.C=[np.asarray(c) for c in payload['C']];self.b=[np.asarray(b) for b in payload['b']]
  self.models=[[np.asarray(t) for t in row] for row in payload['models']]
 def support(self,row,z,ids=None):
  C,b=self.C[row],self.b[row]
  r=linprog(-z,A_ub=C,b_ub=b,bounds=[(None,None)]*self.d,method='highs')
  if not r.success:raise RuntimeError('portable support LP failed')
  return Support(float(z@r.x),r.x,np.array([],dtype=int),np.array([]),0.,0.)

def export_cache(cache):
 a=cache.a;ids=[sorted(s) for s in cache.ids]
 return dict(n=a.n,m=a.m,eta=a.eta.tolist(),umax=a.umax,
  C=[a.C[s].tolist() for s in ids],b=[a.b[i,s].tolist() for i,s in enumerate(ids)],
  models=[[t.tolist() for t in row] for row in cache.models])

def evaluate(payload,x,v):
 a=PortableModel(payload);u,upper,_,supports=filter_model(a,x,v)
 H=[];b=[]
 for i,models in enumerate(a.models):
  for theta in models:
   for sign in (1.,-1.):
    H.append(sign*theta[a.n:]);b.append(1-a.eta[i]-3e-8-sign*theta[:a.n]@x)
 _,_,lower,_=project_qp(v,H,b,a.umax)
 return u,dict(upper=upper,lower=lower,gap=max(0.,upper-lower))

def payload_bytes(payload):
 return len(json.dumps(payload,separators=(',',':'),sort_keys=True).encode('utf-8'))
