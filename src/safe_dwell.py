"""Exact safe dwell design and endpoint set membership for scalar actuator gain.
All bounds use independent per-step Euclidean disturbance balls. A = a Q,
Q orthogonal; only endpoint states are available to the controller.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class Plant:
    A: np.ndarray
    c: np.ndarray
    eta: float
    radius: float = 1.0
    umax: float = 0.25
    hmax: int = 24
    def __post_init__(self):
        self.a = float(np.linalg.norm(self.A, 2))
        assert np.allclose(self.A.T @ self.A, self.a**2*np.eye(len(self.c)), atol=1e-10)
        self.powers = [np.eye(len(self.c))]
        self.q = [np.zeros(len(self.c))]
        self.rho = [0.]
        self.variance = [0.]
        for j in range(1, self.hmax+1):
            self.q.append(self.q[-1] + self.powers[-1] @ self.c)
            self.rho.append(self.rho[-1] + self.eta*self.a**(j-1))
            self.variance.append(self.variance[-1]+self.a**(2*(j-1)))
            self.powers.append(self.powers[-1] @ self.A)
        self.q=np.array(self.q); self.rho=np.array(self.rho)
        self.variance=np.array(self.variance)
    def safe_intervals(self, x, interval):
        """Exact prefix-safe u intervals for each h, cumulatively intersected."""
        lo,hi=-self.umax,self.umax
        out=[]
        for j in range(1,self.hmax+1):
            b=self.powers[j]@x; r=self.radius-self.rho[j]
            for theta in interval:
                d=theta*self.q[j]; dd=float(d@d)
                if r < 0: lo,hi=1.,-1.; break
                if dd < 1e-24:
                    if b@b > r*r+1e-12: lo,hi=1.,-1.
                    continue
                center=-float(b@d)/dd
                rad2=(r*r-float(b@b))/dd+center**2
                if rad2 < -1e-12: lo,hi=1.,-1.; break
                rad=np.sqrt(max(0.,rad2))
                lo=max(lo,center-rad); hi=min(hi,center+rad)
            out.append((lo,hi))
        return np.array(out)
    def posterior(self,x,y,u,h,prior):
        v=u*self.q[h]; z=y-self.powers[h]@x; vv=float(v@v)
        if vv < 1e-24:
            if np.linalg.norm(z)>self.rho[h]+1e-9: raise ValueError('inconsistent endpoint')
            return np.array(prior,dtype=float)
        center=float(z@v)/vv
        perp=z-center*v
        rad2=(self.rho[h]**2-float(perp@perp))/vv
        if rad2 < -1e-8: raise ValueError('inconsistent endpoint')
        rad=np.sqrt(max(0.,rad2))
        result=np.array([max(prior[0],center-rad),min(prior[1],center+rad)])
        if result[0]>result[1]+1e-9: raise ValueError('empty parameter interval')
        return result
    def resolution(self,u,h,width):
        signal=abs(u)*np.linalg.norm(self.q[h])
        if signal<1e-12: return width
        return min(width,2*self.rho[h]/signal)
    def choose(self,x,prior,method,target=0.08,fixed_h=6,remaining=None):
        hs=self.hmax if remaining is None else min(self.hmax,remaining)
        ints=self.safe_intervals(x,prior)[:hs]
        feasible=np.flatnonzero(ints[:,0]<=ints[:,1]+1e-12)
        if not len(feasible): raise ValueError('no safe dwell')
        # Maximal magnitude, deterministic tie toward positive input.
        us=np.array([lo if abs(lo)>abs(hi)+1e-12 else hi for lo,hi in ints])
        width=prior[1]-prior[0]
        ds=np.array([self.resolution(u,j+1,width) for j,u in enumerate(us)])
        if method=='longest': k=feasible[-1]
        elif method=='fixed':
            good=feasible[feasible+1<=fixed_h]; k=good[-1] if len(good) else feasible[0]
        elif method=='one_step': k=feasible[0]
        elif method=='max_resolution':
            # Minimax diameter first, longer duration breaks ties.
            k=min(feasible,key=lambda j:(round(ds[j],12),-j))
        elif method=='gaussian_proxy':
            scores=us**2*np.sum(self.q[1:hs+1]**2,axis=1)/self.variance[1:hs+1]
            k=max(feasible,key=lambda j:(scores[j],j))
        elif method=='resolution_constrained':
            good=feasible[ds[feasible]<=target+1e-12]
            k=good[-1] if len(good) else min(feasible,key=lambda j:(round(ds[j],12),-j))
        else: raise ValueError(method)
        return float(us[k]),int(k+1),float(ds[k])

def rotation(a,angle,eta=0.002,hmax=24,umax=0.25):
    Q=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    return Plant(a*Q,np.array([1.,0.]),eta,umax=umax,hmax=hmax)
