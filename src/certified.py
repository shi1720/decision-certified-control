"""Two-sided decision certificates for full-matrix set-membership safety filters.

The archive stores raw transitions. The controller keeps selected scalar data
constraints and feasible row-model witnesses. LP duals certify support bounds;
QP duals certify lower bounds. No stochastic noise assumption is used.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.optimize import linprog, minimize, nnls

TOL = 2e-8

@dataclass
class Support:
    value: float
    theta: np.ndarray
    ids: np.ndarray
    weights: np.ndarray
    residual: float
    gap: float

class Archive:
    def __init__(self, center, radius, eta, Z, Y):
        self.center = np.asarray(center)
        self.n, self.d = self.center.shape
        self.m = self.d-self.n
        self.radius = np.broadcast_to(radius, self.center.shape).copy()
        self.eta = np.broadcast_to(eta, (self.n,)).copy()
        self.Z = np.asarray(Z); self.Y = np.asarray(Y)
        # Prior inequalities are retained without being counted as data.
        self.C = np.vstack((np.eye(self.d),-np.eye(self.d),self.Z,-self.Z))
        self.b = np.concatenate((self.center+self.radius,-self.center+self.radius,
                                 (self.Y+self.eta).T,(-self.Y+self.eta).T),axis=1)
        self.prior_count = 2*self.d
        self.calls = 0; self.seconds = 0.

    def support(self, row, z, ids=None):
        started=perf_counter()
        ids=np.arange(len(self.C)) if ids is None else np.asarray(ids,dtype=int)
        # Explicit free variables are essential; scipy defaults to nonnegative.
        result=linprog(-np.asarray(z),A_ub=self.C[ids],b_ub=self.b[row,ids],
                       bounds=[(None,None)]*self.d,method='highs',
                       options={'dual_feasibility_tolerance':1e-9,
                                'primal_feasibility_tolerance':1e-9})
        if not result.success: raise RuntimeError('inconsistent or failed support LP: '+result.message)
        weights=-result.ineqlin.marginals
        keep=weights>1e-10
        chosen=ids[keep]; weights=weights[keep]
        residual=float(np.linalg.norm(self.C[chosen].T@weights-z,np.inf))
        gap=float(self.b[row,chosen]@weights-z@result.x)
        # A floating-point solve is accepted only after checking its witnesses.
        primal=float(np.max(self.C[ids]@result.x-self.b[row,ids]))
        if residual>1e-7 or abs(gap)>1e-7 or primal>1e-7 or np.any(weights<0):
            raise RuntimeError('support witness failed validation')
        self.calls+=1; self.seconds+=perf_counter()-started
        return Support(float(z@result.x),result.x,chosen,weights,residual,gap)

    def raw_ids(self, ids):
        a=np.asarray(list(ids),dtype=int)-self.prior_count
        a=a[a>=0]
        return set((a % len(self.Z)).tolist())


def project_qp(v, H, b, umax):
    """Euclidean projection plus a checked dual lower bound and sparse support."""
    v=np.asarray(v);m=len(v)
    H=np.asarray(H,dtype=float).reshape(-1,m);b=np.asarray(b,dtype=float)
    allH=np.vstack((H,np.eye(m),-np.eye(m)))
    allb=np.r_[b,np.full(2*m,umax)]
    clipped=np.clip(v,-umax,umax)
    if np.max(allH@clipped-allb,initial=-np.inf)<=1e-11:
        u=clipped
    else:
        res=minimize(lambda u:.5*np.sum((u-v)**2),np.zeros(m),jac=lambda u:u-v,
            constraints=[{'type':'ineq','fun':lambda u:allb-allH@u,
                          'jac':lambda u:-allH}],method='SLSQP',
            options={'ftol':1e-12,'maxiter':300})
        u=res.x
        if not res.success or np.max(allH@u-allb)>1e-7:
            raise RuntimeError('projection QP failed: '+res.message)
    active=np.flatnonzero(allb-allH@u<2e-7)
    lam=np.zeros(len(allb))
    if np.linalg.norm(u-v)>1e-10:
        # A basic feasible solution uses <=m active normals, a conic
        # Caratheodory certificate. No approximate NNLS value is trusted.
        dual=linprog(np.ones(len(active)),A_eq=allH[active].T,b_eq=v-u,
                     bounds=(0,None),method='highs')
        if dual.success: lam[active]=dual.x
        else:
            coef,_=nnls(allH[active].T,v-u);lam[active]=coef
    stationarity=np.linalg.norm(u-v+allH.T@lam,np.inf)
    lower=float(lam@(allH@v-allb)-.5*np.sum((allH.T@lam)**2))
    value=float(.5*np.sum((u-v)**2))
    if lower>value+1e-7 or stationarity>2e-6:
        raise RuntimeError('QP witness failed validation')
    used=np.flatnonzero(lam[:len(H)]>1e-9)
    return u,value,lower,used


def filter_model(archive,x,v,ids=None,margin=3e-8):
    """Exchange solve of the robust filter for a selected outer model set."""
    n,m=archive.n,archive.m
    if ids is None: ids=[np.arange(len(archive.C)) for _ in range(n)]
    H=[];b=[];u=np.clip(v,-1.,1.) # overwritten by the bounded QP below
    for _ in range(120):
        u,value,lower,_=project_qp(v,H,b,archive.umax)
        z=np.r_[x,u]; supports=[];violations=[]
        for i in range(n):
            for sign in (1.,-1.):
                s=archive.support(i,sign*z,ids[i]);supports.append((i,sign,s))
                violations.append(s.value-(1-archive.eta[i]-margin))
        if max(violations)<=2e-9:
            # The finite-cut QP is a relaxation. Its dual lower bound and this
            # robust feasible candidate sandwich the selected-set optimum.
            return u,value,lower,supports
        for (i,sign,s),violation in zip(supports,violations):
            if violation>2e-9:
                H.append(sign*s.theta[n:]);b.append(1-archive.eta[i]-margin-sign*s.theta[:n]@x)
    raise RuntimeError('robust projection exchange did not terminate')

class DecisionCache:
    def __init__(self, archive, epsilon=1e-5, compact=True):
        self.a=archive;self.epsilon=epsilon;self.compact=compact
        self.ids=[set(range(archive.prior_count)) for _ in range(archive.n)]
        self.models=[[] for _ in range(archive.n)]
        self.refinements=0;self.archive_calls=0

    def optimistic(self,x,v):
        H=[];b=[];mapping=[];a=self.a
        for i,models in enumerate(self.models):
            for j,theta in enumerate(models):
                for sign in (1.,-1.):
                    H.append(sign*theta[a.n:]);b.append(1-a.eta[i]-3e-8-sign*theta[:a.n]@x)
                    mapping.append((i,j))
        u,value,lower,used=project_qp(v,H,b,a.umax)
        return u,value,lower,{mapping[k] for k in used}

    def query(self,x,v):
        a=self.a;oracle_calls=0;rounds=0;started=perf_counter();archive_seconds=0.
        for _ in range(100):
            ids=[np.array(sorted(s)) for s in self.ids]
            u,upper,_,support_outer=filter_model(a,x,v,ids)
            optimistic,_,lower,model_support=self.optimistic(x,v)
            gap=max(0.,upper-lower)
            if gap<=self.epsilon+1e-9:
                if self.compact:
                    # Sparsify upper certificates using selected-set LP duals.
                    newids=[set(range(a.prior_count)) for _ in range(a.n)]
                    for i,sign,s in support_outer:newids[i].update(s.ids.tolist())
                    self.ids=newids
                    # Sparsify lower certificate using the QP dual support.
                    oldmodels=self.models
                    self.models=[[oldmodels[i][j] for ii,j in sorted(model_support) if ii==i]
                                 for i in range(a.n)]
                raw=set().union(*(a.raw_ids(s) for s in self.ids))
                record=dict(upper=upper,lower=lower,gap=gap,
                    scalar_constraints=sum(len(s)-a.prior_count for s in self.ids),
                    raw_transitions=len(raw),model_witnesses=sum(map(len,self.models)),
                    archive_calls=oracle_calls,refinement_rounds=rounds,
                    elapsed=perf_counter()-started,archive_seconds=archive_seconds,
                    model_upper=max(s.value+a.eta[i] for i,sign,s in support_outer))
                return u,record
            # The archive only sees the optimistic state/input query. It
            # returns feasible row witnesses and sparse data dual certificates.
            z=np.r_[x,optimistic];worst=[];now=perf_counter()
            for i in range(a.n):
                for sign in (1.,-1.):
                    s=a.support(i,sign*z);oracle_calls+=1
                    violation=s.value-(1-a.eta[i]-3e-8)
                    worst.append((violation,i,s))
            archive_seconds+=perf_counter()-now
            added=False
            for violation,i,s in worst:
                if violation>1e-9:
                    self.ids[i].update(s.ids.tolist());self.models[i].append(s.theta);added=True
            if not added:
                # Optimistic minimizer is full-data feasible. All support duals
                # make it feasible for the selected outer model too.
                for violation,i,s in worst:
                    self.ids[i].update(s.ids.tolist());self.models[i].append(s.theta)
            rounds+=1;self.refinements+=1
        raise RuntimeError('decision refinement did not close its certificate')

    def screen_witnesses(self, Znew, Ynew):
        """Append-only validity: invalidate models violating any new data slab."""
        removed=0
        for i,models in enumerate(self.models):
            keep=[]
            for theta in models:
                if np.max(np.abs(Ynew[:,i]-Znew@theta)-self.a.eta[i])<=1e-9:keep.append(theta)
                else:removed+=1
            self.models[i]=keep
        return removed

def append_history(cache,new_archive):
    """Move a cache to an extending history; preserve IDs and screen witnesses.

    Constraint IDs use contiguous positive/negative halves, so negative-half
    IDs must be remapped whenever the history length changes.
    """
    old=cache.a;new=new_archive;Nold=len(old.Z);Nnew=len(new.Z)
    if Nnew<Nold or not np.array_equal(old.Z,new.Z[:Nold]) or not np.array_equal(old.Y,new.Y[:Nold]):
        raise ValueError('history must extend the previous data exactly')
    if not np.array_equal(old.center,new.center) or not np.array_equal(old.radius,new.radius) or not np.array_equal(old.eta,new.eta):
        raise ValueError('prior and disturbance model must remain unchanged')
    cache.ids=[{j+(Nnew-Nold) if j>=old.prior_count+Nold else j for j in ids} for ids in cache.ids]
    removed=cache.screen_witnesses(new.Z[Nold:],new.Y[Nold:])
    cache.a=new
    return removed
