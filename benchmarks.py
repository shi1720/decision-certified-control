"""Declared physical and synthetic multivariable benchmark generators."""
import numpy as np
from scipy.linalg import expm
from src.certified import Archive

def plant(name):
    if name=='thermal4':
        L=np.array([[2,-1,0,-1],[-1,2,-1,0],[0,-1,2,-1],[-1,0,-1,2.]])
        A=expm(-.2*np.eye(4)-.12*L)
        B=.4*np.array([[1,0],[.7,.2],[0,1],[.2,.7]])
    elif name=='mechanical4':
        F=np.block([[np.zeros((2,2)),np.eye(2)],[-np.array([[2.,-.4],[-.4,1.5]]),-.9*np.eye(2)]])
        A=expm(.3*F); A*=.72/np.linalg.norm(A,np.inf)
        B=np.vstack((.12*np.eye(2),.55*np.eye(2)))
    elif name=='nonnormal6':
        F=.7*np.eye(6)+.14*np.diag(np.ones(5),1)
        B=1.5*np.array([[.5,0],[.3,.1],[.1,.35],[.35,-.1],[.1,.45],[-.1,.3]])
        K=np.zeros((2,6));K[0,0]=-.6;K[1,4]=-.6
        A=F-B@K
    elif name=='dense8':
        rng=np.random.default_rng(887);A=rng.normal(size=(8,8));A*=.65/np.linalg.norm(A,np.inf)
        B=rng.normal(size=(8,3));B*=.65/np.max(np.sum(abs(B),axis=1))
    else:raise ValueError(name)
    n,m=B.shape;center=np.c_[A,B];radius=.12/(n+m)
    eta=.015;umax=.8
    K=np.zeros((m,n)) if name!='nonnormal6' else K
    assert np.max(np.sum(abs(A+B@K),axis=1)+radius*(n+np.sum(abs(K))))+eta<1
    return center,radius,eta,umax,K

def history(name,seed,regime,N=1024):
    center,radius,eta,umax,K=plant(name);n,d=center.shape;m=d-n
    rng=np.random.default_rng(seed);theta=center+rng.uniform(-.7,.7,size=center.shape)*radius
    x=rng.uniform(-.5,.5,n);Z=[];Y=[];W=[]
    for t in range(N):
        backup=K@x;u=rng.uniform(-umax,umax,m)
        for _ in range(20):
            z=np.r_[x,u]
            if np.max(abs(center@z)+radius*np.sum(abs(z)))+eta<1:break
            u=(u+backup)*.5
        w=rng.uniform(-eta,eta,n) if regime=='uniform' else eta*rng.choice([-1.,1.],n)
        y=theta@z+w
        assert np.max(abs(y))<=1+1e-12
        Z.append(z);Y.append(y);W.append(w);x=y
    archive=Archive(center,radius,eta,np.array(Z),np.array(Y));archive.umax=umax;archive.backup=K
    return archive,theta,np.array(W)

def leverage_ids(Z,k):
    inverse=np.eye(Z.shape[1])*1e6;chosen=[];available=np.ones(len(Z),dtype=bool)
    for _ in range(min(k,len(Z))):
        scores=np.einsum('ij,jk,ik->i',Z,inverse,Z);scores[~available]=-1
        j=int(np.argmax(scores));v=inverse@Z[j]
        inverse-=np.outer(v,v)/(1+Z[j]@v);chosen.append(j);available[j]=False
    return np.array(chosen)

def subset_constraints(a,raw):
    base=np.arange(a.prior_count);raw=np.asarray(raw,dtype=int)
    ids=np.r_[base,a.prior_count+raw,a.prior_count+len(a.Z)+raw]
    return [ids.copy() for _ in range(a.n)]

def query_state(a,rng,t):
    if t%4==0:
        return rng.uniform(-.95,.95,a.n),rng.uniform(-1.8,1.8,a.m)
    row=int(rng.integers(a.n));sign=rng.choice([-1.,1.])
    # Declared boundary stress: align state/reference with one nominal row.
    x=sign*np.sign(a.center[row,:a.n])*rng.uniform(.85,.99,a.n)
    v=sign*np.sign(a.center[row,a.n:])*rng.uniform(1.2,1.8,a.m)
    return x,v
