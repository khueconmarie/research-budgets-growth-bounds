"""User-cost budgets and a uniform research-labor growth restriction.

Quantities are matched BGP model objects, not direct company expense estimates.
"""
from drift_core import kappa, eta
from joint_bound import joint_bound

def user_coefficient(delta,m,k,beta=.96):
    if delta<0 or not 0<=m<=1 or k<=0: raise ValueError('invalid restrictions')
    if delta==0: return kappa(k,beta)
    if delta>=m*k: return 0.
    return min(kappa(m*k-delta,beta)/m,kappa(k-delta,beta))

def user_bound(delta,m=0.,psi=float('inf'),budget=.0192,alpha=.04,beta=.96,k=.02):
    A=1-alpha;top=min(A,psi)
    F=lambda w:w*user_coefficient(delta,max(m,w/psi),k,beta)
    if F(top)<=budget: w=top
    else:
        lo=0.;hi=top
        for _ in range(100):
            mid=(lo+hi)/2
            if F(mid)<=budget:lo=mid
            else:hi=mid
        w=(lo+hi)/2
    return dict(omega=w,multiplier=w/(A-w) if w<A else None)

def labor_bound(delta,h,k=.02,budget=.0192,alpha=.04,beta=.96,sigma=.85):
    if not h>=delta or not k>delta: raise ValueError('labor repair conditions fail')
    a=kappa(k-delta,beta)*eta(k,beta,sigma)
    gap=(1-alpha)*a-budget
    return dict(coefficient=a,multiplier=budget/gap if gap>0 else None)
