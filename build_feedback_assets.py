"""Feedback restrictions for the necessary-condition resource information set.

The new scalar checks certify no planner optimum. The analytical core is read
unchanged from analysis/drift_core.py.
"""
from pathlib import Path
import sys, math, json
import numpy as np
import sympy as sp

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'analysis'))
from drift_core import resource_coefficient,kappa,eta,Candidate


def coefficient_limit(delta,m,k,beta,sigma):
    if m<1:
        return resource_coefficient(delta,m,k,beta,sigma)['coefficient']
    assert abs(m-1)<1e-12
    return 0.0 if delta>=k else kappa(k-delta,beta)*eta(k,beta,sigma)


def feedback_bound(Psi,delta,budget=.0192,alpha=.04,beta=.96,sigma=.85,k=.02):
    A=1-alpha
    end=min(A,Psi)
    def f(w):
        return w*coefficient_limit(delta,w/Psi,k,beta,sigma)
    if f(end)<=budget:
        weight=end
    else:
        lo,hi=0.0,end
        for _ in range(100):
            mid=(lo+hi)/2
            if f(mid)<=budget:lo=mid
            else:hi=mid
        weight=(lo+hi)/2
    return dict(psi_ceiling=Psi,delta=delta,omega_sup=weight,
                multiplier_sup=weight/(A-weight) if weight<A else None,
                endpoint_requirement=f(end),end_weight=end)


def direct_coefficient(delta,m,k,beta,sigma):
    """Independent endpoint evaluation from the primitive budget factor."""
    if delta==0:
        z=k
        return beta*math.expm1(z)/(math.exp(z)-beta)*(math.exp(k)-sigma)/(math.exp(k)/beta-sigma)
    if delta>=m*k:
        return 0.0
    f=lambda u: beta*math.expm1(u*k-delta)/(math.exp(u*k-delta)-beta)*((1-u)/u+(math.exp(k)-sigma)/(math.exp(k)/beta-sigma))
    return min(f(m),f(1.0))


def main():
    a,d,mu,k,delta,Psi=sp.symbols('a d mu k delta Psi',positive=True)
    assert sp.simplify(((1-a-d)/mu).subs(mu,delta/k)-(1-a-d)*k/delta)==0
    Z,lam,psi,T,L,K,H=sp.symbols('Z lam psi T L K H',positive=True)
    Y=T*Z**psi*(L-H)**(1-a)*K**a
    assert sp.simplify(sp.diff(lam*Y,Z,2)-lam*psi*(psi-1)*Y/Z**2)==0

    rng=np.random.default_rng(20260916)
    grid_slack=math.inf; direct_error=0.; classifications=0; grid_count=0
    for _ in range(360):
        al=rng.uniform(.01,.25);b=rng.uniform(.8,.99);s=rng.uniform(.05,.95)
        kk=rng.uniform(.005,.08);dd=rng.uniform(0,.95*kk)
        ps=float(np.exp(rng.uniform(np.log(.2),np.log(400))))
        budget=rng.uniform(.001,.07)
        r=feedback_bound(ps,dd,budget,al,b,s,kk)
        w=r['omega_sup']
        if w<r['end_weight']:
            f=w*direct_coefficient(dd,w/ps,kk,b,s)
            direct_error=max(direct_error,abs(f-budget))
            assert abs(f-budget)<2e-12
        for ww in np.linspace(.001,.999,70)*min(1-al,ps):
            m=ww/ps
            exact=ww*resource_coefficient(dd,m,kk,b,s)['coefficient']
            for u in np.linspace(m,1,41,endpoint=False):
                z=u*kk-dd
                if z>0:
                    grid_count+=1
                    raw=ww*kappa(z,b)*((1-u)/u+eta(kk,b,s))
                    grid_slack=min(grid_slack,raw-exact)
                    assert raw>=exact-2e-12
            if ww<w-1e-10:assert exact<=budget+2e-12
            elif ww>w+1e-10:assert exact>=budget-2e-12
            classifications+=1

    rows=[feedback_bound(p,.0001) for p in [1,2,5,10,100,180,185,192]]
    assert next(r for r in rows if r['psi_ceiling']==180)['multiplier_sup'] is not None
    assert next(r for r in rows if r['psi_ceiling']==185)['multiplier_sup'] is None
    lines=[]
    for r in rows:
        mr=r['multiplier_sup']
        value=f'{mr:.6f}' if mr is not None else r'$+\infty$'
        lines.append(f"{r['psi_ceiling']:.0f} & {100*r['endpoint_requirement']:.6f} & {value} \\\\")
    (P/'generated/feedback_cap_table.tex').write_text('\n'.join(lines)+'\n')

    A=.96;budget=.0192;b=.96;s=.85;kk=.02
    a0=kappa(kk,b)*eta(kk,b,s)
    limits=[]
    for ps in [.01,.05,1,10,100,1000]:
        target=min(ps,budget/a0)
        target_m=target/(A-target)
        seq=[feedback_bound(ps,dd) for dd in [1e-4,1e-6,1e-8,1e-10,1e-12]]
        assert abs(seq[-1]['multiplier_sup']-target_m)<1e-7
        limits.append(dict(psi_ceiling=ps,limit=target_m,sequence=seq))
    # Vanishing-budget witness at the smallest ceiling for the displayed delta.
    cap_witnesses=[]
    for dd in [1e-2,1e-3,1e-4,1e-5]:
        m=dd/kk
        ps=A*kk/dd
        for gap in [.003,.001,.0001]:
            x=Candidate(mu=m,d=gap).primitives_and_initial_state()
            assert x['psi']<ps and abs(x['chi']+dd)<1e-14
            cap_witnesses.append(dict(delta=dd,d=gap,mu=m,psi=x['psi'],psi_limit=ps,e_r=x['e_r'],m_r=x['m_r']))
    labor_a=kappa(kk-.0001,b)*eta(kk,b,s)
    labor_bound=budget/(A*labor_a-budget)
    results=dict(parameters=dict(alpha=.04,beta=b,sigma=s,budget=budget,k=kk,delta=.0001),
                 rows=rows,zero_decline_coefficient=a0,fixed_ceiling_limits=limits,
                 zero_budget_witnesses=cap_witnesses,labor_floor_example=dict(h_floor=.0001,delta=.0001,coefficient=labor_a,multiplier_upper=labor_bound))
    (P/'generated/feedback_results.json').write_text(json.dumps(results,indent=2)+'\n')
    check=dict(symbolic_size_relation=True,symbolic_hamiltonian_curvature=True,
               random_ceiling_cases=360,weight_classifications=classifications,
               primitive_composition_grid_values=grid_count,
               minimum_grid_slack=grid_slack,maximum_independent_budget_error=direct_error,
               fixed_finite_ceiling_limit_checks=len(limits),zero_budget_witnesses=len(cap_witnesses),
               global_optimality_proved=False,scope='resource necessary-condition block')
    (P/'checks/feedback_verification.json').write_text(json.dumps(check,indent=2)+'\n')
    print(json.dumps(check,indent=2))

if __name__=='__main__':main()
