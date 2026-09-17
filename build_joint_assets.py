"""Reproduce combined bounds; check nesting, endpoints and comparative restrictions."""
from pathlib import Path
import sys, math, json
import numpy as np
import sympy as sp
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'analysis'))
from joint_bound import joint_bound, composition_floor, coefficient_limit
from drift_core import resource_coefficient, multiplier_bound, kappa, eta
from build_feedback_assets import feedback_bound


def main():
    # Efficient activity allocation under a Cobb-Douglas research aggregator.
    v,I,n,mu=sp.symbols('v I n mu',positive=True)
    bk=v*I*n*mu; bh=v*I*n*(1-mu)
    assert sp.simplify(bk+bh-v*I*n)==0
    assert sp.simplify(bk/(bk+bh)-mu)==0
    c,r,lam=sp.symbols('c r lam',positive=True)
    assert sp.simplify(lam*r*c/(1-c+r*c)-lam*(r*c/(1-c+r*c)))==0
    rng=np.random.default_rng(2026091607)
    cases=240; grid_count=0; worst_gap=math.inf; nesting_error=0.0
    monotonicity=0
    for i in range(cases):
        al=float(rng.uniform(.01,.25)); b=float(rng.uniform(.82,.99))
        s=float(rng.uniform(.1,.94)); k=float(rng.uniform(.005,.08))
        dd=0.0 if i%17==0 else float(rng.uniform(0,1.2*k))
        m=float(rng.uniform(0,.9)); ps=float(np.exp(rng.uniform(np.log(.03),np.log(300))))
        e=float(rng.uniform(.001,.1))
        args=dict(delta=dd, m_account=m, psi_ceiling=ps, budget=e,alpha=al,beta=b,sigma=s,k=k)
        z=joint_bound(**args)
        # Both old results are exact slices, with independent root implementations.
        old=feedback_bound(ps,dd,e,al,b,s,k)
        new=joint_bound(**{**args,'m_account':0.0})
        nesting_error=max(nesting_error,abs(old['omega_sup']-new['omega_sup']))
        aa=resource_coefficient(dd,m,k,b,s)['coefficient']
        old2=multiplier_bound(e,al,aa)
        new2=joint_bound(**{**args,'psi_ceiling':math.inf})
        expect=min(1-al,e/aa) if aa>0 else 1-al
        nesting_error=max(nesting_error,abs(expect-new2['omega_sup']))
        assert (old2['multiplier_upper'] is None)==(new2['multiplier_sup'] is None)
        # Direct primitive coefficients dominate the conditional envelope.
        for w in np.linspace(.02,.98,15)*min(1-al,ps):
            mm=max(m,w/ps)
            low=w*coefficient_limit(dd,mm,k,b,s)
            for u in np.linspace(mm,1,61,endpoint=False):
                # chi=-delta, H growth zero, service growth k, whenever g_Z>0.
                gz=u*k-dd
                if gz>0:
                    raw=w*kappa(gz,b)*((1-u)/u+eta(k,b,s))
                    worst_gap=min(worst_gap,raw-low); grid_count+=1
                    assert raw>=low-2e-12
            if w<z['omega_sup']-1e-10: assert low<=e+2e-12
            if w>z['omega_sup']+1e-10: assert low>=e-2e-12
        for change,direction in [({'budget':e*1.3},1),({'m_account':(1+m)/2},-1),
                                 ({'psi_ceiling':ps*1.3},1),({'delta':dd+.1*k},1)]:
            other=joint_bound(**{**args,**change})['omega_sup']
            assert direction*(other-z['omega_sup'])>=-2e-12
            monotonicity+=1
    assert nesting_error<2e-13
    # Closed domain endpoints are suprema, not claimed feasible pure-compute paths.
    small=joint_bound(0,psi_ceiling=.01)
    assert small['omega_sup']==.01 and small['multiplier_sup'] is not None
    at_boundary=joint_bound(0,budget=.96*kappa(.02,.96)*eta(.02,.96,.85))
    assert at_boundary['multiplier_sup'] is None
    assert joint_bound(.001,m_account=0,psi_ceiling=math.inf)['multiplier_sup'] is None
    cmin=14633/84377
    specs=[(math.inf,1,0,1),(5,1,0,1),(5,1,.5,1),
           (math.inf,1,1,1),(5,1,1,1),(5,1,1,2),(5,1,1,.5),(5,.5,1,1)]
    rows=[]; lines=[]
    for ps,r,cover,scale in specs:
        floor=composition_floor(cmin,r,cover)
        result=joint_bound(.002,floor,ps,budget=scale*.0192)
        row=dict(relative=r,coverage=cover,absolute_budget_factor=scale,**result)
        rows.append(row)
        ptxt=r'$\infty$' if math.isinf(ps) else f'{ps:g}'
        mtxt=r'$+\infty$' if result['multiplier_sup'] is None else f"{result['multiplier_sup']:.6f}"
        lines.append(f"{ptxt} & {r:g} & {cover:g} & {scale:g} & {floor:.6f} & {mtxt} \\\\")
    # The table displaying separate and joint bounds is written by build_information_assets.py.
    # A reader can call the same function with a model budget or a declared accounting bridge.
    results=dict(accounting_minimum=cmin,accounting_reference_budget=.0192,
                 delta=.002,alpha=.04,beta=.96,sigma=.85,k=.02,rows=rows)
    (P/'generated/joint_results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    check=dict(random_joint_information_sets=cases,nested_result_maximum_weight_error=nesting_error,
               primitive_grid_values=grid_count,minimum_primitive_minus_bound=worst_gap,
               monotonicity_checks=monotonicity,domain_endpoint_cases=3,
               aggregation_bill_identities_symbolically_verified=True,
               empirical_parameters_estimated=False,global_optimality_claim=False)
    (P/'checks/joint_verification.json').write_text(json.dumps(check,indent=2)+'\n')
    print(json.dumps(check,indent=2))
    print(json.dumps(rows,indent=2))

if __name__=='__main__': main()
