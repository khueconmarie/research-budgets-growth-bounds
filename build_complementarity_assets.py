"""Verify Corollary 2 from primitive coefficients and build its current outputs."""
from pathlib import Path
import json
import sys
import math
import numpy as np
import sympy as sp
import mpmath as mp

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P/'analysis'))
from drift_core import kappa, eta, resource_coefficient
from labor_composition import labor_composition_coefficient, labor_composition_bound


def main():
    for folder in ['checks', 'generated']:
        (P/folder).mkdir(exist_ok=True)
    # Differentiate the primitive one-dimensional objective independently.
    mu, s, c = sp.symbols('mu s c', positive=True)
    b = sp.Symbol('b', real=True)
    arg = sp.Symbol('z', real=True)
    K = sp.Function('K'); z=b+s*mu
    kp = sp.Subs(sp.diff(K(arg),arg),arg,z)
    kpp = sp.Subs(sp.diff(K(arg),arg,2),arg,z)
    f=K(z)*(1/mu-c)
    D=s*mu*(1-c*mu)*kp-K(z)
    assert sp.simplify(mu**2*sp.diff(f,mu)-D)==0
    rhs=-2*s*c*mu*kp+s**2*mu*(1-c*mu)*kpp
    assert sp.simplify(sp.diff(D,mu)-rhs)==0

    # Grid and augmented-primitive checks cover both endpoint branches.
    rng=np.random.default_rng(2026091712)
    errors=[]; augmented_slack=[]; recover=[]; counts={}
    for i in range(240):
        beta=float(rng.uniform(.85,.99));sigma=float(rng.uniform(.3,.96))
        k=float(rng.uniform(.005,.08));delta=float(rng.uniform(.02,.95)*k)
        h=float(rng.uniform(0,.995)*delta)
        threshold=(delta-h)/(k-h)
        m=float(threshold+(1-threshold)*rng.uniform(.0001,.98))
        exact=labor_composition_coefficient(delta,m,h,k,beta,sigma)
        mus=np.r_[np.linspace(m,1,4001,endpoint=False),1-1e-11]
        gz=h-delta+mus*(k-h)
        # Raw formulas rather than calling the coefficient routine on the grid.
        E=np.expm1(gz)
        vals=beta*E/(E+1-beta)*((1-mus)/mus+beta*(math.exp(k)-sigma)/(math.exp(k)-beta*sigma))
        err=abs(float(vals.min())-exact['coefficient']);errors.append(err)
        assert float(vals.min())>=exact['coefficient']-1e-12
        assert err<2e-8
        counts[exact['endpoint']]=counts.get(exact['endpoint'],0)+1
        # Increase independent primitive growth restrictions, checking validity.
        for _ in range(5):
            u=float(rng.uniform(m,1));H=h+float(rng.uniform(0,.04))
            Kr=k+float(rng.uniform(0,.04));chi=-delta+float(rng.uniform(0,.02))
            Z=chi+(1-u)*H+u*Kr
            coef=kappa(Z,beta)*((1-u)/u+eta(Kr,beta,sigma))
            augmented_slack.append(coef-exact['coefficient'])
            assert coef>=exact['coefficient']-1e-12
        # h=0 exactly restores Theorem 1, including zero-coefficient cases.
        for floor in [0., m]:
            a=labor_composition_coefficient(delta,floor,0,k,beta,sigma)['coefficient']
            a_old=resource_coefficient(delta,floor,k,beta,sigma)['coefficient']
            recover.append(abs(a-a_old));assert abs(a-a_old)<1e-13
        # The positive-floor limit at h=delta matches the strong labor result.
        strong=labor_composition_coefficient(delta,m,delta,k,beta,sigma)
        assert abs(strong['coefficient']-kappa(k-delta,beta)*eta(k,beta,sigma))<1e-14

    # Exact binary-representable boundary: z_m=0, and either side of it.
    boundary=dict(delta=.03125,h=.015625,k=.078125,m=.25)
    eq=labor_composition_coefficient(**boundary)
    assert eq['knowledge_floor_at_m']==0 and eq['coefficient']==0
    for m in [.25-1e-6,.25+1e-6]:
        a=labor_composition_coefficient(**{**boundary,'m':m})
        assert (a['coefficient']>0)==(m>.25)
    # Explicit sequences with positive knowledge growth establish infimum zero.
    zero_sequences=[]
    for m,h in [(0.,.001),(.5*14633/84377,0.),(.25,.015625)]:
        d,k=(.03125,.078125) if m==.25 else (.002,.02)
        mu0=max(m,(d-h)/(2*(k-h)))
        chi0=-((1-mu0)*h+mu0*k)
        assert mu0>0 and chi0>=-d-1e-15
        vals=[]
        for eps in [1e-5,1e-7,1e-9]:
            chi=chi0+eps;gz=chi+(1-mu0)*h+mu0*k
            vals.append(kappa(gz,.96)*((1-mu0)/mu0+eta(k,.96,.85)))
        assert 0<vals[-1]<vals[1]<vals[0]
        zero_sequences.append(dict(m=m,h=h,values=vals))

    # Independent 70-digit evaluation from the exact disclosed integer ratio.
    mp.mp.dps=70
    b=mp.mpf('.96');sig=mp.mpf('.85');k=mp.mpf('.02');d=mp.mpf('.002')
    h=mp.mpf('.001');m=mp.mpf(14633)/(2*84377);budget=mp.mpf('.0192')
    kap=lambda z:b*mp.expm1(z)/(mp.expm1(z)+1-b)
    et=b*(mp.exp(k)-sig)/(mp.exp(k)-b*sig)
    zm=(1-m)*h+m*k-d
    left=kap(zm)*((1-m)/m+et);right=kap(k-d)*et;a=min(left,right)
    cap=budget/(mp.mpf('.96')*a-budget)
    joint=labor_composition_bound(.002,float(m),.001)
    standalone_c=labor_composition_bound(.002,float(m),0.)
    standalone_h=labor_composition_bound(.002,0.,.001)
    assert standalone_c['multiplier_upper'] is None and standalone_h['multiplier_upper'] is None
    assert abs(joint['multiplier_upper']-float(cap))<1e-14
    # Positive coefficient alone does not imply a finite multiplier, including equality.
    critical=.96*joint['coefficient']
    critical_bound=labor_composition_bound(.002,float(m),.001,budget=critical)
    assert critical_bound['multiplier_upper'] is None
    assert labor_composition_bound(.002,float(m),.001,budget=critical*(1-1e-6))['multiplier_upper']>0
    no_decline=labor_composition_bound(0.,0.,0.)
    assert abs(no_decline['multiplier_upper']-.084117315223)<1e-12
    checks=dict(symbolic_endpoint_derivatives='pass',positive_grid_cases=len(errors),
                primitive_grid_values=240*4002,endpoint_max_error=max(errors),endpoint_branches=counts,
                augmented_primitive_checks=len(augmented_slack),minimum_augmented_slack=min(augmented_slack),
                theorem1_recovery_cases=len(recover),theorem1_max_difference=max(recover),
                positivity_boundary='pass',budget_equality_boundary='pass',zero_sequences=zero_sequences,
                high_precision={key:str(v) for key,v in dict(m=m,z_m=zm,left=left,right=right,coefficient=a,multiplier=cap).items()})
    (P/'checks/complementarity_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    outputs=dict(parameters=dict(alpha=.04,beta=.96,sigma=.85,delta=.002,k=.02,budget=.0192),
                 example=dict(composition=standalone_c,labor=standalone_h,joint=joint,m=float(m),h=.001),
                 high_precision=checks['high_precision'])
    (P/'generated/complementarity_results.json').write_text(json.dumps(outputs,indent=2,allow_nan=False)+'\n')
    (P/'generated/complementarity_macros.tex').write_text(
        '\\newcommand{\\WeakJointBound}{'+f'{float(cap):.6f}'+'}\n'+
        '\\newcommand{\\WeakJointGrowth}{'+f'{float(zm):.7f}'+'}\n')
    # Extend the existing comparison without adding another full-page table.
    table=P/'generated/information_comparison_table.tex'
    rows=table.read_text().split('\\midrule')[0].rstrip()
    rows+='\n\\midrule\n'
    rows+=r'Weak composition only: $m=\underline c_{\mathcal S}/2$ & $\infty$\\'+'\n'
    rows+=r'Weak labor growth only: $g_H\geq.001$ & $\infty$\\'+'\n'
    rows+=r'Weak composition and labor growth together & '+f'{float(cap):.6f}'+r'\\'+'\n'
    table.write_text(rows)
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()
