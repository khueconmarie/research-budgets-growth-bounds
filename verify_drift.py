"""Reproduce rr02's new analytical checks, examples, and TeX table.

Uses symbolic identities, direct resource-block grids, and independently
reconstructed physical laws/costates. No GPU, NLP, or empirical estimation.
"""
from pathlib import Path
import hashlib
import json
import math
import sys

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "analysis"))
from drift_core import (Candidate, drift_cutoff, eta, input_growth_set,
                        inverse_kappa, kappa, multiplier_bound,
                        resource_coefficient)


def symbolic_checks():
    alpha, mu, psi, n, g0, chi, gy, gr, x, z = sp.symbols(
        "alpha mu psi n g0 chi gy gr x z", real=True)
    equations = [z - (chi + n + mu * (x + gr)),
                 (1-alpha)*x - g0 - psi*z - alpha*gy]
    solution = sp.solve(equations, (x, z))
    expected = (g0 + psi*(n+chi) + alpha*gy + mu*psi*gr)/(1-alpha-mu*psi)
    assert sp.simplify(solution[x] - expected) == 0
    # The costate and budget identities use actual knowledge growth, even with drift.
    Gz, beta = sp.symbols("Gz beta", positive=True)
    knowledge_bill = beta * (Gz-1) * psi / (Gz-beta)
    assert sp.simplify(knowledge_bill/psi - beta*(Gz-1)/(Gz-beta)) == 0
    delta, k, c = sp.symbols("delta k c", positive=True)
    arg = mu*k-delta
    zz = sp.symbols("zz", real=True)
    kap = beta*(sp.exp(zz)-1)/(sp.exp(zz)-beta)
    kp, kpp = sp.diff(kap, zz), sp.diff(kap, zz, 2)
    assert sp.simplify(kp-beta*(1-beta)*sp.exp(zz)/(sp.exp(zz)-beta)**2) == 0
    assert sp.simplify(kpp+beta*(1-beta)*sp.exp(zz)*(sp.exp(zz)+beta)
                       /(sp.exp(zz)-beta)**3) == 0
    f = kap.subs(zz,arg)*(1/mu-c)
    D = k*mu*(1-c*mu)*kp.subs(zz,arg)-kap.subs(zz,arg)
    assert sp.simplify(mu**2*sp.diff(f,mu)-D) == 0
    Dprime = -2*k*c*mu*kp.subs(zz,arg) + k*k*mu*(1-c*mu)*kpp.subs(zz,arg)
    assert sp.simplify(sp.diff(D,mu)-Dprime) == 0
    # A full positive-service-growth counterexample, with fixed mu and chi.
    d = sp.symbols("d", positive=True)
    A = 1-alpha
    omega = A-d
    gz = d*d
    gk = k+d*d/mu
    background = A*k+d**3/mu
    assert sp.simplify(-mu*k + mu*gk - gz) == 0
    assert sp.simplify(background+(omega/mu)*gz+alpha*gk-gk) == 0
    assert sp.simplify((background+(omega/mu)*(-mu*k))/d-gk) == 0
    return dict(extended_growth_system=True, knowledge_costate_factor=True,
                kappa_derivatives=True, endpoint_minimization=True,
                counterexample_primitive_growth=True)


def physical_checks(p):
    """Independently evaluate original dated equations and all three costates."""
    a,b,s,m,psi = (p[key] for key in ("alpha","beta","sigma","mu","psi"))
    G,Gz = p["G"],p["Gz"]
    residuals = {}
    def check(name, left, right):
        residuals[name] = abs(left-right)/max(1.0,abs(left),abs(right))
    assert p["c0"] > 0 and 0 < p["h"] < 1 and p["ky0"] > 0 and p["kr0"] > 0
    for t in (0,1,10,40):
        y,c = p["y0"]*G**t,p["c0"]*G**t
        ky,kr,z = p["ky0"]*G**t,p["kr0"]*G**t,Gz**t
        jy,jr = p["jy0"]*G**t,p["jr0"]*G**t
        h=p["h"]
        nu=p["nu0"]*math.exp(p["chi"]*t)
        tech=p["t0"]*math.exp(p["g0"]*t)
        increment=nu*h**(1-m)*kr**m
        check(f"output_{t}",y,tech*z**psi*(1-h)**(1-a)*ky**a)
        check(f"resources_{t}",y,c+jy+jr)
        check(f"knowledge_{t}",z*Gz,z+increment)
        check(f"capital_y_{t}",ky*G,s*ky+jy)
        check(f"capital_r_{t}",kr*G,s*kr+jr)
        lam=1/c
        vz=psi*y/(z*c*(1-b/Gz))
        vz_next=vz/Gz
        vk=G/(b*c)
        vk_next=vk/G
        check(f"knowledge_envelope_{t}",vz,lam*psi*y/z+b*vz_next)
        check(f"production_envelope_{t}",vk,lam*a*y/ky+b*s*vk_next)
        check(f"research_envelope_{t}",vk,b*vz_next*m*increment/kr+b*s*vk_next)
        check(f"labor_foc_{t}",lam*(1-a)*y/(1-h),
              b*vz_next*(1-m)*increment/h)
        check(f"investment_foc_{t}",lam,b*vk_next)
        # These constant products imply discounted TVCs analytically.
        check(f"tvc_z_constant_{t}",vz*z,psi/(p["c0"]*(1-b/Gz)))
        check(f"tvc_y_constant_{t}",vk*ky,G*p["ky0"]/(b*p["c0"]))
        check(f"tvc_r_constant_{t}",vk*kr,G*p["kr0"]/(b*p["c0"]))
    return residuals


def main():
    symbolic = symbolic_checks()
    grid_count=0
    grid_min_slack=float("inf")
    sharp_errors=[]
    for beta in (.90,.96,.99):
        for sigma in (.30,.85):
            for k in (.005,.02,.08):
                for m in (.05,.4,.8):
                    for fraction in (0,.1,.5,.9,.999):
                        delta=fraction*m*k
                        result=resource_coefficient(delta,m,k,beta,sigma)
                        A=result["coefficient"]
                        mus=np.linspace(m,1,301)[:-1]
                        vals=np.array([kappa(float(mu*k-delta),beta)
                            *((1-mu)/mu+eta(k,beta,sigma)) for mu in mus])
                        assert float(vals.min()) >= A-1e-12
                        grid_min_slack=min(grid_min_slack,float(vals.min()-A))
                        grid_count+=len(mus)
                        endpoint_mu=m if result["attained"] else 1-1e-9
                        endpoint_value=kappa(endpoint_mu*k-delta,beta)*(
                            (1-endpoint_mu)/endpoint_mu+eta(k,beta,sigma))
                        sharp_errors.append(abs(endpoint_value-A))
    rng=np.random.default_rng(20260916)
    augmented_count=0
    for _ in range(4000):
        beta=float(rng.uniform(.85,.995));sigma=float(rng.uniform(.1,.95))
        k=float(rng.uniform(.003,.08));m=float(rng.uniform(.02,.85))
        delta=float(rng.uniform(0,.99))*m*k
        mu=float(rng.uniform(m,1))
        gk=k+float(rng.uniform(0,.1));gh=float(rng.uniform(0,.03))
        chi=-delta+float(rng.uniform(0,.01))
        gz=chi+(1-mu)*gh+mu*gk
        coeff=kappa(gz,beta)*((1-mu)/mu+eta(gk,beta,sigma))
        lower=resource_coefficient(delta,m,k,beta,sigma)["coefficient"]
        assert coeff>=lower-1e-12
        augmented_count+=1
    counterexamples=[];worst=0.0;physical_equations=0
    for beta in (.9,.96,.99):
        for sigma in (.3,.85):
            for mu in (.05,.4,.8):
                for d in (.01,.003,.001,.0003):
                    p=Candidate(beta=beta,sigma=sigma,mu=mu,d=d).primitives_and_initial_state()
                    residuals=physical_checks(p)
                    worst=max(worst,max(residuals.values()))
                    physical_equations+=len(residuals)
                    assert max(residuals.values())<2e-11
                    asymptotic=(1-p["alpha"])*beta/(1-beta)*(
                        (1-mu)/mu+eta(p["service_floor"],beta,sigma))
                    counterexamples.append(dict(**p,max_physical_residual=max(residuals.values()),
                                                budget_d2_ratio=p["e_r"]/d**2,
                                                budget_d2_limit=asymptotic))
    # No decline allowance is too small for the unrestricted-composition failure.
    small_drift=[]
    for delta in (1e-2,1e-4,1e-6,1e-8):
        mu=min(.5,delta/(2*.02))
        for d in (1e-4,1e-5,1e-6):
            p=Candidate(mu=mu,d=d).primitives_and_initial_state()
            assert p["chi"]>=-delta and p["g_kr"]>=.02
        small_drift.append(dict(delta=delta,mu=mu,last_budget=p["e_r"],last_multiplier=p["m_r"]))
    beta=.96;sigma=.85;alpha=.04;k=.02;m=.4;budget=.0192
    rows=[]
    for delta in (0,.001,.004,.007,.0075,.0079,.008):
        q=resource_coefficient(delta,m,k,beta,sigma)
        bound=multiplier_bound(budget,alpha,q["coefficient"])
        unrestricted=resource_coefficient(delta,0,k,beta,sigma)
        unrestricted_bound=multiplier_bound(budget,alpha,unrestricted["coefficient"])
        rows.append(dict(delta=delta,**q,**bound,
                         unrestricted_multiplier_upper=unrestricted_bound["multiplier_upper"]))
    assert abs(rows[0]["multiplier_upper"]-.08411731522333218)<1e-13
    cutoff=drift_cutoff(budget,alpha,m,k,beta,sigma)
    for sign in (-1,1):
        q=resource_coefficient(cutoff+sign*1e-7,m,k,beta,sigma)
        bounded=multiplier_bound(budget,alpha,q["coefficient"])["multiplier_upper"] is not None
        assert bounded==(sign<0)
    # Exact endpoint classifications, not rounded comparisons to the threshold.
    threshold=inverse_kappa(.002/(1-.04),.96)
    interval_cases=[
        input_growth_set(.04,.002,.96,0,threshold+.001,threshold+.01),
        input_growth_set(.04,.002,.96,0,threshold,threshold+.01),
        input_growth_set(.04,.002,.96,0,threshold-.001,threshold)]
    assert [r["status"] for r in interval_cases]==["finite","unbounded","empty"]
    # Release-level input hashes are checked by reproduce.py.
    table=[]
    def display(x):
        return r"\textemdash" if x is None else f"{x:.6f}"
    for row in rows:
        table.append(f"{100*row['delta']:.2f} & {row['coefficient']:.6f} & "
                     f"{display(row['multiplier_upper'])} & "
                     f"{display(row['unrestricted_multiplier_upper'])}"+r" \\")
    (ROOT/"generated/drift_table.tex").write_text("\n".join(table)+"\n")
    (ROOT/"generated/drift_macros.tex").write_text(
        r"\newcommand{\DriftFiniteCutoff}{"+f"{100*cutoff:.6f}"+"}\n")
    summary=dict(symbolic=symbolic,endpoint_grid_values=grid_count,
                 minimum_grid_slack=grid_min_slack,
                 maximum_endpoint_approximation_error=max(sharp_errors),
                 augmented_restriction_draws=augmented_count,random_seed=20260916,
                 full_bgp_candidates=len(counterexamples),
                 physical_equation_checks=physical_equations,
                 max_physical_relative_residual=worst,
                 interval_classifications=interval_cases,
                 arbitrarily_small_drift_examples=small_drift,
                 gpu_runs=0,transition_optimizations=0,empirical_estimates=0,
                 global_optimality_proved=False,
                 bound_scope="budget/input-growth moment block")
    results=dict(assumed_inputs=dict(alpha=alpha,beta=beta,sigma=sigma,
                 service_floor=k,mu_floor=m,budget=budget),rows=rows,
                 finite_bound_decline_cutoff=cutoff,
                 counterexample_family=counterexamples)
    (ROOT/"checks/drift_verification.json").write_text(json.dumps(summary,indent=2)+"\n")
    (ROOT/"generated/drift_results.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
