"""Symbolic checks for two restricted optimality lemmas; no full certificate."""
from pathlib import Path
import json
import sympy as sp

X, K, a, mu = sp.symbols('X K a mu', positive=True)
u, v = sp.symbols('u v', real=True)
S = X**mu + a*K**mu
F = S**(1/mu)
actual = (u*u*sp.diff(F,X,2) + 2*u*v*sp.diff(F,X,K)
          + v*v*sp.diff(F,K,2))
expected = -(1-mu)*a*S**(1/mu-2)*X**(mu-2)*K**(mu-2)*(K*u-X*v)**2
ces_residual = sp.simplify(actual-expected)
assert ces_residual == 0

r, J = sp.symbols('r J', positive=True)
b = sp.log(sp.exp(r)-J)
utility_residual = sp.simplify(sp.diff(b,r,2)
                              + sp.exp(r)*J/(sp.exp(r)-J)**2)
assert utility_residual == 0

s, B = sp.symbols('s B', positive=True)
ray = (X**mu+B*s)**(1/mu)
ray_residual = sp.simplify(sp.diff(ray,s,2)
                         -(1-mu)/mu**2*B**2*(X**mu+B*s)**(1/mu-2))
assert ray_residual == 0

report = {
    'fixed_labor_ces_hessian_identity': str(ces_residual),
    'fixed_investment_utility_curvature_identity': str(utility_residual),
    'endogenous_labor_joint_scaling_curvature_identity': str(ray_residual),
    'analytic_domains': {
        'ces': 'X,K,a>0; 0<mu<1',
        'production': 'alpha>0, omega>0, alpha+omega<1',
        'utility': 'J>=0 and exp(r)>J',
        'joint_scaling': 'B,X,s>0; 0<mu<1; local labor interior',
    },
    'original_unrestricted_global_optimality': 'not_proved',
    'original_unrestricted_optimal_set_unboundedness': 'not_proved',
    'suboptimality_of_candidate': 'not_proved',
}
(Path(__file__).resolve().parent/'checks/restricted_optimality_verification.json').write_text(
    json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report, indent=2))
