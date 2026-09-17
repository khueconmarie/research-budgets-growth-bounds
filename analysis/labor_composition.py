"""Sharp resource coefficient from matched labor growth and input composition.

The domain is 0 <= delta < k, h >= 0, and 0 <= m < 1. This is the
necessary-condition resource block; no feedback ceiling is imposed here.
Rates are annual logs. A floor h restricts n on the regular BGP.
"""
from drift_core import kappa, eta, multiplier_bound


def labor_composition_coefficient(delta, m, h, k, beta=.96, sigma=.85):
    if not (0 <= delta < k and h >= 0 and 0 <= m < 1):
        raise ValueError('require 0 <= delta < k, h >= 0, 0 <= m < 1')
    upper = kappa(k-delta, beta) * eta(k, beta, sigma)
    z_floor = (h-delta) + m*(k-h)
    if h >= delta:
        return dict(coefficient=upper, endpoint='mu_to_one',
                    lower_endpoint=None, upper_endpoint=upper,
                    knowledge_floor_at_m=z_floor, attained=False)
    if z_floor <= 0:
        return dict(coefficient=0., endpoint='zero_knowledge_growth',
                    lower_endpoint=None, upper_endpoint=upper,
                    knowledge_floor_at_m=z_floor, attained=False)
    lower = kappa(z_floor, beta) * ((1-m)/m + eta(k, beta, sigma))
    return dict(coefficient=min(lower, upper),
                endpoint='mu_floor' if lower <= upper else 'mu_to_one',
                lower_endpoint=lower, upper_endpoint=upper,
                knowledge_floor_at_m=z_floor, attained=lower <= upper)


def labor_composition_bound(delta, m, h, k=.02, budget=.0192,
                            alpha=.04, beta=.96, sigma=.85):
    result = labor_composition_coefficient(delta, m, h, k, beta, sigma)
    result.update(multiplier_bound(budget, alpha, result['coefficient']))
    return result
