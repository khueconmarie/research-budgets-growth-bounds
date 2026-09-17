"""Joint resource-block bound. No optimization or empirical estimation is performed."""
from __future__ import annotations
import math
from drift_core import resource_coefficient, kappa, eta


def composition_floor(share: float, relative: float, coverage: float) -> float:
    if not 0 <= share < 1 or relative <= 0 or not 0 <= coverage <= 1:
        raise ValueError('invalid accounting bridge')
    return coverage * relative * share / (1 - share + relative * share)


def coefficient_limit(delta, floor, k, beta, sigma):
    """Continuous extension at the excluded pure-compute endpoint mu=1."""
    if floor < 1:
        return resource_coefficient(delta, floor, k, beta, sigma)['coefficient']
    if not math.isclose(floor, 1, abs_tol=1e-13):
        raise ValueError('composition floor exceeds one')
    return 0.0 if delta >= k else kappa(k - delta, beta) * eta(k, beta, sigma)


def joint_bound(delta, m_account=0.0, psi_ceiling=math.inf, budget=.0192,
                alpha=.04, beta=.96, sigma=.85, k=.02):
    """Supremum on the closure of the open regularity/composition domain.

    None means an unbounded multiplier, not a failed solve. A supremum at
    omega=psi_ceiling<1-alpha remains finite and need not be attained.
    """
    if (delta < 0 or not 0 <= m_account < 1 or psi_ceiling <= 0
            or budget <= 0 or not 0 < alpha < 1 or k <= 0):
        raise ValueError('invalid joint restrictions')
    A = 1 - alpha
    end = min(A, psi_ceiling)
    def floor(w): return max(m_account, w / psi_ceiling)
    def requirement(w):
        return w * coefficient_limit(delta, floor(w), k, beta, sigma)
    endpoint = requirement(end)
    if endpoint <= budget:
        weight = end
    else:
        lo, hi = 0.0, end
        for _ in range(100):
            mid = (lo + hi) / 2
            if requirement(mid) <= budget: lo = mid
            else: hi = mid
        weight = (lo + hi) / 2
    mr = weight / (A - weight) if weight < A else None
    fm = floor(weight)
    if math.isclose(m_account, weight / psi_ceiling, abs_tol=1e-12):
        active = 'both'
    else:
        active = 'accounting' if m_account > weight / psi_ceiling else 'feedback'
    return dict(delta=delta, m_account=m_account,
                psi_ceiling=psi_ceiling if math.isfinite(psi_ceiling) else None,
                budget=budget, omega_sup=weight, multiplier_sup=mr,
                m_effective_at_sup=fm, active_floor=active,
                domain_sup=end, endpoint_requirement=endpoint,
                budget_requirement_at_sup=requirement(weight),
                scope='sharp resource-block supremum; not an optimal-economy sharp bound')
