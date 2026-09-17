"""Analytical resource bounds with a trend in normalized research productivity.

All rates are annual logs. This module solves no transition or estimation problem.
The sharp coefficient concerns only the stated budget/input-growth restriction
block. Additional BGP observations may tighten that block.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


def kappa(g: float, beta: float) -> float:
    if g < 0 or not 0 < beta < 1:
        raise ValueError("kappa requires nonnegative knowledge growth and 0<beta<1")
    e = math.expm1(g)
    return beta * e / (e + 1 - beta)


def eta(g: float, beta: float, sigma: float) -> float:
    if not 0 < beta < 1 or not 0 < sigma < 1 or math.exp(g) <= sigma:
        raise ValueError("invalid acquisition-service conversion")
    e = math.expm1(g)
    return beta * (e + 1 - sigma) / (e + 1 - beta * sigma)


def inverse_kappa(value: float, beta: float) -> float:
    if not 0 < value < beta:
        raise ValueError("inverse kappa requires 0<value<beta")
    return math.log(beta * (1 - value) / (beta - value))


def resource_coefficient(delta: float, mu_floor: float, service_floor: float,
                         beta: float, sigma: float) -> dict:
    """Infimum of e_R/omega over the stated resource block.

    chi >= -delta, g_H >= 0, g_K >= service_floor, mu >= mu_floor,
    0 < mu < 1, and g_Z = chi+(1-mu)g_H+mu*g_K > 0.
    The zero floor means mu remains strictly positive but otherwise unrestricted.
    """
    if delta < 0 or not 0 <= mu_floor < 1 or service_floor <= 0:
        raise ValueError("invalid information restrictions")
    acquisition = eta(service_floor, beta, sigma)
    if delta == 0:
        coefficient = kappa(service_floor, beta) * acquisition
        return dict(coefficient=coefficient, endpoint="mu_to_one",
                    lower_endpoint=None, upper_endpoint=coefficient,
                    positive=True, attained=False)
    if delta >= mu_floor * service_floor:
        return dict(coefficient=0.0, endpoint="zero_knowledge_growth",
                    lower_endpoint=None, upper_endpoint=None,
                    positive=False, attained=False)
    left = kappa(mu_floor * service_floor - delta, beta) * (
        (1 - mu_floor) / mu_floor + acquisition)
    right = kappa(service_floor - delta, beta) * acquisition
    return dict(coefficient=min(left, right),
                endpoint="mu_floor" if left <= right else "mu_to_one",
                lower_endpoint=left, upper_endpoint=right, positive=True,
                attained=left <= right)


def multiplier_bound(budget: float, alpha: float, coefficient: float) -> dict:
    if budget <= 0 or not 0 < alpha < 1 or coefficient < 0:
        raise ValueError("invalid bound inputs")
    gap = (1 - alpha) * coefficient - budget
    if coefficient == 0:
        return dict(omega_upper=None, multiplier_upper=None,
                    reason="no_positive_resource_coefficient")
    return dict(omega_upper=budget / coefficient,
                multiplier_upper=budget / gap if gap > 0 else None,
                reason="finite" if gap > 0 else "budget_reaches_regularity_boundary")


def drift_cutoff(budget: float, alpha: float, mu_floor: float,
                 service_floor: float, beta: float, sigma: float) -> float:
    """Largest decline allowance consistent with a finite block-level bound.

    Requires a positive composition floor and a finite zero-drift bound.
    Equality at this cutoff already leaves the multiplier unbounded.
    """
    if not 0 < mu_floor < 1:
        raise ValueError("positive composition floor required")
    acquisition = eta(service_floor, beta, sigma)
    target = budget / (1 - alpha)
    if target >= kappa(service_floor, beta) * acquisition:
        raise ValueError("zero-drift bound must be finite")
    left = mu_floor * service_floor - inverse_kappa(
        target / ((1 - mu_floor) / mu_floor + acquisition), beta)
    right = service_floor - inverse_kappa(target / acquisition, beta)
    return min(left, right)


@dataclass(frozen=True)
class Candidate:
    """Full regular BGP candidate realizing the zero-coefficient construction."""
    alpha: float = .04
    beta: float = .96
    sigma: float = .85
    mu: float = .05
    service_floor: float = .02
    d: float = .003

    def primitives_and_initial_state(self) -> dict:
        a, b, s, m, k, d = (
            self.alpha, self.beta, self.sigma, self.mu, self.service_floor, self.d)
        if not 0 < d < 1 - a or not 0 < m < 1:
            raise ValueError("candidate outside its interior domain")
        omega = 1 - a - d
        psi = omega / m
        chi = -m * k
        gz = d * d
        gk = k + d * d / m
        g0 = (1 - a) * k + d ** 3 / m
        G, Gz = math.exp(gk), math.exp(gz)
        kap, acquisition = kappa(gz, b), eta(gk, b, s)
        labor_bill = (1 - m) * psi * kap
        h = labor_bill / (1 - a + labor_bill)
        rent = G / b - s
        ky, kr = a / rent, omega * kap / rent
        jy, jr = (G - s) * ky, (G - s) * kr
        c = 1 - jy - jr
        z, labor = 1.0, 1.0
        nu = math.expm1(gz) / (h ** (1 - m) * kr ** m)
        background = 1 / ((1 - h) ** (1 - a) * ky ** a)
        return dict(alpha=a, beta=b, sigma=s, mu=m, omega=omega, psi=psi,
                    d=d, service_floor=k, chi=chi, n=0.0, gamma_y=0.0, gamma_r=0.0,
                    g_z=gz, g_kr=gk, g_y=gk, g0=g0, G=G, Gz=Gz,
                    kappa=kap, eta=acquisition, h=h, s_y=a,
                    s_r=omega * kap, s_h=labor_bill, y0=1.0, c0=c,
                    ky0=ky, kr0=kr, z0=z, l0=labor, jy0=jy, jr0=jr,
                    nu0=nu, t0=background,
                    e_r=labor_bill + jr, m_r=omega / d)


def input_growth_set(s_y: float, s_r: float, beta: float, g_in: float,
                     chi_lower: float, chi_upper: float) -> dict:
    """Exact set classification for the matched-cost/input-growth block."""
    if not (0 < s_y < 1 and 0 < s_r < beta * (1 - s_y)
            and chi_lower <= chi_upper):
        raise ValueError("invalid cost moments or trend interval")
    threshold = inverse_kappa(s_r / (1 - s_y), beta)
    lower, upper = g_in + chi_lower, g_in + chi_upper
    if upper <= threshold:
        return dict(status="empty", threshold=threshold, multiplier_upper=None)
    if lower <= threshold:
        return dict(status="unbounded", threshold=threshold, multiplier_upper=None)
    cap = s_r / (kappa(lower, beta) * (1 - s_y) - s_r)
    return dict(status="finite", threshold=threshold, multiplier_upper=cap)
