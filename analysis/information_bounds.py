"""Scalar-bound effects of composition floors and feedback ceilings.

This does not value information in welfare units. Upper or two-sided composition
restrictions are outside the lower-floor experiment characterized here.
"""
from __future__ import annotations
import math
from drift_core import resource_coefficient
from joint_bound import joint_bound


def information_bounds(delta, m_account=0., psi_ceiling=math.inf, budget=.0192,
                       alpha=.04, beta=.96, sigma=.85, k=.02):
    A=1-alpha
    aa=resource_coefficient(delta,m_account,k,beta,sigma)['coefficient']
    wa=A if aa==0 else min(A,budget/aa)
    wf=joint_bound(delta,0.,psi_ceiling,budget,alpha,beta,sigma,k)['omega_sup']
    w=min(wa,wf)
    multiplier=lambda x:x/(A-x) if x<A else None
    return {'omega_composition':wa,'omega_feedback':wf,'omega_joint':w,
            'multiplier_composition':multiplier(wa),
            'multiplier_feedback':multiplier(wf),'multiplier_joint':multiplier(w),
            'composition_coefficient':aa}
