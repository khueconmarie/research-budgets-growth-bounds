"""Frozen-accounting tables and conditional bound illustrations for rr05."""
from pathlib import Path
from fractions import Fraction
import json
import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parent
sys.path.insert(0, str(P/'analysis'))
from drift_core import resource_coefficient, multiplier_bound, drift_cutoff


def main():
    raw = json.loads((P/'data/accounting_observations.json').read_text())['observations']
    beta, sigma, alpha, k, budget = .96, .85, .04, .02, .0192
    rows=[]; headline=[]
    for delta,m in [(0,0),(.0001,0),(.001,0),(.001,.1),(.001,.4)]:
        a=resource_coefficient(delta,m,k,beta,sigma)['coefficient']
        cap=multiplier_bound(budget,alpha,a)['multiplier_upper']
        headline.append(dict(delta=delta,m=m,coefficient=a,multiplier_supremum=cap))
        dlabel='0' if delta==0 else f'{100*delta:.2f}'
        caplabel=r'$+\infty$' if cap is None else f'{cap:.6f}'
        rows.append(f'{dlabel} & {m:.2f} & {caplabel}'+r' \\')
    (P/'generated/headline_table.tex').write_text('\n'.join(rows)+'\n')

    table=[]; acct=[]
    for row in raw:
        c,e=row['compute_service_expense'],row['rd_total']
        assert sum(row['components'].values())==e
        share=Fraction(c,e)
        residual=e-c
        target_r=Fraction(1,9)*Fraction(residual,c)
        acct.append(dict(id=row['id'],share=float(share),r_for_m_point_one=float(target_r)))
        company='MiniMax' if row['company']=='MiniMax' else 'Zhipu'
        period=row['period'].replace('-FY','').replace('-9M',' (9m)').replace('-6M',' (6m)')
        table.append(f'{company} & {period} & {c:,} & {e:,} & {100*float(share):.4f}'+r' \\')
    (P/'generated/accounting_table.tex').write_text('\n'.join(table)+'\n')
    minimum=min(acct,key=lambda x:x['share']); amin=minimum['share']
    scopes=[('All disclosed periods',raw),('Full years only',[r for r in raw if r['months']==12]),
            ('2024 full years only',[r for r in raw if r['period']=='2024-FY'])]
    bridge=[]
    for label,rr in scopes:
        s=min(Fraction(r['compute_service_expense'],r['rd_total']) for r in rr)
        rmin=Fraction(1,9)*(1-s)/s
        bridge.append(f'{label} & {len(rr)} & {float(s):.6f} & {float(rmin):.6f}'+r' \\')
    (P/'generated/bridge_table.tex').write_text('\n'.join(bridge)+'\n')

    out=dict(assumptions=dict(alpha=alpha,beta=beta,sigma=sigma,k=k,budget=budget),
             headline=headline,observed_accounting=acct,observed_minimum=minimum,
             m_point_one_finite_cutoff=drift_cutoff(budget,alpha,.1,k,beta,sigma),
             macro_transfer_assumed_not_estimated=True)
    (P/'generated/paper_assets.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'observations':len(acct),'minimum':minimum,
                      'm_point_one_finite_cutoff':out['m_point_one_finite_cutoff']},indent=2))

if __name__=='__main__': main()
