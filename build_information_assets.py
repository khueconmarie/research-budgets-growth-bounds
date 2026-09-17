"""Information substitution, unchanged-bound regions, and historical-floor costs."""
from pathlib import Path
import sys,json,math
import numpy as np
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'analysis'))
from drift_core import resource_coefficient,kappa,eta
from joint_bound import joint_bound,composition_floor
from information_bounds import information_bounds


def display(x):return r'$+\infty$' if x is None else f'{x:.6f}'


def main():
    rng=np.random.default_rng(2026091608)
    errs=[];fe=[];replacement_checks=0
    # Compare a direct joint root with separate-root minima, including domain caps.
    for i in range(480):
        al=float(rng.uniform(.01,.4));b=float(rng.uniform(.8,.99));s=float(rng.uniform(.1,.97))
        k=float(rng.uniform(.005,.09));delta=0. if i%13==0 else float(rng.uniform(0,1.1*k))
        m=0. if i%11==0 else float(rng.uniform(.001,.98))
        ps=math.inf if i%9==0 else float(np.exp(rng.uniform(np.log(.005),np.log(500))))
        e=float(rng.uniform(.0001,.13))
        args=dict(delta=delta,m_account=m,psi_ceiling=ps,budget=e,alpha=al,beta=b,sigma=s,k=k)
        sep=information_bounds(**args);j=joint_bound(**args)
        errs.append(abs(sep['omega_joint']-j['omega_sup']))
        assert errs[-1]<3e-13
        for w in np.linspace(.04,.96,11)*min(1-al,ps):
            am=resource_coefficient(delta,m,k,b,s)['coefficient']
            ap=resource_coefficient(delta,w/ps,k,b,s)['coefficient']
            aj=resource_coefficient(delta,max(m,w/ps),k,b,s)['coefficient']
            fe.append(abs(w*aj-max(w*am,w*ap)))
        # Stronger data tighten exactly when their stand-alone cap beats the old joint cap.
        m2=(1+m)/2
        new=information_bounds(**{**args,'m_account':m2})
        expect=min(new['omega_composition'],sep['omega_joint'])
        assert abs(expect-new['omega_joint'])<3e-13
        replacement_checks+=1
    assert max(fe)<2e-13
    # Preserve the selected joint cases, now displaying the two individual bounds.
    old=json.loads((P/'generated/joint_results.json').read_text())
    rows=[]
    for row in old['rows']:
        ps=math.inf if row['psi_ceiling'] is None else row['psi_ceiling']
        out=information_bounds(row['delta'],row['m_account'],ps,row['budget'])
        assert abs(out['omega_joint']-row['omega_sup'])<2e-13
        rows.append({**row,**out})
    raw=json.loads((P/'data/accounting_observations.json').read_text())['observations']
    all_min=min(r['compute_service_expense']/r['rd_total'] for r in raw)
    recent=min(r['compute_service_expense']/r['rd_total'] for r in raw if r['period']=='2024-FY')
    temporal=[]
    for delta in [.0001,.002]:
        for lam in [.5,1.]:
            v=[]
            for name,c in [('all_disclosed',all_min),('2024_full_years',recent)]:
                m=composition_floor(c,1,lam)
                r=information_bounds(delta,m,5)
                v.append(r['multiplier_joint'])
                temporal.append(dict(delta=delta,coverage=lam,observed_set=name,share=c,**r))
            if delta==.0001:assert abs(v[0]-v[1])<1e-13
            else:assert v[1]<v[0]-1e-4
    # The coverage cutoffs here are exact because feedback binds at the rising
    # lower-composition endpoint, below the pure-compute limiting coefficient.
    f=information_bounds(.002,0,5)
    mf=f['omega_feedback']/5
    coeff=.0192/f['omega_feedback']
    assert coeff < kappa(.02-.002,.96)*eta(.02,.96,.85)
    thresholds=[]
    for name,c in [('all_disclosed',all_min),('2024_full_years',recent)]:
        lamstar=mf/c
        assert 0<lamstar<1
        at=information_bounds(.002,lamstar*c,5)
        below=information_bounds(.002,(lamstar-1e-5)*c,5)
        above=information_bounds(.002,(lamstar+1e-5)*c,5)
        assert abs(at['omega_joint']-f['omega_feedback'])<1e-13
        assert abs(below['omega_joint']-f['omega_feedback'])<1e-13
        assert above['omega_joint']<f['omega_feedback']-1e-7
        thresholds.append(dict(observed_set=name,coverage_threshold=lamstar,composition_threshold=mf))
    # Display the exact tie separately from its rounded-up decimal approximation.
    crossing=[]
    for label,lam in [('exact_crossing',thresholds[0]['coverage_threshold']),
                      ('rounded_crossing',.624),('above_crossing',.7)]:
        m=composition_floor(all_min,1,lam)
        direct=joint_bound(.002,m,5)
        sep=information_bounds(.002,m,5)
        assert abs(direct['omega_sup']-sep['omega_joint'])<2e-13
        entry=dict(relative=1,coverage=lam,absolute_budget_factor=1,delta=.002,
                   m_account=m,psi_ceiling=5,budget=.0192,
                   crossing_label=label,**sep)
        crossing.append(entry)
    assert abs(crossing[0]['multiplier_composition']-crossing[0]['multiplier_feedback'])<2e-12
    assert crossing[1]['multiplier_composition']<crossing[1]['multiplier_feedback']-1e-3
    assert crossing[2]['multiplier_joint']<crossing[1]['multiplier_joint']
    # Insert in increasing coverage order after the existing half-coverage row.
    insert=next(i+1 for i,row in enumerate(rows)
                if row['psi_ceiling']==5 and row['coverage']==.5
                and row['relative']==1 and row['absolute_budget_factor']==1)
    rows[insert:insert]=crossing
    lines=[]
    for row in rows:
        ps=row['psi_ceiling']
        ptxt=r'$\infty$' if ps is None else f'{ps:g}'
        lamtxt=r'$\lambda^\dagger$' if row.get('crossing_label')=='exact_crossing' else f"{row['coverage']:g}"
        vals=[ptxt,f"{row['relative']:g}",lamtxt,f"{row['absolute_budget_factor']:g}",
              display(row['multiplier_composition']),display(row['multiplier_feedback']),display(row['multiplier_joint'])]
        lines.append(' & '.join(vals)+r' \\')
    (P/'generated/joint_bound_table.tex').write_text('\n'.join(lines)+'\n')
    results={'joint_cases':rows,'historical_floor_comparisons':temporal,
             'coverage_thresholds_at_delta_point002':thresholds,'crossing_rows':crossing,
             'same_other_restrictions':{'alpha':.04,'beta':.96,'sigma':.85,'k':.02,'budget':.0192,'psi_ceiling':5,'relative':1},
             'scope':'Precision of a resource multiplier upper bound; no welfare or net value of measurement.'}
    (P/'generated/information_results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    checks={'separate_vs_joint_root_cases':len(errs),'maximum_weight_difference':max(errs),
            'max_envelope_identity_checks':len(fe),'maximum_envelope_difference':max(fe),
            'stronger_floor_update_checks':replacement_checks,'temporal_comparisons':len(temporal),
            'coverage_threshold_checks':len(thresholds),'crossing_rows_checked':len(crossing),
            'rounded_point_is_not_a_tie':True,'welfare_information_value_estimated':False,
            'unchanged_core_and_joint_root':True}
    (P/'checks/information_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
    print(json.dumps({'thresholds':thresholds,'historical_floor_comparisons':temporal},indent=2))

if __name__=='__main__':main()
