"""v0.11 common-information sensitivity, figure, and budget-definition checks."""
from pathlib import Path
import sys,math,json
import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'analysis'))
from drift_core import kappa,eta,resource_coefficient
from joint_bound import joint_bound,composition_floor
from information_bounds import information_bounds
from budget_extensions import user_coefficient,user_bound,labor_bound

def fmt(v):return r'$+\infty$' if v is None else f'{v:.6f}'
def row(vals):return ' & '.join(map(str,vals))+r' \\'
def main():
    for d in ('generated','figures','checks'): (P/d).mkdir(exist_ok=True)
    c=14633/84377; out={};checks={}
    lines=[];rs=[]
    for dd,pss in [(0.0001,[1,2,5,10,100,180,185,192]),(.002,[1,2,5,10])]:
        if lines:lines.append(r'\midrule')
        for ps in pss:
            z=joint_bound(dd,0,ps);edge=.96*resource_coefficient(dd,.96/ps,.02,.96,.85)['coefficient']
            rs.append(dict(psi=ps,boundary_budget=edge,**z))
            lines.append(row([f'{100*dd:g}',ps,f'{100*edge:.6f}',fmt(z['multiplier_sup'])]))
    (P/'generated/feedback_comparison_table.tex').write_text('\n'.join(lines)+'\n');out['feedback']=rs
    # Keep the existing exact crossing, rounded crossing, and accounting comparisons.
    old=json.loads((P/'generated/information_results.json').read_text())
    rs=old['joint_cases'];table=[]
    for z in rs:
        lam=r'$\lambda^\dagger$' if z.get('crossing_label')=='exact_crossing' else f"{z['coverage']:g}"
        table.append(row([r'$\infty$' if z['psi_ceiling'] is None else f"{z['psi_ceiling']:g}",f"{z['relative']:g}",lam,f"{z['absolute_budget_factor']:g}",fmt(z['multiplier_composition']),fmt(z['multiplier_feedback']),fmt(z['multiplier_joint'])]))
    # Additional perturbations of relative valuation and absolute social-resource budget.
    for ps,r,lam,L in [(5,2,.5,1),(5,1,.5,.5),(5,1,.5,2)]:
        z=information_bounds(.002,composition_floor(c,r,lam),ps,budget=.0192*L)
        rs.append(dict(psi_ceiling=ps,relative=r,coverage=lam,absolute_budget_factor=L,**z))
        table.append(row([ps,r,lam,L,fmt(z['multiplier_composition']),fmt(z['multiplier_feedback']),fmt(z['multiplier_joint'])]))
    (P/'generated/joint_bound_table.tex').write_text('\n'.join(table)+'\n');out['joint']=rs
    # Uniform labor restriction, deliberately distinguished from a joint sharp cap.
    lb=labor_bound(.002,.002);out['labor']=lb
    assert abs(lb['multiplier']-.0909603)<1e-7
    levels=[('Budget only',None),(r'Feedback: $\Psi=5$',joint_bound(.002,0,5)['multiplier_sup']),
            (r'Feedback and transferred floor: $\lambda=.7$',joint_bound(.002,.7*c,5)['multiplier_sup']),
            (r'Research labor: $g_H\geq.002$',lb['multiplier'])]
    (P/'generated/information_comparison_table.tex').write_text('\n'.join(row([l,fmt(v)]) for l,v in levels)+'\n')
    # Inputs often treated as calibration points are explicitly varied at the reference experiment.
    primitive=[]
    for name,values in [('alpha',[.02,.04,.08]),('beta',[.94,.96,.98]),('sigma',[.7,.85,.95]),('k',[.01,.02,.04])]:
        for val in values:
            args={'alpha':.04,'beta':.96,'sigma':.85,'k':.02};args[name]=val
            z=joint_bound(.002,0,5,**args)
            primitive.append(dict(parameter=name,value=val,multiplier=z['multiplier_sup']))
    out['primitive_sensitivity']=primitive
    (P/'generated/primitive_sensitivity_table.tex').write_text('\n'.join(row(['$'+chr(92)+z['parameter']+'$' if z['parameter']!='k' else '$k$',f"{z['value']:g}",fmt(z['multiplier'])]) for z in primitive)+'\n')
    # Acquisition and user-cost ceilings are separately defined experiments.
    rb=[]
    for dd,m,ps in [(0,0,float('inf')),(.002,0,5),(.002,.7*c,5),(.002,c,5)]:
        ac=joint_bound(dd,m,ps)['multiplier_sup'];uc=user_bound(dd,m,ps)['multiplier']
        assert uc<=ac+1e-12
        rb.append(dict(delta=dd,m=m,psi=None if math.isinf(ps) else ps,acquisition=ac,user_cost=uc))
    out['budget_definitions']=rb
    (P/'generated/budget_definitions_table.tex').write_text('\n'.join(row([f"{100*z['delta']:g}",f"{z['m']:.6f}",r'$\infty$' if z['psi'] is None else z['psi'],fmt(z['acquisition']),fmt(z['user_cost'])]) for z in rb)+'\n')
    # Independent primitive grid: user coefficient and labor growth inequality.
    rng=np.random.default_rng(2026091711);usr=[];lbgap=[]
    for _ in range(200):
        b=float(rng.uniform(.85,.99));s=float(rng.uniform(.3,.96));k=float(rng.uniform(.005,.08));m=float(rng.uniform(.02,.95));dd=float(rng.uniform(0,.99*m*k))
        us=np.r_[np.linspace(m,1,3001,endpoint=False),1-1e-11]
        vals=np.array([kappa(u*k-dd,b)/u for u in us])
        target=user_coefficient(dd,m,k,b);usr.append(abs(vals.min()-target))
        assert vals.min()>=target-1e-12
        assert abs(vals.min()-target)<2e-9
        h=dd+float(rng.uniform(0,.04))
        for u in [m,.4,.9,.999999]:
            gz=-dd+(1-u)*h+u*k
            a=kappa(gz,b)*((1-u)/u+eta(k,b,s));low=kappa(k-dd,b)*eta(k,b,s)
            lbgap.append(a-low);assert a>=low-1e-13
    checks.update(user_cost_grid_cases=200,user_cost_grid_values=600400,user_endpoint_max_error=max(usr),labor_inequality_checks=len(lbgap),minimum_labor_slack=min(lbgap))
    mp.mp.dps=60
    kap=lambda g:mp.mpf('.96')*mp.expm1(g)/(mp.expm1(g)+mp.mpf('.04'))
    et=mp.mpf('.96')*(mp.exp(mp.mpf('.02'))-mp.mpf('.85'))/(mp.exp(mp.mpf('.02'))-mp.mpf('.96')*mp.mpf('.85'))
    lbhi=mp.mpf('.0192')/(mp.mpf('.96')*kap(mp.mpf('.018'))*et-mp.mpf('.0192'))
    checks['labor_bound_60_digits']=str(lbhi);assert abs(float(lbhi)-lb['multiplier'])<1e-15
    # Two-panel main figure with styles legible in grayscale.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
    fig,ax=plt.subplots(1,2,figsize=(10.6,4.25),layout='constrained')
    ls=np.linspace(0,1,501);ma=[];joint=[]
    base=information_bounds(.002,0,5);mf=base['multiplier_feedback'];cut=base['omega_feedback']/(5*c)
    for lam in ls:
        z=information_bounds(.002,lam*c,5)
        ma.append(np.nan if z['multiplier_composition'] is None else z['multiplier_composition']);joint.append(z['multiplier_joint'])
    ax[0].plot(ls,ma,color='#6b6b6b',ls='--',lw=2,label='Composition only')
    ax[0].axhline(mf,color='#b15b22',ls=':',lw=2,label=r'Feedback only ($\Psi=5$)')
    ax[0].plot(ls,joint,color='#123f5b',lw=2.3,label='Joint bound')
    ax[0].plot(cut,mf,'o',color='#123f5b',ms=5)
    ax[0].axvline(cut,color='#999999',ls=':',lw=1)
    ax[0].annotate(r'$\lambda^\dagger=0.624$',xy=(cut,mf),xytext=(.64,1.73),arrowprops={'arrowstyle':'-','color':'#444444'})
    ax[0].set(xlim=(0,1),ylim=(0,2.1),xlabel=r'Transferred activity share $\lambda$',ylabel='Multiplier upper bound',title='(a) Which restriction binds?')
    ax[0].legend(loc='upper left',fontsize=8,frameon=False)
    for ps,style,color in [(1,'--','#888888'),(2,'-.','#32765e'),(5,'-','#123f5b'),(10,':','#b15b22')]:
        ys=[information_bounds(.002,lam*c,ps)['multiplier_joint'] for lam in ls]
        ax[1].plot(ls,[np.nan if y is None else y for y in ys],ls=style,color=color,lw=2,label=rf'$\Psi={ps}$')
    ax[1].axhline(lb['multiplier'],color='black',ls=(0,(5,2,1,2)),lw=1.5,label='Labor-growth bound')
    ax[1].set(xlim=(0,1),ylim=(0,2.1),xlabel=r'Transferred activity share $\lambda$',title='(b) The transfer threshold depends on feedback')
    ax[1].legend(loc='upper right',fontsize=8,frameon=False)
    for a in ax:a.grid(alpha=.15)
    fig.savefig(P/'figures/information_frontier.pdf');fig.savefig(P/'figures/information_frontier.png',dpi=180);plt.close(fig)
    out['figure']={'threshold':cut,'feedback_bound':mf,'labor_bound':lb['multiplier'],'delta':.002,'truncated_vertical_range':[0,2.1]}
    (P/'generated/revision_results.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    (P/'checks/revision_assets_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
