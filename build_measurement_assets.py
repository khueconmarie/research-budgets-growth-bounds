"""Coverage and measurement-operator checks; no empirical parameter estimation."""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'analysis'))
from drift_core import kappa,eta,resource_coefficient,drift_cutoff

A,B,S,K,E=.04,.96,.85,.02,.0192
C=14633/84377
RECENT=1552832/2195436

def composition(r,coverage=1,c=C):
    return coverage*r*c/(1-c+r*c)

def cutoff(r,coverage=1,c=C):
    m=composition(r,coverage,c)
    return drift_cutoff(E,A,m,K,B,S) if m>0 else 0.

def witness(chi):
    """Two regular BGP candidates with the same composite-input proxy trend."""
    mu=C;n=.01;gy=.02;gi=.05;gz=gi+chi
    gkr=(gi-(1-mu)*n)/mu;gr=gkr-gy
    kap=kappa(gz,B);er=eta(gkr,B,S)
    omega=E/(kap*((1-mu)/mu+er));psi=omega/mu;d=1-A-omega
    g0=(1-A)*(gy-n)-psi*gz
    sh=(1-mu)*psi*kap;sr=omega*kap
    h=sh/(1-A+sh)
    ry=math.exp(gy)/B-S;rr=math.exp(gkr)/B-S
    ky=A/ry;kr=sr/rr
    jy=math.expm1(gy)*ky+(1-S)*ky
    jr=(math.exp(gkr)-S)*kr;c=1-jy-jr
    nu=math.expm1(gz)/(h**(1-mu)*kr**mu)
    tech=1/((1-h)**(1-A)*ky**A)
    assert c>0 and 0<h<1 and d>0 and g0>0
    checks={}
    def check(name,x,y):
        checks[name]=abs(x-y)/max(1,abs(x),abs(y))
    for t in (0,1,10,40):
        y=math.exp(gy*t);ct=c*y;z=math.exp(gz*t)
        labor=math.exp(n*t);ht=h*labor
        yt=ky*math.exp(gy*t);rt=kr*math.exp(gkr*t)
        q=math.exp(gr*t);jt=jr*y
        inc=nu*math.exp(chi*t)*ht**(1-mu)*rt**mu
        check(f'output_{t}',y,tech*math.exp(g0*t)*z**psi*(labor-ht)**(1-A)*yt**A)
        check(f'resource_{t}',y,ct+jy*y+jt)
        check(f'knowledge_{t}',z*math.exp(gz),z+inc)
        check(f'capital_y_{t}',yt*math.exp(gy),S*yt+jy*y)
        check(f'capital_r_{t}',rt*math.exp(gkr),S*rt+q*jt)
        lam=1/ct
        vz=psi*y/(z*ct*(1-B/math.exp(gz)));vzn=vz/math.exp(gz)
        vy=math.exp(gy)/(B*ct);vyn=vy/math.exp(gy)
        vr=math.exp(gkr)/(B*q*ct);vrn=vr/math.exp(gkr)
        check(f'costate_z_{t}',vz,lam*psi*y/z+B*vzn)
        check(f'costate_y_{t}',vy,lam*A*y/yt+B*S*vyn)
        check(f'costate_r_{t}',vr,B*vzn*mu*inc/rt+B*S*vrn)
        check(f'labor_{t}',lam*(1-A)*y/(labor-ht),B*vzn*(1-mu)*inc/ht)
        check(f'invest_y_{t}',lam,B*vyn)
        check(f'invest_r_{t}',lam,B*q*vrn)
        check(f'tvc_z_{t}',vz*z,psi/(c*(1-B/math.exp(gz))))
        check(f'tvc_y_{t}',vy*yt,math.exp(gy)*ky/(B*c))
        check(f'tvc_r_{t}',vr*rt,math.exp(gkr)*kr/(B*c))
    assert max(checks.values())<1e-11
    check('budget',sh+jr,E)
    return dict(chi=chi,g_z=gz,g_input=gi,n=n,g_y=gy,g_kr=gkr,
                gamma_r=gr,g0=g0,mu=mu,omega=omega,psi=psi,d=d,
                e_r=sh+jr,multiplier=omega/d,c0=c,h=h,ky0=ky,kr0=kr,
                jy0=jy,jr0=jr,nu0=nu,tech0=tech,
                composite_proxy_trend=chi-gz,cost_proxy_trend=-n,
                maximum_relative_residual=max(checks.values()),checks=len(checks))

def main():
    # Symbolic cancellation requires the matched regular BGP; it is not imposed on data.
    chi,mu,gh,gk,n=sp.symbols('chi mu gh gk n',real=True)
    gi=(1-mu)*gh+mu*gk;gz=chi+gi
    assert sp.simplify(chi-gz+gi-n+n)==0
    ph,ga,power=sp.symbols('ph ga power',real=True,nonzero=True)
    assert sp.simplify((1-(1-(1-ph)/power))*power*ga-(1-ph)*ga)==0
    r,c,m,lam=sp.symbols('r c m lam',positive=True)
    mapp=lam*r*c/(1-c+r*c)
    required=m*(1-c)/(c*(lam-m))
    assert sp.simplify(mapp.subs(r,required)-m)==0

    # Exact off-BGP accounting identity, including the annual log-increment correction.
    rng=np.random.default_rng(20260916);operator_errors=[]
    for _ in range(500):
        z=float(rng.uniform(.5,3));nu=float(rng.uniform(.005,.2))
        i=float(rng.uniform(.5,3));den=float(rng.uniform(.5,3))
        ch=float(rng.uniform(-.1,.1));gin=float(rng.uniform(-.1,.2))
        gs=float(rng.uniform(-.1,.2));psi=float(rng.uniform(.02,.5))
        x=nu*i/z;zn=z*(1+x);nn=nu*math.exp(ch)
        inn=i*math.exp(gin);sn=den*math.exp(gs);xn=nn*inn/zn
        prox=psi*math.log1p(x)/den;proxn=psi*math.log1p(xn)/sn
        lhs=math.log(proxn/prox)
        rhs=ch-math.log1p(x)+gin-gs+math.log((math.log1p(xn)/xn)/(math.log1p(x)/x))
        operator_errors.append(abs(lhs-rhs))
    assert max(operator_errors)<1e-12

    rows=[];tables=[]
    rten=(1-C)/(9*C);rforty=(.4/.6)*(1-C)/C
    for label,rr,ll,cc in [('All periods',rten,1,C),('All periods',1,1,C),
                         ('All periods',1,.5,C),('All periods',1,.25,C),
                         ('All periods',rforty,1,C),('2024 only',1,1,RECENT)]:
        mm=composition(rr,ll,cc);fin=cutoff(rr,ll,cc)
        rows.append(dict(sample=label,r=rr,coverage=ll,m=mm,positive_cutoff=mm*K,
                         finite_cutoff=fin,correction_for_five_percent=.05-fin))
        tables.append(f'{label} & {rr:.4f} & {ll:.2f} & {mm:.4f} & {100*mm*K:.4f} & {100*fin:.4f}'+r' \\')
    (P/'generated/coverage_table.tex').write_text('\n'.join(tables)+'\n')

    # Independent bisection of the budget equality, across the joint information map.
    maxerr=0.;gridchecks=0
    for rr in np.geomspace(.005,4,30):
        for ll in np.linspace(.01,1,25):
            mm=composition(float(rr),float(ll));lo=0.;hi=mm*K
            for _ in range(70):
                mid=(lo+hi)/2
                a=resource_coefficient(mid,mm,K,B,S)['coefficient']
                if (1-A)*a>E:lo=mid
                else:hi=mid
            maxerr=max(maxerr,abs((lo+hi)/2-cutoff(rr,ll)))
            gridchecks+=1
    assert maxerr<1e-13
    assert abs(cutoff(1)-.003318221068071958)<1e-14
    assert abs(cutoff(1,.5)-.001660573385184108)<1e-14
    assert abs(cutoff(rten)-.0019147898368626)<1e-14
    correction=.05-cutoff(1)
    phirows=[dict(g_A=g,phi_upper=1-correction/g) for g in [.02,.05,.10]]
    (P/'generated/normalization_table.tex').write_text('\n'.join(
        f'{100*x["g_A"]:.1f} & {x["phi_upper"]:.6f}' + r' \\' for x in phirows)+'\n')

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,
                        'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(7,3.5),layout='constrained')
    rr=np.linspace(0,1,151);ll=np.linspace(0,1,151)
    vals=np.array([[100*cutoff(float(rv),float(lv)) for rv in rr] for lv in ll])
    im=axes[0].pcolormesh(rr,ll,vals,cmap='Blues',shading='auto',vmin=0,vmax=.35,rasterized=True)
    cs=axes[0].contour(rr,ll,vals,levels=[.05,.10,.20,.30],colors='#34495e',linewidths=.7)
    axes[0].clabel(cs,inline=True,fontsize=7,fmt='%.2f')
    axes[0].set(xlabel='Relative valuation lower bound, r',ylabel='Covered cost share lower bound, λ',
                title='(a) Finite-bound decline cutoff, %/yr')
    cb=fig.colorbar(im,ax=axes[0],fraction=.055,pad=.02);cb.ax.tick_params(labelsize=7)
    mm=np.array([composition(1,float(l)) for l in ll])
    pos=100*K*mm;fin=np.array([100*cutoff(1,float(l)) for l in ll])
    ax=axes[1]
    ax.fill_between(ll,0,fin,color='#dce8f2',label='Finite upper bound')
    ax.fill_between(ll,fin,pos,color='#f4dfb7',label='Positive coefficient; no bound')
    ax.fill_between(ll,pos,.37,color='#f1f1f1',label='Zero coefficient')
    ax.plot(ll,fin,color='#234c6b',lw=1.6)
    ax.plot(ll,pos,color='#916c28',lw=1.3,ls='--')
    ax.scatter([.5,1],[100*cutoff(1,.5),100*cutoff(1)],s=18,color='#234c6b',zorder=4)
    ax.set(xlim=(0,1),ylim=(0,.37),xlabel='Covered cost share lower bound, λ',
           ylabel='Permitted decline, 100δ',title='(b) Equal relative valuation: r = 1')
    ax.legend(loc='upper left',fontsize=7.5,frameon=False)
    ax.grid(axis='y',alpha=.15)
    fig.savefig(P/'figures/accounting_bound_region.pdf',dpi=400)
    fig.savefig(P/'figures/accounting_bound_region.png',dpi=240)
    plt.close(fig)

    witnesses=[witness(0),witness(-.01)]
    (P/'generated/proxy_witness_table.tex').write_text('\n'.join(
        f'{100*w["chi"]:.1f} & {100*w["g_z"]:.1f} & {w["omega"]:.6f} & {w["multiplier"]:.6f} & {100*w["g0"]:.4f}'+r' \\'
        for w in witnesses)+'\n')
    result=dict(coverage_rows=rows,operator_stress_rate=-.05,correction=correction,
                normalization_examples=phirows,conditional_bgp_witnesses=witnesses,
                actual_bjvw_aggregate_reported_rates=dict(productivity=-.051,effective_researchers=.043),
                denominator_match_to_model_composite_established=False,
                empirical_macro_floor_estimated=False)
    (P/'generated/measurement_results.json').write_text(json.dumps(result,indent=2)+'\n')
    checks=dict(symbolic_operator_bgp_cancellation=True,normalization_invariance=True,
                coverage_inverse_identity=True,off_bgp_operator_trials=500,
                maximum_off_bgp_operator_error=max(operator_errors),
                joint_coverage_bisection_checks=gridchecks,maximum_cutoff_error=maxerr,
                conditional_bgp_witnesses=2,witness_equations=sum(w['checks'] for w in witnesses),
                max_witness_relative_residual=max(w['maximum_relative_residual'] for w in witnesses),
                no_bjvw_calibration_claim=True,new_estimates=0)
    (P/'checks/measurement_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
