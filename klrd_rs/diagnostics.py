"""Numerical implementation gates and descriptive fixed-key diagnostics."""
import numpy as np
import torch
from scipy.stats import kstest, beta
from .math import rotate, keys, sphere_p
from .io import write_csv


@torch.no_grad()
def geometry_checks(engine, residuals, cfg, adapter):
    r=torch.as_tensor(residuals[:2],device=engine.weight.device,dtype=engine.weight.dtype)
    p,q=engine.p,engine.q
    frame=torch.cat([p,q],dim=1).double()
    orth=float((frame.T@frame-torch.eye(2*cfg.m,device=frame.device)).abs().max())
    a=torch.linspace(-.3,.3,cfg.m,device=r.device,dtype=r.dtype)
    rotated=rotate(r,p,q,a)
    norm_error=float(((rotated.double().norm(dim=-1)-r.double().norm(dim=-1)).abs()/r.double().norm(dim=-1)).max())
    x=torch.flip(r,[-1]); y=r
    skew_errors=[]
    for j in range(cfg.m):
        ax=(x@p[:,j:j+1])*q[:,j]-(x@q[:,j:j+1])*p[:,j]
        ay=(y@p[:,j:j+1])*q[:,j]-(y@q[:,j:j+1])*p[:,j]
        skew_errors.append(float(((x*ay).sum(-1)+(ax*y).sum(-1)).abs().max()))
    lp,phi=engine.full_scores(r)
    mean_zero=float((lp.exp()[:,:,None]*phi).double().sum(1).abs().max())
    relative=[];absolute=[]
    h=cfg.finite_difference_h
    for j in range(cfg.m):
        e=torch.zeros(cfg.m,device=r.device,dtype=r.dtype);e[j]=h
        plus=torch.log_softmax(engine.tail(rotate(r,p,q,e))/cfg.tau,-1)
        minus=torch.log_softmax(engine.tail(rotate(r,p,q,-e))/cfg.tau,-1)
        fd=(plus-minus)/(2*h)
        err=(fd-phi[:,:,j]).abs()
        relative.append(float((err/(cfg.finite_difference_atol+cfg.finite_difference_rtol*phi[:,:,j].abs())).max()))
        absolute.append(float(err.max()))
    native=adapter.native_tail_error(engine,residuals[:2])
    fast_error=float((engine.log_probs(r,a)-torch.log_softmax(engine.tail(rotated)/cfg.tau,-1)).abs().max())
    non_tangent_passed=(orth<cfg.rotation_atol and norm_error<cfg.rotation_atol and mean_zero<cfg.mean_zero_atol
            and native<.0002 and fast_error<.0002 and max(skew_errors)<.0002)
    passed=non_tangent_passed and max(relative)<1
    return {'status':'PASS' if passed else 'FAIL','geometry_and_readout_passed':non_tangent_passed,
        'tangent_fp32_passed':max(relative)<1,'orthogonality_error':orth,
        'norm_relative_error':norm_error,'skew_bilinear_errors':skew_errors,
        'full_vocabulary_score_mean_zero_max':mean_zero,'finite_difference_max_abs_by_direction':absolute,
        'finite_difference_scaled_error_by_direction':relative,'native_tail_max_abs_error':native,
        'fast_teacher_vs_exact_rotation_max_logprob_error':fast_error,'all_directions_tested':cfg.m,
        'token_indexing_rule':'response token i predicted at chat_prefix_length+i-1; all special targets excluded'}


def sphere_math(cfg, vectors):
    import mpmath as mp
    mp.mp.dps=80
    reference=[]
    for t in [-.99,-.6036,0.,.4032,.6036,.9,.99]:
        k=np.zeros(cfg.m);k[0]=t;k[1]=np.sqrt(1-t*t)
        v=np.eye(cfg.m)[0]
        exact=float(mp.betainc(mp.mpf(cfg.m-1)/2,mp.mpf(cfg.m-1)/2,0,(1-mp.mpf(t))/2,regularized=True))
        got=float(sphere_p(v,k));err=abs(got-exact)/max(exact,1e-300)
        reference.append({'cosine':t,'reference':exact,'numeric':got,'relative_error':err})
    bank=keys(cfg.diagnostic_random_keys,cfg.m,cfg.diagnostic_seed)
    checks={}
    for name,v in vectors.items():
        if v is None:checks[name]={'status':'MISSING'};continue
        if np.linalg.norm(v)==0:checks[name]={'status':'ZERO_VECTOR','p':1.};continue
        ps=sphere_p(v,bank)
        ks=kstest(ps,'uniform')
        tail={str(a):{'count':int((ps<=a).sum()),'n':len(ps),'rate':float((ps<=a).mean()),
                'confidence_interval':binomial_ci(int((ps<=a).sum()),len(ps))} for a in [.01,.0001]}
        checks[name]={'status':'PASS' if ks.pvalue>1e-6 else 'FAIL',
            'KS_statistic':float(ks.statistic),'KS_pvalue':float(ks.pvalue),
            'pvalue_quantiles':np.quantile(ps,[.001,.01,.1,.5,.9,.99]).tolist(),'tails':tail}
    passed=max(x['relative_error'] for x in reference)<1e-10 and all(x['status']!='FAIL' for x in checks.values())
    return {'status':'PASS' if passed else 'FAIL','dimension':cfg.m,'high_precision_references':reference,
        'diagnostic_random_keys':cfg.diagnostic_random_keys,'seed':cfg.diagnostic_seed,
        'conditional_fixed_vector_checks':checks,'scope':'key-randomization implementation test, not fixed-key FPR certification'}


def binomial_ci(count,n):
    if n==0:return [0.,1.]
    return [0. if count==0 else float(beta.ppf(.025,count,n-count+1)),
            1. if count==n else float(beta.ppf(.975,count+1,n-count))]


def pilot(rows,vectors,bank,cfg,path):
    from .data import groups
    fields=['key_index','query_budget','group_index','response_count','vector_norm','cosine','p_value']
    csvrows=[];budgets={}
    if rows is None:
        write_csv(path,[],fields);return {'status':'MISSING'}
    for budget in cfg.query_budgets_diagnostic:
        batch,left=groups(rows,budget);n=len(batch)
        probabilities=[]
        for j in range(n):
            v=vectors[j*budget:(j+1)*budget].sum(0);norm=float(np.linalg.norm(v))
            ps=sphere_p(v,bank);cos=bank@v/norm if norm else np.zeros(len(bank))
            probabilities.append(ps)
            csvrows.extend(dict(key_index=i,query_budget=budget,group_index=j,response_count=budget,
                vector_norm=norm,cosine=float(cos[i]),p_value=float(ps[i])) for i in range(len(bank)))
        pvalues=np.asarray(probabilities).reshape(n,len(bank))
        per_key=[]
        for i in range(len(bank)):
            values=pvalues[:,i]
            rates={str(a):{'count':int((values<=a).sum()),'independent_groups':n,
                    'rate':float((values<=a).mean()) if n else None,
                    'CI95':binomial_ci(int((values<=a).sum()),n)} for a in [.01,.0001]}
            per_key.append({'key_index':i,'rates':rates,'pvalue_histogram':np.histogram(values,bins=np.linspace(0,1,11))[0].tolist()})
        summary={}
        for a in [.01,.0001]:
            rates=[k['rates'][str(a)]['rate'] for k in per_key]
            summary[str(a)]={'worst':max(rates) if n else None,'p90':float(np.quantile(rates,.9)) if n else None}
        budgets[str(budget)]={'independent_groups':n,'discarded_remainder':left,'per_key':per_key,
            'across_keys':summary,'tiny_sample_illustration':budget==1000,
            'warning':'Groups, not keys or resampled overlapping responses, are the independent units; too few to certify 1% fixed-key FPR.'}
    write_csv(path,csvrows,fields)
    return {'status':'DESCRIPTIVE_ONLY','budgets':budgets,'pvalue_histogram_edges':np.linspace(0,1,11).tolist()}


def clustered_summary(values,prompt_ids,cfg):
    labels,ix=np.unique(prompt_ids,return_inverse=True)
    totals=np.bincount(ix,weights=values);counts=np.bincount(ix)
    rng=np.random.default_rng(cfg.bootstrap_seed)
    means=[]
    for _ in range(cfg.bootstrap_replicates):
        pick=rng.integers(len(labels),size=len(labels))
        means.append(totals[pick].sum()/counts[pick].sum())
    return {'mean':float(np.mean(values)), 'CI95_prompt_cluster_bootstrap':np.quantile(means,[.025,.975]).tolist(),
        'bootstrap_SE':float(np.std(means,ddof=1)),'contexts':len(values),'independent_prompts':len(labels),
        'bootstrap_replicates':cfg.bootstrap_replicates,'bootstrap_seed':cfg.bootstrap_seed}
