"""Independent FP64 diagnostic for the historical two-context Step C failure.

No tolerance/configuration tuning, vocabulary filtering, or favorable h selection.
Only readout arithmetic is promoted; parameter values and native planes are fixed.
"""
import numpy as np
import torch
from .math import rotate
from .readout import RotationReadout


def policy():
    return {'version':1,'fd_steps':[.002,.005,.01],
        'fd_atol':.003,'fd_rtol':.03,
        'autograd_atol':1e-9,'autograd_rtol':1e-7,
        'richardson_atol':1e-7,'richardson_rtol':1e-5,
        'richardson_pairs':[[.002,.005],[.005,.01]],
        'roundoff_error_improvement_min':10.,
        'roundoff_analytic_error_fraction_max':.1,
        'scaling':'abs(candidate-reference)/(original_atol+original_rtol*abs(analytic_FP64))',
        'acceptance':'All 32 directions, both exact historical contexts, all vocabulary tokens; no subset selection'}


def error_rows(reference,candidate,direction,method,atol,rtol,gate_atol=None,gate_rtol=None):
    reference,candidate=np.asarray(reference,dtype=np.float64),np.asarray(candidate,dtype=np.float64)
    err=np.abs(candidate-reference)
    scaled=err/(atol+rtol*np.abs(reference))
    ga=atol if gate_atol is None else gate_atol
    gr=rtol if gate_rtol is None else gate_rtol
    strict=err/(ga+gr*np.abs(reference))
    rows=[]
    for i in range(len(reference)):
        a=int(np.argmax(err[i]));s=int(np.argmax(scaled[i]))
        rows.append({'method':method,'context_index':i,'direction_index':direction,'direction_one_based':direction+1,
            'max_abs':float(err[i].max()),'p99_abs':float(np.quantile(err[i],.99)),
            'median_abs':float(np.median(err[i])),
            'max_scaled':float(scaled[i].max()),'p99_scaled':float(np.quantile(scaled[i],.99)),
            'median_scaled':float(np.median(scaled[i])),
            'acceptance_scaled_max':float(strict[i].max()),
            'worst_abs_token_id':a,'worst_scaled_token_id':s,
            'worst_abs_context_index':i,'worst_scaled_context_index':i,
            'analytic_at_worst_abs':float(reference[i,a]),'candidate_at_worst_abs':float(candidate[i,a])})
    ca,ta=np.unravel_index(np.argmax(err),err.shape)
    cs,ts=np.unravel_index(np.argmax(scaled),scaled.shape)
    rows.append({'method':method,'context_index':-1,'direction_index':direction,'direction_one_based':direction+1,
        'max_abs':float(err.max()),'p99_abs':float(np.quantile(err,.99)),
        'median_abs':float(np.median(err)),
        'max_scaled':float(scaled.max()),'p99_scaled':float(np.quantile(scaled,.99)),
        'median_scaled':float(np.median(scaled)),'acceptance_scaled_max':float(strict.max()),
        'worst_abs_token_id':int(ta),'worst_scaled_token_id':int(ts),
        'worst_abs_context_index':int(ca),'worst_scaled_context_index':int(cs),
        'analytic_at_worst_abs':float(reference[ca,ta]),'candidate_at_worst_abs':float(candidate[ca,ta])})
    return rows


def _clone64(engine):
    return RotationReadout(engine.weight.to(torch.float64),engine.gain.to(torch.float64),engine.eps,
        engine.p.to(torch.float64),engine.q.to(torch.float64),engine.tau,
        None if engine.bias is None else engine.bias.to(torch.float64))


def precision_diagnostic(engine,residuals,cfg,tokenizer=None,metadata=None):
    """Compare full_scores with independent differentiation of tail(rotate()).

    Original readout stays FP32. FP64 uses its exact floating parameter/plane
    values promoted without fitting, reorthogonalizing or changing directions.
    """
    settings=policy()
    if cfg.finite_difference_atol!=settings['fd_atol'] or cfg.finite_difference_rtol!=settings['fd_rtol']:
        raise ValueError('Frozen FP32 tolerances differ from predeclared diagnostic policy')
    if engine.weight.dtype!=torch.float32:raise ValueError('Historical readout must be native FP32')
    if engine.p.shape[1]!=cfg.m or len(residuals)!=2:raise ValueError('Diagnostic requires exactly two contexts and every frozen direction')
    promoted=_clone64(engine)
    r32=torch.as_tensor(residuals,device=engine.weight.device,dtype=torch.float32)
    r=r32.to(torch.float64)
    with torch.no_grad():
        lp64,phi64=promoted.full_scores(r)
        _,phi32=engine.full_scores(r32)
    reference=phi64.detach().cpu().numpy()
    rows=[]
    # Independent readout derivative: autograd differentiates normalization,
    # rotation and log-softmax. It never calls full_scores/observed/components.
    def independent_logp(a):
        return torch.log_softmax(promoted.tail(rotate(r,promoted.p,promoted.q,a))/cfg.tau,-1)
    for j in range(cfg.m):
        print('FP64 tangent diagnostic direction',j+1,'/',cfg.m,flush=True)
        unit=torch.zeros(cfg.m,device=r.device,dtype=r.dtype);unit[j]=1
        with torch.enable_grad():
            _,auto=torch.func.jvp(independent_logp,(torch.zeros_like(unit),),(unit,))
        auto=auto.detach().cpu().numpy()
        rows.extend(error_rows(reference[:,:,j],auto,j,'autograd_fp64',settings['fd_atol'],settings['fd_rtol'],
            settings['autograd_atol'],settings['autograd_rtol']))
        rows.extend(error_rows(reference[:,:,j],phi32[:,:,j].detach().double().cpu().numpy(),j,'analytic_fp32',
            settings['fd_atol'],settings['fd_rtol']))
        fd64={}
        for h in settings['fd_steps']:
            with torch.no_grad():
                plus=independent_logp(h*unit);minus=independent_logp(-h*unit)
                fd=((plus-minus)/(2*h)).cpu().numpy()
                a32=(h*unit).float()
                plus32=torch.log_softmax(engine.tail(rotate(r32,engine.p,engine.q,a32))/cfg.tau,-1)
                minus32=torch.log_softmax(engine.tail(rotate(r32,engine.p,engine.q,-a32))/cfg.tau,-1)
                fd32=((plus32-minus32)/(2*h)).double().cpu().numpy()
            fd64[h]=fd
            rows.extend(error_rows(reference[:,:,j],fd,j,f'finite_difference_fp64_h{h}',settings['fd_atol'],settings['fd_rtol']))
            rows.extend(error_rows(reference[:,:,j],fd32,j,f'finite_difference_fp32_h{h}',settings['fd_atol'],settings['fd_rtol']))
        for a,b in settings['richardson_pairs']:
            extrapolated=(b*b*fd64[a]-a*a*fd64[b])/(b*b-a*a)
            rows.extend(error_rows(reference[:,:,j],extrapolated,j,f'richardson_fp64_{a}_{b}',
                settings['fd_atol'],settings['fd_rtol'],settings['richardson_atol'],settings['richardson_rtol']))
    finite=all(np.isfinite(row[k]) for row in rows for k in ['max_abs','max_scaled','acceptance_scaled_max'])
    auto_rows=[x for x in rows if x['method']=='autograd_fp64']
    fd_rows=[x for x in rows if x['method'].startswith('finite_difference_fp64')]
    rich_rows=[x for x in rows if x['method'].startswith('richardson_fp64')]
    auto_ok=max(x['acceptance_scaled_max'] for x in auto_rows)<1
    fd_ok=max(x['acceptance_scaled_max'] for x in fd_rows)<1
    rich_ok=max(x['acceptance_scaled_max'] for x in rich_rows)<1
    def maximum(method):return max(x['max_abs'] for x in rows if x['method']==method)
    fp32_error=maximum('finite_difference_fp32_h0.002')
    fp64_error=maximum('finite_difference_fp64_h0.002')
    analytic32_error=maximum('analytic_fp32')
    ratio=fp32_error/max(fp64_error,np.finfo(float).tiny)
    mean_zero=float((lp64.exp()[:,:,None]*phi64).sum(1).abs().max().cpu())
    fp64_ok=finite and auto_ok and fd_ok and rich_ok and mean_zero<cfg.mean_zero_atol
    roundoff=fp64_ok and ratio>=settings['roundoff_error_improvement_min'] and analytic32_error<=settings['roundoff_analytic_error_fraction_max']*fp32_error
    root_cause=('FP32_FINITE_DIFFERENCE_ROUNDOFF' if roundoff else
        'FP64_VALIDATED_BUT_FP32_CAUSE_INCONCLUSIVE' if fp64_ok else 'UNRESOLVED_FORMULA_OR_NUMERICAL_ERROR')
    if metadata is not None:
        for row in rows:
            i=row['worst_abs_context_index'];row.update(prompt_id=str(metadata['prompt_ids'][i]),
                prediction_position=int(metadata['positions'][i]),observed_target_token_id=int(metadata['targets'][i]))
    if tokenizer is not None:
        for row in rows:
            for tag in ['worst_abs','worst_scaled']:
                token=row[tag+'_token_id']
                row[tag+'_token_text']=tokenizer.decode([token],skip_special_tokens=False,clean_up_tokenization_spaces=False)
    worst=max(rows,key=lambda x:x['max_abs'])
    direction23=[x for x in rows if x['direction_index']==22]
    report={'status':'PASS' if roundoff else 'NO_GO','continuation_allowed':bool(roundoff),
        'root_cause':root_cause,'fp64_checks_passed':bool(fp64_ok),'policy':settings,
        'checks':{'finite':finite,'analytic_vs_autograd':auto_ok,'all_FD_steps':fd_ok,
            'both_Richardson_pairs':rich_ok,'mean_zero_max':mean_zero},
        'contexts':2,'directions':cfg.m,'vocabulary_size':int(reference.shape[1]),
        'parameter_values':'Exact native FP32 readout and plane values promoted to FP64; no fitting or reorthogonalization',
        'pooled_context_index':-1,
        'FP32_FD_h002_max_abs':fp32_error,'FP64_FD_h002_max_abs':fp64_error,
        'FP32_analytic_vs_FP64_max_abs':analytic32_error,'FD_error_improvement_ratio':ratio,
        'worst_error_entry':worst,'direction23_entries':direction23,
        'explanation':'Subtracting nearby FP32 log probabilities and dividing by 2h amplifies rounding; FP64 and Richardson independently test the derivative. Classification requires all strict gates, not a tolerance increase.'}
    return report,rows
