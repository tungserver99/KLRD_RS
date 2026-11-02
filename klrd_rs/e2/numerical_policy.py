"""Numerical-only acceptance, independent of scientific ownership thresholds.

Native calibration is a bounded engineering control, not a statistical guarantee
about all future prefixes or a certification of ownership/FPR.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import hashlib,math
from ..io import digest,write_json,read_json


def _finite_nonnegative(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and value>=0


class NumericalBudgetExceeded(ValueError):
    """Optional diagnostic budget exceeded; not a malformed/runtime artifact."""


@dataclass(frozen=True)
class NumericalPolicySpec:
    version: str='native_fp32_distribution_v3'
    unit_roundoff: float=2.**-24
    calibration_prompts: int=32
    holdout_prompts: int=32
    historical_excluded_prompts: int=10
    length_quartiles: int=4
    native_prefixes_per_prompt: int=3
    linear_headroom: float=2.
    quadratic_headroom: float=4.
    raw_probability_atol: float=2e-6

    def absolute_caps(self):
        root=math.sqrt(self.unit_roundoff)
        return {'total_variation':root,'KL_left_to_right':2*self.unit_roundoff,
            'KL_right_to_left':2*self.unit_roundoff,
            'probability_weighted_abs_logprob_error':2*root,
            # Every token event satisfies |p_i-q_i| <= TV(p,q). The same
            # probability budget is therefore consistent with the TV guardrail.
            'significant_token_max_abs_probability_error':root}

    def arithmetic_floors(self):
        u=self.unit_roundoff
        return {'total_variation':64*u,'KL_left_to_right':2*(64*u)**2,
            'KL_right_to_left':2*(64*u)**2,
            'probability_weighted_abs_logprob_error':128*u,
            'significant_token_max_abs_probability_error':64*u}

    def conditional_path_tv_bound(self,horizon):
        # KL chain rule + Pinsker, ONLY under uniform conditional KL bounds.
        return min(1.,math.sqrt(horizon*self.absolute_caps()['KL_left_to_right']/2))

    def resolved(self):
        return dict(asdict(self),absolute_caps=self.absolute_caps(),
            arithmetic_floors=self.arithmetic_floors(),conditional_length512_path_TV=self.conditional_path_tv_bound(512),
            scope='Engineering budgets fixed independently of previous failures; finite controls do not prove uniform conditional KL or scientific guarantees')


def select_contexts(requests,spec,extra_excluded_ids=()):
    ids=[r['prompt_id'] for r in requests]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate numerical candidate prompt IDs')
    previous=sorted(set(extra_excluded_ids))
    excluded=list(dict.fromkeys(ids[:spec.historical_excluded_prompts]+previous))
    blocked=set(excluded)
    candidates=[row for row in requests if row['prompt_id'] not in blocked]
    def order(row):return hashlib.sha256((spec.version+'|'+row['prompt_id']).encode()).hexdigest()
    candidates=sorted(candidates,key=lambda r:(len(r['prompt_ids']),order(r)))
    bins=spec.length_quartiles
    if spec.calibration_prompts%bins or spec.holdout_prompts%bins:raise ValueError('Numerical role counts must divide length strata')
    per_cal=spec.calibration_prompts//bins;per_hold=spec.holdout_prompts//bins
    calibration=[];holdout=[]
    for index in range(bins):
        bucket=candidates[index*len(candidates)//bins:(index+1)*len(candidates)//bins]
        bucket=sorted(bucket,key=order)
        if len(bucket)<per_cal+per_hold:raise ValueError('Insufficient unused frozen prompts for fresh calibration/holdout after historical exclusions')
        calibration.extend(r['prompt_id'] for r in bucket[:per_cal])
        holdout.extend(r['prompt_id'] for r in bucket[per_cal:per_cal+per_hold])
    calibration=[calibration[bin_index*per_cal+i] for i in range(per_cal) for bin_index in range(bins)]
    holdout=[holdout[bin_index*per_hold+i] for i in range(per_hold) for bin_index in range(bins)]
    return {'calibration_prompt_ids':calibration,'holdout_prompt_ids':holdout,
        'excluded_historical_prompt_ids':excluded,'excluded_previous_policy_prompt_ids':previous,
        'selection':'Fixed SHA256 order within four length quartiles, before native/keyed output inspection'}


def historical_policy_contexts(root,spec,code_hashes,current_output):
    """Exclude previously declared policy data using design metadata ONLY.

    No metric, keyed outcome or ownership score is read for this selection.
    Same-policy/same-code replays may keep their deterministic split. Changed
    policy/code must use fresh numerical data, while E2's scientific pool stays.
    """
    root=Path(root);current=(Path(current_output)/'tests/numerical_policy_spec.json').resolve()
    paths=list((root/'experiment2_rs').glob('*/tests/numerical_policy_spec.json'))
    paths+=list((root/'verification/vllm_server').glob('*/tests/numerical_policy_spec.json'))
    excluded=set()
    for path in paths:
        if path.resolve()==current:continue
        old=read_json(path)
        if old.get('spec')==spec.resolved() and old.get('code_sha256')==code_hashes:continue
        selection=old.get('selection',{})
        for role in ['calibration_prompt_ids','holdout_prompt_ids']:
            values=selection.get(role)
            if not isinstance(values,list) or any(not isinstance(value,str) for value in values):
                raise ValueError('Invalid historical numerical design; cannot ensure fresh holdout: '+str(path))
            excluded.update(values)
    return sorted(excluded)


def distribution_metrics(left,right,probability_atol=2e-6):
    """One full-vocabulary pair. Raw normalization checked before FP64 audit.

    FP64 is used for probability measurement only, never generation Linear.
    """
    import torch
    left=left.reshape(-1);right=right.reshape(-1)
    if left.shape!=right.shape or left.numel()<2:raise ValueError('Mismatched numerical comparison vocabulary')
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():raise ValueError('Nonfinite comparison log probabilities')
    a=left.double();b=right.double()
    normalization=max(abs(float(a.exp().sum())-1.),abs(float(b.exp().sum())-1.))
    if normalization>probability_atol:raise ValueError('Unnormalized comparison distribution')
    a=torch.log_softmax(a,-1);b=torch.log_softmax(b,-1)
    p=a.exp();q=b.exp();difference=(a-b).abs();raw=(left.double()-right.double()).abs()
    cutoff=1./left.numel()
    mask=(p>=cutoff)|(q>=cutoff)
    mask[p.argmax()]=True;mask[q.argmax()]=True
    indices=torch.nonzero(mask,as_tuple=False).flatten()
    lp_index=int(indices[difference[mask].argmax()])
    probability_error=(p-q).abs()
    probability_index=int(indices[probability_error[mask].argmax()])
    kl_pq=float((p*(a-b)).sum());kl_qp=float((q*(b-a)).sum())
    if min(kl_pq,kl_qp)<-1e-12:raise ValueError('Materially negative diagnostic KL')
    return {'total_variation':float(.5*(p-q).abs().sum()),
        'KL_left_to_right':max(0.,kl_pq),'KL_right_to_left':max(0.,kl_qp),
        'probability_weighted_abs_logprob_error':float((.5*(p+q)*difference).sum()),
        'weighted_abs_logprob_left':float((p*difference).sum()),'weighted_abs_logprob_right':float((q*difference).sum()),
        'significant_token_max_abs_logprob_error':float(difference[mask].max()),
        'significant_token_max_abs_probability_error':float(probability_error[mask].max()),
        'significant_worst_logprob_token_id':lp_index,
        'significant_worst_logprob_probability_left':float(p[lp_index]),
        'significant_worst_logprob_probability_right':float(q[lp_index]),
        'significant_worst_logprob_abs_probability_error':float(probability_error[lp_index]),
        'significant_worst_probability_token_id':probability_index,
        'significant_token_p99_abs_logprob_error':float(torch.quantile(difference[mask],.99)),
        'significant_token_count':int(mask.sum()),'significant_probability_cutoff':cutoff,
        'significant_mass_left':float(p[mask].sum()),'significant_mass_right':float(q[mask].sum()),
        'max_abs_logprob_error':float(raw.max()),'p99_abs_logprob_error':float(torch.quantile(raw,.99)),
        'raw_probability_normalization_error':normalization}


def fit_native_envelope(rows,spec):
    if not rows or any(row.get('mode')!='native_original' for row in rows):
        raise ValueError('Calibration requires exclusively native original vLLM controls')
    caps=spec.absolute_caps();floors=spec.arithmetic_floors();limits={}
    for key,cap in caps.items():
        values=[row.get(key) for row in rows]
        if any(not _finite_nonnegative(value) for value in values):raise ValueError('Invalid native calibration metric: '+key)
        if max(values)>cap:
            raise NumericalBudgetExceeded(f'Native calibration exceeds independent absolute guardrail: {key}; observed={max(values):.9g}, cap={cap:.9g}, policy={spec.version}')
        factor=spec.quadratic_headroom if key.startswith('KL_') else spec.linear_headroom
        limits[key]=min(cap,factor*max(values)+floors[key])
    return limits


def assess(row,envelope,spec):
    failures=[]
    for key,cap in spec.absolute_caps().items():
        value=row.get(key);bound=envelope.get(key)
        if not _finite_nonnegative(value) or not _finite_nonnegative(bound):
            failures.append({'metric':key,'reason':'missing/nonfinite/negative'})
        elif value>cap or value>bound:
            failures.append({'metric':key,'value':value,'native_envelope':bound,'absolute_guardrail':cap})
    return {'accepted':not failures,'failures':failures}


def freeze_calibration(path,binding,envelopes,evidence=None):
    from filelock import FileLock
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(path)+'.lock'):
        if path.exists():raise ValueError('Numerical calibration already frozen; refuse overwrite/refit')
        body={'schema':2,'binding':binding,'envelopes':envelopes}
        if evidence is not None:body['native_evidence']=evidence
        write_json(path,dict(body,payload_sha256=digest(body)))


def read_calibration(path,binding):
    value=read_json(path)
    body={key:item for key,item in value.items() if key!='payload_sha256'}
    if value.get('payload_sha256')!=digest(body) or value.get('binding')!=binding:
        raise ValueError('Numerical calibration fingerprint/code/engine/data changed')
    return value
