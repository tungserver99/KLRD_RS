"""Disjoint response-group inference; wrong/clean reuse is never extra trials."""
import numpy as np
from ..math import sphere_p
from ..diagnostics import binomial_ci
from ..io import write_csv


def wrong_indices(owner):return [(owner+i)%64 for i in range(1,21)]


def identify(cosines):
    x=np.asarray(cosines);winners=np.flatnonzero(x==x.max())
    return {'correct':bool(len(winners)==1 and winners[0]==0),'tie':bool(len(winners)>1)}


def decide(complete,valid_sampler,conservative_counts,nominal_counts,concerns):
    if not complete or not valid_sampler:return 'E2_INCONCLUSIVE'
    if sum(conservative_counts)>=32 and sum(n>=4 for n in conservative_counts)>=6 and not any(concerns.values()):
        return 'E2_PROMISING_FOR_LORA_REVIEW'
    if sum(conservative_counts)<8 and sum(nominal_counts)<8:return 'E2_NO_GO'
    return 'E2_PARTIAL'


def key_name(i):return f'e1_validation_{i:02}'


def _batch(owner,group,bank,records,vectors):
    ids=group['prompt_ids'];selected=[records[i] for i in ids if i in records]
    complete=len(selected)==group['Q']
    base={'owner_key_id':key_name(owner),'group_id':group['group_id'],'Q':group['Q'],
        'n_prompt_ids':len(selected),'n_eligible_response_tokens':sum(r['eligible_tokens'] for r in selected),
        'roundtrip_mismatch_count':sum(not r['roundtrip_equal'] for r in selected),
        'data_status':'COMPLETE' if complete else 'INCOMPLETE'}
    if not complete:return dict(base,score_norm=None,cosine_T=None,p_spherical=None,detect_at_0_01=None,
        detect_at_1e_4=None,zero_vector=None),None
    v=np.sum([vectors[i] for i in ids],axis=0,dtype=np.float64);norm=float(np.linalg.norm(v))
    t=bank[owner]@v/norm if norm else 0.;p=float(sphere_p(v,bank[owner]))
    return dict(base,score_norm=norm,cosine_T=float(t),p_spherical=p,detect_at_0_01=p<=.01,
        detect_at_1e_4=p<=.0001,zero_vector=norm==0),v


def _distribution(rows,prefix):
    if not rows:return {prefix+'_T_median':None,prefix+'_p_median':None}
    t=[r['cosine_T'] for r in rows];p=[r['p_spherical'] for r in rows]
    return {prefix+'_T_median':float(np.median(t)),prefix+'_T_p05':float(np.quantile(t,.05)),
        prefix+'_T_p95':float(np.quantile(t,.95)),prefix+'_p_median':float(np.median(p)),
        prefix+'_p_p01':float(np.quantile(p,.01)),prefix+'_p_p95':float(np.quantile(p,.95))}


def analyze_groups(groups,bank,observations,out,cfg):
    """observations[label] = (promptID->record, promptID->public score vector)."""
    owner_rows=[];clean_rows=[];wrong_rows=[];identifications=[];summaries=[];wrong_rate_rows=[]
    clean,cv=observations.get('clean',({},{}))
    for owner in cfg.owners:
        records,vectors=observations.get(str(owner),({},{}))
        for group in groups:
            row,v=_batch(owner,group,bank,records,vectors);owner_rows.append(row)
            cr,_=_batch(owner,group,bank,clean,cv);clean_rows.append(cr)
            if v is None:continue
            wrong=wrong_indices(owner);norm=np.linalg.norm(v)
            cos=bank[[owner]+wrong]@v/norm if norm else np.zeros(21)
            identification=identify(cos)
            identifications.append({'owner_key_id':key_name(owner),'group_id':group['group_id'],
                'Q':group['Q'],'correct_identification':identification['correct'],'tie':identification['tie'],
                'correct_cosine':float(cos[0]),'highest_cosine':float(cos.max())})
            for j,k in enumerate(wrong):
                p=float(sphere_p(v,bank[k]))
                wrong_rows.append({**row,'wrong_key_id':key_name(k),'cosine_T':float(cos[j+1]),
                    'p_spherical':p,'detect_at_0_01':p<=.01,'detect_at_1e_4':p<=.0001})
        for Q in cfg.Q:
            valid=lambda rows:[r for r in rows if r['owner_key_id']==key_name(owner) and r['Q']==Q and r.get('data_status','COMPLETE')=='COMPLETE']
            own,neg,wrong=valid(owner_rows),valid(clean_rows),valid(wrong_rows)
            ident=[r for r in identifications if r['owner_key_id']==key_name(owner) and r['Q']==Q]
            s={'owner_key_id':key_name(owner),'Q':Q,'n_prompts':len(own)*Q,
                'n_disjoint_groups':len(own),'planned_disjoint_groups':cfg.prompt_count//Q,
                'n_observed_tokens':sum(r['n_eligible_response_tokens'] for r in own),
                'clean_unique_prompt_count':len(neg)*Q,'clean_disjoint_groups':len(neg),
                'identification_count':sum(r['correct_identification'] for r in ident),
                'identification_trials':len(ident),'identification_ties':sum(r['tie'] for r in ident),
                'identification_rate':sum(r['correct_identification'] for r in ident)/len(ident) if ident else None}
            for k in wrong_indices(owner):
                subset=[r for r in wrong if r['wrong_key_id']==key_name(k)]
                wrong_rate_rows.append({'owner_key_id':key_name(owner),'wrong_key_id':key_name(k),'Q':Q,
                    'n_disjoint_groups':len(subset),'nominal_count':sum(r['detect_at_0_01'] for r in subset),
                    'nominal_rate':sum(r['detect_at_0_01'] for r in subset)/len(subset) if subset else None,
                    'conservative_count':sum(r['detect_at_1e_4'] for r in subset),
                    'conservative_rate':sum(r['detect_at_1e_4'] for r in subset)/len(subset) if subset else None,
                    'dependence':'Same owner outputs reused; wrong keys are correlated controls, not extra deployments'})
            for a,field in [('nominal','detect_at_0_01'),('conservative','detect_at_1e_4')]:
                for label,rows in [('owner',own),('clean',neg)]:
                    count=sum(r[field] for r in rows);n=len(rows);ci=binomial_ci(count,n)
                    s.update({f'{label}_{a}_count':count,f'{label}_{a}_rate':count/n if n else None,
                        f'{label}_{a}_CI95_low':ci[0],f'{label}_{a}_CI95_high':ci[1]})
                rates=[]
                for k in wrong_indices(owner):
                    subset=[r for r in wrong if r['wrong_key_id']==key_name(k)]
                    if subset:rates.append(sum(r[field] for r in subset)/len(subset))
                for quant,fn in [('mean',np.mean),('median',np.median),('p90',lambda v:np.quantile(v,.9)),('worst',np.max)]:
                    s[f'wrong_{a}_{quant}']=float(fn(rates)) if rates else None
            s.update(_distribution(own,'owner'));s.update(_distribution(neg,'clean'));s.update(_distribution(wrong,'wrong'))
            summaries.append(s)
    def save(name,rows):
        if rows:write_csv(out/'scores'/name,rows,list(dict.fromkeys(k for r in rows for k in r)))
        else:write_csv(out/'scores'/name,[],['data_status'])
    for name,rows in [('batch_owner_scores.csv',owner_rows),('batch_clean_scores.csv',clean_rows),
        ('batch_wrong_key_scores.csv',wrong_rows),('closed_set_identification.csv',identifications),
        ('per_wrong_key_rates.csv',wrong_rate_rows),
        ('per_key_summary.csv',summaries)]:save(name,rows)
    complete=all(s['n_disjoint_groups']==s['planned_disjoint_groups'] and s['clean_disjoint_groups']==s['planned_disjoint_groups'] for s in summaries)
    conservative=[s['owner_conservative_count'] for s in summaries if s['Q']==1000]
    nominal=[s['owner_nominal_count'] for s in summaries if s['Q']==1000]
    concerns={'wrong_key_concern':any((s['wrong_conservative_mean'] or 0)>cfg.wrong_mean_concern or
            (s['wrong_nominal_mean'] or 0)>cfg.wrong_mean_concern for s in summaries if s['Q']==1000),
        'clean_null_concern':any(s['clean_conservative_count']>=cfg.clean_Q1000_concern_count for s in summaries if s['Q']==1000)}
    pooled=[]
    for Q in cfg.Q:
        rows=[s for s in summaries if s['Q']==Q]
        pooled.append({'Q':Q,'description':'Descriptive only; shared prompts and clean responses, no pooled independent-trial CI',
            'owner_group_scores':sum(s['n_disjoint_groups'] for s in rows),
            'owner_conservative_detected':sum(s['owner_conservative_count'] for s in rows),
            'owner_nominal_detected':sum(s['owner_nominal_count'] for s in rows),
            'clean_unique_responses':len(clean),'clean_true_disjoint_groups_max':len(clean)//Q,
            'worst_clean_nominal_rate':max((s['clean_nominal_rate'] or 0) for s in rows),
            'p90_clean_nominal_rate':float(np.quantile([(s['clean_nominal_rate'] or 0) for s in rows],.9)),
            'worst_clean_conservative_rate':max((s['clean_conservative_rate'] or 0) for s in rows),
            'p90_clean_conservative_rate':float(np.quantile([(s['clean_conservative_rate'] or 0) for s in rows],.9))})
    return {'complete':complete,'per_key':summaries,'pooled':pooled,
        'conservative_counts':conservative,'nominal_counts':nominal,'concerns':concerns}
