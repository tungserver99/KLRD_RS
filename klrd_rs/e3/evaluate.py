"""E3 ordinary-model returned-text evaluation; historical experiments are read-only."""
from pathlib import Path
import gc
import numpy as np
from ..io import read_json, read_jsonl, write_json, write_csv, digest, sha256
from ..e2.analysis import wrong_indices, identify
from ..diagnostics import binomial_ci
from ..e2.verifier import SphericalBlackBoxVerifier
from ..e2b.source import immutable_rows
from ..e2b.scoring import load_score_batch, save_score_batch
from .provenance import get_frozen, file_manifest


def csv(path, rows):
    write_csv(path, rows, list(dict.fromkeys(k for r in rows for k in r)) if rows else ['status'])


def verify_merged(state):
    out=state['out']; path=out/'merged_student'
    receipt=read_json(out/'provenance/merged_manifest.json')
    if (receipt['identity']!=state['identity'] or not path.is_dir()
            or receipt['files']!=file_manifest(path)):
        raise ValueError('Missing/changed merged checkpoint; run explicit training/merge first')
    config=read_json(path/'config.json')
    if (config.get('model_type')!='qwen2' or config.get('auto_map')
            or any('KLRD' in a for a in config.get('architectures',[]))):
        raise ValueError('Student must be an ordinary Qwen2 checkpoint')
    state['verified_merged_file_manifest']={'path':str(path.resolve()),'files':receipt['files']}
    return receipt


def frozen_e2(state):
    """Load precisely the teacher0 and clean group0 canonical caches, not E2 inference."""
    from ..e2.config import E2Config
    from ..e2.anchor import RECEIPT_SHA256
    root=state['root']; source=root/state['cfg'].e2_run; cfg=E2Config()
    manifest=read_json(source/'provenance/prompt_manifest.json')
    identity=read_json(source/'private/jobs/identity.json')
    environment=read_json(source/'provenance/engine_manifest.json')
    if (identity['config']!=cfg.resolved() or manifest['E2_config']!=cfg.resolved()
            or identity['source_receipt_sha256']!=RECEIPT_SHA256
            or manifest['input_receipt_sha256']!=RECEIPT_SHA256):
        raise ValueError('Frozen E2 scientific identity differs')
    design=digest({'cohort':manifest,'config':cfg.resolved(),'source':RECEIPT_SHA256})
    if design!=identity['design_sha256']:raise ValueError('Frozen E2 design hash changed')
    expected_readout=read_json(root/'experiment1_rs_continuation/numerical/fp64_diagnostic.json')['readout_parameter_sha256']
    if (environment['readout_parameter_sha256']!=expected_readout
            or identity['environment']['readout_parameter_sha256']!=expected_readout
            or environment['generation']!=identity['environment']['generation']
            or environment.get('verification_linear_accumulation')!='float64'
            or environment.get('verification_attention_implementation')!='eager'):
        raise ValueError('Frozen E2 readout/runtime identity mismatch')
    if read_json(source/'report/summary.json').get('sampler_valid') is not True:
        raise ValueError('E2 source does not declare a valid sampler')
    large=[g for g in manifest['groups'] if g['Q']==1000 and g['group_index']==0]
    small=[g for g in manifest['groups'] if g['Q']==100 and g['group_index']==0]
    if len(large)!=1 or len(small)!=10:raise ValueError('Missing frozen group0 partitions')
    ids=large[0]['prompt_ids']
    if len(ids)!=1000 or len(set(ids))!=1000 or [p for g in small for p in g['prompt_ids']]!=ids:
        raise ValueError('Frozen E2 nested partition mismatch')
    prompts=read_jsonl(source/'private/prompts.jsonl'); lookup={r['prompt_id']:r for r in prompts}
    if len(lookup)!=len(prompts) or not set(ids)<=set(lookup):raise ValueError('Missing private E2 prompts')
    observations={}; receipts={}
    for condition,label in [('0','teacher'),('clean','clean')]:
        records=[]; vectors=[]; tokens=[]
        for start in range(0,1000,cfg.batch_size):
            chunk=start//cfg.batch_size; batch_ids=ids[start:start+cfg.batch_size]
            name=f'primary_owner{condition}_g00_chunk{chunk:04}'
            job=read_json(source/'private/jobs'/(name+'.json'))
            expected=dict(owner=condition,group=0,chunk=chunk,prompt_ids=batch_ids,phase='primary',frozen_run_sha256=design)
            rows=job['records']; rh=digest(rows)
            if job['identity']!=expected or job['records_sha256']!=rh or [r['prompt_id'] for r in rows]!=batch_ids:
                raise ValueError('Frozen E2 job identity/order/hash mismatch: '+name)
            path=source/'private/scores'/(name+'.npz'); marker=read_json(path.with_suffix('.sha256.json'))
            if marker['sha256']!=sha256(path) or marker['records_sha256']!=rh or marker.get('primary_path')!='canonical_visible_text_only':
                raise ValueError('Frozen E2 canonical score binding mismatch: '+name)
            with np.load(path,allow_pickle=False) as arrays:
                v=arrays['vectors'].copy(); t=arrays['eligible_tokens'].copy(); p=arrays['prompt_ids'].tolist()
            if p!=batch_ids or v.shape!=(len(rows),32) or t.shape!=(len(rows),) or not np.isfinite(v).all() or (t<0).any():
                raise ValueError('Invalid E2 score cache: '+name)
            for row in rows:
                if (row.get('owner_key_id')!=condition or row.get('group_index')!=0
                        or row.get('chunk_index')!=chunk or row.get('phase')!='primary'
                        or not isinstance(row.get('response'),str)
                        or row['response_id']!=digest({'owner':condition,'prompt':row['prompt_id'],'seed':row['seed'],'tokens':row['generated_token_ids']})):
                    raise ValueError('Frozen E2 response identity mismatch: '+name)
                records.append(dict(row,prompt=lookup[row['prompt_id']]['prompt']))
            vectors.extend(v); tokens.extend(t); receipts[name]=marker['sha256']
        observations[label]=(records,np.array(vectors),np.array(tokens))
    groups=small+large
    for g in small:
        for q in [20,50]:
            for start in range(0,100,q):
                groups.append({'Q':q,'group_id':g['group_id']+f'_Q{q}_{start//q:02}',
                    'prompt_ids':g['prompt_ids'][start:start+q],'parent_group_id':g['group_id']})
    return {'ids':ids,'prompts':[lookup[p] for p in ids],'groups':groups,
        'observations':observations,'environment':environment,'cache_hashes':receipts,'design_sha256':design}


def score_rows(state, rows, label, environment):
    """Checkpoint each returned-text batch; exact unchanged FP64-linear E2 reference."""
    from ..e2b.scoring import scoring_runtime, get_adapter, verifier_rows
    shim=dict(state,manifest={'source_E2_environment':environment},frozen=get_frozen(state))
    runtime=scoring_runtime(shim)
    write_json(state['out']/f'provenance/{label}_verification_runtime.json',runtime)
    directory=state['out']/'private/eval_scores'/label
    adapter=verifier=None; vs=[]; ts=[]; hashes={}
    try:
        for start in range(0,len(rows),state['cfg'].score_batch_size):
            batch=rows[start:start+state['cfg'].score_batch_size]; name=f'batch_{start:05}'
            ids=[r['response_id'] for r in batch]
            identity={'run':state['identity'],'rows':digest(batch),'runtime':runtime}
            values=load_score_batch(directory,name,identity)
            if values is None:
                if adapter is None:adapter,verifier=get_adapter(shim)
                pairs=[dict(r,paraphrased_text=r['response']) for r in batch]
                visible=verifier_rows(pairs,adapter.tokenizer,adapter.model.config.max_position_embeddings)
                vectors,tokens=verifier.score_text_batch(visible)
                values={'vectors':vectors,'eligible_tokens':tokens,'response_ids':np.array(ids)}
                save_score_batch(directory,name,identity,values)
            if (set(values)!={'vectors','eligible_tokens','response_ids'} or values['response_ids'].tolist()!=ids
                    or values['vectors'].shape!=(len(batch),32) or values['eligible_tokens'].shape!=(len(batch),)
                    or not np.isfinite(values['vectors']).all() or (values['eligible_tokens']<0).any()):
                raise ValueError('Invalid E3 returned-text score cache')
            vs.extend(values['vectors']);ts.extend(values['eligible_tokens']);hashes[name]=sha256(directory/(name+'.npz'))
    finally:
        if adapter is not None:
            del adapter,verifier
            gc.collect()
            import torch
            torch.cuda.empty_cache()
    write_json(state['out']/f'provenance/{label}_scores_complete.json',{'identity':state['identity'],
        'rows_sha256':digest(rows),'count':len(rows),'batch_sha256':hashes,'runtime':runtime})
    return np.array(vs,dtype=np.float64),np.array(ts,dtype=np.int64)


def tables(state, observations, groups, directory):
    bank=get_frozen(state)['bank']; candidates=[0]+wrong_indices(0)
    scores=[]; identities=[]; per_response=[]
    for label,(records,vectors,tokens) in observations.items():
        index={r['prompt_id']:i for i,r in enumerate(records)}
        if len(index)!=len(records) or vectors.shape!=(len(records),32) or tokens.shape!=(len(records),):
            raise ValueError('Observation identity/shape mismatch')
        for i,r in enumerate(records):
            key=SphericalBlackBoxVerifier.key_scores(vectors[i],bank)
            per_response.append({'model':label,'response_hash':digest(r['response_id']),
                'eligible_tokens':int(tokens[i]),'zero_vector':key['zero_vector'],
                'cosine':float(key['cosine'][0]),'p_value':float(key['p'][0])})
        for group in groups:
            selected=[index[p] for p in group['prompt_ids']]
            if len(selected)!=group['Q']:raise ValueError('Incomplete frozen group')
            value=vectors[selected].sum(0,dtype=np.float64)
            key=SphericalBlackBoxVerifier.key_scores(value,bank)
            cos=key['cosine'][candidates]; result=identify(cos)
            identities.append({'model':label,'Q':group['Q'],'group_id':group['group_id'],
                'correct':result['correct'],'tie':result['tie'],
                'identified_key':None if result['tie'] else candidates[int(np.argmax(cos))]})
            for k in candidates:
                scores.append({'model':label,'Q':group['Q'],'group_id':group['group_id'],
                    'role':'owner' if k==0 else 'wrong','test_key':k,'responses':len(selected),
                    'eligible_tokens':int(tokens[selected].sum()),'zero_vector':key['zero_vector'],
                    'vector_norm':key['norm'],'cosine':float(key['cosine'][k]),'p_value':float(key['p'][k])})
    summaries=[]; wrong_rates=[]
    for label in observations:
        for q in sorted({g['Q'] for g in groups}):
            own=[r for r in scores if r['model']==label and r['Q']==q and r['test_key']==0]
            ident=[r for r in identities if r['model']==label and r['Q']==q]
            for alpha in [.01,.0001]:
                count=sum(r['p_value']<=alpha for r in own);n=len(own)
                lo,hi=binomial_ci(count,n) if q!=1000 else (None,None)
                rates=[]
                for k in wrong_indices(0):
                    selected=[r for r in scores if r['model']==label and r['Q']==q and r['test_key']==k]
                    c=sum(r['p_value']<=alpha for r in selected);rate=c/len(selected)
                    rates.append(rate)
                    kl,kh=binomial_ci(c,len(selected)) if q!=1000 else (None,None)
                    wrong_rates.append({'model':label,'Q':q,'alpha':alpha,'test_key':k,'groups':len(selected),
                        'detections':c,'rate':rate,'CI95_low':kl,'CI95_high':kh})
                summaries.append({'model':label,'Q':q,'alpha':alpha,'groups':n,'detections':count,
                    'rate':count/n,'CI95_low':lo,'CI95_high':hi,'wrong_mean':float(np.mean(rates)),
                    'wrong_worst':max(rates),'identification_rate':sum(r['correct'] for r in ident)/len(ident),
                    'identification_ties':sum(r['tie'] for r in ident),
                    'cosine_median':float(np.median([r['cosine'] for r in own])),
                    'p_median':float(np.median([r['p_value'] for r in own])),
                    'scope':'single descriptive aggregate; no group CI' if q==1000 else 'groups disjoint within Q; budgets nested/dependent'})
    directory=Path(directory)
    csv(directory/'per_key_group.csv',scores);csv(directory/'per_response.csv',per_response)
    csv(directory/'closed_set_identification.csv',identities);csv(directory/'clean_wrong_controls.csv',wrong_rates)
    csv(directory/'group_summary.csv',summaries)
    return summaries


def utility(observations):
    from ..e2.utility import text_quality
    result={}
    for label,(rows,_,tokens) in observations.items():
        lengths=[len(r['response'].split()) for r in rows]
        values=[text_quality(dict(r,
            native_visible_tokens=r.get('native_visible_tokens',int(tokens[i])),
            eos_terminated=bool(r.get('eos_terminated')),
            max_length_truncated=bool(r.get('max_length_truncated')),
            roundtrip_equal=r.get('roundtrip_equal',False))) for i,r in enumerate(rows)]
        result[label]={'responses':len(rows),'mean_words':float(np.mean(lengths)),
            'median_words':float(np.median(lengths)),'mean_visible_tokens':float(np.mean(tokens)),
            'generation_error_count':sum(r.get('error') is not None for r in rows),
            'empty_count':sum(not r['response'].strip() for r in rows),'very_short_under5_visible_tokens':sum(r['very_short'] for r in values),
            'severe_repeated_4gram_count':sum(r['loop'] for r in values),
            'repeated_4gram_rule':'Unchanged E2 text_quality: native_visible_tokens>=64 and repeated4 fraction>=.4',
            'mean_repeated4_fraction':float(np.mean([r['repeated4'] for r in values])),
            'visible_token_count_fallback':'Returned-text eligible tokens if native count unavailable',
            'eos_terminated_count':sum(bool(r.get('eos_terminated')) for r in rows),
            'max512_truncated_count':sum(bool(r.get('max_length_truncated')) for r in rows),
            'eos_metadata_available':sum('eos_terminated' in r for r in rows),
            'roundtrip_mismatch_count':sum(r.get('roundtrip_equal') is False for r in rows),
            'roundtrip_metadata_available':sum('roundtrip_equal' in r for r in rows)}
    return result


def checked_receipt(state, relative):
    receipt=read_json(state['out']/relative)
    if receipt['identity']!=state['identity']:raise ValueError('E3 receipt run identity changed')
    for name,expected in receipt['products_sha256'].items():
        if sha256(state['out']/name)!=expected:raise ValueError('E3 completed artifact checksum changed: '+name)
    return receipt


def run(state):
    from .data import load_eval_prompts
    from .generation import generate
    merged=verify_merged(state); source=frozen_e2(state); rows=load_eval_prompts(state)
    if [r['prompt_id'] for r in rows]!=source['ids'] or any(r['prompt']!=p['prompt'] for r,p in zip(rows,source['prompts'])):
        raise ValueError('Student evaluation prompts differ from frozen E2 group0')
    student=generate(state,rows,state['out']/'private/eval_responses.jsonl',model_path=state['out']/'merged_student',kind='student')
    baseline=read_json(state['out']/'training/baseline.json')
    if baseline.get('zero_initial_lora_B') is not True:
        raise ValueError('Missing mathematical zero-LoRA initialization evidence for fresh untrained control')
    untrained=generate(state,rows,state['out']/'private/untrained_responses.jsonl',model_path=None,kind='untrained')
    observations=dict(source['observations'])
    for label,generated in [('student',student),('untrained_zero_lora',untrained)]:
        if [r['prompt_id'] for r in generated]!=source['ids']:raise ValueError('Generated population changed')
        texts=[dict(r,prompt=p['prompt']) for r,p in zip(generated,rows)]
        vectors,tokens=score_rows(state,texts,label,source['environment'])
        observations[label]=(texts,vectors,tokens)
    summaries=tables(state,observations,source['groups'],state['out']/'scores')
    write_json(state['out']/'utility/generation_summary.json',utility(observations))
    gap=[]
    for q in [20,50,100,1000]:
        for alpha in [.01,.0001]:
            teacher=next(r for r in summaries if r['model']=='teacher' and r['Q']==q and r['alpha']==alpha)
            child=next(r for r in summaries if r['model']=='student' and r['Q']==q and r['alpha']==alpha)
            gap.append({'Q':q,'alpha':alpha,'teacher_rate':teacher['rate'],'student_rate':child['rate'],
                'student_minus_teacher_pp':100*(child['rate']-teacher['rate']),
                'pairing':'same prompt IDs; newly generated student response, different random sample'})
    csv(state['out']/'scores/teacher_student_gap.csv',gap)
    from ..e2b.quality import _atomic_text, _quoted
    examples=['# Private deterministic comparison examples','',
        'First ten frozen group0 IDs, selected before scoring. Different random samples; no quality filtering.','']
    for i,p in enumerate(rows[:10]):
        examples+=['## '+digest(p['prompt_id'])[:16],_quoted(p['prompt'])]
        for label in ['clean','teacher','student']:
            examples += [label,_quoted(observations[label][0][i]['response'])]
    _atomic_text(state['out']/'private/examples.md','\n'.join(examples)+'\n')
    products=['scores/'+n for n in ['per_key_group.csv','per_response.csv','closed_set_identification.csv',
        'clean_wrong_controls.csv','group_summary.csv','teacher_student_gap.csv']]+[
        'private/eval_responses.jsonl','private/untrained_responses.jsonl','utility/generation_summary.json',
        'provenance/student_scores_complete.json','provenance/untrained_zero_lora_scores_complete.json']
    receipt={'identity':state['identity'],'status':'EVAL_COMPLETE','responses':1000,'summary':summaries,
        'source_E2_cache_hashes':source['cache_hashes'],'merged_manifest_sha256':digest(merged),
        'untrained_control':'Fresh pinned clean-model outputs; exactly equivalent to initialization with every LoRA B zero; not a trained control',
        'products_sha256':{p:sha256(state['out']/p) for p in products}}
    write_json(state['out']/'provenance/evaluation_complete.json',receipt)
    return receipt


def cached_scores(state, rows, label):
    """Read already committed E3 scores without loading any reference model."""
    receipt=read_json(state['out']/f'provenance/{label}_scores_complete.json')
    if receipt['identity']!=state['identity'] or receipt['rows_sha256']!=digest(rows) or receipt['count']!=len(rows):
        raise ValueError('E3 score population changed')
    vectors=[]; tokens=[]; ids=[]
    for name,expected in sorted(receipt['batch_sha256'].items()):
        path=state['out']/'private/eval_scores'/label/(name+'.npz')
        marker=read_json(path.with_suffix('.sha256.json'))
        if sha256(path)!=expected or marker['sha256']!=expected or marker['identity']['run']!=state['identity']:
            raise ValueError('E3 score batch checksum changed')
        with np.load(path,allow_pickle=False) as value:
            vectors.extend(value['vectors']);tokens.extend(value['eligible_tokens']);ids.extend(value['response_ids'].tolist())
    if ids!=[r['response_id'] for r in rows]:raise ValueError('E3 score IDs changed')
    v=np.asarray(vectors,dtype=np.float64);t=np.asarray(tokens,dtype=np.int64)
    if v.shape!=(len(rows),32) or t.shape!=(len(rows),) or not np.isfinite(v).all() or (t<0).any():
        raise ValueError('Invalid E3 cached score arrays')
    return v,t
