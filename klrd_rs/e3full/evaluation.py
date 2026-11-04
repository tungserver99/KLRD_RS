"""Separate 512/1024 returned-text evaluations; historical caches stay read-only."""
import numpy as np

from ..io import digest, read_json, read_jsonl, save_npz, sha256, write_json
from ..e3.data import _write_rows, load_eval_prompts
from ..e3.evaluate import (cached_scores, checked_receipt, csv, frozen_e2,
                          score_rows, tables, verify_merged)
from ..e3.provenance import get_frozen
from .common import checked, complete, phase_state


LABELS=('clean','teacher','student')


def _sampling_rows(source,label):
    previous=source['observations'][label][0]
    if [r['prompt_id'] for r in previous]!=[r['prompt_id'] for r in source['prompts']]:
        raise ValueError('Frozen baseline seed population changed')
    return [dict(p,sampling_seed=int(r['seed'])) for p,r in zip(source['prompts'],previous)]


def _validate_population(source,observations):
    expected=source['ids']
    for label,(rows,vectors,tokens) in observations.items():
        if [r['prompt_id'] for r in rows]!=expected:
            raise ValueError('Evaluation population/order changed: '+label)
        if any(r.get('prompt')!=p['prompt'] for r,p in zip(rows,source['prompts'])):
            raise ValueError('Evaluation prompt text changed: '+label)
        if (vectors.shape!=(len(rows),32) or tokens.shape!=(len(rows),)
                or not np.isfinite(vectors).all() or (tokens<0).any()):
            raise ValueError('Invalid observation shape/finite/token count: '+label)
        if any(not isinstance(r.get('response'),str) or not r.get('response_id') for r in rows):
            raise ValueError('Missing returned response text/identity: '+label)


def _matching_pilot_config(full,pilot):
    names=('model','revision','tau','seed','owner_key_id','strength','m','pca_dim',
           'key_seed','e2_run','max_new_tokens','generation_dtype','generation_backend')
    if pilot.profile!='pilot' or pilot.train_count!=2000 or pilot.validation_count!=200:
        raise ValueError('Pilot cache is not the original 2k/200 pilot')
    if any(getattr(full,k)!=getattr(pilot,k) for k in names):
        raise ValueError('Pilot/full512 sampling, owner, or frozen reference identity differs')


def _pilot_observation(state,source):
    pilot=state['pilot']
    _matching_pilot_config(state['cfg'],pilot['cfg'])
    receipt=checked_receipt(pilot,'provenance/evaluation_complete.json')
    if receipt.get('status')!='EVAL_COMPLETE' or receipt.get('responses')!=1000:
        raise ValueError('Pilot evaluation is incomplete')
    if receipt.get('source_E2_cache_hashes')!=source['cache_hashes']:
        raise ValueError('Pilot evaluation used different frozen E2 caches')
    rows=read_jsonl(pilot['out']/'private/eval_responses.jsonl')
    from ..e3.generation import _seed, _validate
    sampling=read_json(pilot['out']/'private/eval_responses.identity.json')
    settings=sampling['settings']
    if (sampling.get('run')!=pilot['identity'] or settings.get('kind')!='student'
            or settings.get('max_new_tokens')!=512 or settings.get('temperature')!=pilot['cfg'].tau
            or settings.get('seed')!=pilot['cfg'].seed
            or settings.get('backend')!=pilot['cfg'].generation_backend
            or settings.get('dtype')!=pilot['cfg'].generation_dtype
            or any(r.get('seed')!=_seed(pilot['cfg'].seed,r['prompt_id'],'student') for r in rows)):
        raise ValueError('Pilot student journal sampling policy changed')
    _validate(rows,source['prompts'],sampling)
    vectors,tokens=cached_scores(pilot,rows,'student')
    result=(rows,vectors,tokens)
    _validate_population(source,{'pilot512':result})
    return result,{'evaluation_receipt_sha256':sha256(pilot['out']/'provenance/evaluation_complete.json'),
                   'student_responses_sha256':sha256(pilot['out']/'private/eval_responses.jsonl'),
                   'student_sampling_identity_sha256':sha256(pilot['out']/'private/eval_responses.identity.json'),
                   'student_score_receipt_sha256':sha256(pilot['out']/'provenance/student_scores_complete.json'),
                   'identity':pilot['identity']}


def _save_observations(phase,observations):
    arrays={};rows=[]
    for label,(records,vectors,tokens) in observations.items():
        arrays[label+'_vectors']=np.asarray(vectors,dtype=np.float64)
        arrays[label+'_eligible_tokens']=np.asarray(tokens,dtype=np.int64)
        rows.extend({'model':label,'record':r} for r in records)
    _write_rows(phase['out']/'private/observation_rows.jsonl',rows)
    save_npz(phase['out']/'private/observations.npz',**arrays)


def _read_observations(phase,labels):
    grouped={label:[] for label in labels}
    for value in read_jsonl(phase['out']/'private/observation_rows.jsonl'):
        if set(value)!={'model','record'} or value['model'] not in grouped:
            raise ValueError('Unexpected serialized observation label/schema')
        grouped[value['model']].append(value['record'])
    with np.load(phase['out']/'private/observations.npz',allow_pickle=False) as arrays:
        expected={label+suffix for label in labels for suffix in ('_vectors','_eligible_tokens')}
        if set(arrays.files)!=expected:raise ValueError('Unexpected serialized observation arrays')
        return {label:(grouped[label],arrays[label+'_vectors'].copy(),
                       arrays[label+'_eligible_tokens'].copy()) for label in labels}


def generation_utility(observations,length):
    """Native generation length remains distinct from verifier eligible tokens."""
    from ..e2.utility import text_quality
    result={}
    for label,(rows,_,eligible) in observations.items():
        n=len(rows)
        if not n:raise ValueError('Cannot summarize an empty evaluation')
        if any('generated_token_ids' not in r or 'native_visible_tokens' not in r for r in rows):
            raise ValueError('Native generation token metadata required: '+label)
        native=np.asarray([len(r['generated_token_ids']) for r in rows])
        visible=np.asarray([r['native_visible_tokens'] for r in rows])
        values=[text_quality(dict(r,eos_terminated=bool(r.get('eos_terminated')),
            max_length_truncated=bool(r.get('max_length_truncated')),
            roundtrip_equal=r.get('roundtrip_equal',False))) for r in rows]
        timing=[(r['amortized_generation_seconds']-r.get('KL_audit_seconds',0.))
                if r.get('amortized_generation_seconds') is not None else None for r in rows]
        legacy_audit_unknown=any(r.get('KL_samples') and 'KL_audit_seconds' not in r for r in rows)
        timed=all(t is not None and np.isfinite(t) and t>=0 for t in timing)
        seconds=float(sum(timing)) if timed and not legacy_audit_unknown else None
        eos=sum(bool(r.get('eos_terminated')) for r in rows)
        truncated=sum(bool(r.get('max_length_truncated')) for r in rows)
        roundtrip=sum(r.get('roundtrip_equal') is True for r in rows)
        empty=sum(not r['response'].strip() for r in rows)
        loop=sum(r['loop'] for r in values)
        result[label]={'responses':n,'max_new_tokens':length,
            'mean_generated_tokens':float(native.mean()),'median_generated_tokens':float(np.median(native)),
            'native_token_count_definition':'Number of returned native generated token IDs; omitted engine stop tokens are never fabricated',
            'mean_visible_tokens':float(visible.mean()),'median_visible_tokens':float(np.median(visible)),
            'mean_verifier_eligible_tokens':float(np.mean(eligible)),
            'mean_words':float(np.mean([len(r['response'].split()) for r in rows])),
            'median_words':float(np.median([len(r['response'].split()) for r in rows])),
            'eos_terminated_count':eos,'eos_termination_rate':eos/n,
            'max_length_truncated_count':truncated,'max_length_truncation_rate':truncated/n,
            'empty_count':empty,'empty_response_rate':empty/n,
            'severe_repeated_4gram_count':loop,'severe_repetition_rate':loop/n,
            'repeated_4gram_rule':'Unchanged E2: native_visible_tokens>=64 and repeated4 fraction>=.4',
            'mean_repeated4_fraction':float(np.mean([r['repeated4'] for r in values])),
            'roundtrip_mismatch_count':sum(r.get('roundtrip_equal') is False for r in rows),
            'roundtrip_metadata_available':sum('roundtrip_equal' in r for r in rows),
            'text_token_roundtrip_consistency_rate':roundtrip/n,
            'generation_error_count':sum(r.get('error') is not None for r in rows),
            'generation_seconds':seconds,'timed_response_count':sum(t is not None for t in timing),
            'native_generated_tokens_per_second':float(native.sum()/seconds) if seconds and seconds>0 else None,
            'native_visible_tokens_per_second':float(visible.sum()/seconds) if seconds and seconds>0 else None,
            'KL_audit_subtracted_response_count':sum('KL_audit_seconds' in r for r in rows),
            'legacy_KL_audit_timing_unavailable':legacy_audit_unknown,
            'timing_scope':'sum of recorded amortized engine generation time, subtracting recorded legacy KL audit time; initialization excluded; unavailable if audit timing cannot be separated',
            'throughput_comparison_limit':'Interpret with checkpoint, actual worker runtime, hardware and backend; no causal length claim from unmatched execution',
            'generation_backend_counts':{backend:sum(r.get('generation_backend','unrecorded')==backend for r in rows)
                for backend in sorted({r.get('generation_backend','unrecorded') for r in rows})}}
    return result


def _examples(phase,source,observations):
    from ..e2b.quality import _atomic_text, _quoted
    content=['# Private fixed comparison examples','',
        'First ten frozen group0 IDs, selected before generation/scoring. No quality filtering.','']
    for i,prompt in enumerate(source['prompts'][:10]):
        content+=['## '+digest(prompt['prompt_id'])[:16],_quoted(prompt['prompt'])]
        for label in observations:
            content+=[label,_quoted(observations[label][0][i]['response'])]
    _atomic_text(phase['out']/'private/examples.md','\n'.join(content)+'\n')


def _products(phase):
    """Allow only evaluation-owned products; never traverse parent model weights."""
    products=[]
    for name in ('private','scores','utility','provenance'):
        directory=phase['out']/name
        for path in sorted(directory.rglob('*')) if directory.exists() else []:
            # score_rows uses this one working comparison path again during
            # paraphrase. Per-condition *_verification_runtime.json are the
            # immutable runtime evidence; do not bind its mutable scratch copy.
            if (path.is_file() and path.name not in ('evaluation_complete.json','verification_environment_comparison.json')
                    and not path.name.endswith('.tmp')):
                products.append(path.relative_to(phase['out']).as_posix())
    return products


def load_observations(state,length):
    """Read complete bound arrays without generation or loading a reference model."""
    if length not in (512,1024):raise ValueError('Evaluation length must be 512 or 1024')
    phase=phase_state(state,length)
    receipt=checked(phase,'provenance/evaluation_complete.json')
    if receipt is None:raise FileNotFoundError('Complete eval'+str(length)+' observations are required')
    source=frozen_e2(state)
    labels=list(LABELS)+(['pilot512'] if length==512 else [])
    if receipt.get('labels')!=labels or receipt.get('max_new_tokens')!=length:
        raise ValueError('Completed evaluation condition/length changed')
    if receipt.get('source_E2_cache_hashes')!=source['cache_hashes']:
        raise ValueError('Completed evaluation frozen source changed')
    observations=_read_observations(phase,labels)
    _validate_population(source,observations)
    return phase,observations,source['groups'],source


def run(state,length):
    if length not in (512,1024):raise ValueError('Evaluation length must be 512 or 1024')
    from ..e3.generation import generate
    from .teacher import generate as generate_teacher
    _matching_pilot_config(state['cfg'],state['pilot']['cfg'])
    merged=verify_merged(state)
    phase=phase_state(state,length)
    phase['verified_merged_file_manifest']=state['verified_merged_file_manifest']
    if (phase['out']/'provenance/evaluation_complete.json').exists():
        load_observations(state,length)
        return checked(phase,'provenance/evaluation_complete.json')
    source=frozen_e2(state)
    rows=load_eval_prompts(state)
    if ([r['prompt_id'] for r in rows]!=source['ids']
            or any(r['prompt']!=p['prompt'] for r,p in zip(rows,source['prompts']))):
        raise ValueError('Full evaluation prompts differ from frozen group0')
    frozen=get_frozen(state);phase['frozen']=frozen
    observations={};pilot_binding=None
    if length==512:
        observations.update({label:source['observations'][label] for label in ('clean','teacher')})
        pilot,pilot_binding=_pilot_observation(state,source)
    else:
        # Every condition is a fresh 1024 run, never continued/substituted 512 text.
        clean=generate(phase,_sampling_rows(source,'clean'),phase['out']/'private/clean_responses.jsonl',
                       model_path=None,kind='untrained')
        teacher=generate_teacher(phase,_sampling_rows(source,'teacher'),
                                 phase['out']/'private/teacher_responses.jsonl',length)
        # The teacher process is closed before the verifier allocates its model.
        for label,generated in [('clean',clean),('teacher',teacher)]:
            vectors,tokens=score_rows(phase,generated,label,source['environment'])
            observations[label]=(generated,vectors,tokens)
    # The E3 student seed/kind stays exactly the pilot policy at both lengths.
    student=generate(phase,rows,phase['out']/'private/student_responses.jsonl',
                     model_path=state['out']/'merged_student',kind='student')
    vectors,tokens=score_rows(phase,student,'student',source['environment'])
    observations['student']=(student,vectors,tokens)
    if length==512:observations['pilot512']=pilot
    _validate_population(source,observations)
    for label,(records,_,_) in observations.items():
        target=phase['out']/f'private/{label}_responses.jsonl'
        # Generated journals retain their exact on-disk checksums/sidecars.
        if not target.exists():_write_rows(target,records)
        elif read_jsonl(target)!=records:raise ValueError('Response journal changed: '+label)
    summaries=tables(phase,observations,source['groups'],phase['out']/'scores')
    utility=generation_utility(observations,length)
    write_json(phase['out']/'utility/generation_summary.json',utility)
    csv(phase['out']/'utility/generation_summary.csv',[dict(model=k,**v) for k,v in utility.items()])
    # Means supplement the existing unchanged verifier table's medians.
    from ..e2.verifier import SphericalBlackBoxVerifier
    cosine=[]
    for label,(_,vectors,_) in observations.items():
        values=[float(SphericalBlackBoxVerifier.key_scores(v,frozen['bank'])['cosine'][0]) for v in vectors]
        cosine.append({'model':label,'responses':len(values),'correct_key_cosine_mean':float(np.mean(values)),
                       'correct_key_cosine_median':float(np.median(values))})
    csv(phase['out']/'scores/cosine_summary.csv',cosine)
    _save_observations(phase,observations);_examples(phase,source,observations)
    write_json(phase['out']/'provenance/evaluation_protocol.json',{
        'identity':phase['identity'],'max_new_tokens':length,'responses_per_condition':1000,
        'labels':list(observations),'prompt_ids_sha256':digest(source['ids']),
        'groups_sha256':digest(source['groups']),'groups_per_Q':{'20':50,'50':20,'100':10,'1000':1},
        'owner_key':0,'wrong_controls':20,'alphas':[.01,.0001],
        'null':'unchanged analytical spherical random-key null; no empirical clean calibration',
        'verification':'unchanged e3.score_rows / E2 canonical returned-text re-tokenization; no generation hidden states',
        'baseline_policy':'read-only E2 caches at512; newly generated E2 per-row seeds at1024',
        'student_seed_policy':'unchanged e3 _seed(cfg.seed,prompt_id,student), both lengths',
        'checkpoint_selection':'validation KL only; no evaluation-dependent selection',
        'pilot_cache_binding':pilot_binding})
    return complete(phase,'provenance/evaluation_complete.json',_products(phase),
        status='EVAL_COMPLETE',responses=1000,max_new_tokens=length,labels=list(observations),
        summary=summaries,source_E2_cache_hashes=source['cache_hashes'],
        merged_manifest_sha256=digest(merged),pilot_cache_binding=pilot_binding)
