"""Policy v3 orchestration. Generation backend and scientific protocol unchanged."""
from pathlib import Path
import math
import numpy as np
import torch
from ..io import read_json,write_json,digest,sha256
from ..math import rotate
from .sampler import SamplerInvalid
from .jobs import JobStore
from .vllm_gates import _stats,_passed,NumericalToleranceExceeded
from .numerical_policy import NumericalBudgetExceeded
from .native_fp32_validation import prepare_policy,calibrate_native_policy,collect_native_controls,evaluate_rows,collect_keyed_holdout,checked_trace,log_probs


@torch.inference_mode()
def strict_implementation_checks(sampler,requests,bank,cfg):
    device=sampler.readout.weight.device
    tail=[];raw_errors=[];kv=[];norm=[];normalization=[];sampled=[];minimum_kl=0.;details=[]
    for index,request in enumerate(requests[:10]):
        owner=[0,2,7][index%3]
        value=sampler.raw_generate([request],bank[owner],3,probe=True)
        if value.get('teacher_mode')!='exact_rotation':raise SamplerInvalid('Strict check did not exercise the production teacher')
        trace=value['trace'];tokens=value['raw'][0]['token_ids'];sample_lp=value['raw'][0]['sampled_logprobs']
        if len(trace)!=3 or len(tokens)!=3 or sample_lp is None or len(sample_lp)!=3:
            raise SamplerInvalid('Strict trace requires 10 prefill +20 actual cached decode states')
        for step,row in enumerate(trace):
            prefix=request['prompt_ids']+tokens[:step]
            if row['position']!=len(prefix)-1 or row['input_token_id']!=prefix[-1]:raise SamplerInvalid('Strict cached token indexing mismatch')
            r=torch.as_tensor(row['residual'],device=device,dtype=torch.float32)[None,:]
            actual=log_probs(sampler,row,cfg)
            normalization.append(float((actual.exp().sum(-1)-1).abs().max()))
            if not 0<=tokens[step]<actual.shape[-1]:raise SamplerInvalid('Sampled token outside vocabulary')
            sampled.append(abs(float(actual[0,tokens[step]])-sample_lp[step]))
            for test_owner in [None,0,2,7]:
                angle=torch.zeros(cfg.m,device=device) if test_owner is None else torch.tensor(sampler.strength*(sampler.L@bank[test_owner]),device=device,dtype=torch.float32)
                expected=sampler.readout.log_probs(r,angle)
                rotated=rotate(r,sampler.readout.p,sampler.readout.q,angle)
                native=torch.log_softmax(sampler.model.lm_head(sampler.model.model.norm(rotated)).float()/cfg.tau,-1)
                tail.append((expected-native).abs().cpu().numpy())
                if test_owner==owner:
                    tail.append((actual-expected).abs().cpu().numpy())
                    raw=torch.as_tensor(row['raw_logits'],device=device)[None,:]
                    raw_errors.append((raw-sampler.readout.tail(rotated)).abs().cpu().numpy())
                    norm.append((sampler.model.model.norm(rotated)-torch.as_tensor(row['normalized'],device=device)).abs().cpu().numpy())
                kl=sampler.readout.kl(r,angle)
                if not torch.isfinite(kl).all():raise SamplerInvalid('Nonfinite strict KL')
                minimum_kl=min(minimum_kl,float(kl.min()))
                if test_owner is None and float(kl.abs().max())>1e-7:raise NumericalToleranceExceeded('Nonzero zero-rotation KL')
            forced=dict(request,prompt_ids=prefix)
            fresh=checked_trace(sampler.raw_generate([forced],bank[owner],1,probe=True),[forced])[0]
            error=(actual-log_probs(sampler,fresh,cfg)).abs().cpu().numpy()
            kv.append(error)
            details.append({'prompt_id':request['prompt_id'],'owner':owner,'decode_step':step,
                'max_abs_logprob_error':float(error.max()),'p99_abs_logprob_error':float(np.quantile(error,.99))})
    # Explicit EOS-containing input; no alteration of production EOS stopping.
    eos_request=dict(requests[0],prompt_ids=requests[0]['prompt_ids']+[sorted(sampler.eos)[0],17])
    row=checked_trace(sampler.raw_generate([eos_request],bank[0],1,probe=True),[eos_request])[0]
    r=torch.as_tensor(row['residual'],device=device)[None,:]
    angle=torch.tensor(sampler.strength*(sampler.L@bank[0]),device=device,dtype=torch.float32)
    tail.append((log_probs(sampler,row,cfg)-sampler.readout.log_probs(r,angle)).abs().cpu().numpy())
    return {'tail':tail,'raw':raw_errors,'kv':kv,'norm':norm,'normalization':normalization,
        'sampled':sampled,'minimum_kl':minimum_kl,'details':details}


def strict_reports(values,cfg):
    if not all(math.isfinite(value) for name in ['normalization','sampled'] for value in values[name]):
        raise SamplerInvalid('Nonfinite strict norm/probability/sampler metric')
    tail,raw,kv=_stats(values['tail']),_stats(values['raw']),_stats(values['kv'])
    norm=_stats(values['norm'])
    exact=(_passed(tail,cfg) and _passed(raw,cfg) and _passed(norm,cfg)
        and max(values['normalization'])<=cfg.probability_atol and max(values['sampled'])<=cfg.parity_max
        and values['minimum_kl']>=-cfg.negative_KL_atol)
    a={'status':'PASS' if exact else 'SAMPLER_INVALID','max_abs_logprob_error':tail['max'],
        'p99_abs_logprob_error':tail['p99'],'raw_logits_max_error':raw['max'],'raw_logits_p99_error':raw['p99'],
        'final_norm_max_abs_error':norm['max'],'final_norm_p99_abs_error':norm['p99'],
        'probability_normalization_max_abs_error':max(values['normalization']),
        'sampling_distribution_max_abs_logprob_error':max(values['sampled']),
        'minimum_KL':values['minimum_kl'],'prefill_contexts':10,'incremental_decode_contexts':20,
        'owner_indices_tested':[0,2,7],'zero_rotation_formula_tested':True,
        'engine_zero_rotation_report':'native_zero_rotation_parity.json','EOS_prefix_tested':True,
        'temperature_applications':1,'tolerance_max':cfg.parity_max,'tolerance_p99':cfg.parity_p99,
        'scope':'Strict same-context implementation checks, never relaxed by native calibration'}
    b={'status':'PASS' if _passed(kv,cfg) else 'SAMPLER_INVALID','max_abs_logprob_error':kv['max'],
        'p99_abs_logprob_error':kv['p99'],'per_context':values['details'],
        'cached_vs_full_prefix':True,'actual_vllm_decode_trace':True,
        'tolerance_max':cfg.parity_max,'tolerance_p99':cfg.parity_p99}
    return a,b


def family_report(family,native,keyed,calibration,spec):
    rows=native+keyed
    accepted=bool(native) and bool(keyed) and all(row['acceptance']['accepted'] for row in rows)
    return {'status':'PASS' if accepted else 'SAMPLER_INVALID','policy_version':spec.version,
        'classification':'consistent_with_native_fp32_on_holdout' if accepted else 'numerical_holdout_mismatch',
        'comparison_family':family,'native_holdout_comparisons':len(native),'keyed_holdout_comparisons':len(keyed),
        'frozen_native_envelope':calibration['envelopes'][family],'absolute_guardrails':spec.absolute_caps(),
        'calibration_payload_sha256':calibration['payload_sha256'],
        'full_vocab_abs_error_diagnostics':{'max_over_contexts':max((row['max_abs_logprob_error'] for row in rows),default=None),
            'max_of_context_p99':max((row['p99_abs_logprob_error'] for row in rows),default=None),
            'scope':'Raw full-vocabulary errors reported, not cross-shape/backend acceptance thresholds'},
        'max_distribution_metrics':{name:max((row[name] for row in rows),default=None) for name in spec.absolute_caps()},
        'per_context':rows,
        'interpretation':'Finite native/keyed holdout evidence only; does not prove uniform future-prefix accuracy or ownership/FPR'}


def resume_check(sampler,requests,bank,out):
    probe=requests[:8];first=sampler.generate(probe,bank[0],8);repeat=sampler.generate(probe,bank[0],8)
    tokens=[r['generated_token_ids'] for r in first]
    if tokens!=[r['generated_token_ids'] for r in repeat]:raise SamplerInvalid('vLLM fixed-batch seeded reproduction failed')
    identity={'phase':'v3_numerical_atomic_resume','prompt_ids':[r['prompt_id'] for r in probe]}
    store=JobStore(out/'private/validation_resume_vllm_policy_v3',{'phase':'policy_v3_atomic_resume'})
    if store.load('probe',identity) is None:store.commit('probe',identity,first)
    restored=JobStore(out/'private/validation_resume_vllm_policy_v3',{'phase':'policy_v3_atomic_resume'}).load('probe',identity)
    if tokens!=[r['generated_token_ids'] for r in restored]:raise SamplerInvalid('Committed vLLM resume sequence changed')
    result={'status':'PASS','records':len(first),'seeded_sequences_identical':True,
        'simulation':'Atomic commit and fresh store reopen; no primary observations',
        'token_sequence_sha256':digest(tokens)}
    write_json(out/'tests/resume_integrity.json',result)
    return result


APPROVAL_FILES=['exact_sampler_parity.json','kv_cache_parity.json','batch_prefill_parity.json',
    'reference_engine_parity.json','native_fp32_holdout.json','native_zero_rotation_parity.json','resume_integrity.json']


def all_policy_gates_pass(out):
    from .numerical_policy import read_calibration,NumericalPolicySpec
    out=Path(out);proof=out/'tests/numerical_policy_acceptance.json'
    if not proof.is_file():return False
    try:
        value=read_json(proof)
        if value.get('status')!='PASS' or value.get('policy_version')!=NumericalPolicySpec().version:return False
        spec=read_json(out/'tests/numerical_policy_spec.json')
        calibration=read_calibration(out/'tests/native_fp32_calibration.json',spec)
        if value['calibration_payload_sha256']!=calibration['payload_sha256']:return False
        if value['binding_sha256']!=digest(spec):return False
        for name in APPROVAL_FILES:
            path=out/'tests'/name
            if not path.is_file() or read_json(path).get('status')!='PASS' or value['reports_sha256'].get(name)!=sha256(path):return False
        for name,expected in spec['code_sha256'].items():
            if sha256(Path(__file__).parent/name)!=expected:return False
        return True
    except (OSError,ValueError,KeyError,TypeError):return False


@torch.inference_mode()
def validate_vllm_sampler(sampler,requests,bank,cfg,out):
    out=Path(out)
    if not np.array_equal(bank,sampler.bank):raise SamplerInvalid('Numerical validation key bank differs from frozen sampler bank')
    spec,binding,calibration_requests,holdout_requests=prepare_policy(requests,sampler,cfg,out)
    try:calibration=calibrate_native_policy(sampler,calibration_requests,cfg,spec,binding,out)
    except NumericalBudgetExceeded as exc:
        write_json(out/'tests/numerical_policy_rejection.json',{'status':'NATIVE_CALIBRATION_INVALID','reason':str(exc)})
        raise NumericalToleranceExceeded('Native numerical calibration exceeds diagnostic budget: '+str(exc))
    # Frozen envelope already exists BEFORE any keyed holdout comparison.
    strict=strict_implementation_checks(sampler,holdout_requests[:10],bank,cfg)
    a,b=strict_reports(strict,cfg)
    write_json(out/'tests/exact_sampler_parity.json',a);write_json(out/'tests/kv_cache_parity.json',b)
    if a['status']!='PASS' or b['status']!='PASS':raise NumericalToleranceExceeded('Optional implementation/KV numerical comparison did not pass')
    native_rows,states=collect_native_controls(sampler,holdout_requests,cfg,spec,'holdout')
    native=evaluate_rows(native_rows,calibration['envelopes'],spec)
    native_pass=all(row['acceptance']['accepted'] for rows in native.values() for row in rows)
    write_json(out/'tests/native_fp32_holdout.json',{'status':'PASS' if native_pass else 'NATIVE_BASELINE_UNSTABLE',
        'policy_version':spec.version,'per_family':native,'calibration_payload_sha256':calibration['payload_sha256']})
    if not native_pass:raise NumericalToleranceExceeded('Independent native FP32 holdout exceeds diagnostic envelope')
    rows,tail,raw,norm,zero=collect_keyed_holdout(sampler,states,bank,cfg,spec)
    strict['tail'].extend(tail);strict['raw'].extend(raw);strict['norm'].extend(norm)
    a,b=strict_reports(strict,cfg)
    write_json(out/'tests/exact_sampler_parity.json',a);write_json(out/'tests/kv_cache_parity.json',b)
    zero_stats=_stats(zero)
    z={'status':'PASS' if _passed(zero_stats,cfg) else 'SAMPLER_INVALID',
        'max_abs_logprob_error':zero_stats['max'],'p99_abs_logprob_error':zero_stats['p99'],
        'native_mode':'Original vLLM fused/non-fused RMSNorm, no rotation',
        'tolerance_max':cfg.parity_max,'tolerance_p99':cfg.parity_p99,'contexts':len(states)}
    write_json(out/'tests/native_zero_rotation_parity.json',z)
    keyed=evaluate_rows(rows,calibration['envelopes'],spec)
    c=family_report('batch',native['batch'],keyed['batch'],calibration,spec)
    d=family_report('reference',native['reference'],keyed['reference'],calibration,spec)
    write_json(out/'tests/batch_prefill_parity.json',c);write_json(out/'tests/reference_engine_parity.json',d)
    if any(item['status']!='PASS' for item in [a,b,c,d,z]):
        raise NumericalToleranceExceeded('Optional held-out numerical comparison failed: '+', '.join(
            name+'='+item['status'] for name,item in [('strict',a),('KV',b),('native-zero',z),('batch',c),('HF reference',d)])+
            '; no refit or scientific-threshold changes. See tests/*.json')
    resume=resume_check(sampler,holdout_requests,bank,out)
    proof={'status':'PASS','policy_version':spec.version,'binding_sha256':digest(binding),
        'calibration_payload_sha256':calibration['payload_sha256'],
        'reports_sha256':{name:sha256(out/'tests'/name) for name in APPROVAL_FILES},
        'scope':'Numerical engineering approval on independent holdout, not scientific E2 success or uniform prefix guarantee'}
    write_json(out/'tests/numerical_policy_acceptance.json',proof)
    if not all_policy_gates_pass(out):raise SamplerInvalid('Numerical approval integrity verification failed')
    print('Optional numerical diagnostics PASS; these reports are not required by Experiment 2.',flush=True)
    return {'sampler':a,'KV':b,'batch':c,'reference':d,'native_zero':z,'seeded_resume':resume['status'],
        'numerical_policy':{'status':'PASS','version':spec.version,'calibration_payload_sha256':calibration['payload_sha256']}}
