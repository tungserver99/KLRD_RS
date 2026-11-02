"""Native-only calibration and independent validation; no generation changes."""
from pathlib import Path
import numpy as np
import torch
from ..io import digest,sha256,read_json,write_json
from .cohort import generation_seed
from .numerical_policy import NumericalPolicySpec,select_contexts,historical_policy_contexts,distribution_metrics,fit_native_envelope,assess,freeze_calibration,read_calibration
from .sampler import SamplerInvalid
from .vllm_gates import NumericalToleranceExceeded


def validate_coverage(rows,prompt_ids,spec,role,mode,owners=(None,)):
    """Require the complete Cartesian design, not merely all present rows PASS."""
    if set(rows)!={'batch','reference'}:raise ValueError('Missing numerical comparison family')
    prefix_hashes={}
    for family,values in rows.items():
        comparisons=[None] if family=='batch' else ['single_vs_HF','batch_vs_HF']
        expected={(pid,step,owner,comparison) for pid in prompt_ids
            for step in range(spec.native_prefixes_per_prompt) for owner in owners for comparison in comparisons}
        actual=[]
        for row in values:
            if row.get('role')!=role or row.get('mode')!=mode:raise ValueError('Wrong numerical role or native/custom source')
            case=(row.get('prompt_id'),row.get('prefix_step'),row.get('owner'),row.get('comparison'))
            actual.append(case)
            identity=(row.get('prompt_id'),row.get('prefix_step'))
            value=row.get('prefix_sha256')
            if not isinstance(value,str) or len(value)!=64:raise ValueError('Missing numerical prefix identity')
            if identity in prefix_hashes and prefix_hashes[identity]!=value:raise ValueError('Numerical comparisons use different prefixes')
            prefix_hashes[identity]=value
        if len(actual)!=len(expected) or set(actual)!=expected:
            raise ValueError('Incomplete/duplicate numerical prompt-prefix-key-comparison coverage: '+family)


def prepare_policy(requests,sampler,cfg,out):
    if cfg.batch_size!=8:raise SamplerInvalid('Frozen E2 numerical validation requires batch_size=8; batch-vs-single must not degenerate')
    environment=sampler.environment
    if (environment.get('dtype')!='float32' or environment.get('TF32') is not False
        or environment.get('matmul_precision')!='highest'):
        raise SamplerInvalid('Native FP32 policy requires FP32, TF32 disabled and highest matmul precision')
    if any(value not in (None,'0') for value in environment.get('TF32_environment',{}).values()):
        raise SamplerInvalid('TF32 environment override conflicts with native FP32 numerical policy')
    if torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision()!='highest':
        raise SamplerInvalid('HF FP32 reference precision does not match the declared numerical policy')
    spec=NumericalPolicySpec()
    by_id={r['prompt_id']:r for r in requests}
    source_dir=Path(__file__).parent
    code={name:sha256(source_dir/name) for name in ['numerical_policy.py','native_fp32_validation.py',
        'vllm_gates.py','vllm_policy_gates.py','vllm_model.py','vllm_backend.py','sampler.py',
        '../math.py','../readout.py']}
    path=out/'tests/numerical_policy_spec.json'
    if path.exists():
        old=read_json(path)
        if old.get('spec')!=spec.resolved() or old.get('code_sha256')!=code:
            raise SamplerInvalid('Numerical policy/code changed; use a fresh run and new numerical holdout')
        exclusions=old['selection']['excluded_previous_policy_prompt_ids']
    else:
        exclusions=historical_policy_contexts(source_dir.parent.parent,spec,code,out)
    selection=select_contexts(requests,spec,exclusions)
    selected=selection['calibration_prompt_ids']+selection['holdout_prompt_ids']
    binding={'spec':spec.resolved(),'selection':selection,'code_sha256':code,
        'engine':sampler.environment,'reference_torch':torch.__version__,'CUDA':torch.version.cuda,
        'reference_attention_implementation':getattr(sampler.model.config,'_attn_implementation','unknown'),
        'reference_matmul_precision':torch.get_float32_matmul_precision(),
        'reference_TF32':torch.backends.cuda.matmul.allow_tf32,
        'device':torch.cuda.get_device_name(sampler.readout.weight.device),
        'model':cfg.model,'revision':cfg.revision,'tau':cfg.tau,'strength':sampler.strength,
        'batch_size':cfg.batch_size,
        'strict_limits':{'max':cfg.parity_max,'p99':cfg.parity_p99,'normalization':cfg.probability_atol},
        'native_generation_seed_base':cfg.generation_seed,
        'key_bank_sha256':digest(np.asarray(sampler.bank).tolist()),
        'input_token_hashes':{pid:digest(by_id[pid]['prompt_ids']) for pid in selected},
        'geometry_sha256':{name:digest(np.asarray(value).tolist()) for name,value in
            [('P',sampler.readout.p.detach().cpu().numpy()),('Q',sampler.readout.q.detach().cpu().numpy()),('L',sampler.L)]}}
    if path.exists():
        if read_json(path)!=binding:raise SamplerInvalid('Numerical policy spec/data/code/engine changed; use a fresh run')
    else:write_json(path,binding) # before any native measurement or custom holdout
    def role_requests(role):
        return [dict(by_id[pid],seed=generation_seed(cfg.generation_seed,'native_control',pid,spec.version))
            for pid in selection[role+'_prompt_ids']]
    return spec,binding,role_requests('calibration'),role_requests('holdout')


def log_probs(sampler,row,cfg):
    value=torch.as_tensor(row['raw_logits'],device=sampler.readout.weight.device,dtype=torch.float32)[None,:]
    if not torch.isfinite(value).all():raise SamplerInvalid('Nonfinite actual vLLM logits')
    lp=torch.log_softmax(value/cfg.tau,-1)
    if not torch.isfinite(lp).all():raise SamplerInvalid('Nonfinite actual vLLM log probabilities')
    if float((lp.exp().sum(-1)-1).abs().max())>cfg.probability_atol:raise NumericalToleranceExceeded('Optional probability normalization tolerance exceeded')
    return lp


def checked_trace(result,requests,native=False):
    if result.get('teacher_mode')!=('native_original' if native else 'exact_rotation'):
        raise SamplerInvalid('Wrong native/custom diagnostic path')
    rows=result['trace']
    if len(rows)!=len(requests):raise SamplerInvalid('Incomplete prefill trace')
    for row,request in zip(rows,requests):
        if row['position']!=len(request['prompt_ids'])-1 or row['input_token_id']!=request['prompt_ids'][-1]:
            raise SamplerInvalid('Numerical trace/prefix indexing mismatch')
    return rows


@torch.inference_mode()
def hf_residual(sampler,prefix):
    from .vllm_gates import fp32_reference_linears
    ids=torch.tensor([prefix],device=sampler.readout.weight.device);captured=[]
    def capture(module,args):captured.append(args[0][:,-1,:].detach().clone())
    hook=sampler.model.model.norm.register_forward_pre_hook(capture)
    try:
        with fp32_reference_linears(sampler.model):
            sampler.model.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,return_dict=True)
    finally:hook.remove()
    if len(captured)!=1:raise SamplerInvalid('Missing ordinary FP32 HF reference residual')
    return captured[0]


@torch.inference_mode()
def collect_native_controls(sampler,requests,cfg,spec,role):
    """Three native-visited states per prompt. No keyed data in calibration."""
    generated={};initial={}
    for index,request in enumerate(requests):
        value=sampler.raw_generate([request],None,3,probe=True,native_baseline=True)
        if value.get('teacher_mode')!='native_original':raise SamplerInvalid('Calibration did not use original native Qwen2 path')
        tokens=value['raw'][0]['token_ids']
        if len(tokens)!=3 or len(value['trace'])!=3:raise SamplerInvalid('Incomplete native prefix probe')
        generated[request['prompt_id']]=tokens;initial[request['prompt_id']]=value['trace'][0]
        if (index+1)%8==0:print(f'Native FP32 {role}: {index+1}/{len(requests)} prompts...',flush=True)
    states=[];rows={'batch':[],'reference':[]}
    for step in range(spec.native_prefixes_per_prompt):
        for request in requests:
            prefix=request['prompt_ids']+generated[request['prompt_id']][:step]
            forced=dict(request,prompt_ids=prefix)
            trace=initial[request['prompt_id']] if step==0 else checked_trace(
                sampler.raw_generate([forced],None,1,probe=True,native_baseline=True),[forced],True)[0]
            if trace['position']!=len(prefix)-1 or trace['input_token_id']!=prefix[-1]:raise SamplerInvalid('Native prefix state indexing mismatch')
            native_lp=log_probs(sampler,trace,cfg)
            residual=hf_residual(sampler,prefix)
            reference_lp=sampler.readout.log_probs(residual)
            meta={'prompt_id':request['prompt_id'],'prefix_step':step,'prefix_sha256':digest(prefix),
                'prompt_length':len(prefix),'mode':'native_original','role':role}
            rows['reference'].append(dict(distribution_metrics(native_lp,reference_lp,cfg.probability_atol),comparison='single_vs_HF',**meta))
            states.append({'request':forced,'meta':meta,'native_lp':native_lp.cpu(),
                'hf_residual':residual.cpu(),'native_residual':np.asarray(trace['residual'])})
    for start in range(0,len(states),cfg.batch_size):
        chunk=states[start:start+cfg.batch_size];batch=[state['request'] for state in chunk]
        result=sampler.raw_generate(batch,None,1,probe=True,native_baseline=True)
        for state,trace in zip(chunk,checked_trace(result,batch,True)):
            native_batch=log_probs(sampler,trace,cfg)
            row=dict(distribution_metrics(native_batch,state['native_lp'].to(native_batch.device),cfg.probability_atol),**state['meta'])
            rows['batch'].append(row)
            reference=sampler.readout.log_probs(state['hf_residual'].to(native_batch.device))
            rows['reference'].append(dict(distribution_metrics(native_batch,reference,cfg.probability_atol),
                comparison='batch_vs_HF',**state['meta']))
    validate_coverage(rows,[r['prompt_id'] for r in requests],spec,role,'native_original')
    return rows,states


def calibrate_native_policy(sampler,calibration_requests,cfg,spec,binding,out):
    path=out/'tests/native_fp32_calibration.json'
    if path.exists():
        calibration=read_calibration(path,binding)
        if not isinstance(calibration.get('native_evidence'),dict):raise ValueError('Frozen numerical calibration lacks native evidence')
        validate_coverage(calibration['native_evidence'],[r['prompt_id'] for r in calibration_requests],spec,'calibration','native_original')
        expected={family:fit_native_envelope(values,spec) for family,values in calibration['native_evidence'].items()}
        if calibration['envelopes']!=expected:raise ValueError('Frozen native envelope differs from recorded native evidence')
        return calibration
    rows,_=collect_native_controls(sampler,calibration_requests,cfg,spec,'calibration')
    write_json(out/'tests/native_fp32_calibration_measurements.json',rows)
    envelopes={family:fit_native_envelope(values,spec) for family,values in rows.items()}
    freeze_calibration(path,binding,envelopes,evidence=rows)
    return read_calibration(path,binding)


def evaluate_rows(rows,envelopes,spec):
    assessed={}
    for family,values in rows.items():
        assessed[family]=[dict(row,acceptance=assess(row,envelopes[family],spec)) for row in values]
    return assessed


@torch.inference_mode()
def collect_keyed_holdout(sampler,states,bank,cfg,spec):
    """All keys use the same predeclared native-visited holdout prefixes."""
    device=sampler.readout.weight.device
    rows={'batch':[],'reference':[]};tail_errors=[];raw_errors=[];norm_errors=[];zero_errors=[]
    for owner in [0,2,7]:
        angle=torch.tensor(sampler.strength*(sampler.L@bank[owner]),device=device,dtype=torch.float32)
        for start in range(0,len(states),cfg.batch_size):
            chunk=states[start:start+cfg.batch_size];batch=[state['request'] for state in chunk]
            batched=checked_trace(sampler.raw_generate(batch,bank[owner],1,probe=True),batch)
            for state,batch_trace in zip(chunk,batched):
                request=state['request']
                single=checked_trace(sampler.raw_generate([request],bank[owner],1,probe=True),[request])[0]
                lp_batch=log_probs(sampler,batch_trace,cfg);lp_single=log_probs(sampler,single,cfg)
                reference=sampler.readout.log_probs(state['hf_residual'].to(device),angle)
                meta=dict(state['meta'],mode='keyed_custom',owner=owner)
                rows['batch'].append(dict(distribution_metrics(lp_batch,lp_single,cfg.probability_atol),**meta))
                rows['reference'].append(dict(distribution_metrics(lp_single,reference,cfg.probability_atol),comparison='single_vs_HF',**meta))
                rows['reference'].append(dict(distribution_metrics(lp_batch,reference,cfg.probability_atol),comparison='batch_vs_HF',**meta))
                for trace,actual in [(batch_trace,lp_batch),(single,lp_single)]:
                    r=torch.as_tensor(trace['residual'],device=device)[None,:]
                    expected=sampler.readout.log_probs(r,angle)
                    tail_errors.append((actual-expected).abs().cpu().numpy())
                    from ..math import rotate
                    rotated=rotate(r,sampler.readout.p,sampler.readout.q,angle)
                    raw=torch.as_tensor(trace['raw_logits'],device=device)[None,:]
                    raw_errors.append((raw-sampler.readout.tail(rotated)).abs().cpu().numpy())
                    expected_norm=sampler.model.model.norm(rotated)
                    norm_errors.append((expected_norm-torch.as_tensor(trace['normalized'],device=device)).abs().cpu().numpy())
            print(f'Keyed numerical holdout owner {owner}: {min(start+cfg.batch_size,len(states))}/{len(states)} prefixes...',flush=True)
    # Genuine original-native vs disabled-custom rotation on independent inputs.
    for state in states:
        request=state['request']
        custom=checked_trace(sampler.raw_generate([request],None,1,probe=True),[request])[0]
        zero_errors.append((log_probs(sampler,custom,cfg)-state['native_lp'].to(device)).abs().cpu().numpy())
    ids=list(dict.fromkeys(state['meta']['prompt_id'] for state in states))
    validate_coverage(rows,ids,spec,'holdout','keyed_custom',owners=(0,2,7))
    return rows,tail_errors,raw_errors,norm_errors,zero_errors
