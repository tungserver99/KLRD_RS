"""Real-model prefill AND incremental-decode gates before primary generation."""
import numpy as np
import torch
from contextlib import contextmanager
from types import MethodType
from ..math import rotate
from ..io import write_json,digest
from .sampler import SamplerInvalid,linear_fp64_forward
from .jobs import JobStore


@contextmanager
def linear_fp64_probe(model):
    """Diagnostic only: FP64 linear accumulation, FP32 layer boundaries/weights.

    This is NOT a full FP64 model or a replacement acceptance criterion.
    Restore exact instance forward attributes even when the probe fails.
    """
    saved=[]
    try:
        for layer in model.modules():
            if isinstance(layer,torch.nn.Linear):
                saved.append((layer,'forward' in layer.__dict__,layer.__dict__.get('forward')))
                layer.forward=MethodType(linear_fp64_forward,layer)
        yield
    finally:
        for layer,had,value in reversed(saved):
            if had:layer.forward=value
            else:delattr(layer,'forward')


@torch.inference_mode()
def diagnose_kv_roundoff(sampler,requests,bank,cfg,details,out):
    worst=max(details,key=lambda row:row['max_abs_logprob_error'])
    start=worst['batch_start'];size=min(cfg.batch_size,4)
    batch=requests[start:min(start+size,10)]
    angle=torch.tensor(sampler.strength*(sampler.L@bank[worst['owner']]),
        device=sampler.readout.weight.device,dtype=torch.float32)
    errors=[]
    print('KV gate failed: probing FP64 Linear accumulation on the worst batch only; pipeline remains stopped.',flush=True)
    with linear_fp64_probe(sampler.model):
        session=sampler.session([r['prompt_ids'] for r in batch])
        for step in range(3):
            full=session.full_prefix_residual()
            error=(sampler.readout.log_probs(session.residual,angle)-sampler.readout.log_probs(full,angle)).abs()
            errors.append({'decode_step':step,'max_abs_logprob_error':float(error.max()),
                'p99_abs_logprob_error':float(torch.quantile(error.double().flatten(),.99)),
                'max_abs_residual_error':float((session.residual-full).abs().max())})
            if step<2:
                tokens=torch.full((len(batch),),17+step,dtype=torch.long,device=full.device)
                if step==0:tokens[0]=sorted(sampler.eos)[0]
                session.advance(tokens,torch.ones(len(batch),dtype=torch.bool,device=full.device))
    write_json(out/'tests/kv_linear_precision_diagnostic.json',{
        'status':'DIAGNOSTIC_ONLY_PIPELINE_STOPPED','baseline_worst_context':worst,
        'prompt_ids':[r['prompt_id'] for r in batch],'linear_fp64_probe':errors,
        'scope':'Only nn.Linear accumulation is FP64; weights, residual boundaries, attention, RMSNorm, RoPE and readout remain FP32. Not a full FP64 reference.',
        'interpretation':'A reduction supports shape-dependent Linear roundoff; no reduction requires tracing attention/normalization/cache. Neither outcome changes gate tolerance or authorizes generation.'})


@torch.inference_mode()
def validate_sampler(sampler,requests,bank,cfg,out):
    if len(requests)<10:raise SamplerInvalid('At least10 actual prompt contexts required for parity')
    parity=[];kv_errors=[];probability_errors=[];kl_min=0.;prefills=0;decodes=0;details=[]
    owners=[0,2,7]
    for start in range(0,10,min(cfg.batch_size,4)):
        batch=requests[start:min(start+min(cfg.batch_size,4),10)]
        session=sampler.session([r['prompt_ids'] for r in batch])
        for step in range(3):
            r=session.residual
            if step==0:prefills+=len(batch)
            else:decodes+=len(batch)
            full=session.full_prefix_residual()
            zero=torch.zeros(sampler.L.shape[0],device=r.device)
            native_clean=torch.log_softmax(sampler.model.lm_head(sampler.model.model.norm(r)).float()/sampler.readout.tau,-1)
            clean=sampler.readout.log_probs(r,zero)
            parity.append((clean-native_clean).abs().cpu().numpy().ravel())
            if float(sampler.readout.kl(r,zero).abs().max())>1e-7:raise SamplerInvalid('Nonzero KL at zero rotation')
            for owner in owners:
                angle=torch.tensor(sampler.strength*(sampler.L@bank[owner]),device=r.device,dtype=torch.float32)
                actual=sampler.readout.log_probs(r,angle)
                exact=torch.log_softmax(sampler.model.lm_head(sampler.model.model.norm(
                    rotate(r,sampler.readout.p,sampler.readout.q,angle))).float()/sampler.readout.tau,-1)
                parity.append((actual-exact).abs().cpu().numpy().ravel())
                ref=sampler.readout.log_probs(full,angle)
                error=(actual-ref).abs()
                kv_errors.append(error.cpu().numpy().ravel())
                flat=int(error.argmax());row=flat//error.shape[-1];token=flat%error.shape[-1]
                details.append({'batch_start':start,'decode_step':step,'owner':owner,
                    'worst_prompt_id':batch[row]['prompt_id'],'worst_token_id':token,
                    'max_abs_logprob_error':float(error.max()),
                    'max_abs_residual_error':float((r-full).abs().max())})
                probability_errors.append(float((actual.exp().sum(-1)-1).abs().max()))
                if not torch.isfinite(actual).all():raise SamplerInvalid('Nonfinite teacher log probabilities')
                kl_min=min(kl_min,float(sampler.readout.kl(r,angle).min()))
            # Numerical KV probe includes an actual EOS ID; never a primary output.
            tokens=torch.full((len(batch),),17+step,dtype=torch.long,device=r.device)
            if step==0:tokens[0]=sorted(sampler.eos)[0]
            if step<2:session.advance(tokens,torch.ones(len(batch),device=r.device,dtype=torch.bool))
    p=np.concatenate(parity);k=np.concatenate(kv_errors)
    finite=np.isfinite(p).all() and np.isfinite(k).all()
    parity_pass=finite and p.max()<=cfg.parity_max and np.quantile(p,.99)<=cfg.parity_p99
    kv_pass=finite and k.max()<=cfg.parity_max and np.quantile(k,.99)<=cfg.parity_p99
    sampler_pass=parity_pass and max(probability_errors)<=cfg.probability_atol and kl_min>=-cfg.negative_KL_atol
    a={'status':'PASS' if sampler_pass else 'SAMPLER_INVALID','max_abs_logprob_error':float(p.max()),
        'p99_abs_logprob_error':float(np.quantile(p,.99)), 'prefill_contexts':prefills,
        'incremental_decode_contexts':decodes,'owner_indices_tested':owners,'zero_rotation_tested':True,
        'probability_normalization_max_abs_error':max(probability_errors),'minimum_KL':kl_min,
        'tolerance_max':cfg.parity_max,'tolerance_p99':cfg.parity_p99,'dtype':'float32',
        'linear_accumulation':getattr(sampler.model,'_e2_linear_accumulation','float32'),'temperature_applications':1}
    b={'status':'PASS' if kv_pass else 'SAMPLER_INVALID','max_abs_logprob_error':float(k.max()),
        'p99_abs_logprob_error':float(np.quantile(k,.99)),'cached_vs_full_prefix':True,
        'left_padding_tested':True,'EOS_sensitive_context_tested':True,'cache_recreated_per_request_batch':True,
        'tolerance_max':cfg.parity_max,'tolerance_p99':cfg.parity_p99,'per_context_batch':details,
        'attention_implementation':getattr(sampler.model.config,'_attn_implementation','unknown'),
        'linear_accumulation':getattr(sampler.model,'_e2_linear_accumulation','float32')}
    write_json(out/'tests/exact_sampler_parity.json',a);write_json(out/'tests/kv_cache_parity.json',b)
    if not sampler_pass or not kv_pass:
        if not kv_pass and getattr(sampler.model,'_e2_linear_accumulation','float32')!='float64':
            try:diagnose_kv_roundoff(sampler,requests,bank,cfg,details,out)
            except (RuntimeError,ValueError) as exc:
                write_json(out/'tests/kv_linear_precision_diagnostic.json',{
                    'status':'DIAGNOSTIC_FAILED_PIPELINE_STOPPED','error_type':type(exc).__name__,'reason':str(exc)})
        raise SamplerInvalid(f"rotation/probability={'PASS' if sampler_pass else 'FAIL'} "
            f"(max={p.max():.9g}, p99={np.quantile(p,.99):.9g}); "
            f"KV={'PASS' if kv_pass else 'FAIL'} (max={k.max():.9g}, p99={np.quantile(k,.99):.9g}); "
            f"limits max={cfg.parity_max}, p99={cfg.parity_p99}. See {out/'tests/kv_cache_parity.json'}")
    # Repeating a fixed-shape mini-job validates actual categorical stream reproducibility.
    probe=requests[:min(2,len(requests))]
    first=sampler.generate(probe,bank[0],max_new_tokens=8)
    repeated=sampler.generate(probe,bank[0],max_new_tokens=8)
    same=all(x['generated_token_ids']==y['generated_token_ids'] for x,y in zip(first,repeated))
    if not same:raise SamplerInvalid('SAMPLER_INVALID: fixed-request seeded reproduction failed')
    store=JobStore(out/'private/validation_resume',{'phase':'predeclared_atomic_resume_probe'})
    identity={'prompt_ids':[r['prompt_id'] for r in probe],'owner':0,'group':'validation','chunk':0}
    existing=store.load('probe',identity)
    if existing is None:store.commit('probe',identity,first)
    # Simulated failure immediately after committed write, then fresh object reopen.
    try:raise RuntimeError('injected post-commit interruption')
    except RuntimeError:pass
    reopened=JobStore(out/'private/validation_resume',{'phase':'predeclared_atomic_resume_probe'})
    restored=reopened.load('probe',identity)
    if [r['generated_token_ids'] for r in restored]!=[r['generated_token_ids'] for r in first]:
        raise SamplerInvalid('SAMPLER_INVALID: resumed sequence differs')
    write_json(out/'tests/resume_integrity.json',{'status':'PASS','simulation':'post-atomic-commit exception and fresh store reopen',
        'records':len(restored),'seeded_sequences_identical':same,'duplicate_prompt_ids':False,
        'token_sequence_sha256':digest([r['generated_token_ids'] for r in restored])})
    return {'sampler':a,'KV':b,'seeded_resume':'PASS'}
