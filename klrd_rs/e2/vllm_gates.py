"""Optional numerical diagnostics, separate from production runtime validity."""
import numpy as np
import torch
from contextlib import contextmanager
from types import MethodType
from .sampler import SamplerInvalid
from .jobs import JobStore
from ..io import write_json,read_json,digest
from ..math import rotate


class NumericalToleranceExceeded(SamplerInvalid):
    """Diagnostic tolerance result, never a production generation veto."""


def runtime_configuration_valid(out):
    """Read the existing initialization receipt, without any numerical report.

    sampler_valid retains its output field but describes configuration validity,
    not numerical parity. Initialization already checks these frozen hashes.
    """
    from pathlib import Path
    try:
        environment=read_json(Path(out)/'provenance/engine_manifest.json')
        expected=environment['readout_parameter_sha256']
        from .vllm_runtime import runtime_identity
        generation=environment['generation']
        return (all(generation.get(k)==v for k,v in runtime_identity().items())
            and environment['generation']['readout_hashes']==[expected]
            and environment['generation']['dtype']=='float32')
    except (OSError,ValueError,KeyError,TypeError):return False


@contextmanager
def fp32_reference_linears(model):
    """Restore ordinary E1 FP32 Linear arithmetic for the fixed-prefix reference.

    The retained verifier uses FP64 Linear accumulation; that must not silently
    redefine the E1 FP32 comparison. Restore the verifier methods afterwards.
    """
    saved=[]
    try:
        for layer in model.modules():
            if isinstance(layer,torch.nn.Linear) and getattr(layer,'_e2_fp64_accumulation',False):
                saved.append((layer,layer.forward))
                layer.forward=MethodType(torch.nn.Linear.forward,layer)
        yield
    finally:
        for layer,forward in reversed(saved):layer.forward=forward


def _stats(errors):
    values=np.concatenate([np.asarray(e,dtype=np.float64).ravel() for e in errors])
    if not np.isfinite(values).all():raise SamplerInvalid('Nonfinite actual vLLM/reference parity errors')
    return {'max':float(values.max()),'p99':float(np.quantile(values,.99)),'finite':bool(np.isfinite(values).all())}


def _passed(stats,cfg):
    return stats['finite'] and stats['max']<=cfg.parity_max and stats['p99']<=cfg.parity_p99


def distribution_diagnostic(left,right):
    """Measurement only: quantify the distributions, never replace the gate.

    FP64 here is only a small probability summary, not Linear/model inference.
    Normalize the supplied FP32 log probabilities again in FP64 for this audit.
    """
    a=torch.log_softmax(left.double(),-1);b=torch.log_softmax(right.double(),-1)
    p=a.exp();q=b.exp()
    return {'total_variation':float((.5*(p-q).abs().sum(-1)).max()),
        'KL_left_to_right':float((p*(a-b)).sum(-1).max()),
        'probability_weighted_abs_logprob_error':float((p*(left.double()-right.double()).abs()).sum(-1).max())}


def all_vllm_gates_pass(out):
    from .vllm_policy_gates import all_policy_gates_pass
    return all_policy_gates_pass(out)


def validate_vllm_sampler(sampler,requests,bank,cfg,out,*,enabled=False):
    """Explicit opt-in only. Tolerance failures are diagnostic, runtime errors raise."""
    if not enabled:
        return {'status':'NOT_RUN','blocking':False,'reason':'Optional numerical diagnostics disabled'}
    from .vllm_policy_gates import validate_vllm_sampler as validate
    try:
        with sampler.diagnostic_mode():
            result=validate(sampler,requests,bank,cfg,out)
    except NumericalToleranceExceeded as exc:
        result={'status':'NUMERICAL_DIAGNOSTIC_NOT_PASSED','blocking':False,'reason':str(exc)}
        write_json(out/'tests/optional_numerical_diagnostics.json',result)
        print('Optional numerical diagnostic (non-blocking): '+str(exc),flush=True)
        return result
    return {'status':'PASS','blocking':False,'diagnostics':result}
