"""Out-of-tree exact teacher for vLLM 0.10.2. Imported by the GPU worker only.

Upstream: v0.10.2/model_executor/models/qwen2.py and layers/layernorm.py.
Their Apache-2.0 implementations remain installed, unmodified and inherited.
"""
from functools import partial
import torch
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from ..math import rotate


def rotate_final_state(hidden,residual,p,q,angles):
    if hidden.dtype!=torch.float32:raise RuntimeError('Exact vLLM teacher requires FP32 residuals')
    combined=hidden if residual is None else hidden+residual
    return rotate(combined,p,q,angles),combined


class RotationFinalRMSNorm(RMSNorm):
    @classmethod
    def from_native(cls,native,m):
        if native.variance_size_override is not None:raise RuntimeError('Unsupported final RMSNorm override')
        result=cls(native.hidden_size,eps=native.variance_epsilon,dtype=native.weight.dtype)
        # Preserve model.norm.weight and original Qwen2 load_weights behavior.
        result.weight=native.weight
        result.register_buffer('P',torch.zeros(native.hidden_size,m,device=native.weight.device,dtype=torch.float32),persistent=False)
        result.register_buffer('Q',torch.zeros_like(result.P),persistent=False)
        result.register_buffer('angles',torch.zeros(m,device=native.weight.device,dtype=torch.float32),persistent=False)
        result.capture=False;result.last_capture=None;result.native_baseline=False
        return result

    def configure(self,p,q,angles):
        if self.weight.dtype!=torch.float32:raise RuntimeError('vLLM teacher weights must be FP32')
        for target,value in [(self.P,p),(self.Q,q),(self.angles,angles)]:
            value=torch.as_tensor(value,device=target.device,dtype=target.dtype)
            if value.shape!=target.shape or not torch.isfinite(value).all():raise RuntimeError('Invalid frozen rotation geometry')
            target.copy_(value)

    def forward(self,hidden,residual=None):
        if self.native_baseline:
            # Validation control only: the genuine original fused/non-fused
            # vLLM RMSNorm path. Production leaves native_baseline=False.
            original_hidden=hidden.detach().clone() if self.capture and residual is None else None
            result=super().forward(hidden,residual)
            if self.capture:
                normalized,combined=result if residual is not None else (result,original_hidden)
                self.last_capture={'residual':combined.detach(),'normalized':normalized.detach()}
            return result
        rotated,combined=rotate_final_state(hidden,residual,self.P,self.Q,self.angles)
        # Original norm kernel, residual=None: no double residual addition.
        normalized=super().forward(rotated)
        if self.capture:
            self.last_capture={'residual':combined.detach(),'normalized':normalized.detach()}
        return normalized if residual is None else (normalized,combined)


class KLRDRSQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self,*,vllm_config,prefix=''):
        parallel=vllm_config.parallel_config
        if parallel.tensor_parallel_size!=1 or parallel.pipeline_parallel_size!=1:
            raise RuntimeError('This implementation targets TP=PP=1 only')
        if vllm_config.quant_config is not None:raise RuntimeError('No quantization in exact E2 teacher')
        super().__init__(vllm_config=vllm_config,prefix=prefix)
        self.model.norm=RotationFinalRMSNorm.from_native(self.model.norm,32)
        self._klrd_trace_enabled=False;self._klrd_trace=[]

    def forward(self,input_ids,positions,intermediate_tensors=None,inputs_embeds=None):
        output=super().forward(input_ids,positions,intermediate_tensors,inputs_embeds)
        if self._klrd_trace_enabled:
            flat=positions.reshape(-1)
            # Probe mode permits one decoding stream, or a prefill-only batch.
            ends=torch.where(flat[1:]==0)[0].tolist()+[len(flat)-1]
            captured=self.model.norm.last_capture
            self._klrd_pending_trace_rows=len(ends)
            for index in ends:
                self._klrd_trace.append({'residual':captured['residual'][index].float().cpu().numpy(),
                    'normalized':captured['normalized'][index].float().cpu().numpy(),
                    'position':int(flat[index].item()),'input_token_id':int(input_ids.reshape(-1)[index].item())})
            self.model.norm.last_capture=None
        return output

    def compute_logits(self,hidden_states,sampling_metadata):
        logits=super().compute_logits(hidden_states,sampling_metadata)
        if logits is not None and not torch.isfinite(logits).all():
            raise RuntimeError('Nonfinite vLLM logits during generation')
        if self._klrd_trace_enabled and logits is not None:
            if not self._klrd_trace:raise RuntimeError('Missing actual forward trace')
            count=self._klrd_pending_trace_rows
            if logits.shape[0]!=count:raise RuntimeError('Unexpected probe logits/trace shape')
            for row,value in zip(self._klrd_trace[-count:],logits):
                row['raw_logits']=value.detach().float().cpu().numpy()
        # Temperature belongs exclusively to SamplingParams, never here.
        return logits


def configure_model(model,p,q):
    norm=model.model.norm
    if not isinstance(norm,RotationFinalRMSNorm):raise RuntimeError('Custom Qwen2 was not selected')
    norm.configure(p,q,torch.zeros(32))
    from ..continuation import tensor_sha256
    return {'weight':tensor_sha256(model.lm_head.weight[:model.config.vocab_size]),
        'gain':tensor_sha256(norm.weight),'bias':None}


def set_owner(model,angles):
    norm=model.model.norm
    norm.configure(norm.P,norm.Q,angles)
    return {'fixed_owner_set':True}


def set_native_baseline(model,enabled):
    norm=model.model.norm
    if enabled and bool(torch.count_nonzero(norm.angles)):
        raise RuntimeError('Native baseline requires rotation disabled')
    norm.native_baseline=bool(enabled)
    return {'native_original_norm':norm.native_baseline}


def start_trace(model):
    model._klrd_trace=[];model._klrd_trace_enabled=True
    model.model.norm.capture=True


def take_trace(model):
    result=model._klrd_trace
    model._klrd_trace=[];model._klrd_trace_enabled=False
    model.model.norm.capture=False;model.model.norm.last_capture=None
    return result


def start_linear_shape_probe(model):
    """Diagnostic only: identical input/weights, native batched GEMM vs one row.

    The original output is returned unchanged. No FP64, weight edit or new
    inference implementation is used. Probe methods are restored after the call.
    """
    model._klrd_linear_probe=[];model._klrd_linear_saved=[]
    names={id(layer):name for name,layer in model.named_modules()};seen=set()
    for name,layer in model.named_modules():
        method=getattr(layer,'quant_method',None)
        if method is None or type(method).__name__ not in ['UnquantizedLinearMethod','UnquantizedEmbeddingMethod']:continue
        if id(method) in seen:continue
        seen.add(id(method))
        original=method.apply
        model._klrd_linear_saved.append((method,'apply' in method.__dict__,method.__dict__.get('apply')))
        def apply(current,x,bias=None,_original=original):
            output=_original(current,x,bias)
            if x.ndim==2 and x.shape[0]>1:
                separate=_original(current,x[-1:],bias)
                difference=(output[-1]-separate[-1]).abs()
                index=int(difference.argmax())
                model._klrd_linear_probe.append({'module':names.get(id(current),'unknown'),
                    'input_shape':list(x.shape),'weight_shape':list(current.weight.shape),
                    'input_dtype':str(x.dtype),'weight_dtype':str(current.weight.dtype),
                    'same_input_row_index':x.shape[0]-1,'worst_output_index':index,
                    'max_abs_error':float(difference.max()),
                    'batch_output_at_worst':float(output[-1,index]),
                    'single_output_at_worst':float(separate[-1,index]),
                    'TF32':torch.backends.cuda.matmul.allow_tf32,
                    'matmul_precision':torch.get_float32_matmul_precision()})
            return output
        method.apply=apply


def stop_linear_shape_probe(model):
    for method,had,value in reversed(getattr(model,'_klrd_linear_saved',[])):
        if had:method.apply=value
        else:delattr(method,'apply')
    model._klrd_linear_saved=[]
    rows=getattr(model,'_klrd_linear_probe',[]);model._klrd_linear_probe=[]
    return rows
