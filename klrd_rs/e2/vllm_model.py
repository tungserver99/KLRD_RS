"""Out-of-tree exact teacher for vLLM 0.32.0. Imported by the GPU worker only.

Upstream: v0.32.0/model_executor/models/qwen2.py and layers/layernorm.py.
Their Apache-2.0 implementations remain installed, unmodified and inherited.
"""
from vllm.forward_context import get_forward_context
import torch
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from ..math import rotate


def rotate_final_state(hidden,residual,p,q,angles):
    if hidden.dtype!=torch.float32:raise RuntimeError('Exact vLLM teacher requires FP32 residuals')
    combined=hidden if residual is None else hidden+residual
    return rotate(combined,p,q,angles),combined


def trace_query_spans(metadata,token_count):
    """Actual scheduled queries only; V1 may append zero-length padding slots."""
    actual=int(metadata.num_actual_tokens)
    boundaries=metadata.query_start_loc.detach().cpu().tolist()
    if not boundaries or boundaries[0]!=0 or not 0<actual<=token_count:
        raise RuntimeError('Invalid V1 diagnostic attention metadata')
    spans=[]
    for start,end in zip(boundaries,boundaries[1:]):
        if not 0<=start<=end<=actual:
            raise RuntimeError('Invalid V1 diagnostic query boundary')
        if end>start:spans.append((start,end))
    if not spans or spans[-1][1]!=actual:
        raise RuntimeError('Incomplete V1 diagnostic query coverage')
    return spans

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
        self.register_buffer('_klrd_logits_finite',torch.ones((),dtype=torch.bool,
            device=self.model.norm.weight.device),persistent=False)
        self._klrd_trace_enabled=False;self._klrd_trace=[]
        self._klrd_diagnostic_eager=bool(vllm_config.model_config.enforce_eager)

    def forward(self,input_ids,positions,intermediate_tensors=None,inputs_embeds=None):
        output=super().forward(input_ids,positions,intermediate_tensors,inputs_embeds)
        if self._klrd_trace_enabled:
            flat=positions.reshape(-1)
            metadata=get_forward_context().attn_metadata
            if not isinstance(metadata,dict) or not metadata:
                raise RuntimeError('Missing actual V1 attention metadata for diagnostic')
            metadata=next(iter(metadata.values()))
            spans=trace_query_spans(metadata,len(flat))
            captured=self.model.norm.last_capture
            if captured is None:raise RuntimeError('Missing eager final-norm capture')
            self._klrd_pending_trace_rows=len(spans)
            for start,end in spans:
                index=end-1
                self._klrd_trace.append({'residual':captured['residual'][index].float().cpu().numpy(),
                    'normalized':captured['normalized'][index].float().cpu().numpy(),
                    'position':int(flat[index].item()),'input_token_id':int(input_ids.reshape(-1)[index].item()),
                    'query_token_ids':input_ids.reshape(-1)[start:end].cpu().tolist()})
            self.model.norm.last_capture=None
        return output

    def compute_logits(self,hidden_states):
        logits=super().compute_logits(hidden_states)
        if logits is not None:
            # GPU-only reduction/update: no Tensor->Python bool synchronization
            # per token. The worker checks this flag once after engine.generate.
            self._klrd_logits_finite.logical_and_(torch.isfinite(logits).all())
        if self._klrd_trace_enabled and logits is not None:
            if not self._klrd_trace:raise RuntimeError('Missing actual forward trace')
            count=self._klrd_pending_trace_rows
            if logits.shape[0]!=count:raise RuntimeError('Unexpected probe logits/trace shape')
            for row,value in zip(self._klrd_trace[-count:],logits):
                row['raw_logits']=value.detach().float().cpu().numpy()
        # Temperature belongs exclusively to SamplingParams, never here.
        return logits


@torch.inference_mode()
def configure_model(model,p,q):
    norm=model.model.norm
    if not isinstance(norm,RotationFinalRMSNorm):raise RuntimeError('Custom Qwen2 was not selected')
    norm.configure(p,q,torch.zeros(32))
    model._klrd_logits_finite.fill_(True) # discard initialization/profile probes
    from ..continuation import tensor_sha256
    return {'weight':tensor_sha256(model.lm_head.weight[:model.config.vocab_size]),
        'gain':tensor_sha256(norm.weight),'bias':None}


@torch.inference_mode()
def set_owner(model,angles):
    norm=model.model.norm
    norm.configure(norm.P,norm.Q,angles)
    return {'fixed_owner_set':True}


def check_logits_finite(model):
    """One CPU/GPU synchronization per completed batch, never per decode step."""
    if not bool(model._klrd_logits_finite.item()):
        raise RuntimeError('Nonfinite vLLM logits during generation; batch rejected')
    return {'logits_finite':True}


def set_native_baseline(model,enabled):
    norm=model.model.norm
    if enabled and bool(torch.count_nonzero(norm.angles)):
        raise RuntimeError('Native baseline requires rotation disabled')
    if not model._klrd_diagnostic_eager:raise RuntimeError('Native baseline requires an eager diagnostic engine')
    norm.native_baseline=bool(enabled)
    return {'native_original_norm':norm.native_baseline}


def start_trace(model):
    if not model._klrd_diagnostic_eager:raise RuntimeError('Trace requires an eager diagnostic engine')
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



@torch.inference_mode()
def check_rotation_graph(model,angles_a,angles_b):
    """Server diagnostic only: prove captured norm reads mutable owner buffers.

    This captures the actual installed final norm and exact rotation on a small
    input. It does not claim to observe vLLM's whole-model graph dispatch.
    """
    norm=model.model.norm
    if norm.capture or norm.native_baseline or model._klrd_trace_enabled:
        raise RuntimeError('Graph check requires ordinary exact-rotation mode')
    original=norm.angles.detach().clone()
    addresses=[norm.P.data_ptr(),norm.Q.data_ptr(),norm.angles.data_ptr()]
    hidden=(torch.linspace(-1.,1.,norm.hidden_size,device=norm.weight.device,dtype=torch.float32)
        +norm.P.sum(-1)+.3*norm.Q.sum(-1))[None,:].repeat(2,1)
    residual=torch.ones_like(hidden)*.25
    keys=[torch.as_tensor(a,device=norm.angles.device,dtype=torch.float32) for a in [angles_a,angles_b]]
    if torch.equal(keys[0],keys[1]):raise RuntimeError('Graph check needs distinct owner angles')
    def forward():return norm(hidden,residual)[0]
    try:
        norm.configure(norm.P,norm.Q,keys[0])
        stream=torch.cuda.Stream(device=hidden.device)
        stream.wait_stream(torch.cuda.current_stream(hidden.device))
        with torch.cuda.stream(stream):
            for _ in range(3):forward()
        torch.cuda.current_stream(hidden.device).wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):captured=forward()
        outputs=[];errors=[]
        for angles in [keys[0],keys[1],keys[0]]:
            norm.configure(norm.P,norm.Q,angles)
            expected=forward().clone()
            graph.replay()
            observed=captured.clone()
            if not bool(torch.isfinite(observed).all()):raise RuntimeError('Nonfinite rotation graph output')
            torch.testing.assert_close(observed,expected,rtol=1e-6,atol=1e-6)
            errors.append(float((observed-expected).abs().max()))
            outputs.append(observed)
        if not torch.equal(outputs[0],outputs[2]):raise RuntimeError('Owner A was not restored during graph replay')
        if torch.equal(outputs[0],outputs[1]):raise RuntimeError('Captured rotation did not respond to owner B')
        if addresses!=[norm.P.data_ptr(),norm.Q.data_ptr(),norm.angles.data_ptr()]:
            raise RuntimeError('Rotation buffer storage changed after capture')
        return {'status':'PASS','scope':'Actual final-norm graph only; not whole-engine dispatch proof',
            'A_B_A_replay_matches_eager':True,'owner_outputs_differ':True,
            'buffer_addresses_preserved':True,'max_abs_error':max(errors)}
    finally:
        norm.configure(norm.P,norm.Q,original)
