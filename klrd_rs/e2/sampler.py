"""Exact rotated-tail categorical sampler with independent streams and KV cache."""
import time
import numpy as np
import torch
from types import MethodType
from .cohort import kl_selected
from ..model import ModelAdapter


def linear_fp64_forward(layer,x):
    """Accumulate the unchanged linear map in FP64; retain FP32 boundaries.

    Cast weights per call to avoid retaining a second full model on the GPU.
    No activation, cache or parameter is stored in FP64.
    """
    bias=None if layer.bias is None else layer.bias.double()
    return torch.nn.functional.linear(x.double(),layer.weight.double(),bias).to(x.dtype)


def stabilize_linear_accumulation(model):
    count=0
    for layer in model.modules():
        if isinstance(layer,torch.nn.Linear) and not getattr(layer,'_e2_fp64_accumulation',False):
            if layer.weight.dtype!=torch.float32:raise ValueError('E2 Linear weights must remain FP32')
            layer.forward=MethodType(linear_fp64_forward,layer)
            layer._e2_fp64_accumulation=True
            count+=1
    model._e2_linear_accumulation='float64'
    return count


class E2ModelAdapter(ModelAdapter):
    """E2 FP32 eager loader, kept here so runtime needs no new loader module."""
    def __init__(self,cfg):
        from transformers import AutoModelForCausalLM,AutoTokenizer
        if not torch.cuda.is_available():raise RuntimeError('E2 requires a CUDA GPU')
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        print('Loading pinned HF reference/verifier in FP32; this model is not used for vLLM generation.',flush=True)
        self.tokenizer=AutoTokenizer.from_pretrained(cfg.model,revision=cfg.revision,trust_remote_code=False)
        self.model=AutoModelForCausalLM.from_pretrained(cfg.model,revision=cfg.revision,
            torch_dtype=torch.float32,attn_implementation='eager',trust_remote_code=False).to('cuda').eval()
        self.model.requires_grad_(False)
        if self.model.config.model_type!='qwen2':raise ValueError('Unsupported final normalization architecture')
        count=stabilize_linear_accumulation(self.model)
        print(f'Verifier numerical policy: FP32 boundaries, FP64 accumulation in {count} Linear modules. The separate vLLM generation process uses FP32.',flush=True)
        self.norm,self.head=self.model.model.norm,self.model.get_output_embeddings()
        self.cfg,self.device=cfg,torch.device('cuda')


class SamplerInvalid(RuntimeError):pass


class CachedSession:
    def __init__(self,model,tokenizer,prompts):
        self.model=model;self.device=model.get_input_embeddings().weight.device
        length=max(map(len,prompts));pad=tokenizer.pad_token_id
        if pad is None:pad=tokenizer.eos_token_id
        self.input_ids=torch.full((len(prompts),length),pad,dtype=torch.long,device=self.device)
        self.mask=torch.zeros_like(self.input_ids)
        for i,prompt in enumerate(prompts):
            self.input_ids[i,-len(prompt):]=torch.tensor(prompt,device=self.device);self.mask[i,-len(prompt):]=1
        self.cache=None;self.decode_calls=0
        self.residual,self.cache=self._forward(self.input_ids,self.mask,None,True)
        if self.cache is None:raise SamplerInvalid('KV cache was not returned')

    @torch.inference_mode()
    def _forward(self,ids,mask,cache,use_cache):
        captured=[]
        def capture(module,args):captured.append(args[0][:,-1,:].detach().clone())
        hook=self.model.model.norm.register_forward_pre_hook(capture)
        positions=(mask.cumsum(-1)-1).clamp_min(0)
        if cache is not None:positions=positions[:,-ids.shape[1]:]
        try:
            output=self.model.model(input_ids=ids,attention_mask=mask,position_ids=positions,
                past_key_values=cache,use_cache=use_cache,return_dict=True)
        finally:hook.remove()
        if len(captured)!=1:raise SamplerInvalid('Expected one pre-final-norm residual')
        return captured[0],getattr(output,'past_key_values',None)

    @torch.inference_mode()
    def advance(self,tokens,active):
        self.input_ids=torch.cat([self.input_ids,tokens[:,None]],-1)
        self.mask=torch.cat([self.mask,active.to(self.mask.dtype)[:,None]],-1)
        self.residual,self.cache=self._forward(tokens[:,None],self.mask,self.cache,True)
        self.decode_calls+=1

    @torch.inference_mode()
    def full_prefix_residual(self):
        return self._forward(self.input_ids,self.mask,None,False)[0]


class ExactRotationTeacherSampler:
    def __init__(self,model,tokenizer,rotation_readout,L,lambda_frozen,kl_stride=16,generation_seed=2026110102):
        self.model,self.tokenizer,self.readout=model,tokenizer,rotation_readout
        self.L=np.asarray(L);self.strength=lambda_frozen;self.kl_stride=kl_stride;self.generation_seed=generation_seed
        if self.readout.weight.dtype!=torch.float32:raise SamplerInvalid('E2 reference requires FP32 readout')
        eos=getattr(model.generation_config,'eos_token_id',None)
        self.eos=set(eos if isinstance(eos,(list,tuple)) else [] if eos is None else [eos])
        if tokenizer.eos_token_id is not None:self.eos.add(tokenizer.eos_token_id)
        if not self.eos:raise SamplerInvalid('No native EOS stopping IDs')

    def session(self,prompts):return CachedSession(self.model,self.tokenizer,prompts)

    @torch.inference_mode()
    def generate(self,requests,key,max_new_tokens=512):
        device=self.readout.weight.device
        generators=[torch.Generator(device=device).manual_seed(int(r['seed'])) for r in requests]
        if device.type=='cuda':torch.cuda.synchronize(device)
        start=time.perf_counter()
        session=self.session([r['prompt_ids'] for r in requests])
        angle=torch.zeros(self.L.shape[0],device=device,dtype=torch.float32) if key is None else torch.tensor(
            self.strength*(self.L@np.asarray(key)),device=device,dtype=torch.float32)
        active=torch.ones(len(requests),device=device,dtype=torch.bool)
        ids=[[] for _ in requests];events=[[] for _ in requests]
        for step in range(max_new_tokens):
            logq=self.readout.log_probs(session.residual,angle).float()
            probabilities=logq.exp()
            if not torch.isfinite(logq).all() or float((probabilities.sum(-1)-1).abs().max())>2e-6:
                raise SamplerInvalid('Nonfinite or nonnormalized exact teacher probabilities')
            sample_rows=[i for i,r in enumerate(requests) if active[i] and kl_selected(self.generation_seed,r['prompt_id'],step,self.kl_stride)]
            if sample_rows:
                lp=self.readout.log_probs(session.residual[sample_rows]).float()
                kl=(probabilities[sample_rows]*(logq[sample_rows]-lp)).sum(-1)
                if not torch.isfinite(kl).all() or float(kl.min()) < -1e-6:raise SamplerInvalid('Invalid full-vocabulary visited-prefix KL')
                for i,v in zip(sample_rows,kl.cpu().tolist()):events[i].append({'position':step,'KL':max(0.,v)})
            chosen=torch.full((len(requests),),self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,dtype=torch.long,device=device)
            for i in range(len(requests)):
                if not bool(active[i]):continue
                token=int(torch.multinomial(probabilities[i],1,generator=generators[i]).item())
                if not 0<=token<probabilities.shape[-1]:raise SamplerInvalid('Sampled token outside vocabulary')
                ids[i].append(token);chosen[i]=token
                if token in self.eos:active[i]=False
            if not bool(active.any()) or step+1==max_new_tokens:break
            session.advance(chosen,active)
        if device.type=='cuda':torch.cuda.synchronize(device)
        elapsed=time.perf_counter()-start
        records=[];special=set(self.tokenizer.all_special_ids)
        for i,r in enumerate(requests):
            text=self.tokenizer.decode(ids[i],skip_special_tokens=True,clean_up_tokenization_spaces=False)
            visible=[t for t in ids[i] if t not in special]
            recovered=list(self.tokenizer.encode(text,add_special_tokens=False))
            records.append({'prompt_id':r['prompt_id'],'seed':int(r['seed']),'response':text,
                'generated_token_ids':ids[i],'eos_terminated':bool(ids[i] and ids[i][-1] in self.eos),
                'max_length_truncated':len(ids[i])==max_new_tokens and ids[i][-1] not in self.eos,
                'roundtrip_equal':visible==recovered,'native_visible_tokens':len(visible),
                'recovered_tokens':len(recovered),'KL_samples':events[i],
                'sampling_temperature_applications':1,'amortized_generation_seconds':elapsed/len(requests),
                'batch_size':len(requests),'decode_steps':session.decode_calls})
        return records


class CleanSampler:
    def __init__(self,sampler):self.sampler=sampler
    def generate(self,requests,max_new_tokens=512):return self.sampler.generate(requests,None,max_new_tokens=max_new_tokens)
