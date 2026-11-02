"""Pinned model replay. No generation, training, adapter or API calls."""
import time
import torch
from .readout import RotationReadout


class ModelAdapter:
    def __init__(self,cfg):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError('Full Qwen replay requires a CUDA GPU; CPU tests use --smoke in a separate output.')
        # Full-vocabulary finite differences need genuine FP32 matmuls, not TF32.
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        print('Loading pinned Qwen and tokenizer in FP32...',flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model,revision=cfg.revision,trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(cfg.model,revision=cfg.revision,
            torch_dtype=torch.float32,attn_implementation='sdpa',trust_remote_code=False).to('cuda').eval()
        self.model.requires_grad_(False)
        if self.model.config.model_type!='qwen2': raise ValueError('Unsupported final normalization architecture')
        self.norm, self.head = self.model.model.norm, self.model.get_output_embeddings()
        self.cfg, self.device = cfg, torch.device('cuda')

    def readout(self,p,q):
        return RotationReadout(self.head.weight,self.norm.weight,self.norm.variance_epsilon,
            torch.as_tensor(p),torch.as_tensor(q),self.cfg.tau,self.head.bias)

    @torch.no_grad()
    def replay(self,encoded):
        """Return only selected pre-final-norm residuals in input row order."""
        result = [None]*len(encoded)
        order = sorted(range(len(encoded)),key=lambda i:len(encoded[i]['input_ids']))
        batches, batch = [],[]
        for i in order:
            n=len(encoded[i]['input_ids'])
            if n > self.model.config.max_position_embeddings: raise ValueError('Replay text exceeds model context window')
            if batch and (len(batch)>=self.cfg.replay_batch_size or (len(batch)+1)*n>self.cfg.replay_batch_tokens):
                batches.append(batch);batch=[]
            batch.append(i)
        if batch:batches.append(batch)
        last=time.monotonic()
        for ids in batches:
            size=max(len(encoded[i]['input_ids']) for i in ids)
            pad=self.tokenizer.pad_token_id
            if pad is None: pad=self.tokenizer.eos_token_id
            tokens=torch.full((len(ids),size),pad,dtype=torch.long,device=self.device)
            mask=torch.zeros_like(tokens)
            for j,i in enumerate(ids):
                n=len(encoded[i]['input_ids']); tokens[j,:n]=torch.tensor(encoded[i]['input_ids'],device=self.device);mask[j,:n]=1
            captured=[]
            def capture(module,args):
                x=args[0]
                for j,i in enumerate(ids):
                    result[i]=x[j,encoded[i]['positions']].detach().float().cpu().numpy()
                captured.append(True)
            hook=self.norm.register_forward_pre_hook(capture)
            try:self.model.model(input_ids=tokens,attention_mask=mask,use_cache=False,return_dict=True)
            finally:hook.remove()
            if len(captured)!=1: raise RuntimeError('Final norm input capture failed')
            if time.monotonic()-last>=self.cfg.log_interval_seconds:
                print('Replay residuals:',sum(x is not None for x in result),'/',len(encoded),flush=True);last=time.monotonic()
        return result

    @torch.no_grad()
    def native_tail_error(self,engine,residuals):
        r=torch.as_tensor(residuals,device=self.device,dtype=self.head.weight.dtype)
        native=self.head(self.norm(r))
        return float((native-engine.tail(r)).abs().max().cpu())
