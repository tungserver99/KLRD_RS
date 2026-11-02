"""Unchanged full-vocabulary KL on native vLLM-visited prefixes, after generation.

This is model replay for measurement, not HF generation. It uses sampled token
IDs, including EOS, and the existing key-independent hashed 1/16 position mask.
"""
import torch
from .cohort import kl_selected
from .sampler import SamplerInvalid


@torch.inference_mode()
def audit_native_prefix_kl(sampler,requests,records,key):
    device=sampler.readout.weight.device
    selected=[];encodings=[]
    for request,record in zip(requests,records):
        tokens=record['generated_token_ids'];prefix=request['prompt_ids']
        positions=[i for i in range(len(tokens)) if kl_selected(sampler.generation_seed,request['prompt_id'],i,sampler.kl_stride)]
        selected.append(positions);encodings.append(prefix+tokens)
    if not any(selected):return
    maximum=max(map(len,encodings));pad=sampler.tokenizer.pad_token_id
    if pad is None:pad=sampler.tokenizer.eos_token_id
    ids=torch.full((len(records),maximum),pad,dtype=torch.long,device=device)
    mask=torch.zeros_like(ids)
    for i,tokens in enumerate(encodings):
        ids[i,:len(tokens)]=torch.tensor(tokens,device=device);mask[i,:len(tokens)]=1
    captured=[]
    def capture(module,args):
        values=[]
        for i,positions in enumerate(selected):
            indices=[len(requests[i]['prompt_ids'])+position-1 for position in positions]
            values.append(args[0][i,indices].detach().clone())
        captured.append(values)
    hook=sampler.model.model.norm.register_forward_pre_hook(capture)
    try:sampler.model.model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
    finally:hook.remove()
    if len(captured)!=1:raise SamplerInvalid('Native-prefix KL capture failed')
    angle=torch.zeros(sampler.L.shape[0],device=device) if key is None else torch.tensor(
        sampler.strength*(sampler.L@key),dtype=torch.float32,device=device)
    for i,positions in enumerate(selected):
        events=[]
        # Retain the original score replay chunk size and full-vocabulary formula.
        for start in range(0,len(positions),4):
            r=captured[0][i][start:start+4]
            lp=sampler.readout.log_probs(r);lq=sampler.readout.log_probs(r,angle)
            values=(lq.exp()*(lq-lp)).sum(-1)
            if not torch.isfinite(values).all() or float(values.min()) < -1e-6:
                raise SamplerInvalid('Invalid full-vocabulary native-prefix KL')
            events.extend({'position':pos,'KL':max(0.,float(value))}
                for pos,value in zip(positions[start:start+4],values.cpu().tolist()))
        records[i]['KL_samples']=events
