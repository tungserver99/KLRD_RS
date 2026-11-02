"""Canonical observable-text primary scoring, no generation-side features."""
import numpy as np
import torch
from ..data import encode_response,prompt_ids
from ..math import sphere_p


class SphericalBlackBoxVerifier:
    def __init__(self,model,tokenizer,readout,L,chunk_size=4):
        self.model,self.tokenizer,self.readout=model,tokenizer,readout
        self.L=np.asarray(L);self.chunk_size=chunk_size;self.device=readout.weight.device

    @torch.inference_mode()
    def _score(self,encodings):
        output=np.zeros((len(encodings),self.L.shape[0]),dtype=np.float64);counts=[]
        if not encodings:return output,np.zeros(0,dtype=np.int64)
        maximum=max(len(e['input_ids']) for e in encodings)
        pad=self.tokenizer.pad_token_id
        if pad is None:pad=self.tokenizer.eos_token_id
        ids=torch.full((len(encodings),maximum),pad,device=self.device,dtype=torch.long);mask=torch.zeros_like(ids)
        for j,e in enumerate(encodings):
            n=len(e['input_ids']);ids[j,:n]=torch.tensor(e['input_ids'],device=self.device);mask[j,:n]=1
        captured=[]
        def capture(module,args):captured.append([args[0][j,e['positions']].detach() for j,e in enumerate(encodings)])
        hook=self.model.model.norm.register_forward_pre_hook(capture)
        try:self.model.model(input_ids=ids,attention_mask=mask,use_cache=False,return_dict=True)
        finally:hook.remove()
        if len(captured)!=1:raise ValueError('Verifier pre-final-norm capture failed')
        for j,e in enumerate(encodings):
            counts.append(len(e['positions']))
            if not e['positions']:continue
            value=np.zeros(self.L.shape[0])
            for start in range(0,len(e['positions']),self.chunk_size):
                targets=torch.tensor(e['targets'][start:start+self.chunk_size],device=self.device)
                # Native E1 FP32 score evaluation; accumulation promoted to FP64.
                score=self.readout.observed(captured[0][j][start:start+self.chunk_size],targets)
                value+=score.double().cpu().numpy().sum(0)
            output[j]=self.L@value
        return output,np.array(counts,dtype=np.int64)

    def score_text_batch(self,rows):
        return self._score([encode_response(self.tokenizer,r['prompt'],r['response']) for r in rows])

    def score_native_batch(self,rows):
        special=set(self.tokenizer.all_special_ids);encoded=[]
        for r in rows:
            prefix=prompt_ids(self.tokenizer,r['prompt']);tokens=r['generated_token_ids']
            pairs=[(len(prefix)+i-1,t) for i,t in enumerate(tokens) if t not in special]
            encoded.append({'input_ids':prefix+tokens,'positions':[p for p,t in pairs],'targets':[t for p,t in pairs]})
        return self._score(encoded)

    @staticmethod
    def key_scores(v,bank):
        norm=float(np.linalg.norm(v));cos=bank@v/norm if norm else np.zeros(len(bank))
        return {'cosine':cos,'p':sphere_p(v,bank),'norm':norm,'zero_vector':norm==0}
