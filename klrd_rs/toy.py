"""Deterministic toy fixture for CPU smoke tests; never research evidence."""
import hashlib
import numpy as np
import torch
from .readout import RotationReadout


class ToyTokenizer:
    all_special_ids=[0]
    chat_template='synthetic'
    def apply_chat_template(self,messages,**kwargs):return [1,2]+self.encode(messages[0]['content'])+[3]
    def encode(self,text,**kwargs):return [4+ord(c)%27 for c in text]


class ToyAdapter:
    def __init__(self):
        self.tokenizer=ToyTokenizer()
        rng=np.random.default_rng(18)
        self.weight=torch.tensor(rng.normal(size=(31,12))*.3,dtype=torch.float64)
        self.gain=torch.ones(12,dtype=torch.float64)
    def readout(self,p,q):return RotationReadout(self.weight,self.gain,1e-6,torch.tensor(p),torch.tensor(q),.8)
    def replay(self,encoded):
        result=[]
        for e in encoded:
            rows=[]
            for pos in e['positions']:
                seed=int.from_bytes(hashlib.sha256(bytes(e['input_ids'][:pos+1])).digest()[:8],'little')
                rows.append(np.random.default_rng(seed).normal(size=12))
            result.append(np.array(rows))
        return result
    def native_tail_error(self,engine,residuals):return 0.


def toy_inputs():
    rows=[{'prompt_id':str(i),'prompt':'toy prompt '+str(i),
        'response':'toy response for mathematical integration check '+str(i)*3} for i in range(100)]
    return {'calibration':rows[:80],'original':rows[80:],'paraphrase':None,
        'basis':np.eye(12),'mean':np.zeros(12),'index':{'synthetic':True,'seed':18},'manifest':{}}
