"""No-download integration against a real Qwen2 architecture with tiny random weights."""
from dataclasses import replace
import numpy as np
import torch


def test_qwen_batched_prefinal_norm_replay_matches_native_logits():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from klrd_rs.config import Config
    from klrd_rs.model import ModelAdapter
    from klrd_rs.math import planes
    torch.manual_seed(9)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=35,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=64,
        attention_dropout=0.,rms_norm_eps=1e-6)).eval()
    model.requires_grad_(False)
    adapter=ModelAdapter.__new__(ModelAdapter)
    adapter.model=model;adapter.norm=model.model.norm;adapter.head=model.get_output_embeddings()
    adapter.device=torch.device('cpu');adapter.cfg=replace(Config(),replay_batch_size=4)
    class Tok:pad_token_id=0
    adapter.tokenizer=Tok()
    examples=[{'input_ids':[1,2,3,4,5],'positions':[2,3],'targets':[4,5]},
              {'input_ids':[6,7,8],'positions':[0,1],'targets':[7,8]}]
    captured=adapter.replay(examples)
    p,q=planes(np.eye(16),3,41);engine=adapter.readout(p,q)
    with torch.no_grad():
        for item,r in zip(examples,captured):
            out=model(input_ids=torch.tensor([item['input_ids']])).logits[0,item['positions']]
            torch.testing.assert_close(engine.tail(torch.tensor(r)),out,atol=1e-7,rtol=1e-5)
            assert adapter.native_tail_error(engine,r)<1e-7
