"""Predeclared context selection and resumable text-score aggregation."""
import time
import numpy as np
import torch
from .data import encode_response


def contexts(rows, adapter, cap, seed, store, name):
    saved=store.load(name)
    if saved is not None:return saved
    excluded=[r['prompt_id'] for r in rows if r.get('text_roundtrip_equal') is False]
    rows=[r for r in rows if r.get('text_roundtrip_equal') is not False]
    encoded=[encode_response(adapter.tokenizer,r['prompt'],r['response']) for r in rows]
    counts=np.array([len(x['positions']) for x in encoded]);total=int(counts.sum())
    if total<max(2,cap//10): raise ValueError('Insufficient recoverable clean prefix contexts')
    chosen=np.sort(np.random.default_rng(seed).choice(total,min(total,cap),replace=False))
    offsets=np.r_[0,counts.cumsum()]
    pending=[];owners=[];targets=[];positions=[]
    for j,(e,row) in enumerate(zip(encoded,rows)):
        pick=chosen[(chosen>=offsets[j])&(chosen<offsets[j+1])]-offsets[j]
        if not len(pick):continue
        pos=[e['positions'][i] for i in pick];tok=[e['targets'][i] for i in pick]
        # Prefix through the last selected target is sufficient for replay.
        pending.append({'input_ids':e['input_ids'][:max(pos)+2],'positions':pos,'targets':tok})
        owners.extend([row['prompt_id']]*len(pick));targets.extend(tok);positions.extend(pos)
    residuals=adapter.replay(pending)
    result={'residuals':np.concatenate(residuals), 'prompt_ids':np.array(owners,dtype=str),
            'targets':np.array(targets,dtype=np.int64),'positions':np.array(positions,dtype=np.int64),
            'excluded_prompt_ids':np.array(excluded,dtype=str),
            'eligible_prompt_ids':np.array([r['prompt_id'] for r in rows],dtype=str)}
    store.save(name,result)
    return result


@torch.no_grad()
def sampled_scores(data,engine,cfg,seed):
    rng=torch.Generator(device=engine.weight.device).manual_seed(seed)
    arrays=[];last=time.monotonic()
    for start in range(0,len(data['residuals']),cfg.readout_chunk):
        x=torch.as_tensor(data['residuals'][start:start+cfg.readout_chunk],device=engine.weight.device,dtype=engine.weight.dtype)
        arrays.append(engine.observed(x,generator=rng).double().cpu().numpy())
        if time.monotonic()-last>=cfg.log_interval_seconds:
            print('Fisher sampled score contexts:',start,'/',len(data['residuals']),flush=True);last=time.monotonic()
    return np.concatenate(arrays)


@torch.no_grad()
def response_scores(rows,adapter,engine,cfg,store,name):
    if rows is None:return None
    saved=store.load(name)
    if saved is not None:return saved['vectors']
    vectors=[];lengths=[];last=time.monotonic()
    # Per-response checkpoints avoid replaying an entire cohort after interruption.
    for j,row in enumerate(rows):
        cached=store.load(f'{name}_{j:05}')
        if cached is not None:
            vectors.append(cached['vector']);lengths.append(int(cached['tokens']));continue
        e=encode_response(adapter.tokenizer,row['prompt'],row['response'])
        value=np.zeros(engine.p.shape[1],dtype=np.float64)
        if e['positions']:
            residuals=adapter.replay([e])[0]
            for start in range(0,len(residuals),cfg.readout_chunk):
                x=torch.as_tensor(residuals[start:start+cfg.readout_chunk],device=engine.weight.device,dtype=engine.weight.dtype)
                targets=torch.tensor(e['targets'][start:start+cfg.readout_chunk],device=x.device)
                value+=engine.observed(x,targets).double().cpu().numpy().sum(0)
        store.save(f'{name}_{j:05}',{'vector':value,'tokens':np.array(len(e['positions']))})
        vectors.append(value);lengths.append(len(e['positions']))
        if time.monotonic()-last>=cfg.log_interval_seconds:
            print(name,j+1,'/',len(rows),'responses scored',flush=True);last=time.monotonic()
    vectors=np.stack(vectors)
    store.save(name,{'vectors':vectors,'token_counts':np.array(lengths),'prompt_ids':np.array([r['prompt_id'] for r in rows])})
    return vectors
