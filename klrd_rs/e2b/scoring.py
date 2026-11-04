"""Checkpointed server-only paraphrase scoring through the unchanged E2 verifier."""
from pathlib import Path
from importlib.metadata import version
import numpy as np
from klrd_scw.artifacts import ensure_manifest
from ..io import read_json,write_json,sha256,digest,save_npz
from .config import SCORE_BATCH_SIZE


def load_score_batch(directory,name,identity):
    directory=Path(directory);path=directory/(name+'.npz');marker=path.with_suffix('.sha256.json')
    if not marker.exists():return None
    receipt=read_json(marker)
    if receipt['identity']!=identity:raise ValueError('Paraphrase score text/IDs/runtime changed')
    if not path.exists() or sha256(path)!=receipt['sha256']:raise ValueError('Paraphrase score cache checksum mismatch')
    with np.load(path,allow_pickle=False) as value:return {k:value[k].copy() for k in value.files}


def save_score_batch(directory,name,identity,arrays):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True,mode=0o700)
    previous=load_score_batch(directory,name,identity)
    if previous is not None:
        if set(previous)!=set(arrays) or any(not np.array_equal(previous[k],v) for k,v in arrays.items()):
            raise ValueError('Refuse replacement of committed paraphrase scores')
        return
    path=directory/(name+'.npz');save_npz(path,**arrays)
    write_json(path.with_suffix('.sha256.json'),{'identity':identity,'sha256':sha256(path)})


def scoring_runtime(state):
    import torch
    from ..e2.config import E2Config
    cfg=E2Config()
    source=state['manifest']['source_E2_environment']
    values={'torch':str(torch.__version__),'transformers':version('transformers'),
        'numpy':version('numpy'),'scipy':version('scipy'),'CUDA':torch.version.cuda}
    differences={name:{'E2':source.get(name),'E2B':actual}
        for name,actual in values.items() if actual!=source.get(name)}
    write_json(state['out']/'provenance/verification_environment_comparison.json',{
        'E2':{name:source.get(name) for name in values},'E2B':values,
        'differences':differences,'policy':'Version differences are recorded warnings, not a scoring gate',
        'comparison_note':'Original vectors are reused from E2; software differences may affect numerical comparison'})
    for name,change in differences.items():
        print(f'E2B warning: {name}: E2={change["E2"]}; current={change["E2B"]}. Continuing verification.',flush=True)
    if not torch.cuda.is_available():raise RuntimeError('E2B verification requires the GPU server; no CPU substitution')
    return dict(values,GPU=torch.cuda.get_device_name(0),batch_size=SCORE_BATCH_SIZE,
        verifier_source_sha256=sha256(state['root']/'klrd_rs/e2/verifier.py'),
        model=cfg.model,revision=cfg.revision,
        verification_attention_implementation='eager',weights_activations_dtype='float32',
        linear_accumulation='float64',vector_accumulation='float64',score_execution_chunk_minimum=64,
        prompt_policy='Exact unchanged original prompt',text_policy='Full canonical paraphrase text; no length clipping')


def get_adapter(state):
    import torch
    from ..config import Config
    from ..e2.sampler import E2ModelAdapter
    from ..continuation import tensor_sha256
    from ..e2.config import E2Config
    from ..e2.verifier import SphericalBlackBoxVerifier
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    adapter=E2ModelAdapter(Config());frozen=state['frozen']
    readout=adapter.readout(frozen['P'],frozen['Q'])
    actual={'weight':tensor_sha256(readout.weight),'gain':tensor_sha256(readout.gain),
        'bias':None if readout.bias is None else tensor_sha256(readout.bias)}
    environment=state['manifest']['source_E2_environment']
    if actual!=environment['readout_parameter_sha256']:
        raise ValueError('E2B verifier readout differs from frozen E2/E1')
    if (adapter.tokenizer.chat_template!=frozen['chat_template']
        or digest(adapter.tokenizer.get_vocab())!=environment['generation']['tokenizer_vocab_sha256']):
        raise ValueError('E2B tokenizer/chat template differs from source E2')
    verifier=SphericalBlackBoxVerifier(adapter.model,adapter.tokenizer,readout,frozen['L'],E2Config().score_chunk)
    return adapter,verifier


def verifier_rows(pairs,tokenizer,max_context):
    """Use the verifier's own complete encoding, including original chat prefix."""
    from ..data import encode_response
    rows=[{'prompt':r['prompt'],'response':r['paraphrased_text']} for r in pairs]
    for row in rows:
        if len(encode_response(tokenizer,row['prompt'],row['response'])['input_ids'])>max_context:
            raise RuntimeError('Paraphrase exceeds the pinned verifier context window; no truncation, filtering or reduced-cohort result')
    return rows

def score(state,pairs):
    from .quality import write_quality
    import torch
    if len(pairs)!=3600:raise ValueError('Exactly3600 valid paraphrases are required before scoring')
    runtime=scoring_runtime(state)
    ensure_manifest(state['out']/'provenance/verification_runtime.json',runtime)
    directory=state['out']/'private/paraphrase_scores';all_vectors=[];all_tokens=[];cache_hashes={}
    adapter=verifier=None
    for start in range(0,len(pairs),SCORE_BATCH_SIZE):
        batch=pairs[start:start+SCORE_BATCH_SIZE];name=f'batch_{start//SCORE_BATCH_SIZE:04}'
        expected_ids=[r['response_id'] for r in batch]
        identity={'source_manifest_sha256':digest(state['manifest']),'pairs_sha256':digest(batch),'runtime':runtime}
        arrays=load_score_batch(directory,name,identity)
        if arrays is None:
            if adapter is None:adapter,verifier=get_adapter(state)
            rows=verifier_rows(batch,adapter.tokenizer,adapter.model.config.max_position_embeddings)

            vectors,tokens=verifier.score_text_batch(rows)
            arrays={'vectors':vectors,'eligible_tokens':tokens,'response_ids':np.array(expected_ids)}
            save_score_batch(directory,name,identity,arrays)
        if (set(arrays)!={'vectors','eligible_tokens','response_ids'} or arrays['response_ids'].tolist()!=expected_ids
            or arrays['vectors'].shape!=(len(batch),32) or arrays['eligible_tokens'].shape!=(len(batch),)
            or not np.isfinite(arrays['vectors']).all() or (arrays['eligible_tokens']<0).any()):
            raise ValueError('Invalid committed paraphrase score array/ordering')
        all_vectors.extend(arrays['vectors']);all_tokens.extend(arrays['eligible_tokens'])
        cache_hashes[name]=sha256(directory/(name+'.npz'))
        if (start//SCORE_BATCH_SIZE+1)%10==0 or start+len(batch)==len(pairs):
            print(f'E2B verification:{start+len(batch)}/3600 paired responses cached.',flush=True)
    target=state['out']/'private/paraphrase_vectors.npz'
    values={'vectors':np.array(all_vectors,dtype=np.float64),'eligible_tokens':np.array(all_tokens,dtype=np.int64),
        'response_ids':np.array([r['response_id'] for r in pairs])}
    if target.exists():
        with np.load(target,allow_pickle=False) as existing:
            if set(existing.files)!=set(values) or any(not np.array_equal(existing[k],v) for k,v in values.items()):
                raise ValueError('Committed complete paraphrase vectors changed')
    else:save_npz(target,**values)
    ensure_manifest(state['out']/'provenance/verification_complete.json',{'status':'COMPLETE','responses':3600,
        'pairs_sha256':digest(pairs),'vectors_sha256':sha256(target),'batch_cache_sha256':cache_hashes,
        'runtime':runtime,'original_vector_source':'Checksummed original E2 canonical text-only score caches'})
    if adapter is not None:tokenizer=adapter.tokenizer
    else:
        from transformers import AutoTokenizer
        from ..e2.config import E2Config
        cfg=E2Config()
        tokenizer=AutoTokenizer.from_pretrained(cfg.model,revision=cfg.revision,trust_remote_code=False,local_files_only=True)
        if (tokenizer.chat_template!=state['frozen']['chat_template']
            or digest(tokenizer.get_vocab())!=state['manifest']['source_E2_environment']['generation']['tokenizer_vocab_sha256']):
            raise ValueError('Cached quality tokenizer differs from source E2')
    write_quality(state,pairs,tokenizer)
    return values


def load_vectors(state,pairs):
    receipt=read_json(state['out']/'provenance/verification_complete.json')
    if receipt['runtime']!=read_json(state['out']/'provenance/verification_runtime.json'):
        raise ValueError('Complete verification runtime receipt changed')
    path=state['out']/'private/paraphrase_vectors.npz'
    if receipt['status']!='COMPLETE' or receipt['responses']!=3600 or receipt['pairs_sha256']!=digest(pairs) or receipt['vectors_sha256']!=sha256(path):
        raise ValueError('Complete paraphrase score receipt is missing/incompatible')
    with np.load(path,allow_pickle=False) as value:after={k:value[k].copy() for k in value.files}
    with np.load(state['out']/'private/original_vectors.npz',allow_pickle=False) as value:before={k:value[k].copy() for k in value.files}
    expected=[r['response_id'] for r in pairs]
    if any(a['response_ids'].tolist()!=expected or a['vectors'].shape!=(3600,32) or not np.isfinite(a['vectors']).all()
           or a['eligible_tokens'].shape!=(3600,) or (a['eligible_tokens']<0).any() for a in [before,after]):
        raise ValueError('Original/paraphrase vectors do not match the frozen population')
    return before,after
