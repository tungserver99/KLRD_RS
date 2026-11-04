"""Ordinary clean/merged Qwen inference in a short-lived, key-free process.

This module never loads the rotation, verifier, geometry or owner material.
Each prompt has its own seed; committed responses are never requested again.
Engine-level failures stop with an explicit private diagnostic for resume.
"""
from pathlib import Path
import hashlib
import os
import subprocess
import sys
import time
from ..io import digest, read_json, write_json, sha256


def _journal(path):
    from klrd_scw.artifacts import read_jsonl
    return read_jsonl(path, repair_tail=True) if Path(path).exists() else []


def _seed(base, prompt_id, kind):
    return int(hashlib.sha256(f'{base}|{kind}|{prompt_id}'.encode()).hexdigest()[:15], 16)


def _request_seed(settings,row):
    """Explicit frozen per-model seeds for matched-length studies; no reseeding."""
    value=row.get('sampling_seed',_seed(settings['seed'],row['prompt_id'],settings['kind']))
    if not isinstance(value,int) or isinstance(value,bool) or not 0<=value<2**63:
        raise ValueError('Per-prompt sampling seed must be a nonnegative signed64-bit integer')
    return value


def _same_sampling_identity(previous,current):
    """Allow batching changes, never a backend/dtype/model/prompt/seed change."""
    if set(previous)!=set(current):return False
    if any(previous[k]!=current[k] for k in current if k!='settings'):return False
    old=dict(previous['settings']);new=dict(current['settings'])
    for name in ('requested_batch_size','actual_request_batch_size'):
        old.pop(name,None);new.pop(name,None)
    return old==new


def _execution(cfg):
    from .vllm_execution import tuning
    values=tuning() if cfg.generation_backend=='vllm' else {'request_batch_size':1}
    return dict(values,backend=cfg.generation_backend,
        cudagraph_mode='FULL_DECODE_ONLY' if cfg.generation_backend=='vllm' else 'NONE',
        precision_note='FP32 remains FP32; sampled token identity across backends/batch shapes is not asserted')


def generate(state, rows, output_path, model_path=None, kind='student'):
    """Resume an immutable sampling journal, then wait for GPU process exit."""
    cfg=state['cfg']; output_path=Path(output_path).resolve()
    if kind=='student' and model_path is None:
        raise ValueError('Student generation requires an explicitly merged model')
    if kind not in ('student','untrained','clean_train','smoke_clean'):
        raise ValueError('Unknown ordinary generation condition: '+kind)
    prompts=[dict(r, prompt_id=str(r['prompt_id'])) for r in rows]
    if len({r['prompt_id'] for r in prompts})!=len(prompts):
        raise ValueError('Duplicate generation prompt IDs')
    model=str(Path(model_path).resolve()) if model_path is not None else cfg.model
    model_files=None
    if model_path is not None:
        # Evaluate has just verified all merged weight hashes under run.lock.
        # Reuse that same-process result instead of reading the GB checkpoint twice.
        verified=state.get('verified_merged_file_manifest',{})
        if verified.get('path')==model:
            model_files=verified['files']
        else:
            from .provenance import file_manifest
            model_files=file_manifest(model_path)
    execution=_execution(cfg)
    settings={'model':model,'revision':cfg.revision if model_path is None else None,
        'tokenizer':cfg.model,'tokenizer_revision':cfg.revision,
        'backend':cfg.generation_backend,'dtype':'float32' if kind in ('clean_train','smoke_clean') else cfg.generation_dtype,
        'temperature':cfg.tau,'top_p':1.,'top_k':0,'max_new_tokens':cfg.max_new_tokens,
        'seed':cfg.seed,'kind':kind,'requested_batch_size':cfg.generation_batch_size,
        'actual_request_batch_size':execution['request_batch_size'],
        'model_files':model_files}
    identity={'run':state['identity'],'settings':settings,'prompts_sha256':digest(prompts)}
    binding=output_path.with_suffix('.identity.json')
    if binding.exists():
        previous=read_json(binding)
        if previous!=identity:
            if not _same_sampling_identity(previous,identity):
                raise ValueError('Generation input/model/sampling identity changed: '+str(output_path))
            # Keep historical digests and every cached record unchanged. New
            # rows record their actual batching in generation_execution.
            identity=previous
    else:
        if output_path.exists():raise ValueError('Unbound generation journal: '+str(output_path))
        write_json(binding,identity)
    request=output_path.with_suffix('.request.json')
    write_json(request,{'identity':identity,'rows':prompts,'output':str(output_path),'execution':execution})
    from .journal import load_generation_journal
    manifest_name='train_manifest.jsonl' if output_path.stem=='train_clean' else 'val_manifest.jsonl'
    recoverable=(kind=='clean_train' and output_path.stem in ('train_clean','validation_clean')
        and not (state['out']/'private'/manifest_name).exists()
        and not (state['out']/'provenance/data_complete.json').exists()
        and not (state['out']/'training/eligibility.json').exists())
    existing=(load_generation_journal(output_path,prompts,identity,_validate,allow_recovery=recoverable)
        if kind=='clean_train' else _journal(output_path))
    _validate(existing,prompts,identity)
    write_json(output_path.with_suffix('.execution.json'),dict(execution,
        pending_at_start=len(prompts)-len(existing),worker_needed=len(existing)<len(prompts)))
    print(f'E3 ordinary generation ({kind}): {len(existing)}/{len(prompts)} saved; {len(prompts)-len(existing)} pending.',flush=True)
    if len(existing)<len(prompts):
        env=os.environ.copy(); env['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
        env['TRITON_F32_DEFAULT']='ieee';env['VLLM_FLOAT32_MATMUL_PRECISION']='highest'
        env['NVIDIA_TF32_OVERRIDE']='0';env['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']='0'
        env.pop('VLLM_USE_V1',None);env.pop('VLLM_ATTENTION_BACKEND',None)
        subprocess.run([sys.executable,'-u','-m','klrd_rs.e3.generation','--request',str(request)],
            cwd=str(state['root']),env=env,check=True)
    result=_journal(output_path); _validate(result,prompts,identity)
    if len(result)!=len(prompts):raise ValueError('Generation worker exited with an incomplete journal')
    write_json(output_path.with_suffix('.complete.json'),{'identity':digest(identity),
        'sha256':sha256(output_path),'rows':len(result),'errors':sum(r.get('error') is not None for r in result),
        'gpu_lifecycle':'dedicated subprocess exited before caller proceeds'})
    return _ordered_records(result,prompts)


def _validate(records,prompts,identity):
    if len(records)>len(prompts):raise ValueError('Generation journal contains unexpected rows')
    expected={r['prompt_id']:r for r in prompts};seen=set()
    if len(expected)!=len(prompts):raise ValueError('Duplicate requested generation prompt IDs')
    for row in records:
        if not isinstance(row,dict):raise ValueError('Generation journal record is not an object')
        pid=row.get('prompt_id')
        if not isinstance(pid,str) or pid not in expected or pid in seen:
            raise ValueError('Generation journal has duplicate/unexpected prompt IDs')
        seen.add(pid);prompt=expected[pid]
        body={k:v for k,v in row.items() if k!='record_sha256'}
        if (row.get('prompt_id')!=prompt['prompt_id'] or row.get('prompt')!=prompt['prompt']
                or row.get('generation_identity')!=digest(identity) or row.get('record_sha256')!=digest(body)
                or row.get('seed')!=_request_seed(identity['settings'],prompt)):
            raise ValueError('Generation journal prompt/identity/hash/seed mismatch')


def _pending_rows(prompts,records):
    seen={r['prompt_id'] for r in records}
    return [row for row in prompts if row['prompt_id'] not in seen]


def _ordered_records(records,prompts):
    index={row['prompt_id']:row for row in records}
    if len(index)!=len(prompts):raise ValueError('Generation journal has missing responses')
    return [index[row['prompt_id']] for row in prompts]


def _record(row,ids,tokenizer,settings,identity,error=None,finish_reason=None,stop_reason=None,execution=None,elapsed_seconds=None):
    ids=[int(i) for i in ids]; special=set(tokenizer.all_special_ids)
    eos=settings['eos_token_ids']; terminal=ids[-1] if ids and ids[-1] in eos else None
    # vLLM may remove its native stop token from returned sampled IDs. Keep the
    # engine's explicit stop provenance; never fabricate a token in the sequence.
    engine_eos=stop_reason if isinstance(stop_reason,int) and stop_reason in eos else None
    terminated=terminal is not None or (finish_reason=='stop' and engine_eos is not None)
    visible=[i for i in ids if i not in special]
    text=tokenizer.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    roundtrip=tokenizer.encode(text,add_special_tokens=False)
    seed=_request_seed(settings,row)
    result=dict(row,response=text,seed=seed,generated_token_ids=ids,
        native_visible_token_ids=visible,native_visible_tokens=len(visible),
        eos_token_ids=eos,eos_terminated=terminated,native_eos_token_id=terminal if terminal is not None else engine_eos,
        eos_token_present_in_generated_ids=terminal is not None,
        max_length_truncated=not terminated and len(ids)>=settings['max_new_tokens'],
        finish_reason=finish_reason,stop_reason=stop_reason,roundtrip_equal=roundtrip==visible,
        returned_text_token_ids=roundtrip,error=error,generation_identity=digest(identity),
        generation_backend=settings['backend'],generation_dtype=settings['dtype'],
        sampling={'temperature':settings['temperature'],'top_p':1.,'top_k':0,'native_eos':True},
        response_id=digest({'generation':digest(identity),'prompt_id':row['prompt_id'],'seed':seed,'tokens':ids,'error':error}))
    if execution is not None:result['generation_execution']=dict(execution)
    if elapsed_seconds is not None:
        result['amortized_generation_seconds']=float(elapsed_seconds)
        result['generation_timing_scope']='engine inference wall time; excludes model loading/graph capture'
    result['record_sha256']=digest(result)
    return result


def _vllm_options(settings,execution):
    """Standard Qwen engine: no custom model, rotation, key or logits hook."""
    from .vllm_execution import capture_sizes
    concurrent=execution.get('max_num_seqs',execution['request_batch_size'])
    graph_mode=execution.get('cudagraph_mode','FULL_DECODE_ONLY')
    options=dict(model=settings['model'],revision=settings['revision'],tokenizer=settings['tokenizer'],
        tokenizer_revision=settings['tokenizer_revision'],dtype=settings['dtype'],trust_remote_code=False,
        max_model_len=execution.get('max_model_len',2048),max_num_seqs=concurrent,
        max_num_batched_tokens=execution.get('max_num_batched_tokens',8192),
        enforce_eager=graph_mode=='NONE',gpu_memory_utilization=execution.get('gpu_memory_utilization',.85),
        generation_config='vllm',enable_prefix_caching=True,enable_chunked_prefill=True,
        compilation_config={'mode':0,'cudagraph_mode':graph_mode,
            'cudagraph_capture_sizes':[] if graph_mode=='NONE' else capture_sizes(concurrent)})
    if settings['dtype']=='float32':options['attention_config']={'backend':'TRITON_ATTN'}
    return options


def _vllm_generate_pending(model,tokenizer,rows,settings,identity,execution,params_type,commit_batch):
    """Batch only missing prompts, one independently seeded sample per prompt.

    Validate the whole batch mapping before committing any of its rows. An
    engine-level failure propagates; it never produces synthetic empty samples
    or a silent backend fallback. Already committed rows remain resumable.
    """
    from ..data import prompt_ids
    batch_size=execution['request_batch_size']
    for start in range(0,len(rows),batch_size):
        batch=rows[start:start+batch_size]
        prefixes=[prompt_ids(tokenizer,r['prompt']) for r in batch]
        params=[params_type(n=1,temperature=settings['temperature'],top_p=1.,top_k=-1,
            max_tokens=settings['max_new_tokens'],seed=_request_seed(settings,r),
            ignore_eos=False,stop_token_ids=settings['eos_token_ids'],detokenize=True,
            skip_special_tokens=True,repetition_penalty=1.,presence_penalty=0.,frequency_penalty=0.) for r in batch]
        print(f'E3 vLLM batch {start//batch_size+1}: START {len(batch)} requests; '
            f'max concurrent={execution.get("max_num_seqs",batch_size)}.',flush=True)
        started=time.perf_counter()
        outputs=model.generate([{'prompt_token_ids':p} for p in prefixes],sampling_params=params,use_tqdm=True)
        elapsed=time.perf_counter()-started
        if len(outputs)!=len(batch):raise ValueError('vLLM returned an incomplete batch; no rows from this batch committed')
        for output,prefix in zip(outputs,prefixes):
            if (list(output.prompt_token_ids)!=prefix or len(output.outputs)!=1):
                raise ValueError('vLLM returned unexpected prompt/order/sample count; no rows from this batch committed')
        records=[]
        for row,output in zip(batch,outputs):
            answer=output.outputs[0]
            actual=dict(execution,actual_request_batch_size=len(batch))
            records.append(_record(row,list(answer.token_ids),tokenizer,settings,identity,
                finish_reason=answer.finish_reason,stop_reason=answer.stop_reason,execution=actual,
                elapsed_seconds=elapsed/len(batch)))
        commit_batch(records)
        tokens=sum(r['native_visible_tokens'] for r in records)
        print(f'E3 vLLM batch: {tokens} visible tokens; generation={elapsed:.1f}s '
            f'({tokens/max(elapsed,1e-9):.1f} visible tok/s).',flush=True)


def _worker(request_path):
    from transformers import AutoTokenizer, AutoConfig, GenerationConfig
    from .provenance import append_event,append_events
    request=read_json(request_path); identity=request['identity']; settings=dict(identity['settings'])
    rows=request['rows']; output=Path(request['output']); existing=_journal(output)
    execution=request.get('execution',{'backend':settings['backend'],'request_batch_size':1,'cudagraph_mode':'NONE'})
    _validate(existing,rows,identity)
    pending=_pending_rows(rows,existing)
    tokenizer=AutoTokenizer.from_pretrained(settings['tokenizer'],revision=settings['tokenizer_revision'],trust_remote_code=False)
    model_config=AutoConfig.from_pretrained(settings['model'],revision=settings['revision'],trust_remote_code=False)
    if model_config.model_type!='qwen2' or getattr(model_config,'auto_map',None):
        raise ValueError('Only standard Qwen2 inference is permitted')
    gen_config=GenerationConfig.from_pretrained(settings['model'],revision=settings['revision'])
    eos=gen_config.eos_token_id
    settings['eos_token_ids']=[int(eos)] if isinstance(eos,int) else [int(i) for i in (eos or [])]
    if not settings['eos_token_ids']:raise ValueError('Missing native model EOS configuration')
    from ..data import prompt_ids
    # Capacity is checked on actual chat-template token prefixes, never by
    # truncating prompts or excluding long examples. Longer evaluation states
    # can extend the vLLM limit within the native pinned model capacity.
    maximum=max((len(prompt_ids(tokenizer,row['prompt'])) for row in rows),default=0)
    required=maximum+settings['max_new_tokens']
    native=getattr(model_config,'max_position_embeddings',None)
    if native is not None and required>native:
        raise ValueError('Prompt plus requested generation exceeds native model context; no truncation permitted')
    execution=dict(execution,max_model_len=max(2048,required),max_prompt_tokens=maximum)
    if settings['backend']=='hf':
        import torch
        from transformers import AutoModelForCausalLM
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        dtype={'float32':torch.float32,'bfloat16':torch.bfloat16}[settings['dtype']]
        model=AutoModelForCausalLM.from_pretrained(settings['model'],revision=settings['revision'],
            torch_dtype=dtype,trust_remote_code=False,attn_implementation='eager').to('cuda').eval()
    elif settings['backend']=='vllm':
        import torch,triton
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        triton.knobs.language.fp32_default='ieee'
        from vllm import LLM, SamplingParams
        options=_vllm_options(settings,execution)
        model=LLM(**options)
        print(f'E3 vLLM ready: dtype={settings["dtype"]}; queue={execution["request_batch_size"]}; '
            f'max concurrent={options["max_num_seqs"]}; token budget={options["max_num_batched_tokens"]}; '
            f'GPU memory budget={options["gpu_memory_utilization"]:.0%}.',flush=True)
        from importlib.metadata import version
        resolved=getattr(model.llm_engine,'vllm_config',None)
        graph=getattr(getattr(resolved,'compilation_config',None),'cudagraph_mode',None)
        attention=getattr(getattr(resolved,'attention_config',None),'backend',None)
        write_json(output.with_suffix('.worker_runtime.json'),{
            'vllm':version('vllm'),'torch':str(torch.__version__),'CUDA':torch.version.cuda,
            'GPU':torch.cuda.get_device_name(0),'execution':execution,
            'engine_options':options,'TF32':False,'triton_fp32':'ieee',
            'resolved_dtype':str(getattr(getattr(resolved,'model_config',None),'dtype',None)),
            'resolved_cudagraph_mode':getattr(graph,'name',str(graph)),
            'resolved_attention_backend':getattr(attention,'name',str(attention))})
        saved=len(existing)
        saved_ids={r['prompt_id'] for r in existing}
        def commit_batch(records):
            nonlocal saved
            append_events(output,records);saved+=len(records)
            saved_ids.update(r['prompt_id'] for r in records)
            print(f'E3 vLLM generation ({settings["kind"]}): {saved}/{len(rows)} saved.',flush=True)
        try:
            _vllm_generate_pending(model,tokenizer,pending,settings,identity,
                execution,SamplingParams,commit_batch)
        except Exception as exc:
            append_event(output.with_suffix('.failures.jsonl'),{
                'generation_identity':digest(identity),'saved_rows':saved,
                'pending_prompt_ids':[r['prompt_id'] for r in rows if r['prompt_id'] not in saved_ids],
                'error_type':type(exc).__name__,'error':str(exc),
                'policy':'Committed responses retained; uncommitted IDs remain pending; no artificial empty outputs'})
            raise
        return
    else:raise ValueError('Unsupported ordinary generation backend')
    for number,row in enumerate(pending,len(existing)+1):
        ids=[]; error=None; reason=stop=None
        started=time.perf_counter()
        try:
            prefix=prompt_ids(tokenizer,row['prompt']); seed=_request_seed(settings,row)
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            inputs=torch.tensor([prefix],device='cuda',dtype=torch.long)
            with torch.inference_mode():
                sampled=model.generate(input_ids=inputs,attention_mask=torch.ones_like(inputs),
                    generation_config=gen_config,do_sample=True,temperature=settings['temperature'],
                    top_p=1.,top_k=0,max_new_tokens=settings['max_new_tokens'],
                    eos_token_id=settings['eos_token_ids'],pad_token_id=tokenizer.pad_token_id or settings['eos_token_ids'][0],
                    repetition_penalty=1.,no_repeat_ngram_size=0,num_beams=1,num_return_sequences=1)
            ids=sampled[0,len(prefix):].tolist()
            reason='stop' if ids and ids[-1] in settings['eos_token_ids'] else 'length'
            stop=ids[-1] if reason=='stop' else None
            record=_record(row,ids,tokenizer,settings,identity,finish_reason=reason,stop_reason=stop,
                execution=execution,elapsed_seconds=time.perf_counter()-started)
        except Exception as exc:
            error=type(exc).__name__+': '+str(exc)
            record=_record(row,ids,tokenizer,settings,identity,error=error,finish_reason='error')
        append_event(output,record)
        if number==len(existing)+1 or number%10==0 or number==len(rows):
            print(f'E3 ordinary generation ({settings["kind"]}): {number}/{len(rows)} saved.',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True)
    _worker(parser.parse_args().request)
