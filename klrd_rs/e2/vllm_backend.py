"""Exact vLLM generation; HF is used only for the existing post-generation audit.

No vLLM import or CUDA initialization at module import. No HF generation fallback.
All model/runtime methods are intended to be invoked on the user's GPU server.
"""
import atexit
import multiprocessing as mp
import time,os,warnings
# Set before torch import, including when a spawned worker imports this module.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
from pathlib import Path
from functools import partial
from importlib.metadata import version,PackageNotFoundError
import numpy as np
import torch
from .sampler import SamplerInvalid
from .cohort import kl_selected
from ..io import digest

TARGET_VERSION='0.10.2'
ARCHITECTURE='KLRDRSQwen2ForCausalLM'


def require_vllm():
    try:installed=version('vllm')
    except PackageNotFoundError:raise SamplerInvalid('vLLM 0.10.2 required; install requirements-vllm.txt on the GPU server. No HF generation fallback.')
    if installed!=TARGET_VERSION:
        raise SamplerInvalid(f'Unsupported vLLM {installed}; target is exactly {TARGET_VERSION}. See README_VLLM_EXPERIMENT2.md')
    return installed


def sampling_kwargs(tau,seed,max_tokens,eos,probe=False):
    return dict(n=1,temperature=tau,top_p=1.,top_k=-1,min_p=0.,
        presence_penalty=0.,frequency_penalty=0.,repetition_penalty=1.,
        seed=int(seed),max_tokens=max_tokens,min_tokens=0,
        stop_token_ids=[] if probe else sorted(eos),ignore_eos=probe,
        include_stop_str_in_output=True,detokenize=False,logprobs=0 if probe else None)


def result_records(requests,raw,tokenizer,eos,max_tokens,elapsed):
    if len(requests)!=len(raw):raise SamplerInvalid('Incomplete vLLM response batch')
    records=[]
    for request,value in zip(requests,raw):
        tokens=[int(t) for t in value['token_ids']]
        if value['stop_reason'] in eos and (not tokens or tokens[-1]!=value['stop_reason']):
            raise SamplerInvalid('vLLM omitted native stop token; refuse to fabricate tokens')
        text=tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
        visible=[t for t in tokens if t not in set(tokenizer.all_special_ids)]
        recovered=tokenizer.encode(text,add_special_tokens=False)
        records.append({'prompt_id':request['prompt_id'],'seed':int(request['seed']),'response':text,
            'generated_token_ids':tokens,'eos_terminated':bool(tokens and tokens[-1] in eos),
            'max_length_truncated':len(tokens)==max_tokens and bool(tokens) and tokens[-1] not in eos,
            'roundtrip_equal':visible==list(recovered),'native_visible_tokens':len(visible),
            'recovered_tokens':len(recovered),'KL_samples':[],
            'sampling_temperature_applications':1,'amortized_generation_seconds':elapsed/len(requests),
            'batch_size':len(requests),'decode_steps':max(0,max((len(r['token_ids']) for r in raw),default=0)-1),
            'generation_backend':'vllm','vllm_finish_reason':value['finish_reason']})
    return records


def _gpu_worker(connection,settings,p,q):
    """Spawn target: one resident engine; a fixed key for each synchronous call."""
    engine=None
    try:
        # vLLM's lazy architecture inspection uses another Python subprocess.
        # Propagate the repository root there as well as through multiprocessing.
        root=str(Path(__file__).resolve().parents[2])
        os.environ['PYTHONPATH']=root+(os.pathsep+os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
        os.environ['VLLM_USE_V1']='0'
        os.environ['VLLM_ATTENTION_BACKEND']='XFORMERS'
        os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
        require_vllm()
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        from vllm import LLM,SamplingParams,ModelRegistry
        from .vllm_model import configure_model,set_owner,set_native_baseline,start_trace,take_trace,start_linear_shape_probe,stop_linear_shape_probe
        ModelRegistry.register_model(ARCHITECTURE,'klrd_rs.e2.vllm_model:KLRDRSQwen2ForCausalLM')
        engine=LLM(model=settings['model'],revision=settings['revision'],
            tokenizer=settings['model'],tokenizer_revision=settings['revision'],
            dtype='float32',tensor_parallel_size=1,quantization=None,
            hf_overrides={'architectures':[ARCHITECTURE]},trust_remote_code=False,
            gpu_memory_utilization=.40,max_model_len=2048,max_num_seqs=8,
            max_num_batched_tokens=8192,enforce_eager=True,compilation_config=0,
            enable_prefix_caching=False,enable_chunked_prefill=False,
            generation_config='vllm',seed=settings['generation_seed'],
            distributed_executor_backend='uni',swap_space=0)
        checks=engine.apply_model(partial(configure_model,p=p,q=q))
        tokenizer=engine.get_tokenizer()
        connection.send({'ok':True,'data':{'readout_hashes':checks,'tokenizer_vocab_sha256':digest(tokenizer.get_vocab()),
            'chat_template':tokenizer.chat_template,'engine':'vllm','version':TARGET_VERSION,
            'dtype':'float32','linear_accumulation':'float32','attention_backend':'XFORMERS',
            'engine_version':'V0','max_num_seqs':8,'max_model_len':2048,'gpu_memory_utilization':.40,
            'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'TF32':torch.backends.cuda.matmul.allow_tf32,
            'matmul_precision':torch.get_float32_matmul_precision(),
            'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'TF32_environment':{name:os.environ.get(name) for name in ['NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']},
            'temperature_location':'SamplingParams only','custom_architecture':ARCHITECTURE}})
        while True:
            command=connection.recv()
            if command['op']=='close':break
            if command['op']!='generate':raise RuntimeError('Unknown vLLM worker command')
            engine.apply_model(partial(set_owner,angles=command['angles']))
            native_baseline=command.get('native_baseline',False)
            if native_baseline and not command['probe']:raise RuntimeError('Native baseline is diagnostic only')
            if native_baseline:
                confirmation=engine.apply_model(partial(set_native_baseline,enabled=True))
                if confirmation!=[{'native_original_norm':True}]:raise RuntimeError('Native baseline mode was not confirmed by the actual worker')
            if command['probe']:
                if len(command['requests'])!=1 and command['max_tokens']!=1:
                    raise RuntimeError('Actual-logit trace supports single-stream decode or a prefill-only batch')
                engine.apply_model(start_trace)
            prompts=[{'prompt_token_ids':r['prompt_ids']} for r in command['requests']]
            params=[SamplingParams(**sampling_kwargs(settings['tau'],r['seed'],command['max_tokens'],
                settings['eos'],command['probe'])) for r in command['requests']]
            linear_probe=command.get('linear_probe',False)
            if linear_probe:
                if command['max_tokens']!=1:raise RuntimeError('Linear shape diagnostic is prefill-only')
                engine.apply_model(start_linear_shape_probe)
            begin=time.perf_counter()
            try:outputs=engine.generate(prompts,params,use_tqdm=False)
            finally:
                linear_rows=engine.apply_model(stop_linear_shape_probe)[0] if linear_probe else []
                if native_baseline:engine.apply_model(partial(set_native_baseline,enabled=False))
            elapsed=time.perf_counter()-begin
            raw=[]
            for request,output in zip(command['requests'],outputs):
                if list(output.prompt_token_ids)!=list(request['prompt_ids']):raise RuntimeError('vLLM changed request ordering/prompt tokens')
                if len(output.outputs)!=1:raise RuntimeError('Expected one completion per prompt')
                result=output.outputs[0]
                sampled=None
                if command['probe']:
                    sampled=[float(row[int(token)].logprob) for row,token in zip(result.logprobs,result.token_ids)]
                raw.append({'token_ids':list(result.token_ids),'finish_reason':result.finish_reason,
                    'stop_reason':result.stop_reason,'sampled_logprobs':sampled})
            traces=engine.apply_model(take_trace)[0] if command['probe'] else []
            connection.send({'ok':True,'data':{'raw':raw,'trace':traces,'seconds':elapsed,
                'linear_shape_probe':linear_rows,
                'teacher_mode':'native_original' if native_baseline else 'exact_rotation',
                'peak_worker_CUDA_allocated_bytes':torch.cuda.max_memory_allocated()}})
    except BaseException as exc:
        try:connection.send({'ok':False,'error_type':type(exc).__name__,'reason':str(exc)})
        except (EOFError,OSError):pass
    finally:
        cleanup_worker(engine)
        connection.close()


def cleanup_worker(engine):
    """Best-effort orderly teardown, including after a failed parity gate."""
    if engine is not None:
        try:engine.llm_engine.model_executor.shutdown()
        except Exception as exc:warnings.warn('vLLM executor cleanup: '+type(exc).__name__)
    try:
        from vllm.distributed.parallel_state import destroy_model_parallel,destroy_distributed_environment
    except ImportError:return
    for cleanup in [destroy_model_parallel,destroy_distributed_environment]:
        try:cleanup()
        except Exception as exc:warnings.warn('vLLM distributed cleanup: '+type(exc).__name__)
    if torch.distributed.is_initialized():
        try:torch.distributed.destroy_process_group()
        except Exception as exc:warnings.warn('NCCL cleanup: '+type(exc).__name__)


class VLLMExactRotationTeacherSampler:
    def __init__(self,adapter,readout,frozen,cfg):
        require_vllm()
        self.model=adapter.model;self.tokenizer=adapter.tokenizer;self.readout=readout
        self.L=np.asarray(frozen['L']);self.strength=frozen['strength'];self.bank=frozen['bank']
        self.kl_stride=cfg.kl_stride;self.generation_seed=cfg.generation_seed;self.tau=cfg.tau
        eos=self.model.generation_config.eos_token_id
        self.eos=set(eos if isinstance(eos,(tuple,list)) else [eos])
        if self.tokenizer.eos_token_id is not None:self.eos.add(self.tokenizer.eos_token_id)
        if not self.eos or None in self.eos:raise SamplerInvalid('Invalid native EOS policy')
        self.process=None;self.connection=None;self.closed=False;self.peak_worker_bytes=0
        context=mp.get_context('spawn');self.connection,child=context.Pipe()
        settings={'model':cfg.model,'revision':cfg.revision,'generation_seed':cfg.generation_seed,
            'tau':cfg.tau,'eos':sorted(self.eos)}
        self.process=context.Process(target=_gpu_worker,args=(child,settings,frozen['P'],frozen['Q']))
        self.process.start();child.close();atexit.register(self.close)
        self.environment=self._receive(1800)
        if self.environment['tokenizer_vocab_sha256']!=digest(self.tokenizer.get_vocab()) or self.environment['chat_template']!=self.tokenizer.chat_template:
            self.close();raise SamplerInvalid('Pinned vLLM/HF tokenizer mismatch')
        print(f'Custom vLLM {TARGET_VERSION} exact rotation teacher ready: FP32 generation, native KV cache, no HF generation fallback.',flush=True)

    def _receive(self,timeout=1800):
        begin=time.monotonic();last=begin
        while not self.connection.poll(1):
            if not self.process.is_alive():raise SamplerInvalid('vLLM worker exited without a result')
            if time.monotonic()-begin>timeout:raise SamplerInvalid('vLLM worker timeout; no HF generation fallback')
            if time.monotonic()-last>30:
                print('Waiting for vLLM worker...',flush=True);last=time.monotonic()
        try:value=self.connection.recv()
        except EOFError:raise SamplerInvalid('vLLM worker connection closed')
        if not value['ok']:raise SamplerInvalid('vLLM '+value['error_type']+': '+value['reason'])
        return value['data']

    def raw_generate(self,requests,key,max_new_tokens=512,probe=False,linear_probe=False,native_baseline=False):
        if native_baseline and (key is not None or not probe):
            raise SamplerInvalid('Native control requires a zero-key diagnostic request')
        angle=np.zeros(self.L.shape[0],dtype=np.float32) if key is None else np.asarray(self.strength*(self.L@np.asarray(key)),dtype=np.float32)
        self.connection.send({'op':'generate','requests':requests,'angles':angle,'max_tokens':max_new_tokens,
            'probe':probe,'linear_probe':linear_probe,'native_baseline':native_baseline})
        result=self._receive();self.peak_worker_bytes=max(self.peak_worker_bytes,result['peak_worker_CUDA_allocated_bytes'])
        return result

    def generate(self,requests,key,max_new_tokens=512):
        if not requests:return []
        raw=self.raw_generate(requests,key,max_new_tokens)
        records=result_records(requests,raw['raw'],self.tokenizer,self.eos,max_new_tokens,raw['seconds'])
        from .vllm_kl import audit_native_prefix_kl
        begin=time.perf_counter()
        audit_native_prefix_kl(self,requests,records,key)
        elapsed=time.perf_counter()-begin
        for record in records:
            record['amortized_generation_seconds']+=elapsed/len(records)
            record['KL_audit_seconds']=elapsed/len(records)
        return records

    def close(self):
        if self.closed:return
        self.closed=True
        try:
            if self.connection is not None and self.process is not None and self.process.is_alive():self.connection.send({'op':'close'})
        except (BrokenPipeError,EOFError,OSError):pass
        if self.process is not None:
            self.process.join(timeout=5)
            if self.process.is_alive():self.process.terminate();self.process.join(timeout=5)
        if self.connection is not None:self.connection.close()
