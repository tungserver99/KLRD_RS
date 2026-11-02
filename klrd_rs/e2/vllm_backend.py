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
from contextlib import contextmanager
from collections import defaultdict,deque
from importlib.metadata import version,PackageNotFoundError
import numpy as np
import torch
from .sampler import SamplerInvalid
from .cohort import kl_selected
from ..io import digest

from .vllm_runtime import TARGET_VERSION,ARCHITECTURE,engine_options,runtime_identity


def require_vllm():
    try:installed=version('vllm')
    except PackageNotFoundError:raise SamplerInvalid(f'vLLM {TARGET_VERSION} required; install requirements-vllm.txt on the GPU server. No HF generation fallback.')
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


def ordered_prefill_traces(rows,requests):
    """V1 can reorder scheduled requests; bind by full query, not last token."""
    pending=defaultdict(deque)
    for row in rows:pending[tuple(row['query_token_ids'])].append(row)
    ordered=[]
    for request in requests:
        queue=pending[tuple(request['prompt_ids'])]
        if not queue:raise SamplerInvalid('Missing actual V1 prefill trace for requested prefix')
        ordered.append(queue.popleft())
    if any(pending.values()):raise SamplerInvalid('Unexpected extra V1 prefill traces')
    return ordered


def _gpu_worker(connection,settings,p,q):
    """One resident V1 engine inside this spawned process; idle for key changes."""
    engine=None
    try:
        root=str(Path(__file__).resolve().parents[2])
        os.environ['PYTHONPATH']=root+(os.pathsep+os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
        # V0 is gone. Remove obsolete inherited switches before importing vLLM.
        os.environ.pop('VLLM_USE_V1',None)
        os.environ.pop('VLLM_ATTENTION_BACKEND',None)
        # Our process already provides isolation. In-process V1 core + uni
        # executor make shutdown/process exit release the entire teacher engine.
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING']='0'
        os.environ['VLLM_USE_V2_MODEL_RUNNER']='1'
        os.environ['VLLM_FLOAT32_MATMUL_PRECISION']='highest'
        if any(os.environ.get(name) not in (None,'0') for name in ['NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']):
            raise RuntimeError('TF32 environment override conflicts with the exact FP32 teacher')
        os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
        require_vllm()
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        # tl.dot has its own FP32 precision policy, independent of torch TF32.
        # Pin both the environment and cached knob inside this worker only.
        os.environ['TRITON_F32_DEFAULT']='ieee'
        import triton
        triton.knobs.language.fp32_default='ieee'
        from vllm import LLM,SamplingParams,ModelRegistry
        from .vllm_model import configure_model,set_owner,set_native_baseline,start_trace,take_trace,start_linear_shape_probe,stop_linear_shape_probe,check_logits_finite,check_rotation_graph
        ModelRegistry.register_model(ARCHITECTURE,'klrd_rs.e2.vllm_model:KLRDRSQwen2ForCausalLM')
        eager=bool(settings['diagnostic_eager'])
        options=engine_options(settings,eager=eager)
        engine=LLM(**options)
        resolved=engine.llm_engine.vllm_config
        resolved_graph=resolved.compilation_config.cudagraph_mode.name
        if resolved.model_config.dtype!=torch.float32:
            raise RuntimeError('vLLM changed the required FP32 model dtype')
        if torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision()!='highest':
            raise RuntimeError('vLLM changed the required FP32 matmul precision')
        checks=engine.apply_model(partial(configure_model,p=p,q=q))
        tokenizer=engine.get_tokenizer()
        environment=dict(runtime_identity(),readout_hashes=checks,
            tokenizer_vocab_sha256=digest(tokenizer.get_vocab()),chat_template=tokenizer.chat_template,
            engine='vllm',version=TARGET_VERSION,dtype='float32',linear_accumulation='float32',
            enforce_eager=eager,cudagraph_mode=options['compilation_config']['cudagraph_mode'],
            enable_chunked_prefill=options['enable_chunked_prefill'],
            CUDA_graph_decode_requested=not eager,CUDA_graph_decode_observed=None,
            execution_mode='eager_diagnostic' if eager else 'production_decode_graph',
            engine_core_multiprocessing=False,
            triton_f32_default=triton.knobs.language.fp32_default,
            resolved_cudagraph_mode=resolved_graph,
            resolved_compilation_mode=int(resolved.compilation_config.mode),
            resolved_attention_backend=resolved.attention_config.backend.name,
            resolved_enable_chunked_prefill=resolved.scheduler_config.enable_chunked_prefill,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            TF32=torch.backends.cuda.matmul.allow_tf32,matmul_precision=torch.get_float32_matmul_precision(),
            CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            TF32_environment={name:os.environ.get(name) for name in ['NVIDIA_TF32_OVERRIDE','TORCH_ALLOW_TF32_CUBLAS_OVERRIDE']},
            temperature_location='SamplingParams only',custom_architecture=ARCHITECTURE)
        connection.send({'ok':True,'data':environment})
        while True:
            command=connection.recv()
            if command['op']=='close':break
            if command['op']=='rotation_graph_check':
                if eager:raise RuntimeError('Graph validation requires the production engine')
                from vllm.config import set_current_vllm_config
                with set_current_vllm_config(engine.llm_engine.vllm_config):
                    result=engine.apply_model(partial(check_rotation_graph,angles_a=command['angles_a'],angles_b=command['angles_b']))
                connection.send({'ok':True,'data':result[0]})
                continue
            if command['op']!='generate':raise RuntimeError('Unknown vLLM worker command')
            probe=command['probe'];linear_probe=command.get('linear_probe',False)
            native_baseline=command.get('native_baseline',False)
            if (probe or linear_probe or native_baseline) and not eager:
                raise RuntimeError('Python diagnostics require a dedicated eager engine')
            if native_baseline and not probe:raise RuntimeError('Native baseline is diagnostic only')
            if probe and len(command['requests'])!=1 and command['max_tokens']!=1:
                raise RuntimeError('Actual-logit trace supports single-stream decode or prefill-only batches')
            if linear_probe and command['max_tokens']!=1:
                raise RuntimeError('Linear shape diagnostic is prefill-only')
            engine.apply_model(partial(set_owner,angles=command['angles']))
            if native_baseline:
                confirmation=engine.apply_model(partial(set_native_baseline,enabled=True))
                if confirmation!=[{'native_original_norm':True}]:raise RuntimeError('Native baseline mode was not confirmed by the actual worker')
            if probe:engine.apply_model(start_trace)
            prompts=[{'prompt_token_ids':r['prompt_ids']} for r in command['requests']]
            params=[SamplingParams(**sampling_kwargs(settings['tau'],r['seed'],command['max_tokens'],settings['eos'],probe))
                for r in command['requests']]
            if linear_probe:engine.apply_model(start_linear_shape_probe)
            begin=time.perf_counter()
            try:outputs=engine.generate(prompts,params,use_tqdm=False)
            finally:
                linear_rows=engine.apply_model(stop_linear_shape_probe)[0] if linear_probe else []
                if native_baseline:engine.apply_model(partial(set_native_baseline,enabled=False))
            engine.apply_model(check_logits_finite)
            elapsed=time.perf_counter()-begin
            if len(outputs)!=len(command['requests']):raise RuntimeError('Incomplete vLLM response batch')
            raw=[]
            for request,output in zip(command['requests'],outputs):
                if list(output.prompt_token_ids)!=list(request['prompt_ids']):raise RuntimeError('vLLM changed request ordering/prompt tokens')
                if len(output.outputs)!=1:raise RuntimeError('Expected one completion per prompt')
                result=output.outputs[0];sampled=None
                if probe:
                    if result.logprobs is None or len(result.logprobs)!=len(result.token_ids):
                        raise RuntimeError('Missing actual sampled-token log probabilities')
                    sampled=[float(row[int(token)].logprob) for row,token in zip(result.logprobs,result.token_ids)]
                raw.append({'token_ids':list(result.token_ids),'finish_reason':result.finish_reason,
                    'stop_reason':result.stop_reason,'sampled_logprobs':sampled})
            traces=engine.apply_model(take_trace)[0] if probe else []
            if probe and command['max_tokens']==1:traces=ordered_prefill_traces(traces,command['requests'])
            connection.send({'ok':True,'data':{'raw':raw,'trace':traces,'seconds':elapsed,
                'linear_shape_probe':linear_rows,'execution_mode':environment['execution_mode'],
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
        try:engine.llm_engine.engine_core.shutdown()
        except Exception as exc:warnings.warn('vLLM V1 engine core cleanup: '+type(exc).__name__)
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
        self._worker_eager=None;self._diagnostic_depth=0
        self._settings={'model':cfg.model,'revision':cfg.revision,'generation_seed':cfg.generation_seed,
            'tau':cfg.tau,'eos':sorted(self.eos)}
        self._geometry=(frozen['P'],frozen['Q'])
        self.environment=None
        atexit.register(self.close)
        self._start_worker(False)
        print(f'Custom vLLM {TARGET_VERSION} V1 exact rotation teacher ready: FP32, Triton attention, decode graphs requested.',flush=True)

    def _start_worker(self,eager):
        context=mp.get_context('spawn');self.connection,child=context.Pipe()
        settings=dict(self._settings,diagnostic_eager=bool(eager))
        self.process=context.Process(target=_gpu_worker,args=(child,settings,*self._geometry))
        try:
            self.process.start();child.close()
            environment=self._receive(1800)
            if (environment['tokenizer_vocab_sha256']!=digest(self.tokenizer.get_vocab())
                or environment['chat_template']!=self.tokenizer.chat_template):
                raise SamplerInvalid('Pinned vLLM/HF tokenizer mismatch')
            if self.environment is not None and environment['readout_hashes']!=self.environment['readout_hashes']:
                raise SamplerInvalid('vLLM model changed while switching diagnostic engine')
            self.environment=environment;self._worker_eager=bool(eager)
        except BaseException:
            child.close();self._stop_worker();raise

    def _stop_worker(self):
        process=self.process;connection=self.connection
        try:
            if connection is not None and process is not None and process.is_alive():connection.send({'op':'close'})
        except (BrokenPipeError,EOFError,OSError):pass
        if process is not None and process.pid is not None:
            process.join(timeout=30)
            if process.is_alive():process.terminate();process.join(timeout=5)
            if process.is_alive():
                raise SamplerInvalid('Previous vLLM worker did not exit; refuse a second resident engine')
            process.close()
        if connection is not None:connection.close()
        self.process=None;self.connection=None;self._worker_eager=None

    def _ensure_worker(self,eager):
        if self._worker_eager==bool(eager):return
        # Wait for exit before allocating another engine on the shared GPU.
        self._stop_worker();self._start_worker(bool(eager))

    @contextmanager
    def diagnostic_mode(self):
        previous=self._worker_eager
        self._diagnostic_depth+=1
        try:
            self._ensure_worker(True)
            yield
        finally:
            self._diagnostic_depth-=1
            if self._diagnostic_depth==0 and previous is not None:
                self._ensure_worker(previous)

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
        if self.closed:raise SamplerInvalid('vLLM sampler is closed')
        self._ensure_worker(bool(probe or linear_probe or native_baseline or self._diagnostic_depth))
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

    def validate_rotation_graph(self,key_a,key_b):
        """Explicit server diagnostic: actual final-norm CUDA graph A/B/A replay."""
        if self.closed:raise SamplerInvalid('vLLM sampler is closed')
        self._ensure_worker(False)
        angles=[np.asarray(self.strength*(self.L@np.asarray(key)),dtype=np.float32) for key in [key_a,key_b]]
        self.connection.send({'op':'rotation_graph_check','angles_a':angles[0],'angles_b':angles[1]})
        return self._receive()

    def close(self):
        if self.closed:return
        self.closed=True
        self._stop_worker()
