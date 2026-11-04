"""Exact E2 rotation in an isolated FP32 vLLM worker, without HF generation/audits.

The compatibility overrides are the recorded 0.32/0.33 version allowlist,
context capacity and execution-only queue/memory tuning. E2 mathematical
implementation, EOS sampling, TRITON_ATTN and IEEE FP32 policies are reused.
"""
import atexit
import multiprocessing as mp
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..io import digest, read_json, sha256, write_json
from ..e3.provenance import append_event, append_events, get_frozen
from ..e3.vllm_execution import tuning,capture_sizes


def require_supported_vllm():
    installed=version('vllm')
    if installed not in ('0.32.0','0.33.0'):
        raise RuntimeError('E3 full exact teacher requires vLLM 0.32.0 or 0.33.0; found '+installed)
    return installed


def _execution_options(base,settings,*,eager=False):
    """Change scheduling/memory only; retain original teacher model/math options."""
    execution=settings['execution']
    result=dict(base,max_model_len=int(settings['max_model_len']),
        max_num_seqs=execution['max_num_seqs'],
        max_num_batched_tokens=execution['max_num_batched_tokens'],
        gpu_memory_utilization=execution['gpu_memory_utilization'],
        enable_prefix_caching=False,enable_chunked_prefill=not eager)
    if eager:
        result['max_num_batched_tokens']=max(result['max_num_batched_tokens'],result['max_model_len'])
    result['compilation_config']=dict(base.get('compilation_config',{}),mode=0,
        cudagraph_mode='NONE' if eager else 'FULL_DECODE_ONLY',
        cudagraph_capture_sizes=[] if eager else capture_sizes(execution['max_num_seqs']))
    return result


def _worker_entry(connection,settings,p,q):
    """Patch the imported module only inside this dedicated child process."""
    from ..e2 import vllm_backend as backend
    installed=require_supported_vllm()
    original_options=backend.engine_options
    original_runtime=backend.runtime_identity
    def options(values,*,eager=False):
        return _execution_options(original_options(values,eager=eager),values,eager=eager)
    def runtime():
        return dict(original_runtime(),vllm_version=installed,
                    max_model_len=int(settings['max_model_len']),
                    **settings['execution'])
    backend.require_vllm=require_supported_vllm
    backend.TARGET_VERSION=installed
    backend.engine_options=options
    backend.runtime_identity=runtime
    backend._gpu_worker(connection,settings,p,q)


def exact_sampler_class():
    """Lazy import keeps evaluation cache/report reads independent of torch."""
    from ..e2.vllm_backend import VLLMExactRotationTeacherSampler
    from ..e2.sampler import SamplerInvalid

    class ExactTeacherSampler(VLLMExactRotationTeacherSampler):
        def __init__(self,adapter,readout,frozen,cfg,*,max_model_len=2048,execution=None):
            installed=require_supported_vllm()
            self.model=adapter.model;self.tokenizer=adapter.tokenizer;self.readout=readout
            self.L=np.asarray(frozen['L']);self.strength=frozen['strength'];self.bank=frozen['bank']
            self.kl_stride=cfg.kl_stride;self.generation_seed=cfg.generation_seed;self.tau=cfg.tau
            native=self.model.generation_config.eos_token_id
            self.eos=set(native if isinstance(native,(list,tuple)) else [native])
            if self.tokenizer.eos_token_id is not None:self.eos.add(self.tokenizer.eos_token_id)
            if not self.eos or None in self.eos:raise SamplerInvalid('Invalid native EOS policy')
            self.process=None;self.connection=None;self.closed=False;self.peak_worker_bytes=0
            self._worker_eager=None;self._diagnostic_depth=0
            self._settings={'model':cfg.model,'revision':cfg.revision,'generation_seed':cfg.generation_seed,
                'tau':cfg.tau,'eos':sorted(self.eos),'max_model_len':int(max_model_len),
                'execution':dict(execution) if execution is not None else tuning()}
            self._geometry=(frozen['P'],frozen['Q']);self.environment=None
            self.actual_vllm_version=installed
            atexit.register(self.close)
            self._start_worker(False)

        def _start_worker(self,eager):
            context=mp.get_context('spawn');self.connection,child=context.Pipe()
            settings=dict(self._settings,diagnostic_eager=bool(eager))
            self.process=context.Process(target=_worker_entry,args=(child,settings,*self._geometry))
            try:
                self.process.start();child.close()
                environment=self._receive(1800)
                if (environment['tokenizer_vocab_sha256']!=digest(self.tokenizer.get_vocab())
                        or environment['chat_template']!=self.tokenizer.chat_template):
                    raise SamplerInvalid('Pinned worker/canonical tokenizer mismatch')
                if (environment['version']!=self.actual_vllm_version
                        or environment['dtype']!='float32' or environment['TF32']
                        or environment['triton_f32_default']!='ieee'):
                    raise SamplerInvalid('Exact teacher runtime/FP32 identity mismatch')
                if environment['resolved_attention_backend']!='TRITON_ATTN':
                    raise SamplerInvalid('Exact teacher FP32 attention backend changed')
                execution_notes=[]
                if (environment['resolved_compilation_mode']!=0
                        or (not eager and (environment['resolved_cudagraph_mode']!='FULL_DECODE_ONLY'
                            or environment['resolved_enable_chunked_prefill'] is not True))):
                    execution_notes.append('Engine changed requested compilation/graph/chunked-prefill settings')
                    print('E3 teacher: engine execution settings differ; recorded in runtime metadata.',flush=True)
                if self.environment is not None and environment['readout_hashes']!=self.environment['readout_hashes']:
                    raise SamplerInvalid('Exact teacher readout changed')
                self.environment=dict(environment,version_allowlist=['0.32.0','0.33.0'],
                    mathematical_implementation='unchanged klrd_rs.e2.vllm_backend._gpu_worker/vllm_model',
                    context_capacity=self._settings['max_model_len'],
                    generation_execution=self._settings['execution'],
                    parity_or_KL_audit_run=False,execution_notes=execution_notes,
                    compatibility_note='0.33 accepted explicitly; no cross-version sampled-token equivalence claim')
                self._worker_eager=bool(eager)
            except BaseException:
                child.close();self._stop_worker();raise

    return ExactTeacherSampler


def ExactTeacherSampler(*args,**kwargs):
    """Factory retaining the E2 sampler interface for the ARC adapter."""
    return exact_sampler_class()(*args,**kwargs)


def _validate(records,rows,identity):
    if len(records)>len(rows):raise ValueError('Unexpected exact teacher journal rows')
    for saved,row in zip(records,rows):
        body={k:v for k,v in saved.items() if k!='record_sha256'}
        if (saved.get('prompt_id')!=row['prompt_id'] or saved.get('prompt')!=row['prompt']
                or saved.get('seed')!=row['sampling_seed']
                or saved.get('generation_identity')!=digest(identity)
                or saved.get('record_sha256')!=digest(body)):
            raise ValueError('Exact teacher journal identity/seed/order/hash mismatch')


def generate(state,rows,output_path,max_new_tokens):
    """Resume native-EOS exact teacher samples using frozen per-prompt E2 seeds."""
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    from ..data import prompt_ids
    from ..e2.config import E2Config
    from ..e2.vllm_backend import result_records
    from ..e3.generation import _journal
    if max_new_tokens not in (512,1024):raise ValueError('Expected 512 or 1024 tokens')
    cfg=state['cfg'];path=Path(output_path)
    exact_cfg=E2Config()
    if cfg.owner_key_id!=0 or any(getattr(cfg,k)!=getattr(exact_cfg,k) for k in ('model','revision','tau','strength')):
        raise ValueError('Full teacher configuration differs from frozen E2 scientific settings')
    if len({r['prompt_id'] for r in rows})!=len(rows) or any('sampling_seed' not in r for r in rows):
        raise ValueError('Exact teacher requires unique IDs and frozen per-row seeds')
    frozen=get_frozen(state)
    identity={'run':state['identity'],'prompts_sha256':digest(rows),'max_new_tokens':max_new_tokens,
        'model':cfg.model,'revision':cfg.revision,'tau':cfg.tau,'owner_key_id':0,
        'geometry_sha256':digest({k:np.asarray(frozen[k]).tolist() for k in ('P','Q','L','bank')}),
        'strength':frozen['strength'],'implementation':'E2 exact FP32 rotation; raw_generate; no KL audit'}
    binding=path.with_suffix('.identity.json')
    if binding.exists():
        if read_json(binding)!=identity:raise ValueError('Exact teacher sampling identity changed')
    else:
        if path.exists():raise ValueError('Unbound exact teacher journal')
        write_json(binding,identity)
    existing=_journal(path);_validate(existing,rows,identity)
    if len(existing)<len(rows):
        runtime_path=path.with_suffix('.worker_runtime.json')
        if existing and (not runtime_path.exists()
                or read_json(runtime_path).get('version')!=require_supported_vllm()):
            raise ValueError('Exact teacher resume runtime version changed or original worker receipt missing')
        tokenizer=AutoTokenizer.from_pretrained(cfg.model,revision=cfg.revision,trust_remote_code=False)
        config=AutoConfig.from_pretrained(cfg.model,revision=cfg.revision,trust_remote_code=False)
        generation=GenerationConfig.from_pretrained(cfg.model,revision=cfg.revision)
        source=read_json(state['root']/cfg.e2_run/'provenance/engine_manifest.json')
        if (tokenizer.chat_template!=frozen['chat_template']
                or digest(tokenizer.get_vocab())!=source['generation']['tokenizer_vocab_sha256']):
            raise ValueError('Canonical teacher tokenizer/chat template differs from frozen E2')
        prefixes=[prompt_ids(tokenizer,r['prompt']) for r in rows]
        required=max((len(p) for p in prefixes),default=0)+max_new_tokens
        capacity=max(2048,required)
        if capacity>config.max_position_embeddings:
            raise ValueError('Frozen prompt plus generation exceeds native model capacity; truncation prohibited')
        # This holder has configuration only, never HF parameters/model allocation.
        adapter=SimpleNamespace(tokenizer=tokenizer,model=SimpleNamespace(config=config,generation_config=generation))
        execution=tuning()
        print(f'E3 teacher: FP32; queue={execution["request_batch_size"]}; '
            f'max concurrent={execution["max_num_seqs"]}; token budget={execution["max_num_batched_tokens"]}; '
            f'GPU memory budget={execution["gpu_memory_utilization"]:.0%}.',flush=True)
        sampler=ExactTeacherSampler(adapter,None,frozen,exact_cfg,max_model_len=capacity,execution=execution)
        try:
            if (sampler.environment['readout_hashes']!=source['generation']['readout_hashes']
                    or sorted(sampler.eos)!=sorted(source['native_EOS_ids'])):
                raise ValueError('Exact teacher native readout/EOS differs from frozen E2')
            write_json(runtime_path,dict(sampler.environment,
                longest_prompt_tokens=max(map(len,prefixes),default=0),required_context=required,
                GPU_lifecycle='one isolated E2 worker; no resident HF model'))
            request_batch_size=execution['request_batch_size']
            for start in range(len(existing),len(rows),request_batch_size):
                chunk=rows[start:start+request_batch_size]
                requests=[{'prompt_id':r['prompt_id'],'prompt_ids':prefixes[start+i],
                           'seed':int(r['sampling_seed'])} for i,r in enumerate(chunk)]
                print(f'E3 teacher batch: START {len(chunk)} requests; {start}/{len(rows)} saved.',flush=True)
                result=sampler.raw_generate(requests,frozen['bank'][0],max_new_tokens)
                generated=result_records(requests,result['raw'],tokenizer,sampler.eos,max_new_tokens,result['seconds'])
                records=[]
                for row,record in zip(chunk,generated):
                    record=dict(row,**record,generation_identity=digest(identity),error=None,
                        generation_dtype='float32',generation_vllm_version=sampler.actual_vllm_version,
                        generation_execution=dict(execution,actual_request_batch_size=len(chunk)),
                        response_id=digest({'generation':digest(identity),'prompt_id':row['prompt_id'],
                            'seed':record['seed'],'tokens':record['generated_token_ids']}))
                    record['record_sha256']=digest(record)
                    records.append(record)
                append_events(path,records)
                tokens=sum(r['native_visible_tokens'] for r in records)
                print(f'E3 full exact teacher: {min(start+len(chunk),len(rows))}/{len(rows)} saved.',flush=True)
                print(f'E3 teacher batch: {tokens} visible tokens; generation={result["seconds"]:.1f}s '
                    f'({tokens/max(result["seconds"],1e-9):.1f} visible tok/s).',flush=True)
        except BaseException as exc:
            append_event(path.with_suffix('.failures.jsonl'),{'generation_identity':digest(identity),
                'error_type':type(exc).__name__,'error':str(exc),
                'policy':'Committed samples retained; no fabricated errors/empty responses; resume missing IDs only'})
            raise
        finally:sampler.close()
    result=_journal(path);_validate(result,rows,identity)
    if len(result)!=len(rows):raise ValueError('Exact teacher journal incomplete')
    write_json(path.with_suffix('.complete.json'),{'identity':digest(identity),'rows':len(result),
        'sha256':sha256(path),'errors':0,'gpu_lifecycle':'dedicated worker closed before verification'})
    return result
