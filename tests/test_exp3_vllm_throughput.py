"""SERVER ONLY. These regression sources are not executed on the local machine."""
from pathlib import Path
from types import SimpleNamespace
import pytest


ENV_NAMES=('REQUEST_BATCH_SIZE','MAX_NUM_SEQS','MAX_NUM_BATCHED_TOKENS','GPU_MEMORY_UTILIZATION')


@pytest.fixture(autouse=True)
def clean_execution_environment(monkeypatch):
    for suffix in ENV_NAMES:monkeypatch.delenv('KLRD_RS_VLLM_'+suffix,raising=False)


def test_queue_and_concurrency_are_independent_and_fp32_unchanged():
    from klrd_rs.e3.generation import _execution,_vllm_options
    cfg=SimpleNamespace(generation_backend='vllm',generation_batch_size=8)
    execution=_execution(cfg)
    assert execution['request_batch_size']==128
    assert execution['max_num_seqs']==64
    options=_vllm_options({'model':'native','revision':'frozen','tokenizer':'native',
        'tokenizer_revision':'frozen','dtype':'float32'},execution)
    assert options['max_num_seqs']==64 and options['max_num_batched_tokens']==8192
    assert options['gpu_memory_utilization']==.85 and options['enable_prefix_caching'] is True
    assert options['enable_chunked_prefill'] is True
    assert options['dtype']=='float32' and options['attention_config']=={'backend':'TRITON_ATTN'}
    assert options['compilation_config']['cudagraph_capture_sizes']==[1,2,4,8,16,32,64]
    assert cfg.generation_batch_size==8  # Saved config/checkpoint identity stays intact.


def test_execution_overrides_can_reduce_vram_without_changing_sampling(monkeypatch):
    from klrd_rs.e3.vllm_execution import tuning
    monkeypatch.setenv('KLRD_RS_VLLM_REQUEST_BATCH_SIZE','32')
    monkeypatch.setenv('KLRD_RS_VLLM_MAX_NUM_SEQS','16')
    monkeypatch.setenv('KLRD_RS_VLLM_MAX_NUM_BATCHED_TOKENS','4096')
    monkeypatch.setenv('KLRD_RS_VLLM_GPU_MEMORY_UTILIZATION','.80')
    values=tuning()
    assert values['request_batch_size']==32 and values['max_num_seqs']==16
    assert values['max_num_batched_tokens']==4096 and values['gpu_memory_utilization']==.8
    assert not any(k in values for k in ('dtype','seed','temperature','top_p','top_k'))


@pytest.mark.parametrize('suffix,value',[
    ('REQUEST_BATCH_SIZE','0'),('MAX_NUM_SEQS','-1'),('MAX_NUM_BATCHED_TOKENS','oops'),
    ('GPU_MEMORY_UTILIZATION','nan'),('GPU_MEMORY_UTILIZATION','1')])
def test_invalid_execution_override_is_rejected(suffix,value,monkeypatch):
    from klrd_rs.e3.vllm_execution import tuning
    monkeypatch.setenv('KLRD_RS_VLLM_'+suffix,value)
    with pytest.raises(ValueError):tuning()


def test_batch_is_prepared_before_single_commit_and_resume_keeps_seeds():
    from klrd_rs.e3 import generation
    class Tokenizer:
        all_special_ids=[99]
        def apply_chat_template(self,messages,**kwargs):return [10,int(messages[0]['content'])]
        def decode(self,ids,**kwargs):return str(ids[0])
        def encode(self,text,**kwargs):return [int(text)]
    calls=[];commits=[]
    class Engine:
        def generate(self,prompts,sampling_params,use_tqdm):
            calls.append(sampling_params)
            return [SimpleNamespace(prompt_token_ids=p['prompt_token_ids'],outputs=[SimpleNamespace(
                token_ids=[20],finish_reason='length',stop_reason=None)]) for p in prompts]
    rows=[{'prompt_id':str(i),'prompt':str(i),'sampling_seed':40+i} for i in range(5)]
    settings={'temperature':.8,'max_new_tokens':512,'seed':17,'kind':'student',
        'eos_token_ids':[99],'backend':'vllm','dtype':'float32'}
    generation._vllm_generate_pending(Engine(),Tokenizer(),rows[1:],settings,{'run':'fixed'},
        {'request_batch_size':3,'max_num_seqs':2},lambda **kw:SimpleNamespace(**kw),commits.append)
    assert [len(batch) for batch in commits]==[3,1]
    assert [p.seed for batch in calls for p in batch]==[41,42,43,44]
    assert [r['prompt_id'] for batch in commits for r in batch]==['1','2','3','4']


def test_late_decode_failure_does_not_commit_half_batch():
    from klrd_rs.e3 import generation
    class Tokenizer:
        all_special_ids=[]
        def apply_chat_template(self,*args,**kwargs):return [10]
        def decode(self,ids,**kwargs):
            if ids==[2]:raise ValueError('bad second response')
            return 'ok'
        def encode(self,*args,**kwargs):return [1]
    class Engine:
        def generate(self,prompts,**kwargs):
            return [SimpleNamespace(prompt_token_ids=[10],outputs=[SimpleNamespace(
                token_ids=[i+1],finish_reason='length',stop_reason=None)]) for i in range(2)]
    settings={'temperature':.8,'max_new_tokens':512,'seed':17,'kind':'student',
        'eos_token_ids':[99],'backend':'vllm','dtype':'float32'}
    commits=[]
    with pytest.raises(ValueError,match='second response'):
        generation._vllm_generate_pending(Engine(),Tokenizer(),[
            {'prompt_id':'a','prompt':'a'},{'prompt_id':'b','prompt':'b'}],settings,{},
            {'request_batch_size':128},lambda **kw:kw,commits.append)
    assert commits==[]


def test_batch_append_repairs_crash_tail_once_and_retains_committed_rows(tmp_path):
    from klrd_rs.e3.provenance import append_events
    from klrd_rs.io import read_jsonl
    path=tmp_path/'responses.jsonl'
    path.write_bytes(b'{"id":0}\n{"interrupted":')
    append_events(path,[{'id':1},{'id':2}])
    assert read_jsonl(path)==[{'id':0},{'id':1},{'id':2}]


def test_teacher_options_share_queue_limits_without_prefix_cache():
    from klrd_rs.e3full.teacher import _execution_options
    settings={'max_model_len':3072,'execution':{
        'request_batch_size':128,'max_num_seqs':64,'max_num_batched_tokens':8192,
        'gpu_memory_utilization':.85}}
    production=_execution_options({'dtype':'float32'},settings,eager=False)
    assert production['dtype']=='float32' and production['max_num_seqs']==64
    assert production['max_model_len']==3072 and production['max_num_batched_tokens']==8192
    assert production['gpu_memory_utilization']==.85 and production['enable_prefix_caching'] is False
    assert production['compilation_config']['cudagraph_capture_sizes']==[1,2,4,8,16,32,64]
    eager=_execution_options({'dtype':'float32'},settings,eager=True)
    assert eager['compilation_config']['cudagraph_capture_sizes']==[]


def test_large_queue_resume_does_not_change_cached_sampling_identity():
    from klrd_rs.e3.generation import _same_sampling_identity
    old={'run':'frozen','prompts_sha256':'same','settings':{'backend':'vllm','dtype':'float32',
        'seed':17,'temperature':.8,'requested_batch_size':8,'actual_request_batch_size':8}}
    new=dict(old,settings=dict(old['settings'],actual_request_batch_size=128))
    assert _same_sampling_identity(old,new)
    assert not _same_sampling_identity(old,dict(new,settings=dict(new['settings'],dtype='bfloat16')))
