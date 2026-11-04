"""SERVER ONLY contracts. Authored without executing Python or tests locally."""
from pathlib import Path
import pytest
from klrd_rs.e3.config import E3Config
from klrd_rs.e3 import data,generation
from klrd_rs.io import write_json,read_json,digest,sha256
from klrd_rs.e2.exclusions import UsedHistoryIndex


def test_heldout_comparison_catches_id_aliases_and_nfkc_text():
    index=UsedHistoryIndex()
    index.consume('used',{'prompt_id':'train_gen:held','prompt':'Ｆｏｏ  BAR'})
    assert data._overlap({'prompt_id':'held','prompt':'unrelated'},index)
    assert data._overlap({'prompt_id':'different','prompt':'foo bar'},index)
    assert not data._overlap({'prompt_id':'new','prompt':'different text'},index)


def test_eval_original_text_hash_binding_and_frozen_order(tmp_path):
    cfg=E3Config();source=tmp_path/cfg.e2_run
    rows=[{'prompt_id':str(i),'prompt':f'original {i}'} for i in range(1000)]
    path=source/'private/prompts.jsonl';data._write_rows(path,list(reversed(rows)))
    manifest={'groups':[{'Q':1000,'group_index':0,'prompt_ids':[r['prompt_id'] for r in rows]}],
        'private_prompts_sha256':sha256(path)}
    write_json(source/'provenance/prompt_manifest.json',manifest)
    state={'root':tmp_path,'cfg':cfg}
    assert data.load_eval_prompts(state)==rows
    rows[0]['prompt']='altered text with unchanged ID';data._write_rows(path,rows)
    with pytest.raises(ValueError,match='checksum'):data.load_eval_prompts(state)


def test_profile_uses_full_partition_prefix_and_retains_empty_errors(tmp_path,monkeypatch):
    selected={s:[{'prompt_id':f'{s}{i}','prompt':f'prompt {i}','source':'train_gen','split':s}
        for i in range(n)] for s,n in [('train',10000),('validation',1000)]}
    monkeypatch.setattr(data,'_selection',lambda state:selected)
    calls=[]
    def fake_generate(state,rows,path,kind):
        calls.append((kind,len(rows)))
        return [dict(r,response='',response_id=digest(r),error='recorded failure' if i==0 else None)
            for i,r in enumerate(rows)]
    monkeypatch.setattr(generation,'generate',fake_generate)
    state={'root':tmp_path,'out':tmp_path/'pilot','cfg':E3Config(),'identity':'fixed'}
    result=data.prepare(state)
    assert calls==[('clean_train',2000),('clean_train',200)]
    for split,n in [('train',2000),('validation',200)]:
        assert [r['prompt_id'] for r in result[split]]==[r['prompt_id'] for r in selected[split][:n]]
        assert len(result[split])==n and result[split][0]['eligible_context_status']=='error'
    audit=read_json(state['out']/'provenance/data_complete.json')
    assert audit['errors']=={'train':1,'validation':1}
    calls.clear();state['out']=tmp_path/'smoke'
    data.prepare_smoke(state)
    assert calls==[('smoke_clean',2),('smoke_clean',2)]


def test_eos_stop_metadata_does_not_invent_sampled_eos():
    class Tokenizer:
        all_special_ids=[99]
        def decode(self,ids,**kwargs):return 'visible'
        def encode(self,text,**kwargs):return [10,11]
    settings={'eos_token_ids':[99],'seed':7,'kind':'clean_train','backend':'vllm',
        'dtype':'float32','temperature':.8,'max_new_tokens':512}
    record=generation._record({'prompt_id':'id','prompt':'prompt'},[10,11],Tokenizer(),settings,
        {'identity':'fixed'},finish_reason='stop',stop_reason=99)
    assert record['eos_terminated'] and record['native_eos_token_id']==99
    assert record['generated_token_ids']==[10,11]
    assert record['eos_token_present_in_generated_ids'] is False
    assert record['roundtrip_equal'] is True and record['max_length_truncated'] is False


def test_generation_journal_rejects_changed_prompt_even_with_same_id():
    identity={'run':'fixed','settings':{'seed':17,'kind':'clean_train'}}
    rows=[{'prompt_id':'id','prompt':'original'}]
    record=dict(rows[0],generation_identity=digest(identity),response='',error='retained error',
        seed=generation._seed(17,'id','clean_train'))
    record['record_sha256']=digest(record)
    generation._validate([record],rows,identity)
    with pytest.raises(ValueError,match='identity/hash/seed'):
        generation._validate([record],[{'prompt_id':'id','prompt':'changed'}],identity)


def test_historical_artifacts_exclude_eval_roles_but_not_train_or_reserve(tmp_path):
    index=UsedHistoryIndex();legacy=tmp_path/'KLRD'
    family=legacy/'artifacts/stage3a_wmle_implantation/run_001'
    data._write_rows(family/'data/implant_eval_prompts.jsonl',[
        {'prompt_id':'ultrachat:train_gen:old-eval','source_id':'old-eval','prompt':'held out text','split':'implant_eval'}])
    data._write_rows(family/'data/prompts.jsonl',[
        {'prompt_id':'train','prompt':'permitted training','split':'implant_train'},
        {'prompt_id':'dev','prompt':'old heldout validation','split':'implant_dev'}])
    data._write_rows(family/'data/train_reserve_prompts.jsonl',[
        {'prompt_id':'reserve','prompt':'unused inventory','split':'implant_eval'}])
    sources,audit=data.historical_heldout(index,legacy)
    assert 'old-eval' in index.ids and 'dev' in index.ids
    assert 'train' not in index.ids and 'reserve' not in index.ids
    assert len(sources)==2 and any(r.get('selected_heldout_rows')==1 for r in audit)
    assert data._overlap({'prompt_id':'other','prompt':'HELD   OUT TEXT'},index)


def test_missing_historical_artifacts_are_explicit_limitations(tmp_path):
    sources,audit=data.historical_heldout(UsedHistoryIndex(),tmp_path/'not-synced')
    assert not sources and audit[0]['status']=='MISSING_OPTIONAL_HISTORY'

def test_historical_role_manifest_retains_ids_without_raw_text():
    rows=data._heldout_rows({'splits':{'implant_eval':['train_gen:used',{'source_id':'other'}],
        'implant_train':['permitted'],'reserve':['unused']}})
    index=UsedHistoryIndex();index.consume('role-manifest',rows)
    assert {'used','other'}<=index.ids
    assert 'permitted' not in index.ids and 'unused' not in index.ids


def test_vllm_batches_keep_seed_sampling_order_and_final_short_batch():
    from types import SimpleNamespace
    class Tokenizer:
        all_special_ids=[99]
        def apply_chat_template(self,messages,**kwargs):return [10,int(messages[0]['content'])]
        def decode(self,ids,**kwargs):return str(ids[0])
        def encode(self,text,**kwargs):return [int(text)]
    seen=[]
    def params_type(**kwargs):return SimpleNamespace(**kwargs)
    class Engine:
        def generate(self,prompts,sampling_params,use_tqdm):
            seen.append((prompts,sampling_params))
            return [SimpleNamespace(prompt_token_ids=p['prompt_token_ids'],outputs=[
                SimpleNamespace(token_ids=[20+p['prompt_token_ids'][-1]],finish_reason='length',stop_reason=None)]) for p in prompts]
    rows=[{'prompt_id':str(i),'prompt':str(i)} for i in range(5)]
    settings={'temperature':.8,'max_new_tokens':512,'seed':2026110301,'kind':'clean_train',
        'eos_token_ids':[99],'backend':'vllm','dtype':'float32'}
    output=[]
    generation._vllm_generate_pending(Engine(),Tokenizer(),rows,settings,{'run':'fixed'},
        {'request_batch_size':2},params_type,output.extend)
    assert [len(prompts) for prompts,params in seen]==[2,2,1]
    assert [r['prompt_id'] for r in output]==[r['prompt_id'] for r in rows]
    params=[p for prompts,values in seen for p in values]
    assert [p.seed for p in params]==[generation._seed(settings['seed'],r['prompt_id'],'clean_train') for r in rows]
    assert all(p.n==1 and p.temperature==.8 and p.top_p==1 and p.top_k==-1
        and p.max_tokens==512 and p.stop_token_ids==[99] and not p.ignore_eos for p in params)
    # A resume schedules only the suffix without revisiting committed IDs.
    seen.clear();remaining=[]
    generation._vllm_generate_pending(Engine(),Tokenizer(),rows[3:],settings,{'run':'fixed'},
        {'request_batch_size':2},params_type,remaining.extend)
    assert [r['prompt_id'] for r in remaining]==['3','4']
    assert len(seen)==1


def test_batch_execution_change_preserves_identity_but_sampling_change_is_rejected():
    original={'run':'frozen','prompts_sha256':'same','settings':{'backend':'vllm','dtype':'float32',
        'temperature':.8,'seed':17,'actual_request_batch_size':1,'requested_batch_size':8}}
    current=dict(original,settings=dict(original['settings'],actual_request_batch_size=8))
    assert generation._same_sampling_identity(original,current)
    for name,value in [('backend','hf'),('dtype','bfloat16'),('temperature',1.),('seed',18)]:
        changed=dict(current,settings=dict(current['settings'],**{name:value}))
        assert not generation._same_sampling_identity(original,changed)


def test_new_cli_runs_default_to_vllm_without_modifying_legacy_config_constructor():
    import main_exp3
    assert main_exp3.NEW_RUN_GENERATION_BACKEND=='vllm'
    assert main_exp3.parser().parse_args([]).generation_backend is None


def test_vllm_engine_uses_ordinary_fp32_graph_decode_and_bounded_batch():
    settings={'model':'ordinary-merged','revision':None,'tokenizer':'pinned-clean',
        'tokenizer_revision':'frozen','dtype':'float32'}
    options=generation._vllm_options(settings,{'request_batch_size':8})
    assert options['max_num_seqs']==8
    assert options['dtype']=='float32' and options['enforce_eager'] is False
    assert options['attention_config']=={'backend':'TRITON_ATTN'}
    assert options['compilation_config']['cudagraph_mode']=='FULL_DECODE_ONLY'
    assert 'hf_overrides' not in options and 'logits_processors' not in options


@pytest.mark.parametrize('failure',['engine','order'])
def test_vllm_failed_or_mismatched_batch_never_publishes_fake_empty_results(failure):
    from types import SimpleNamespace
    class Tokenizer:
        def apply_chat_template(self,*args,**kwargs):return [10,11]
    class Engine:
        def generate(self,*args,**kwargs):
            if failure=='engine':raise RuntimeError('simulated GPU failure')
            return [SimpleNamespace(prompt_token_ids=[777],outputs=[SimpleNamespace()])]
    output=[]
    settings={'temperature':.8,'max_new_tokens':512,'seed':17,'kind':'student','eos_token_ids':[99]}
    with pytest.raises((ValueError,RuntimeError)):
        generation._vllm_generate_pending(Engine(),Tokenizer(),[{'prompt_id':'p','prompt':'same'}],
            settings,{}, {'request_batch_size':8},lambda **kwargs:kwargs,output.extend)
    assert output==[]
