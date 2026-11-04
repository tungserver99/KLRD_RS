"""SERVER ONLY contracts for E2B; do not execute on the local machine."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from klrd_rs.e2b.protocol import PARAPHRASE_PROMPT


def test_selection_is_order_independent_fixed_400_and_ignores_results():
    from klrd_rs.e2b.config import select_prompt_ids,make_groups
    candidates=[f'p{i}' for i in range(1000)]
    chosen=select_prompt_ids(candidates)
    assert chosen==select_prompt_ids(list(reversed(candidates)))
    assert len(chosen)==len(set(chosen))==400
    groups=make_groups(chosen)
    for q,count in [(20,20),(50,8),(100,4)]:
        selected=[g for g in groups if g['Q']==q]
        assert len(selected)==count
        assert [pid for g in selected for pid in g['prompt_ids']]==chosen
    with pytest.raises(ValueError):select_prompt_ids(candidates[:-1])
    with pytest.raises(ValueError):select_prompt_ids(['duplicate']*1000)


def test_transport_requests_never_include_prompt_keys_or_scores():
    from klrd_rs.e2b.transport import requests_for
    rows=[{'condition':'0','prompt_id':'p','response_id':'r0','original_text':'ONLY RESPONSE',
        'prompt':'PRIVATE PROMPT','key':[1,2],'verifier_score':.123},
        {'condition':'1','prompt_id':'p','response_id':'r1','original_text':'OTHER RESPONSE'}]
    request=requests_for(rows,'0')[0]
    assert set(request)=={'split','prompt_id','response_id','source_text','source_index'}
    assert request==dict(split='fingerprint',prompt_id='p',response_id='r0',source_text='ONLY RESPONSE',source_index=0)
    assert requests_for(rows,'1')[0]['response_id']=='r1'


def test_success_requires_full_source_mapping_and_exact_model():
    from klrd_rs.e2b.transport import checked_results,IncompleteParaphrases
    from klrd_scw.scw_paraphrase import SCW_MODEL
    source=dict(split='clean',prompt_id='p',response_id='r',source_text='original',source_index=0)
    good=dict(source,status='ok',paraphrased_text=' rewrite\n ',paraphraser_model=SCW_MODEL,
        paraphrase_prompt=PARAPHRASE_PROMPT,api_response_model=SCW_MODEL)
    assert checked_results([source],[good])[0]['paraphrased_text']==' rewrite\n '
    with pytest.raises(IncompleteParaphrases):checked_results([source],[])
    with pytest.raises(IncompleteParaphrases):checked_results([source],[dict(good,status='failed')])
    with pytest.raises(ValueError):checked_results([source],[dict(good,source_text='changed')])
    with pytest.raises(ValueError):checked_results([source],[dict(good,api_response_model='gpt-5')])


def test_output_is_independent_of_e2_and_original_klrd(tmp_path):
    from klrd_rs.e2b.source import resolve_output
    root=tmp_path/'KLRD_RS';root.mkdir()
    assert resolve_output(root,None)==root/'experiment2_paraphrase/run_002'
    for value in [root/'experiment2_rs/run_012',tmp_path/'KLRD',root/'experiment2_paraphrase']:
        with pytest.raises(ValueError):resolve_output(root,value)


def test_score_cache_checks_text_ids_runtime_and_file_hash(tmp_path):
    from klrd_rs.e2b.scoring import load_score_batch,save_score_batch
    identity={'texts':'frozen','runtime':'frozen'}
    arrays={'vectors':np.zeros((2,32)),'eligible_tokens':np.array([2,3]),'response_ids':np.array(['a','b'])}
    save_score_batch(tmp_path,'batch',identity,arrays)
    result=load_score_batch(tmp_path,'batch',identity)
    assert result['response_ids'].tolist()==['a','b']
    with pytest.raises(ValueError):load_score_batch(tmp_path,'batch',dict(identity,texts='changed'))
    with (tmp_path/'batch.npz').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError):load_score_batch(tmp_path,'batch',identity)


def test_pair_comparison_uses_fixed_groups_and_unchanged_spherical_thresholds():
    from klrd_rs.e2b.analysis import summarize_comparisons
    rows=[]
    for stage,p in [('original',1e-6),('paraphrased',.5)]:
        for group in range(4):
            for role,test_key in [('correct',0),('clean',0),('wrong',1)]:
                rows.append(dict(stage=stage,owner=0,Q=100,group_index=group,role=role,
                    test_key=test_key,p_value=p if role=='correct' else 1.,cosine=.2 if role=='correct' else 0.))
    result=summarize_comparisons(rows)
    assert len(result)==2
    for row in result:
        assert row['groups']==4 and row['TPR_original']==1. and row['TPR_paraphrased']==0.
        assert row['TPR_drop_percentage_points']==100.
        assert row['clean_FPR_paraphrased']==row['wrong_key_mean_paraphrased']==0.


def test_vendor_hash_and_env_location_match_copy_receipt():
    import json,hashlib
    from klrd_scw import scw_paraphrase
    root=Path(__file__).resolve().parents[1]
    receipt=json.loads((root/'docs/superpowers/references/e2b_scw_protocol.json').read_text())
    for item in receipt['copied_files']:
        assert hashlib.sha256((root/item['copy']).read_bytes()).hexdigest()==item['sha256']
    assert scw_paraphrase.PROJECT_ENV_PATH==root/'.env'
    assert receipt['sdk_version']=='2.54.0'


def test_e2_launcher_and_verifier_are_not_edited_for_e2b():
    root=Path(__file__).resolve().parents[1]
    assert (root/'run_experiment2.sh').read_text().strip()=='python main_exp2.py --pilot-only'
    assert (root/'run_experiment2_paraphrase.sh').read_text().strip()=='python -u tools/run_e2b_with_progress.py "$@"'


def test_context_limit_counts_original_chat_prompt_and_never_clips_rewrite():
    from klrd_rs.e2b.scoring import verifier_rows
    class Tokenizer:
        all_special_ids=[]
        def apply_chat_template(self,messages,**kwargs):
            assert messages==[{'role':'user','content':'ORIGINAL TASK'}]
            return [10,11,12]
        def encode(self,text,**kwargs):
            assert text=='full paraphrase text'
            assert kwargs=={'add_special_tokens':False}
            return [1,2,3,4,5]
    pairs=[{'prompt':'ORIGINAL TASK','paraphrased_text':'full paraphrase text'}]
    with pytest.raises(RuntimeError,match='context window'):verifier_rows(pairs,Tokenizer(),7)
    assert verifier_rows(pairs,Tokenizer(),8)==[{'prompt':'ORIGINAL TASK','response':'full paraphrase text'}]


def test_launcher_progress_wrapper_preserves_original_pipeline_arguments(monkeypatch,capsys):
    # SERVER ONLY; mock runpy so neither preparation nor API/GPU code executes.
    import importlib.util,sys
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('e2b_progress_wrapper',root/'tools/run_e2b_with_progress.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    observed={}
    monkeypatch.setattr(sys,'argv',['run_e2b_with_progress.py','--stage','paraphrase','--workers','8'])
    def fake_run(path,run_name):
        observed.update(path=path,run_name=run_name,argv=list(sys.argv))
        raise SystemExit(7)
    monkeypatch.setattr(module.runpy,'run_path',fake_run)
    with pytest.raises(SystemExit) as caught:module.main()
    assert caught.value.code==7
    assert observed['path']==str(root/'main_exp2_paraphrase.py')
    assert observed['argv']==[str(root/'main_exp2_paraphrase.py'),'--stage','paraphrase','--workers','8']
    assert observed['run_name']=='__main__'
    output=capsys.readouterr().out
    assert output.splitlines()==['Starting E2B: loading the selected-response pipeline.']


@pytest.fixture
def prepared_e2b_snapshot(tmp_path,monkeypatch):
    """Server-only fixture: E2B snapshot with NO source E1/E2 directories."""
    from klrd_rs.e2b import source
    from klrd_rs.e2b.config import CONDITIONS,SOURCE_RUN,SELECTION_DOMAIN,select_prompt_ids,make_groups
    from klrd_rs.io import write_json,save_npz,sha256,digest
    root=tmp_path/'KLRD_RS';root.mkdir();out=source.resolve_output(root,None)
    ids=select_prompt_ids([f'p{i}' for i in range(1000)])
    population=[dict(condition=c,prompt_id=pid,response_id=f'{c}:{pid}',
        prompt='original prompt',original_text='original response') for c in CONDITIONS for pid in ids]
    source.immutable_rows(out/'private/population.jsonl',population)
    save_npz(out/'private/original_vectors.npz',vectors=np.zeros((3600,32)),
        eligible_tokens=np.ones(3600,dtype=np.int64),response_ids=np.array([r['response_id'] for r in population]))
    manifest=dict(source_run=SOURCE_RUN,conditions=list(CONDITIONS),selection_domain=SELECTION_DOMAIN,
        prompt_ids=ids,groups=make_groups(ids),population_sha256=digest(population),
        original_vectors_sha256=sha256(out/'private/original_vectors.npz'),
        # Deliberately nonexistent old source/code files must not be scanned.
        source_sha256={'experiment2_rs/run_012/unavailable':'old'},
        integration_code_sha256={'old_pipeline.py':'old'})
    write_json(out/'provenance/population_manifest.json',manifest)
    write_json(out/'provenance/prepare_complete.json',
        {'population_manifest_sha256':sha256(out/'provenance/population_manifest.json')})
    monkeypatch.setattr(source,'protocol_receipt',lambda root:{})
    def forbidden(*args,**kwargs):raise AssertionError('Verifier inputs loaded before verification')
    monkeypatch.setattr(source,'load_verifier_inputs',forbidden)
    return root,out,population,manifest


def test_resume_uses_selected_snapshot_without_original_trees_or_code_reaudit(prepared_e2b_snapshot):
    from klrd_rs.e2b.source import load_prepared
    from klrd_rs.io import digest
    root,out,population,manifest=prepared_e2b_snapshot
    state=load_prepared(root,out)
    assert state['population']==population
    assert state['frozen'] is None
    # Preserves old scoring-cache identities and paid API receipt mappings.
    assert state['manifest']==manifest and digest(state['manifest'])==digest(manifest)


def test_light_resume_still_rejects_changed_original_response(prepared_e2b_snapshot):
    import json
    from klrd_rs.e2b.source import load_prepared
    root,out,population,_=prepared_e2b_snapshot
    population[0]['original_text']='changed text'
    (out/'private/population.jsonl').write_text(
        ''.join(json.dumps(r)+'\n' for r in population),encoding='utf-8')
    with pytest.raises(ValueError,match='population changed'):load_prepared(root,out)


def test_light_resume_still_rejects_changed_original_vectors(prepared_e2b_snapshot):
    from klrd_rs.e2b.source import load_prepared
    root,out,_,_=prepared_e2b_snapshot
    with (out/'private/original_vectors.npz').open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='vector cache changed'):load_prepared(root,out)

@pytest.fixture
def failed_api_resume(tmp_path):
    from klrd_rs.e2b.transport import receipts_path,SOURCE_FIELDS
    from klrd_scw import scw_paraphrase as scw
    from klrd_scw.artifacts import ensure_manifest,digest
    state={'out':tmp_path.resolve()}
    requests=[dict(split='fingerprint',prompt_id=f'p{i}',response_id=f'r{i}',
        source_text=f'original {i}',source_index=i) for i in range(3)]
    path=receipts_path(state,'0');path.mkdir(parents=True)
    ensure_manifest(path/'api_manifest.json',dict(scw_commit=scw.SCW_COMMIT,
        paraphraser_model=scw.SCW_MODEL,paraphrase_prompt=PARAPHRASE_PROMPT,
        source_population_sha256=digest([{field:r[field] for field in SOURCE_FIELDS} for r in requests])))
    ok=dict(requests[0],status='ok',paraphrased_text='PAID RESULT, KEEP VERBATIM\n',
        paraphraser_model=scw.SCW_MODEL,paraphrase_prompt=PARAPHRASE_PROMPT,attempts=1,
        api_response_model=scw.SCW_MODEL,api_response_id='paid-result')
    failed=dict(requests[1],status='failed',paraphrased_text='',
        paraphraser_model=scw.SCW_MODEL,paraphrase_prompt=PARAPHRASE_PROMPT,
        attempts=5,error='OpenAI request failed (HTTP 429); upstream detail omitted')
    scw._write_jsonl(path/'api_results.jsonl',[ok,failed])
    return state,requests,path,ok,failed


def test_resume_calls_api_only_for_failed_and_missing_and_keeps_paid_result(failed_api_resume):
    from klrd_rs.e2b.transport import prepare_condition_resume,run_copied_transport
    from klrd_scw import scw_paraphrase as scw
    from klrd_scw.artifacts import read_jsonl
    state,requests,path,ok,_=failed_api_resume
    original=(path/'api_results.jsonl').read_bytes()
    plan=prepare_condition_resume(state,'0',requests)
    assert (plan['kept'],plan['retry_failed'],plan['missing'])==(1,1,1)
    assert (state['out']/plan['archive']).read_bytes()==original
    assert read_jsonl(path/'api_results.jsonl')==[ok]
    calls=[]
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_text='new rewrite',model=scw.SCW_MODEL,id='new',status='completed')
    client=SimpleNamespace(responses=SimpleNamespace(create=create))
    results=run_copied_transport(requests,path,workers=1,client=client)
    assert calls==[dict(model=scw.SCW_MODEL,input=PARAPHRASE_PROMPT+r['source_text']) for r in requests[1:]]
    assert results[0]==ok
    assert [r['response_id'] for r in results]==['r0','r1','r2']
    assert all(r['status']=='ok' for r in results)
    # A second resume performs ZERO API calls, including client creation.
    current=(path/'api_results.jsonl').read_bytes()
    assert prepare_condition_resume(state,'0',requests)==dict(kept=3,retry_failed=0,missing=0)
    assert (path/'api_results.jsonl').read_bytes()==current
    calls.clear()
    assert run_copied_transport(requests,path,workers=1,client=client)==results
    assert calls==[]


def test_resume_backup_precedes_journal_replacement(failed_api_resume,monkeypatch):
    from klrd_rs.e2b.transport import prepare_condition_resume,run_copied_transport
    from klrd_scw import scw_paraphrase as scw
    state,requests,path,_,_=failed_api_resume
    original=(path/'api_results.jsonl').read_bytes()
    def interrupted(*args):raise OSError('simulated interruption before publish')
    monkeypatch.setattr(scw,'_write_jsonl',interrupted)
    with pytest.raises(OSError):prepare_condition_resume(state,'0',requests)
    assert (path/'api_results.jsonl').read_bytes()==original
    backups=list((path/'retry_history').glob('*.jsonl'))
    assert len(backups)==1 and backups[0].read_bytes()==original


def test_resume_refuses_wrong_source_before_requeue(failed_api_resume):
    from klrd_rs.e2b.transport import prepare_condition_resume,run_copied_transport
    from klrd_scw import scw_paraphrase as scw
    state,requests,path,ok,failed=failed_api_resume
    scw._write_jsonl(path/'api_results.jsonl',[ok,dict(failed,source_text='WRONG SOURCE')])
    original=(path/'api_results.jsonl').read_bytes()
    with pytest.raises(ValueError,match='Incompatible'):prepare_condition_resume(state,'0',requests)
    assert (path/'api_results.jsonl').read_bytes()==original
    assert not (path/'retry_history').exists()


def test_resume_archives_partial_tail_and_preserves_completed_output(failed_api_resume):
    from klrd_rs.e2b.transport import prepare_condition_resume,run_copied_transport
    from klrd_scw import scw_paraphrase as scw
    from klrd_scw.artifacts import read_jsonl
    state,requests,path,ok,_=failed_api_resume
    scw._write_jsonl(path/'api_results.jsonl',[ok])
    with (path/'api_results.jsonl').open('ab') as stream:stream.write(b'{"interrupted":')
    original=(path/'api_results.jsonl').read_bytes()
    plan=prepare_condition_resume(state,'0',requests)
    assert plan['kept']==1 and plan['missing']==2 and plan['recovered_partial_tail']
    assert (state['out']/plan['archive']).read_bytes()==original
    assert read_jsonl(path/'api_results.jsonl')==[ok]

def test_e2_version_difference_warns_and_records_actual_runtime_without_blocking(tmp_path,monkeypatch,capsys):
    # SERVER ONLY: replace torch with a stub; no model, CUDA operation or API call.
    import sys
    from klrd_rs.e2b import scoring
    source={'torch':'old-torch','transformers':'old-transformers','numpy':'old-numpy',
        'scipy':'old-scipy','CUDA':'old-cuda'}
    fake=SimpleNamespace(__version__='current-torch',version=SimpleNamespace(cuda='current-cuda'),
        cuda=SimpleNamespace(is_available=lambda:True,get_device_name=lambda index:'test-GPU'))
    monkeypatch.setitem(sys.modules,'torch',fake)
    monkeypatch.setattr(scoring,'version',lambda name:'current-'+name)
    monkeypatch.setattr(scoring,'sha256',lambda path:'unchanged-verifier')
    records={}
    monkeypatch.setattr(scoring,'write_json',lambda path,value:records.update({path.name:value}))
    state={'root':tmp_path,'out':tmp_path,'manifest':{'source_E2_environment':source}}
    runtime=scoring.scoring_runtime(state)
    assert runtime['torch']=='current-torch' and runtime['CUDA']=='current-cuda'
    assert runtime['linear_accumulation']=='float64'
    comparison=records['verification_environment_comparison.json']
    assert comparison['differences']['torch']=={'E2':'old-torch','E2B':'current-torch'}
    assert 'Continuing verification.' in capsys.readouterr().out
    # Matching versions do not emit warnings.
    state['manifest']['source_E2_environment']={name:runtime[name] for name in source}
    scoring.scoring_runtime(state)
    assert capsys.readouterr().out==''
    assert records['verification_environment_comparison.json']['differences']=={}
    # A real missing execution capability still gives an actionable error.
    fake.cuda.is_available=lambda:False
    with pytest.raises(RuntimeError,match='requires the GPU server'):scoring.scoring_runtime(state)

def test_high_fidelity_prompt_replaces_only_instruction_and_restores_original(monkeypatch,tmp_path):
    from klrd_rs.e2b.protocol import PARAPHRASE_PROMPT,prompt_for,PROTOCOL_VARIANT
    from klrd_rs.e2b.transport import run_copied_transport,checked_results
    from klrd_scw import scw_paraphrase as scw
    historical=scw.PARAPHRASE_PROMPT
    assert PROTOCOL_VARIANT=='high_fidelity_v1'
    assert PARAPHRASE_PROMPT.startswith('You are a high-fidelity text paraphraser.\n\n')
    assert PARAPHRASE_PROMPT.endswith('Text to paraphrase:\n\n')
    assert 'do not continue or complete the missing content' in PARAPHRASE_PROMPT
    assert '7. Preserve the distinction between facts, opinions, hypotheses, and recommendations.' in PARAPHRASE_PROMPT
    assert prompt_for({})==PARAPHRASE_PROMPT
    assert prompt_for({'manifest':{'protocol':{'prompt':historical}}})==historical
    def fake_transport(requests,path,**kwargs):
        assert scw.PARAPHRASE_PROMPT==PARAPHRASE_PROMPT
        raise RuntimeError('simulated interrupted API run')
    monkeypatch.setattr(scw,'run_paraphrases',fake_transport)
    with pytest.raises(RuntimeError):run_copied_transport([],tmp_path,workers=1)
    assert scw.PARAPHRASE_PROMPT==historical
    source=dict(split='clean',prompt_id='p',response_id='r',source_text='text',source_index=0)
    row=dict(source,status='ok',paraphrased_text='rewritten',paraphraser_model=scw.SCW_MODEL,
        paraphrase_prompt=historical)
    with pytest.raises(ValueError,match='instruction changed'):checked_results([source],[row])
    assert checked_results([source],[row],historical)==[row]