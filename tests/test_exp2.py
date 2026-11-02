from pathlib import Path
from dataclasses import replace
import json
import numpy as np
import pytest
import torch


def tiny_sampler():
    from transformers import Qwen2Config,Qwen2ForCausalLM
    from klrd_rs.math import planes
    from klrd_rs.readout import RotationReadout
    from klrd_rs.e2.sampler import ExactRotationTeacherSampler
    torch.manual_seed(431)
    model=Qwen2ForCausalLM(Qwen2Config(vocab_size=41,hidden_size=16,intermediate_size=32,
        num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,max_position_embeddings=128,
        attention_dropout=0.,rms_norm_eps=1e-6)).eval()
    model.requires_grad_(False)
    class Tokenizer:
        pad_token_id=0;eos_token_id=40;all_special_ids=[0,40];chat_template='toy'
        def decode(self,ids,**kw):return ''.join(chr(65+i) for i in ids if i not in self.all_special_ids)
        def encode(self,text,**kw):return [ord(c)-65 for c in text]
        def apply_chat_template(self,messages,**kw):return [1,2,3]
    p,q=map(torch.tensor,planes(np.eye(16),3,19))
    engine=RotationReadout(model.lm_head.weight,model.model.norm.weight,1e-6,p,q,.8)
    return ExactRotationTeacherSampler(model,Tokenizer(),engine,np.eye(3),.09,kl_stride=2)


def test_e2_key_reconstruction_and_source_receipt():
    from klrd_rs.e2.frozen import load_frozen
    root=Path(__file__).resolve().parents[1]
    if not (root/'e2_inputs/source_receipt.json').exists():pytest.skip('Frozen E2 input receipt not prepared')
    frozen=load_frozen(root)
    assert frozen['bank'].shape==(64,32)
    assert frozen['strength']==.09271215438842773
    assert frozen['P'].shape==(2048,32)


def test_e2_cohort_excludes_normalized_duplicates_and_has_exact_nested_groups():
    from klrd_rs.e2.cohort import select_cohort,exclusion_sets,make_groups
    class Tok:
        def encode(self,s,**kw):return list(range(20))
    prior=[{'prompt_id':'used','prompt':'Hello   WORLD'}]
    ex=exclusion_sets(prior)
    datasets={'test_gen':[{'prompt_id':'used','prompt':'different'},
        {'prompt_id':'alias','prompt':' hello world '}]+[{'prompt_id':str(i),'prompt':'fresh '+str(i)} for i in range(5001)],'train_gen':[]}
    rows=select_cohort(datasets,Tok(),ex,5000,2026110101)
    assert len(rows)==5000 and not any(r['prompt_id'] in ('used','alias') for r in rows)
    groups=make_groups(rows,2026110103)
    assert sum(g['Q']==1000 for g in groups)==5 and sum(g['Q']==100 for g in groups)==50
    large=[g for g in groups if g['Q']==1000]
    assert len(set(x for g in large for x in g['prompt_ids']))==5000
    for group in large:
        nested=[g for g in groups if g['Q']==100 and g['parent_group_id']==group['group_id']]
        assert set(x for g in nested for x in g['prompt_ids'])==set(group['prompt_ids'])


def test_e2_prompt_shortage_stops_instead_of_shrinking_design():
    from klrd_rs.e2.cohort import select_cohort,exclusion_sets
    class Tok:
        def encode(self,*a,**kw):return [1]*20
    with pytest.raises(ValueError,match='INSUFFICIENT_ELIGIBLE_FRESH_PROMPTS'):
        select_cohort({'test_gen':[{'prompt_id':'one','prompt':'Only one'}],'train_gen':[]},Tok(),exclusion_sets([]),5000,1)


def test_e2_failed_kv_precision_diagnostic_is_only_a_probe(tmp_path):
    from klrd_rs.e2.gates import diagnose_kv_roundoff
    from klrd_rs.e2.config import E2Config
    from klrd_rs.math import keys
    import json
    sampler=tiny_sampler()
    requests=[{'prompt_id':str(i),'prompt_ids':[1,2,3+i%2]} for i in range(10)]
    weights=sampler.model.lm_head.weight.clone()
    details=[{'batch_start':4,'owner':7,'max_abs_logprob_error':.001}]
    diagnose_kv_roundoff(sampler,requests,keys(64,3,7),E2Config(),details,tmp_path)
    report=json.loads((tmp_path/'tests/kv_linear_precision_diagnostic.json').read_text())
    assert report['status']=='DIAGNOSTIC_ONLY_PIPELINE_STOPPED'
    assert len(report['linear_fp64_probe'])==3
    assert report['prompt_ids']==['4','5','6','7']
    assert torch.equal(weights,sampler.model.lm_head.weight)
    assert all('forward' not in layer.__dict__ for layer in sampler.model.modules() if isinstance(layer,torch.nn.Linear))


def test_e2_cached_sampler_native_rotation_parity_prefill_decode_padding():
    sampler=tiny_sampler()
    from klrd_rs.math import rotate
    prompts=[[1,2,3],[4,5,6,7,8]]
    session=sampler.session(prompts)
    for step in range(3):
        r=session.residual
        angles=torch.tensor([.1,-.04,.2])
        actual=sampler.readout.log_probs(r,angles)
        native=sampler.model.lm_head(sampler.model.model.norm(rotate(r,sampler.readout.p,sampler.readout.q,angles)))
        torch.testing.assert_close(actual,torch.log_softmax(native/.8,-1),atol=1e-5,rtol=1e-5)
        full=session.full_prefix_residual()
        torch.testing.assert_close(r,full,atol=1e-6,rtol=1e-5)
        torch.testing.assert_close(sampler.readout.log_probs(r,torch.zeros(3)),
            torch.log_softmax(sampler.model.lm_head(sampler.model.model.norm(r))/.8,-1),atol=1e-6,rtol=1e-5)
        session.advance(torch.tensor([9,10]),torch.tensor([True,True]))
    assert session.cache is not None and session.decode_calls==3


def test_e2_generation_seed_reproducibility_and_contexts_are_not_replayed(monkeypatch):
    sampler=tiny_sampler()
    lengths=[]
    def trace(module,args,kwargs):lengths.append(kwargs['input_ids'].shape[-1])
    hook=sampler.model.model.register_forward_pre_hook(trace,with_kwargs=True)
    requests=[{'prompt_id':'a','prompt_ids':[1,2,3],'seed':13},
              {'prompt_id':'b','prompt_ids':[4,5,6,7],'seed':14}]
    one=sampler.generate(requests,np.array([1.,0,0]),max_new_tokens=8)
    hook.remove()
    assert lengths[0]==4 and all(n==1 for n in lengths[1:])
    two=sampler.generate(requests,np.array([1.,0,0]),max_new_tokens=8)
    assert [r['generated_token_ids'] for r in one]==[r['generated_token_ids'] for r in two]
    assert all(r['sampling_temperature_applications']==1 for r in one)
    assert all(all(0<=i<41 for i in r['generated_token_ids']) for r in one)


def test_e2_eos_is_sampled_and_empty_response_is_retained(monkeypatch):
    sampler=tiny_sampler()
    original=torch.multinomial
    def eos(*args,**kw):return torch.tensor([40],device=args[0].device)
    monkeypatch.setattr(torch,'multinomial',eos)
    result=sampler.generate([{'prompt_id':'a','prompt_ids':[1,2,3],'seed':7}],None,8)[0]
    assert result['generated_token_ids']==[40] and result['response']==''
    assert result['eos_terminated'] and not result['max_length_truncated']


def test_e2_atomic_jobs_resume_without_regeneration_and_reject_corruption(tmp_path):
    from klrd_rs.e2.jobs import JobStore
    store=JobStore(tmp_path,{'runtime':'one'})
    identity={'owner':0,'group':0,'chunk':0,'prompt_ids':['a']}
    assert store.load('one',identity) is None
    record={'prompt_id':'a','generated_token_ids':[1,2,40],'seed':4}
    store.commit('one',identity,[record])
    assert store.load('one',identity)==[record]
    with pytest.raises(ValueError):store.load('one',dict(identity,owner=1))
    path=store.path('one');x=json.loads(path.read_text());x['records'][0]['seed']=123;path.write_text(json.dumps(x))
    with pytest.raises(ValueError,match='checksum'):store.load('one',identity)


def test_e2_black_box_primary_does_not_use_hidden_token_ids():
    from klrd_rs.e2.verifier import SphericalBlackBoxVerifier
    sampler=tiny_sampler()
    verifier=SphericalBlackBoxVerifier(sampler.model,sampler.tokenizer,sampler.readout,np.eye(3),chunk_size=4)
    v,n=verifier.score_text_batch([{'prompt':'P','response':'FG'}])
    assert v.shape==(1,3) and n.tolist()==[2]
    k=np.eye(3)
    from klrd_rs.math import sphere_p
    assert np.allclose(verifier.key_scores(v.sum(0),k)['p'],sphere_p(v.sum(0),k))


def test_e2_wrong_key_rule_is_fixed_and_21_way_ties_are_incorrect():
    from klrd_rs.e2.analysis import wrong_indices,identify
    assert wrong_indices(0)==list(range(1,21))
    assert wrong_indices(63)==list(range(20))
    assert identify([.5,.5,.4])=={'correct':False,'tie':True}
    assert identify([.6,.5,.4])=={'correct':True,'tie':False}


def test_e2_incomplete_or_sampler_failure_can_never_be_scientific_go():
    from klrd_rs.e2.analysis import decide
    assert decide(False,False,[5]*8,[5]*8,{})=='E2_INCONCLUSIVE'
    assert decide(True,False,[5]*8,[5]*8,{})=='E2_INCONCLUSIVE'
    assert decide(True,True,[4]*8,[5]*8,{'wrong_key_concern':False})=='E2_PROMISING_FOR_LORA_REVIEW'
    assert decide(True,True,[4]*8,[5]*8,{'wrong_key_concern':True})=='E2_PARTIAL'
    assert decide(True,True,[0]*8,[0]*8,{})=='E2_NO_GO'


def test_e2_runtime_gates_include_ten_prefills_twenty_decodes_and_resume(tmp_path):
    from klrd_rs.e2.gates import validate_sampler
    from klrd_rs.e2.config import E2Config
    from klrd_rs.math import keys
    sampler=tiny_sampler()
    from klrd_rs.e2.sampler import stabilize_linear_accumulation
    stabilize_linear_accumulation(sampler.model)
    requests=[{'prompt_id':str(i),'prompt_ids':[1,2,3]+[4]*(i%3),'seed':i+9} for i in range(10)]
    result=validate_sampler(sampler,requests,keys(64,3,7),replace(E2Config(),batch_size=4),tmp_path)
    assert result['sampler']['status']=='PASS'
    assert result['sampler']['linear_accumulation']=='float64'
    assert result['sampler']['prefill_contexts']==10
    assert result['sampler']['incremental_decode_contexts']==20
    assert result['KV']['status']=='PASS' and result['seeded_resume']=='PASS'


def test_e2_full_group_analysis_uses_five_not_bootstrapped_Q1000_trials(tmp_path):
    from klrd_rs.e2.cohort import make_groups
    from klrd_rs.e2.config import E2Config
    from klrd_rs.e2.analysis import analyze_groups
    from klrd_rs.math import keys
    import csv
    cfg=E2Config();bank=keys(64,32,2026102803)
    rows=[{'prompt_id':str(i)} for i in range(5000)]
    records={r['prompt_id']:{'eligible_tokens':2,'roundtrip_equal':True} for r in rows}
    observations={'clean':(records,{str(i):np.zeros(32) for i in range(5000)})}
    for k in range(8):observations[str(k)]=(records,{str(i):bank[k] for i in range(5000)})
    result=analyze_groups(make_groups(rows,cfg.group_seed),bank,observations,tmp_path,cfg)
    assert result['complete'] and result['conservative_counts']==[5]*8
    assert all(r['n_disjoint_groups']==(5 if r['Q']==1000 else 50) for r in result['per_key'])
    wrong=list(csv.DictReader((tmp_path/'scores/batch_wrong_key_scores.csv').open()))
    assert len(wrong)==8*55*20
    assert all(p['clean_unique_responses']==5000 for p in result['pooled'])
    assert result['pooled'][1]['clean_true_disjoint_groups_max']==5


def test_e2_main_protects_all_old_runs_and_frozen_input_subtrees(tmp_path,monkeypatch):
    import main_exp2 as main
    root=tmp_path/'project';old=root/'experiment1_rs_continuation';old.mkdir(parents=True)
    marker=old/'keep';marker.write_text('unchanged')
    monkeypatch.setattr(main,'__file__',str(root/'main_exp2.py'))
    with pytest.raises(SystemExit):main.main(['--output',str(old/'nested')])
    assert list(old.iterdir())==[marker]


def test_e2_job_identity_is_created_once_and_concurrent_initializers_agree(tmp_path):
    from klrd_rs.e2.jobs import JobStore
    import os
    from concurrent.futures import ThreadPoolExecutor
    JobStore(tmp_path,{'one':'frozen'})
    p=tmp_path/'identity.json';os.utime(p,ns=(1600000000000000000,1600000000000000000))
    before=p.stat().st_mtime_ns
    JobStore(tmp_path,{'one':'frozen'})
    assert p.stat().st_mtime_ns==before
    with ThreadPoolExecutor(max_workers=8) as pool:
        stores=list(pool.map(lambda i:JobStore(tmp_path/'concurrent',{'one':'frozen'}),range(24)))
    assert len(stores)==24


def test_e2_padded_batch_matches_independent_request_and_verifier_padding():
    sampler=tiny_sampler();prompts=[[1,2,3],[4,5,6,7,8]]
    batch=sampler.session(prompts)
    for i,prompt in enumerate(prompts):
        single=sampler.session([prompt])
        torch.testing.assert_close(batch.residual[i],single.residual[0],atol=1e-6,rtol=1e-5)
    from klrd_rs.e2.verifier import SphericalBlackBoxVerifier
    v=SphericalBlackBoxVerifier(sampler.model,sampler.tokenizer,sampler.readout,np.eye(3))
    rows=[{'prompt':'P','response':'FG'},{'prompt':'P','response':'FGHIJ'}]
    both,n=v.score_text_batch(rows)
    for i,row in enumerate(rows):
        one,_=v.score_text_batch([row]);np.testing.assert_allclose(both[i],one[0],rtol=1e-5,atol=1e-6)
    assert n.tolist()==[2,5]


def test_e2_analysis_refuses_changed_nested_groups_without_regenerating(tmp_path):
    from klrd_rs.e2.runner import validate_design
    from klrd_rs.e2.config import E2Config
    from klrd_rs.io import digest
    cfg=E2Config();receipt='frozen'
    manifest={'groups':[{'group_id':'g0','prompt_ids':['a','b']}],'E2_config':cfg.resolved(),
        'input_receipt_sha256':receipt}
    identity={'design_sha256':digest({'cohort':manifest,'config':cfg.resolved(),'source':receipt})}
    validate_design(manifest,cfg,receipt,identity)
    changed=dict(manifest,groups=[{'group_id':'g0','prompt_ids':['b','a']}])
    with pytest.raises(ValueError,match='design|group'):
        validate_design(changed,cfg,receipt,identity)


def test_e2_partial_single_owner_report_and_private_zip_are_safe(tmp_path,monkeypatch):
    from klrd_rs.e2 import runner
    from klrd_rs.e2.config import E2Config
    from klrd_rs.e2.cohort import make_groups
    from klrd_rs.io import write_json
    from klrd_rs.math import keys
    import zipfile,uuid
    root=Path(__file__).resolve().parents[1];cfg=E2Config();bank=keys(64,32,2)
    prompts=[{'prompt_id':str(i),'prompt':'A test prompt'} for i in range(5000)]
    groups=make_groups(prompts,cfg.group_seed);ids=groups[0]['prompt_ids']
    record={'eligible_tokens':2,'roundtrip_equal':True,'response':'ordinary text','native_visible_tokens':2,
        'eos_terminated':True,'max_length_truncated':False,'KL_samples':[{'position':0,'KL':.005}],
        'generated_token_ids':[1,2,40],'amortized_generation_seconds':.01}
    observed={x:({}, {}) for x in ['clean',*map(str,range(8)),'23','59']}
    observed['clean']=({i:dict(record) for i in ids},{i:np.zeros(32) for i in ids})
    observed['1']=({i:dict(record) for i in ids},{i:bank[1] for i in ids})
    monkeypatch.setattr(runner,'collect',lambda *a:(observed,[],{'clean':1000,'1':1000}))
    private=tmp_path/'private';private.mkdir()
    (private/'prompts.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in prompts))
    secret=uuid.uuid4().hex;(private/'secret_key.txt').write_text(secret)
    for name in ['exact_sampler_parity','kv_cache_parity','batch_prefill_parity','reference_engine_parity']:
        write_json(tmp_path/('tests/'+name+'.json'),{'status':'PASS'})
    manifest={'groups':groups,'quality_prompt_ids':[str(i) for i in range(100)],'source_split_counts':{'test_gen':5000},
        'cohort_provenance_status':'VERIFIED_LOADED_MANIFESTS','unknown_prior_prompt_text_count':0}
    result=runner.analyze(root,tmp_path,{'bank':bank},manifest,None,cfg,'base')
    assert result['decision']=='E2_INCONCLUSIVE' and not result['complete']
    with zipfile.ZipFile(tmp_path/'KLRD_RS_experiment2_review.zip') as z:
        assert not any('/private/' in name for name in z.namelist())
        assert all(secret.encode() not in z.read(name) for name in z.namelist())
