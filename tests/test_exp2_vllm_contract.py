"""SERVER ONLY: written but intentionally NOT executed on the development PC."""
import numpy as np
import pytest
import torch


@pytest.mark.parametrize('fused',[False,True])
def test_rotation_is_before_norm_and_residual_is_added_exactly_once(fused):
    from klrd_rs.e2.vllm_model import rotate_final_state
    from klrd_rs.math import rotate
    p=torch.eye(6)[:,:2];q=torch.eye(6)[:,2:4]
    x=torch.tensor([[1.,2.,3.,4.,5.,6.]])
    residual=torch.ones_like(x)*3 if fused else None
    angle=torch.tensor([.2,-.3])
    combined=x if residual is None else x+residual
    actual,total=rotate_final_state(x,residual,p,q,angle)
    torch.testing.assert_close(total,combined)
    torch.testing.assert_close(actual,rotate(combined,p,q,angle))
    assert torch.equal(x,torch.tensor([[1.,2.,3.,4.,5.,6.]]))


def test_sampling_applies_temperature_once_and_uses_original_seed():
    from klrd_rs.e2.vllm_backend import sampling_kwargs
    kwargs=sampling_kwargs(0.8,123456789012345,512,[2,3])
    assert kwargs['temperature']==.8 and kwargs['seed']==123456789012345
    assert kwargs['top_k']==-1 and kwargs['top_p']==1. and kwargs['min_p']==0.
    assert kwargs['repetition_penalty']==1. and kwargs['n']==1
    assert kwargs['stop_token_ids']==[2,3] and not kwargs['ignore_eos']
    probe=sampling_kwargs(.8,7,3,[2,3],probe=True)
    assert probe['ignore_eos'] and probe['stop_token_ids']==[]


def test_records_preserve_native_eos_and_empty_output():
    from klrd_rs.e2.vllm_backend import result_records
    class Tok:
        all_special_ids=[0,2];eos_token_id=2
        def decode(self,ids,**kwargs):return ''.join('a' for i in ids if i not in self.all_special_ids)
        def encode(self,text,**kwargs):return [1]*len(text)
    requests=[{'prompt_id':'a','seed':3},{'prompt_id':'b','seed':4}]
    raw=[{'token_ids':[2],'finish_reason':'stop','stop_reason':2},
         {'token_ids':[1,1],'finish_reason':'length','stop_reason':None}]
    records=result_records(requests,raw,Tok(),{2},2,2.)
    assert records[0]['response']=='' and records[0]['generated_token_ids']==[2]
    assert records[0]['eos_terminated'] and records[1]['max_length_truncated']
    assert [r['seed'] for r in records]==[3,4]
    assert records[0]['sampling_temperature_applications']==1


def test_missing_native_stop_token_fails_instead_of_fabricating_tokens():
    from klrd_rs.e2.vllm_backend import result_records
    with pytest.raises(RuntimeError,match='stop token'):
        result_records([{'prompt_id':'a','seed':3}],
            [{'token_ids':[1],'finish_reason':'stop','stop_reason':2}],None,{2},512,0.)


def test_tail_wrapper_keeps_weight_name_and_native_norm_kernel(monkeypatch):
    # Needs the pinned server vLLM installation; no import/skip on local machine.
    from vllm.model_executor.layers.layernorm import RMSNorm
    from klrd_rs.e2.vllm_model import RotationFinalRMSNorm
    # Unit-test dispatch only; no default model configuration or weights needed.
    from types import SimpleNamespace
    import vllm.model_executor.custom_op as custom_op
    dispatch=SimpleNamespace(level=0,use_inductor=False,custom_ops=[],
        enabled_custom_ops=set(),disabled_custom_ops=set())
    monkeypatch.setattr(custom_op,'get_cached_compilation_config',lambda:dispatch)
    original=RMSNorm(64,eps=1e-6,dtype=torch.float32).to('cuda')
    wrapped=RotationFinalRMSNorm.from_native(original,2)
    assert dict(wrapped.named_parameters())['weight'] is original.weight
    p=torch.eye(64,device='cuda')[:,:2];q=torch.eye(64,device='cuda')[:,2:4]
    wrapped.configure(p,q,torch.tensor([.2,-.1]))
    x=torch.arange(1,65,device='cuda',dtype=torch.float32)[None,:]
    residual=torch.ones_like(x)
    from klrd_rs.math import rotate
    expected=original(rotate(x+residual,p,q,wrapped.angles))
    actual,total=wrapped(x,residual)
    torch.testing.assert_close(actual,expected)
    torch.testing.assert_close(total,x+residual)
    torch.testing.assert_close(wrapped(x),original(rotate(x,p,q,wrapped.angles)))
    wrapped.configure(p,q,torch.zeros(2))
    native,_=original(x.clone(),residual.clone())
    clean,_=wrapped(x,residual)
    torch.testing.assert_close(clean,native)
    # Native calibration must call the original fused path, not rotation(0).
    wrapped.native_baseline=True
    native_y,native_residual=original(x.clone(),residual.clone())
    control_y,control_residual=wrapped(x.clone(),residual.clone())
    torch.testing.assert_close(control_y,native_y)
    torch.testing.assert_close(control_residual,native_residual)
    wrapped.native_baseline=False


def test_missing_engine_never_selects_hf_generation(monkeypatch):
    import klrd_rs.e2.vllm_backend as backend
    def missing(name):raise backend.PackageNotFoundError(name)
    monkeypatch.setattr(backend,'version',missing)
    with pytest.raises(RuntimeError,match='No HF generation fallback'):
        backend.require_vllm()


def test_unreviewed_vllm_version_is_rejected(monkeypatch):
    import klrd_rs.e2.vllm_backend as backend
    monkeypatch.setattr(backend,'version',lambda name:'0.11.0')
    with pytest.raises(RuntimeError,match='exactly 0.10.2'):backend.require_vllm()


def test_engine_change_keeps_the_exact_frozen_cohort(tmp_path):
    from klrd_rs.e2.vllm_cohort import reuse_frozen_cohort
    from klrd_rs.e2.cohort import make_groups
    from klrd_rs.e2.config import E2Config
    from klrd_rs.io import write_json,sha256,read_json
    from dataclasses import replace
    import json
    cfg=replace(E2Config(),prompt_count=1000)
    source=tmp_path/'experiment2_rs/run_005';pool=source/'private/prompts.jsonl'
    pool.parent.mkdir(parents=True)
    rows=[{'prompt_id':str(i),'prompt':'unit fixture '+str(i),'prompt_ids':[1,2,i]} for i in range(1000)]
    pool.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    before=sha256(pool)
    manifest={'private_prompts_sha256':before,'input_receipt_sha256':'frozen',
        'E2_config':dict(cfg.resolved(),engine='HF_PREVIOUS_ENGINE'),
        'groups':make_groups(rows,cfg.group_seed),'quality_prompt_ids':['9','8'],
        'smoke_prompt_ids':['9']}
    write_json(source/'provenance/prompt_manifest.json',manifest)
    out=tmp_path/'experiment2_rs/run_006'
    assert reuse_frozen_cohort(tmp_path,out,{'receipt_sha256':'frozen'},cfg)
    copied=read_json(out/'provenance/prompt_manifest.json')
    assert copied['groups']==manifest['groups']
    assert copied['quality_prompt_ids']==manifest['quality_prompt_ids']
    assert sha256(out/'private/prompts.jsonl')==before==sha256(pool)
    assert not (out/'private/jobs').exists()


def test_kl_audit_uses_native_prefix_positions_including_eos():
    from types import SimpleNamespace
    from klrd_rs.e2.vllm_kl import audit_native_prefix_kl
    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__();self.norm=torch.nn.Identity();self.inputs=None
        def forward(self,input_ids,attention_mask,**kwargs):
            self.inputs=input_ids.clone()
            return self.norm(input_ids.float()[:,:,None])
    class Readout:
        weight=torch.zeros(2,1)
        def __init__(self):self.observed=[]
        def log_probs(self,r,angles=None):
            self.observed.extend(r[:,0].tolist())
            adjustment=0. if angles is None else angles[0]
            return torch.log_softmax(torch.stack([r[:,0]+adjustment,torch.zeros(len(r))],-1),-1)
    backbone=Backbone();readout=Readout()
    sampler=SimpleNamespace(model=SimpleNamespace(model=backbone),readout=readout,
        tokenizer=SimpleNamespace(pad_token_id=0,eos_token_id=2),
        generation_seed=1,kl_stride=1,L=np.eye(1),strength=.1)
    requests=[{'prompt_id':'native','prompt_ids':[10,11,12]}]
    records=[{'generated_token_ids':[1,2],'KL_samples':[]}]
    audit_native_prefix_kl(sampler,requests,records,np.array([1.]))
    assert backbone.inputs.tolist()==[[10,11,12,1,2]]
    assert readout.observed==[12.,1.,12.,1.]
    assert [event['position'] for event in records[0]['KL_samples']]==[0,1]


def test_fp32_reference_probe_restores_existing_verifier_policy():
    from klrd_rs.e2.vllm_gates import fp32_reference_linears
    from klrd_rs.e2.sampler import stabilize_linear_accumulation
    layer=torch.nn.Linear(3,1,bias=False)
    with torch.no_grad():layer.weight.copy_(torch.tensor([[1e8,1.,-1e8]]))
    x=torch.ones(1,3);expected=torch.nn.functional.linear(x,layer.weight)
    stabilize_linear_accumulation(layer)
    assert layer(x).item()==1.
    with pytest.raises(RuntimeError):
        with fp32_reference_linears(layer):
            assert torch.equal(layer(x),expected)
            raise RuntimeError('reference interruption')
    assert layer(x).item()==1.


def test_batch_and_reference_failures_cannot_be_hidden_by_kv_pass(tmp_path):
    from klrd_rs.e2.vllm_gates import all_vllm_gates_pass
    from klrd_rs.io import write_json
    for name in ['exact_sampler_parity','kv_cache_parity']:
        write_json(tmp_path/('tests/'+name+'.json'),{'status':'PASS'})
    assert not all_vllm_gates_pass(tmp_path)
    write_json(tmp_path/'tests/batch_prefill_parity.json',{'status':'SAMPLER_INVALID'})
    write_json(tmp_path/'tests/reference_engine_parity.json',{'status':'PASS'})
    assert not all_vllm_gates_pass(tmp_path)
    write_json(tmp_path/'tests/batch_prefill_parity.json',{'status':'PASS'})
    # Historical four PASS files are insufficient without the new frozen-native
    # evidence, holdout, strict native-zero and the hashed approval marker.
    assert not all_vllm_gates_pass(tmp_path)


def test_worker_cleanup_tears_down_distributed_groups(monkeypatch):
    from types import SimpleNamespace
    from klrd_rs.e2.vllm_backend import cleanup_worker
    import vllm.distributed.parallel_state as parallel
    calls=[]
    engine=SimpleNamespace(llm_engine=SimpleNamespace(model_executor=SimpleNamespace(shutdown=lambda:calls.append('executor'))))
    monkeypatch.setattr(parallel,'destroy_model_parallel',lambda:calls.append('model_parallel'))
    monkeypatch.setattr(parallel,'destroy_distributed_environment',lambda:calls.append('distributed'))
    monkeypatch.setattr(torch.distributed,'is_initialized',lambda:False)
    cleanup_worker(engine)
    assert calls==['executor','model_parallel','distributed']


def test_diagnostic_and_worker_set_cublas_environment_before_torch_import():
    # Source-order regression: no model initialization needed by this test.
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    for name in ['tools/check_vllm_exp2.py','klrd_rs/e2/vllm_backend.py']:
        source=(root/name).read_text(encoding='utf-8')
        assert source.index("os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')")<source.index('import torch')


def test_native_linear_shape_probe_preserves_output_and_restores_method():
    from klrd_rs.e2.vllm_model import start_linear_shape_probe,stop_linear_shape_probe
    class UnquantizedLinearMethod:
        def apply(self,layer,x,bias=None):return torch.nn.functional.linear(x,layer.weight,bias)
    class Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.arange(16,dtype=torch.float32).reshape(4,4),requires_grad=False)
            self.quant_method=UnquantizedLinearMethod()
        def forward(self,x):return self.quant_method.apply(self,x,None)
    model=Layer();x=torch.arange(12,dtype=torch.float32).reshape(3,4)
    expected=model(x);weight=model.weight.clone()
    start_linear_shape_probe(model)
    try:actual=model(x)
    finally:rows=stop_linear_shape_probe(model)
    assert torch.equal(actual,expected) and torch.equal(model.weight,weight)
    assert 'apply' not in model.quant_method.__dict__
    assert len(rows)==1 and rows[0]['same_input_row_index']==2
    assert rows[0]['input_dtype']=='torch.float32'
    assert rows[0]['input_shape']==[3,4]
