"""GPU SERVER ONLY: migration contracts; intentionally not executed locally."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import inspect
import numpy as np
import pytest
import torch


def test_v1_runtime_preserves_fp32_and_uses_supported_decode_graph_configuration():
    from klrd_rs.e2.vllm_runtime import engine_options, runtime_identity
    settings={'model':'pinned-qwen','revision':'pinned-revision','generation_seed':42}
    normal=engine_options(settings)
    eager=engine_options(settings,eager=True)
    assert normal['dtype']=='float32' and normal['kv_cache_dtype']=='auto'
    assert normal['attention_config']=={'backend':'TRITON_ATTN'}
    assert normal['logprobs_mode']=='processed_logprobs'
    assert eager['logprobs_mode']=='processed_logprobs'
    assert normal['compilation_config']['mode']==0
    assert normal['compilation_config']['cudagraph_mode']=='FULL_DECODE_ONLY'
    assert normal['compilation_config']['cudagraph_capture_sizes']==list(range(1,9))
    assert normal['enable_prefix_caching'] is False
    assert normal['enable_chunked_prefill'] is True
    assert eager['enforce_eager'] is True and eager['enable_chunked_prefill'] is False
    assert eager['compilation_config']['cudagraph_mode']=='NONE'
    assert not {'swap_space','attention_backend'} & normal.keys()
    assert runtime_identity()['vllm_version']=='0.32.0'
    assert runtime_identity()['engine_version']=='V1'


def test_resolved_identity_matches_runtime_and_launcher_is_unchanged():
    from klrd_rs.e2.config import E2Config
    from klrd_rs.e2.vllm_runtime import runtime_identity
    resolved=E2Config().resolved()
    assert all(resolved[k]==v for k,v in runtime_identity().items())
    root=Path(__file__).resolve().parents[1]
    assert (root/'run_experiment2.sh').read_text().strip()=='python main_exp2.py'
    source=(root/'main_exp2.py').read_text()
    assert "experiment2_rs/run_012" in source and 'range(1,12)' in source


def test_version_guard_accepts_only_reviewed_vllm(monkeypatch):
    import klrd_rs.e2.vllm_backend as backend
    monkeypatch.setattr(backend,'version',lambda name:'0.32.0')
    assert backend.require_vllm()=='0.32.0'
    for version in ['0.10.2','0.31.0','0.32.1']:
        monkeypatch.setattr(backend,'version',lambda name,v=version:v)
        with pytest.raises(RuntimeError,match='exactly 0.32.0'):backend.require_vllm()


def test_logits_override_matches_v1_signature():
    from klrd_rs.e2.vllm_model import KLRDRSQwen2ForCausalLM
    assert list(inspect.signature(KLRDRSQwen2ForCausalLM.compute_logits).parameters)==['self','hidden_states']


def test_padded_query_boundaries_exclude_zero_length_padding():
    from klrd_rs.e2.vllm_model import trace_query_spans
    metadata=SimpleNamespace(query_start_loc=torch.tensor([0,3,5,5,5]),num_actual_tokens=5)
    assert trace_query_spans(metadata,8)==[(0,3),(3,5)]
    with pytest.raises(RuntimeError):
        trace_query_spans(SimpleNamespace(query_start_loc=torch.tensor([0,6]),num_actual_tokens=5),8)


def test_reordered_prefill_traces_match_full_prefix_not_just_last_token():
    from klrd_rs.e2.vllm_backend import ordered_prefill_traces
    requests=[{'prompt_ids':[4,7]},{'prompt_ids':[3,7]},{'prompt_ids':[4,7]}]
    rows=[{'query_token_ids':[3,7],'tag':'b'}, {'query_token_ids':[4,7],'tag':'a'},
          {'query_token_ids':[4,7],'tag':'c'}]
    assert [r['tag'] for r in ordered_prefill_traces(rows,requests)]==['a','b','c']
    with pytest.raises(RuntimeError):ordered_prefill_traces(rows[:2],requests)


def test_diagnostics_restart_one_worker_and_restore_production_after_exception():
    from klrd_rs.e2.vllm_backend import VLLMExactRotationTeacherSampler
    sampler=VLLMExactRotationTeacherSampler.__new__(VLLMExactRotationTeacherSampler)
    sampler._diagnostic_depth=0;sampler._worker_eager=False;events=[]
    def ensure(eager):
        events.append(bool(eager));sampler._worker_eager=bool(eager)
    sampler._ensure_worker=ensure
    with pytest.raises(ValueError):
        with sampler.diagnostic_mode():
            assert sampler._diagnostic_depth==1 and sampler._worker_eager
            with sampler.diagnostic_mode():assert sampler._diagnostic_depth==2
            raise ValueError('probe interruption')
    assert sampler._diagnostic_depth==0 and sampler._worker_eager is False
    assert events==[True,True,False]


def test_worker_switch_stops_before_starting_and_checks_model_identity():
    from klrd_rs.e2.vllm_backend import VLLMExactRotationTeacherSampler
    sampler=VLLMExactRotationTeacherSampler.__new__(VLLMExactRotationTeacherSampler)
    sampler._worker_eager=False;events=[]
    sampler._stop_worker=lambda:events.append('stop')
    def start(eager):events.append(('start',eager))
    sampler._start_worker=start
    sampler._ensure_worker(True)
    assert events==['stop',('start',True)]


def test_runtime_configuration_rejects_old_vllm_receipt(tmp_path):
    from klrd_rs.e2.vllm_gates import runtime_configuration_valid
    from klrd_rs.e2.vllm_runtime import runtime_identity
    from klrd_rs.io import write_json
    expected={'weight':'w','gain':'g','bias':None}
    generation=dict(runtime_identity(),readout_hashes=[expected],dtype='float32')
    path=tmp_path/'provenance/engine_manifest.json'
    write_json(path,{'generation':generation,'readout_parameter_sha256':expected})
    assert runtime_configuration_valid(tmp_path)
    generation['vllm_version']='0.10.2'
    write_json(path,{'generation':generation,'readout_parameter_sha256':expected})
    assert not runtime_configuration_valid(tmp_path)


def test_disabled_diagnostics_do_not_switch_engine(tmp_path):
    from klrd_rs.e2.vllm_gates import validate_vllm_sampler
    def forbidden():raise AssertionError('diagnostics must remain disabled')
    sampler=SimpleNamespace(diagnostic_mode=forbidden)
    assert validate_vllm_sampler(sampler,[],[],None,tmp_path)['status']=='NOT_RUN'


def test_rotation_graph_replay_reads_updated_owner_buffers(monkeypatch):
    from vllm.model_executor.layers.layernorm import RMSNorm
    import vllm.model_executor.custom_op as custom_op
    from klrd_rs.e2.vllm_model import RotationFinalRMSNorm,check_rotation_graph
    dispatch=SimpleNamespace(mode=0,backend='eager',custom_ops=['all'],
        enabled_custom_ops=set(),disabled_custom_ops=set())
    monkeypatch.setattr(custom_op,'get_cached_compilation_config',lambda:dispatch)
    norm=RotationFinalRMSNorm.from_native(RMSNorm(64,dtype=torch.float32).cuda(),2)
    norm.configure(torch.eye(64,device='cuda')[:,:2],torch.eye(64,device='cuda')[:,2:4],torch.zeros(2))
    model=SimpleNamespace(model=SimpleNamespace(norm=norm),_klrd_trace_enabled=False)
    before=[norm.P.data_ptr(),norm.Q.data_ptr(),norm.angles.data_ptr()]
    original=norm.angles.clone()
    result=check_rotation_graph(model,np.array([.1,.2]),np.array([-.2,.3]))
    assert result['status']=='PASS' and result['owner_outputs_differ']
    assert result['buffer_addresses_preserved'] and result['A_B_A_replay_matches_eager']
    assert before==[norm.P.data_ptr(),norm.Q.data_ptr(),norm.angles.data_ptr()]
    torch.testing.assert_close(norm.angles,original,rtol=0,atol=0)



@pytest.mark.parametrize('eager',[False,True])
def test_runtime_options_parse_with_pinned_upstream_engineargs(eager):
    # Construct configuration only; does not resolve/load the fake model.
    from vllm.engine.arg_utils import EngineArgs
    from klrd_rs.e2.vllm_runtime import engine_options
    options=engine_options({'model':'migration-api-contract','revision':'pinned','generation_seed':7},eager=eager)
    parsed=EngineArgs(**options)
    assert parsed.compilation_config.mode.value==0
    assert parsed.compilation_config.cudagraph_mode.name==('NONE' if eager else 'FULL_DECODE_ONLY')
    assert parsed.attention_config.backend.name=='TRITON_ATTN'
    assert parsed.logprobs_mode=='processed_logprobs'


def test_worker_pins_triton_ieee_before_importing_vllm():
    from klrd_rs.e2.vllm_runtime import runtime_identity
    assert runtime_identity()['triton_f32_default']=='ieee'
    source=(Path(__file__).resolve().parents[1]/'klrd_rs/e2/vllm_backend.py').read_text()
    assert source.index("os.environ['TRITON_F32_DEFAULT']='ieee'")<source.index('from vllm import LLM')
    assert "triton.knobs.language.fp32_default='ieee'" in source
    assert source.index("os.environ['VLLM_FLOAT32_MATMUL_PRECISION']='highest'")<source.index('from vllm import LLM')


def test_start_worker_rejects_changed_readout_and_cleans_up(monkeypatch):
    import klrd_rs.e2.vllm_backend as backend
    sampler=backend.VLLMExactRotationTeacherSampler.__new__(backend.VLLMExactRotationTeacherSampler)
    sampler._settings={};sampler._geometry=([],[])
    sampler.environment={'readout_hashes':['original']}
    sampler.tokenizer=SimpleNamespace(chat_template='template',get_vocab=lambda:{'a':0})
    events=[]
    child=SimpleNamespace(close=lambda:events.append('child_closed'))
    process=SimpleNamespace(start=lambda:events.append('started'))
    context=SimpleNamespace(Pipe=lambda:(object(),child),Process=lambda **kwargs:process)
    monkeypatch.setattr(backend.mp,'get_context',lambda method:context)
    sampler._receive=lambda timeout:{'tokenizer_vocab_sha256':backend.digest({'a':0}),
        'chat_template':'template','readout_hashes':['changed']}
    sampler._stop_worker=lambda:events.append('stopped')
    with pytest.raises(RuntimeError,match='model changed'):sampler._start_worker(True)
    assert events[-1]=='stopped' and events[0]=='started'


def test_unstoppable_old_worker_prevents_second_engine():
    from klrd_rs.e2.vllm_backend import VLLMExactRotationTeacherSampler
    sampler=VLLMExactRotationTeacherSampler.__new__(VLLMExactRotationTeacherSampler)
    events=[];sampler._worker_eager=False
    sampler.connection=SimpleNamespace(send=lambda value:events.append('close_sent'))
    sampler.process=SimpleNamespace(pid=1,is_alive=lambda:True,join=lambda **kwargs:None,
        terminate=lambda:events.append('terminated'))
    sampler._start_worker=lambda eager:events.append('started')
    with pytest.raises(RuntimeError,match='refuse a second resident engine'):sampler._ensure_worker(True)
    assert events==['close_sent','terminated']
