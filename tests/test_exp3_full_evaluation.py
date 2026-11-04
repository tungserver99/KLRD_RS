"""Source-only regression cases; run only on the authorized experiment server."""
import numpy as np
import pytest

from klrd_rs.e3full import evaluation


def test_sampling_rows_reuse_frozen_per_response_seeds():
    prompts=[{'prompt_id':'a','prompt':'hello'}]
    source={'prompts':prompts,'observations':{'clean':([{'prompt_id':'a','seed':42}],None,None)}}
    rows=evaluation._sampling_rows(source,'clean')
    assert rows==[dict(prompts[0],sampling_seed=42)]
    assert 'sampling_seed' not in prompts[0]


def test_population_rejects_changed_text_order_and_empty_vectors():
    source={'ids':['a'],'prompts':[{'prompt_id':'a','prompt':'original'}]}
    valid=([{'prompt_id':'a','prompt':'original','response':'answer','response_id':'r'}],
           np.zeros((1,32)),np.array([3]))
    evaluation._validate_population(source,{'student':valid})
    changed=([dict(valid[0][0],prompt='different')],valid[1],valid[2])
    with pytest.raises(ValueError,match='prompt text'):
        evaluation._validate_population(source,{'student':changed})
    with pytest.raises(ValueError,match='shape'):
        evaluation._validate_population(source,{'student':(valid[0],np.zeros((0,32)),valid[2])})


def test_utility_uses_native_tokens_not_verifier_eligible_count():
    row={'prompt_id':'a','response':'hello','generated_token_ids':[8,9,2],
         'native_visible_tokens':2,'eos_terminated':True,'max_length_truncated':False,
         'roundtrip_equal':True,'amortized_generation_seconds':.5}
    result=evaluation.generation_utility({'student':([row],np.zeros((1,32)),np.array([1]))},1024)['student']
    assert result['mean_generated_tokens']==3
    assert result['mean_visible_tokens']==2
    assert result['median_generated_tokens']==3
    assert result['eos_termination_rate']==1
    assert result['max_length_truncation_rate']==0
    assert result['native_generated_tokens_per_second']==6
    assert result['text_token_roundtrip_consistency_rate']==1


def test_utility_does_not_invent_throughput_for_unrecorded_runtime():
    row={'response':'x','generated_token_ids':[1],'native_visible_tokens':1,
         'eos_terminated':False,'max_length_truncated':True,'roundtrip_equal':False}
    result=evaluation.generation_utility({'student':([row],np.zeros((1,32)),np.array([1]))},512)['student']
    assert result['generation_seconds'] is None
    assert result['native_generated_tokens_per_second'] is None
    assert result['max_length_truncation_rate']==1


def test_utility_subtracts_old_posthoc_KL_audit_time():
    row={'response':'x','generated_token_ids':[1,2],'native_visible_tokens':1,
         'eos_terminated':True,'max_length_truncated':False,'roundtrip_equal':True,
         'amortized_generation_seconds':3.,'KL_audit_seconds':2.,'KL_samples':[{'KL':.1}]}
    observation={'teacher':([row],np.zeros((1,32)),np.array([1]))}
    result=evaluation.generation_utility(observation,512)['teacher']
    assert result['generation_seconds']==1
    assert result['native_generated_tokens_per_second']==2
    assert result['KL_audit_subtracted_response_count']==1
    row.pop('KL_audit_seconds')
    assert evaluation.generation_utility(observation,512)['teacher']['generation_seconds'] is None


def test_observation_serialization_preserves_labels_and_native_metadata(tmp_path):
    rows=[{'prompt_id':'a','prompt':'q','response_id':'r','response':'a',
           'generated_token_ids':[3,2],'native_visible_tokens':1}]
    observations={'student':(rows,np.ones((1,32)),np.array([1],dtype=np.int64))}
    state={'out':tmp_path,'identity':'run'}
    evaluation._save_observations(state,observations)
    restored=evaluation._read_observations(state,['student'])
    assert restored['student'][0]==rows
    assert np.array_equal(restored['student'][1],observations['student'][1])
    assert restored['student'][2].dtype==np.int64


def test_evaluation_products_exclude_shared_paraphrase_runtime_scratch(tmp_path):
    provenance=tmp_path/'provenance';provenance.mkdir()
    for name in ('student_verification_runtime.json','verification_environment_comparison.json'):
        (provenance/name).write_text('{}',encoding='utf-8')
    products=evaluation._products({'out':tmp_path})
    assert products==['provenance/student_verification_runtime.json']


def test_invalid_length_fails_before_loading_a_model():
    with pytest.raises(ValueError,match='512 or 1024'):
        evaluation.run({},768)


def test_pilot_config_must_match_sampling_without_changing_train_size():
    from klrd_rs.e3.config import E3Config
    from dataclasses import replace
    cfg=E3Config(profile='full')
    evaluation._matching_pilot_config(cfg,replace(cfg,profile='pilot'))
    with pytest.raises(ValueError,match='sampling'):
        evaluation._matching_pilot_config(cfg,replace(cfg,profile='pilot',seed=7))


@pytest.mark.parametrize('installed',['0.32.0','0.33.0'])
def test_exact_teacher_version_allowlist_records_actual_version(monkeypatch,installed):
    from klrd_rs.e3full import teacher
    monkeypatch.setattr(teacher,'version',lambda name:installed)
    assert teacher.require_supported_vllm()==installed


def test_exact_teacher_rejects_unreviewed_version(monkeypatch):
    from klrd_rs.e3full import teacher
    monkeypatch.setattr(teacher,'version',lambda name:'0.34.0')
    with pytest.raises(RuntimeError,match='0.32.0 or 0.33.0'):
        teacher.require_supported_vllm()


def test_exact_teacher_resume_rejects_changed_frozen_seed():
    from klrd_rs.e3full import teacher
    from klrd_rs.io import digest
    identity={'run':'x'}
    prompt={'prompt_id':'a','prompt':'question','sampling_seed':17}
    row=dict(prompt,response='answer',seed=17,generation_identity=digest(identity))
    row['record_sha256']=digest(row)
    teacher._validate([row],[prompt],identity)
    with pytest.raises(ValueError,match='seed'):
        teacher._validate([row],[dict(prompt,sampling_seed=18)],identity)
