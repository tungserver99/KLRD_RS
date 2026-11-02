"""Prepared for server execution only. Intentionally not run by Codex locally."""
from dataclasses import replace
import math
import pytest
import torch


def test_precision_guardrails_derive_from_fp32_unit_roundoff():
    from klrd_rs.e2.numerical_policy import NumericalPolicySpec
    spec=NumericalPolicySpec();caps=spec.absolute_caps()
    assert spec.unit_roundoff==2**-24
    assert caps['total_variation']==math.sqrt(2**-24)
    assert caps['KL_left_to_right']==2*2**-24
    assert spec.conditional_path_tv_bound(512)==math.sqrt(512*2**-24)


def test_distribution_metrics_cover_both_directions_significant_tokens_and_tails():
    from klrd_rs.e2.numerical_policy import distribution_metrics
    a=torch.log(torch.tensor([.8,.19,.01],dtype=torch.float64))
    b=torch.log(torch.tensor([.79,.20,.01],dtype=torch.float64))
    row=distribution_metrics(a,b)
    assert row['total_variation']==pytest.approx(.01)
    assert row['KL_left_to_right']>0 and row['KL_right_to_left']>0
    assert row['significant_token_count']>=1
    assert row['significant_token_max_abs_logprob_error']>0
    assert row['significant_token_max_abs_probability_error']<=row['total_variation']+1e-15


def test_relative_token_error_remains_diagnostic_with_probability_budget():
    from klrd_rs.e2.numerical_policy import NumericalPolicySpec,distribution_metrics,fit_native_envelope,assess
    spec=NumericalPolicySpec()
    p=torch.full((1000,),.998/999,dtype=torch.float64);p[0]=.002
    q=p.clone();q[0]*=math.exp(.005);q[1]-=q[0]-p[0]
    row=distribution_metrics(p.log(),q.log())
    assert row['significant_token_max_abs_logprob_error']>4*math.sqrt(spec.unit_roundoff)
    assert 'significant_token_max_abs_logprob_error' not in spec.absolute_caps()
    assert row['significant_token_max_abs_probability_error']<=row['total_variation']+1e-15
    assert row['significant_worst_logprob_probability_left']>0
    envelope=fit_native_envelope([dict(row,mode='native_original')],spec)
    assert assess(row,envelope,spec)['accepted']


def test_revised_policy_excludes_previous_declared_roles(tmp_path):
    from klrd_rs.e2.numerical_policy import NumericalPolicySpec,select_contexts,historical_policy_contexts
    from klrd_rs.io import write_json
    spec=NumericalPolicySpec()
    previous={'calibration_prompt_ids':['10','11'],'holdout_prompt_ids':['12','13']}
    write_json(tmp_path/'experiment2_rs/run_old/tests/numerical_policy_spec.json',
        {'spec':{'version':'older'},'code_sha256':{},'selection':previous})
    excluded=historical_policy_contexts(tmp_path,spec,{},tmp_path/'experiment2_rs/run_new')
    assert set(excluded)=={'10','11','12','13'}
    requests=[{'prompt_id':str(i),'prompt_ids':[1]*(16+i)} for i in range(120)]
    split=select_contexts(requests,spec,excluded)
    assert not set(excluded)&set(split['calibration_prompt_ids']+split['holdout_prompt_ids'])


def test_rare_tail_error_is_reported_and_does_not_replace_strict_same_context_gate():
    from klrd_rs.e2.numerical_policy import distribution_metrics
    a=torch.tensor([0.,-100.,-110.],dtype=torch.float64)
    b=torch.tensor([0.,-100.2,-110.],dtype=torch.float64)
    row=distribution_metrics(a,b)
    assert row['max_abs_logprob_error']==pytest.approx(.2)
    assert row['total_variation']<1e-40
    assert row['significant_token_max_abs_logprob_error']==pytest.approx(0.)


def test_policy_split_is_deterministic_disjoint_and_excludes_historical_contexts():
    from klrd_rs.e2.numerical_policy import select_contexts,NumericalPolicySpec
    requests=[{'prompt_id':str(i),'prompt_ids':[1]*(16+i),'seed':i} for i in range(100)]
    result=select_contexts(requests,NumericalPolicySpec())
    again=select_contexts(requests,NumericalPolicySpec())
    assert result==again
    calibration=set(result['calibration_prompt_ids']);holdout=set(result['holdout_prompt_ids'])
    assert len(calibration)==32 and len(holdout)==32
    assert not calibration&holdout
    assert not (calibration|holdout)&{str(i) for i in range(10)}


def test_native_baseline_cannot_learn_away_a_material_error():
    from klrd_rs.e2.numerical_policy import NumericalPolicySpec,fit_native_envelope
    spec=NumericalPolicySpec()
    bad={key:value*100 for key,value in spec.absolute_caps().items()}
    with pytest.raises(ValueError,match='absolute'):
        fit_native_envelope([dict(bad,mode='native_original')],spec)
    with pytest.raises(ValueError,match='native'):
        fit_native_envelope([dict(bad,mode='keyed')],spec)


def test_independent_holdout_failure_never_expands_native_envelope():
    from klrd_rs.e2.numerical_policy import NumericalPolicySpec,fit_native_envelope,assess
    spec=NumericalPolicySpec()
    native={key:value/100 for key,value in spec.absolute_caps().items()}
    envelope=fit_native_envelope([dict(native,mode='native_original')],spec)
    before=dict(envelope)
    assert assess(native,envelope,spec)['accepted']
    bad=dict(native,total_variation=spec.absolute_caps()['total_variation']*.9)
    assert not assess(bad,envelope,spec)['accepted']
    assert envelope==before


def test_wrong_temperature_or_token_permutation_fails_distribution_policy():
    from klrd_rs.e2.numerical_policy import NumericalPolicySpec,fit_native_envelope,distribution_metrics,assess
    spec=NumericalPolicySpec()
    clean={key:0. for key in spec.absolute_caps()}
    envelope=fit_native_envelope([dict(clean,mode='native_original')],spec)
    logits=torch.tensor([2.,1.,0.],dtype=torch.float64)
    lp=torch.log_softmax(logits/.8,-1)
    for other in [torch.log_softmax(logits/.4,-1),lp.flip(0)]:
        assert not assess(distribution_metrics(lp,other),envelope,spec)['accepted']


def test_missing_nonfinite_or_unnormalized_distributions_fail_closed():
    from klrd_rs.e2.numerical_policy import NumericalPolicySpec,distribution_metrics,assess
    spec=NumericalPolicySpec();caps=spec.absolute_caps()
    assert not assess({},caps,spec)['accepted']
    for value in [torch.tensor([float('nan'),0.]),torch.tensor([0.,0.])]:
        with pytest.raises(ValueError):distribution_metrics(value,value)


def test_frozen_calibration_refuses_changed_binding_or_overwrite(tmp_path):
    from klrd_rs.e2.numerical_policy import freeze_calibration,read_calibration
    path=tmp_path/'calibration.json';binding={'policy':'v2','engine':'pinned','ids':['a']}
    freeze_calibration(path,binding,{'batch':{'total_variation':1e-5}})
    assert read_calibration(path,binding)['envelopes']['batch']['total_variation']==1e-5
    with pytest.raises(ValueError):read_calibration(path,dict(binding,engine='changed'))
    with pytest.raises(ValueError):freeze_calibration(path,binding,{'batch':{}})


def test_coverage_requires_every_prompt_prefix_key_and_reference_path():
    from klrd_rs.e2.native_fp32_validation import validate_coverage
    from klrd_rs.e2.numerical_policy import NumericalPolicySpec
    spec=NumericalPolicySpec();rows={'batch':[],'reference':[]}
    for step in range(3):
        base={'prompt_id':'a','prefix_step':step,'prefix_sha256':str(step)*64,
            'mode':'native_original','role':'calibration'}
        rows['batch'].append(dict(base))
        for comparison in ['single_vs_HF','batch_vs_HF']:
            rows['reference'].append(dict(base,comparison=comparison))
    validate_coverage(rows,['a'],spec,'calibration','native_original')
    rows['batch'].pop()
    with pytest.raises(ValueError,match='coverage'):validate_coverage(rows,['a'],spec,'calibration','native_original')


def test_batch_size_one_cannot_degenerate_the_native_batch_control(tmp_path):
    from klrd_rs.e2.native_fp32_validation import prepare_policy
    from klrd_rs.e2.config import E2Config
    with pytest.raises(RuntimeError,match='batch_size=8'):
        prepare_policy([],None,replace(E2Config(),batch_size=1),tmp_path)


def test_strict_rmsnorm_p99_remains_an_independent_veto():
    import numpy as np
    from klrd_rs.e2.vllm_policy_gates import strict_reports
    from klrd_rs.e2.config import E2Config
    zero=[np.zeros((1,3))]
    values={'tail':zero,'raw':zero,'kv':zero,'norm':[np.full((1,16),1e-4)],
        'normalization':[0.],'sampled':[0.],'minimum_kl':0.,'details':[]}
    result,_=strict_reports(values,E2Config())
    assert result['final_norm_max_abs_error']<E2Config().parity_max
    assert result['final_norm_p99_abs_error']>E2Config().parity_p99
    assert result['status']=='SAMPLER_INVALID'
