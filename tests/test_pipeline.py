from dataclasses import replace
import numpy as np
import pytest
import torch


def test_math_checks_and_reports_under_synthetic_pipeline(tmp_path):
    from klrd_rs.config import Config
    from klrd_rs.pipeline import run_experiment
    from klrd_rs.toy import toy_inputs, ToyAdapter
    cfg = replace(Config(), m=3, pca_dim=12, calibration_keys=8, validation_keys=12,
        fisher_context_sample_cap=600, covariance_contexts=120, strength_contexts=64,
        local_contexts=80, diagnostic_random_keys=20000, bootstrap_replicates=100,
        readout_chunk=8, query_budgets_diagnostic=(20,100,1000), finite_difference_h=.0001,
        finite_difference_atol=.001, mean_zero_atol=1e-8)
    result = run_experiment(cfg, toy_inputs(), ToyAdapter(), tmp_path, synthetic=True)
    assert result['status'] == 'SYNTHETIC_CHECKS_PASSED'
    assert result['scientific_success'] is False
    assert result['math_checks_passed'] is True
    from klrd_rs.io import read_json
    assert read_json(tmp_path/'null/fixed_key_pilot.json')['populations']['clean_paraphrase']['status'] == 'MISSING'
    for path in ['provenance/resolved_config.json','geometry/basis.npz','geometry/fisher.npz',
        'geometry/whitening.npz','geometry/geometry_checks.json','keys/calibration_key_ids.json',
        'keys/validation_key_ids.json','null/sphere_math_tests.json','null/clean_original_key_pvalues.csv',
        'null/clean_paraphrase_key_pvalues.csv','teacher/global_lambda.json',
        'teacher/local_capacity_by_key.csv','teacher/KL_by_key.csv','report/summary.json',
        'report/experiment1_report.md']:
        assert (tmp_path/path).is_file(), path
    result2 = run_experiment(cfg,toy_inputs(),ToyAdapter(),tmp_path,synthetic=True)
    assert result2 == result
    strength_file=tmp_path/'teacher/global_lambda.json'
    frozen=read_json(strength_file);frozen['lambda']*=2
    from klrd_rs.io import write_json
    write_json(strength_file,frozen)
    with pytest.raises(ValueError,match='strength|checksum'):
        run_experiment(cfg,toy_inputs(),ToyAdapter(),tmp_path,synthetic=True)
    with pytest.raises(ValueError, match='changed|incompatible'):
        run_experiment(replace(cfg, strength_contexts=32),toy_inputs(),ToyAdapter(),tmp_path,synthetic=True)


def test_real_run_cannot_be_promoted_with_missing_diagnostics():
    from klrd_rs.pipeline import decision
    assert decision(True, True, 1., False, False) == 'INCONCLUSIVE'
    assert decision(True, True, 1., True, False) == 'VALID_IMPLEMENTATION'
    assert decision(False, True, 1., True, False) == 'NO_GO'
    assert decision(True, True, .5, True, False) == 'NO_GO'


def test_bundle_is_real_matched_data_and_portable():
    from pathlib import Path
    from klrd_rs.inputs import load_inputs
    from klrd_rs.config import Config
    root = Path(__file__).resolve().parents[1]/'inputs'
    if not root.exists(): pytest.skip('Portable historical inputs not installed')
    x=load_inputs(root,Config())
    assert len(x['calibration'])==2000 and len(x['original'])==2500 and len(x['paraphrase'])==2500
    assert x['basis'].shape==(2048,128)
