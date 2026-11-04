"""SERVER ONLY source contracts; delivered without local execution.

These checks inspect source and do not synthesize experiment measurements or
import model/API modules. Run only with the separately authorized server checks.
"""
import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(name):
    return (ROOT/'klrd_rs/e3full'/name).read_text(encoding='utf-8')


def _constant(name, target):
    tree = ast.parse(_source(name))
    return next(node.value for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(item, ast.Name) and item.id == target for item in node.targets))


def test_paraphrase_uses_all_three_fresh_1024_conditions():
    source = _source('paraphrase.py')
    assert ast.literal_eval(_constant('paraphrase.py', 'MODELS')) == ('clean', 'teacher', 'student')
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'load_observations']
    assert len(calls) == 1 and isinstance(calls[0].args[1], ast.Constant) and calls[0].args[1].value == 1024
    assert 'load_pairs' not in source and 'load_vectors' not in source
    assert 'NO_ORIGINAL_STUDENT_SIGNAL' not in source
    assert 'historical_paraphrase_texts_reused=0' in source
    assert 'responses=1200' in source and 'responses_per_model=400' in source


def test_copied_transport_has_no_new_caps_or_payload_policies():
    source = _source('paraphrase.py'); tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == 'run_copied_transport']
    assert len(calls) == 1
    assert {item.arg for item in calls[0].keywords} == {'prompt', 'workers', 'max_attempts'}
    assert 'prepare_condition_resume' in source and 'read_condition' in source
    assert "namespace = out/'paraphrase'/model" in source
    assert "resumes[model]['kept'] == 400" in source
    assert "protocol['model'] != 'gpt-5-mini-2025-08-07'" in source
    assert "protocol['prompt'] != PARAPHRASE_PROMPT" in source
    assert "protocol.get('protocol_variant') != 'high_fidelity_v1'" in source


def test_quality_retains_provider_incomplete_status_and_usage():
    source = _source('paraphrase.py')
    assert 'incomplete_response_status_count' in source
    assert "api_usage=_sum_usage" in source
    assert 'All400 retained' in source and 'PENDING_HUMAN_REVIEW' in source
    assert 'cosine_mean_original' in source and 'cosine_median_paraphrased' in source
    assert 'original_clean_FPR' in source and 'wrong_worst_paraphrased' in source


def test_report_reads_evidence_without_inference_or_api_imports():
    tree = ast.parse(_source('report.py'))
    imports = [node.module or '' for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    assert not any(any(token in module for token in
        ('transformers', 'openai', 'generation', 'train', 'paraphrase', 'evaluate')) for module in imports)
    source = _source('report.py')
    assert 'scientific_success=None' in source
    assert 'SCORED_PHASE_INCOMPLETE' in source
    assert 'models=condition_status' in source and 'models=arc_models' in source
    assert 'effective_training_counts' in source


def test_report_provides_six_tables_and_null_scope_caveats():
    tables = ast.literal_eval(_constant('report.py', 'TABLE_NAMES'))
    assert tables == ('training_pilot_vs_full512', 'fingerprint1024', 'paraphrase1024',
                      'arc_c_utility', 'generation_utility1024', 'same_checkpoints512_vs1024')
    source = _source('report.py')
    assert 'analytical spherical random-key null' in source
    assert 'not empirical calibration' in source
    assert 'Q1000 is one descriptive aggregate' in source
    assert 'Q20/Q50/Q100 share nested prompt populations' in source
    assert 'historicalBF16' in source


def test_training_and_length_tables_include_saved_runtime_evidence():
    source = _source('report.py')
    for name in ('selected_training_teacher_KL_nats_per_token', 'selected_global_step',
                 'elapsed_training_seconds', 'generation_seconds512', 'generation_seconds1024',
                 'native_generated_tokens_per_second512', 'native_generated_tokens_per_second1024',
                 'timing_scope512', 'timing_scope1024', 'timing_metadata_status512',
                 'timing_metadata_status1024'):
        assert source.count(name) >= 2  # Producer field and main table declaration.
    assert 'unavailable: generation timing was not recorded for all responses' in source


def test_public_package_uses_named_tables_and_excludes_private_files():
    source = _source('report.py')
    constants = ast.parse(source)
    for node in constants.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [item.id for item in node.targets if isinstance(item, ast.Name)]
        if not set(names) & {'ROOT_PUBLIC', 'PHASE_PUBLIC', 'PARAPHRASE_PUBLIC'}:
            continue
        strings = [item.value for item in ast.walk(node.value)
                   if isinstance(item, ast.Constant) and isinstance(item.value, str)]
        assert not any(item.startswith('private/') or '/private/' in item
                       or '.safetensors' in item or 'api_results' in item for item in strings)
    assert "(base/relative).resolve()" in source and 'base not in candidate.parents' in source
    assert "archive.testzip()" in source
    assert "set(archive.namelist()) != set(files)" in source
