"""Saved-evidence-only full experiment report and private-safe review ZIP.

Reporting never imports generation, training, model loaders or API clients.
Incomplete and invalid evidence remains explicit; no scientific pass is inferred.
"""
from pathlib import Path
import csv
import zipfile

from ..io import read_json, sha256, write_csv, write_json
from .common import checked, phase_state


SCORE_TABLES = ('per_key_group.csv', 'per_response.csv', 'closed_set_identification.csv',
                'clean_wrong_controls.csv', 'group_summary.csv', 'cosine_summary.csv')
TABLE_NAMES = ('training_pilot_vs_full512', 'fingerprint1024', 'paraphrase1024',
               'arc_c_utility', 'generation_utility1024', 'same_checkpoints512_vs1024')
ROOT_PUBLIC = ('provenance/resolved_config.json', 'provenance/environment.json',
    'provenance/input_manifest.json', 'provenance/current_source_manifest.json',
    'provenance/full_protocol.json', 'provenance/pilot_clean_reuse.json',
    'provenance/historical_heldout_audit.json', 'provenance/merged_manifest.json',
    'training/selection.json', 'training/baseline.json', 'training/epochs.json',
    'training/validation.jsonl', 'training/steps.jsonl', 'training/eligibility.json',
    'arc_c/complete.json', 'arc_c/comparison.csv', 'arc_c/protocol.json',
    'arc_c/request_binding.json', 'arc_c/summary.json',
    'report/summary.json', 'report/full_experiment_report.md')
PHASE_PUBLIC = ('provenance/evaluation_complete.json', 'provenance/evaluation_protocol.json',
    'utility/generation_summary.json', 'utility/generation_summary.csv') + tuple(
        'scores/' + name for name in SCORE_TABLES)
PARAPHRASE_PUBLIC = ('paraphrase/complete.json', 'paraphrase/protocol.json', 'paraphrase/manifest.json',
    'paraphrase/api_execution.json', 'paraphrase/metrics.csv', 'paraphrase/quality.csv',
    'paraphrase/quality_summary.json') + tuple('paraphrase/scores/' + name
        for name in SCORE_TABLES if name != 'cosine_summary.csv') + tuple(
        'paraphrase/' + model + '/complete.json' for model in ('clean', 'teacher', 'student'))


def _rows(path):
    if not Path(path).is_file():
        return []
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _json(path, default=None):
    return read_json(path) if Path(path).is_file() else default


def _text(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(value, encoding='utf-8'); temporary.replace(path)


def _write_table(out, name, rows, fields):
    # Empty tables still have an explicit schema rather than invented zero rows.
    columns = list(dict.fromkeys(fields + [key for row in rows for key in row]))
    write_csv(out/'report/tables'/(name + '.csv'), rows, columns)


def _receipt(state, relative, errors, label):
    try:
        result = checked(state, relative)
        return result, 'COMPLETE' if result is not None else 'MISSING'
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(label + ': ' + str(error))
        return None, 'INVALID'


def _training(state, errors, label):
    """Inspect committed selection and export evidence, without weight loading."""
    result = dict(status='MISSING', selection=None, effective_counts=None)
    try:
        selection = _json(state['out']/'training/selection.json')
        if selection is None:
            return result
        if selection.get('identity') != state['identity']:
            raise ValueError('Training selection identity mismatch')
        merged = read_json(state['out']/'provenance/merged_manifest.json')
        if merged.get('identity') != state['identity']:
            raise ValueError('Merged checkpoint identity mismatch')
        if selection.get('selection_rule') != 'minimum_validation_teacher_forward_KL':
            raise ValueError('Checkpoint selection is not the frozen validation-KL rule')
        for directory, files in ((state['out']/'checkpoints/best_adapter', selection['adapter_files']),
                                 (state['out']/'merged_student', merged['files'])):
            if not files:
                raise ValueError('Empty selected/export file manifest')
            for name, expected in files.items():
                target = (directory/name).resolve()
                if directory.resolve() not in target.parents or not target.is_file():
                    raise ValueError('Missing/outside selected checkpoint file: ' + name)
                if target.suffix == '.json' and sha256(target) != expected:
                    raise ValueError('Selected checkpoint JSON content changed: ' + name)
        if 'config.json' not in merged['files']:
            raise ValueError('Merged checkpoint config is missing')
        eligibility = _json(state['out']/'training/eligibility.json', {})
        result.update(status='COMPLETE', selection=selection,
            effective_counts={key: eligibility.get(key) for key in
                ('eligible_sequences', 'eligible_tokens', 'excluded_sequence_count')})
        if 'excluded' in eligibility:
            result['effective_counts']['excluded_sequence_count'] = len(eligibility['excluded'])
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append(label + ': ' + str(error)); result['status'] = 'INVALID'
    return result


def _phase(state, length, errors):
    result = dict(state=None, receipt=None, status='MISSING', rows=[], utility={}, cosine=[])
    try:
        phase = phase_state(state, length)
        result['state'] = phase
        receipt, status = _receipt(phase, 'provenance/evaluation_complete.json', errors, 'eval' + str(length))
        result.update(receipt=receipt, status=status)
        if receipt is not None:
            labels = ['clean', 'teacher', 'student'] + (['pilot512'] if length == 512 else [])
            if (receipt.get('status') != 'EVAL_COMPLETE' or receipt.get('max_new_tokens') != length
                    or receipt.get('responses') != 1000 or receipt.get('labels') != labels):
                raise ValueError('Evaluation length, condition or population receipt mismatch')
            result.update(rows=receipt['summary'],
                utility=read_json(phase['out']/'utility/generation_summary.json'),
                cosine=_rows(phase['out']/'scores/cosine_summary.csv'))
        elif status == 'MISSING' and (phase['out']/'private').exists():
            result['status'] = 'PARTIAL'
    except FileNotFoundError as error:
        result.update(receipt=None, status='MISSING', rows=[], utility={}, cosine=[])
        # An absent plan is expected in a report of an unstarted phase. Missing
        # products after a completion receipt are caught by _receipt as INVALID.
        if result['state'] is not None:
            errors.append('eval' + str(length) + ': ' + str(error)); result['status'] = 'INVALID'
    except (OSError, ValueError, KeyError, TypeError) as error:
        errors.append('eval' + str(length) + ': ' + str(error))
        result.update(receipt=None, status='INVALID', rows=[], utility={}, cosine=[])
    return result


def _model_status(phase, model):
    if phase['receipt'] is not None:
        return dict(status='COMPLETE', responses=1000)
    state = phase['state']
    if state is None:
        return dict(status=phase['status'], responses=0)
    score_path = state['out']/('provenance/' + model + '_scores_complete.json')
    if score_path.is_file():
        try:
            receipt = read_json(score_path)
            if receipt['identity'] != state['identity']:
                raise ValueError('Partial score identity changed')
            for name, expected in receipt['batch_sha256'].items():
                path = state['out']/'private/eval_scores'/model/(name + '.npz')
                if state['out'].resolve() not in path.resolve().parents or sha256(path) != expected:
                    raise ValueError('Partial score batch changed')
            return dict(status='SCORED_PHASE_INCOMPLETE', responses=receipt['count'])
        except (OSError, ValueError, KeyError, TypeError):
            return dict(status='INVALID', responses=None)
    started = (state['out']/('private/' + model + '_responses.jsonl')).exists()
    return dict(status='PARTIAL' if started else 'MISSING', responses=None if started else 0)


def _find(rows, model, q, alpha):
    return next((row for row in rows if row['model'] == model and int(row['Q']) == int(q)
                 and float(row['alpha']) == float(alpha)), None)


def _teacher_kl(training):
    selection = training.get('selection')
    return None if selection is None else selection['selected']['teacher_kl']['mean']


def _training_comparison(pilot, full, phase):
    rows = []
    for model, label, size, validation_size, evidence in (
            ('pilot512', 'Pilot LoRA 2k', 2000, 200, pilot),
            ('student', 'Full LoRA 10k', 10000, 1000, full)):
        utility = phase['utility'].get(model, {})
        response_cos = next((row for row in phase['cosine'] if row['model'] == model), {})
        selected = (evidence['selection'] or {}).get('selected', {})
        for metric in (row for row in phase['rows'] if row['model'] == model):
            clean = _find(phase['rows'], 'clean', metric['Q'], metric['alpha'])
            rows.append(dict(model=label, train_sequences=size, validation_sequences=validation_size,
                max_new_tokens=512, validation_KL_nats_per_token=_teacher_kl(evidence),
                selected_epoch=selected.get('epoch'),
                selected_training_teacher_KL_nats_per_token=selected.get('training_teacher_kl', {}).get('mean'),
                selected_global_step=selected.get('global_step'),
                elapsed_training_seconds=selected.get('elapsed_training_seconds'),
                Q=metric['Q'], alpha=metric['alpha'], groups=metric['groups'],
                detections=metric['detections'], owner_TPR=metric['rate'],
                observed_clean_FPR=clean['rate'] if clean else None,
                correct_key_cosine_median=metric['cosine_median'], wrong_mean=metric['wrong_mean'],
                per_response_cosine_mean=response_cos.get('correct_key_cosine_mean'),
                per_response_cosine_median=response_cos.get('correct_key_cosine_median'),
                wrong_worst=metric['wrong_worst'], CI95_low=metric['CI95_low'], CI95_high=metric['CI95_high'],
                mean_generated_tokens=utility.get('mean_generated_tokens'),
                median_generated_tokens=utility.get('median_generated_tokens'),
                truncation_rate=utility.get('max_length_truncation_rate'),
                eos_rate=utility.get('eos_termination_rate')))
    for row in rows:
        other = next((other for other in rows if other['model'] != row['model']
            and other['Q'] == row['Q'] and other['alpha'] == row['alpha']), None)
        row['full_minus_pilot_TPR_pp'] = (100 * (row['owner_TPR'] - other['owner_TPR'])
            if other and row['model'] == 'Full LoRA 10k' else None)
    return rows


def _fingerprint(phase):
    rows = []
    for metric in phase['rows']:
        clean = _find(phase['rows'], 'clean', metric['Q'], metric['alpha'])
        response_cos = next((r for r in phase['cosine'] if r['model'] == metric['model']), {})
        rows.append(dict(metric, max_new_tokens=1024,
            rate_kind='observed_clean_FPR' if metric['model'] == 'clean' else 'owner_TPR',
            observed_clean_FPR=clean['rate'] if clean else None,
            per_response_cosine_mean=response_cos.get('correct_key_cosine_mean'),
            per_response_cosine_median=response_cos.get('correct_key_cosine_median')))
    return rows


def _length_comparison(short, long):
    rows = []
    for model in ('clean', 'teacher', 'student'):
        for q in (20, 50, 100, 1000):
            for alpha in (.01, .0001):
                a = _find(short['rows'], model, q, alpha)
                b = _find(long['rows'], model, q, alpha)
                if a is None or b is None:
                    continue
                u, v = short['utility'][model], long['utility'][model]
                rows.append(dict(model=model, Q=q, alpha=alpha, groups=a['groups'],
                    rate512=a['rate'], rate1024=b['rate'], delta_rate_pp=100*(b['rate']-a['rate']),
                    cosine_median512=a['cosine_median'], cosine_median1024=b['cosine_median'],
                    truncation_rate512=u['max_length_truncation_rate'],
                    truncation_rate1024=v['max_length_truncation_rate'],
                    mean_generated_tokens512=u['mean_generated_tokens'],
                    mean_generated_tokens1024=v['mean_generated_tokens'],
                    generation_seconds512=u.get('generation_seconds'),
                    generation_seconds1024=v.get('generation_seconds'),
                    native_generated_tokens_per_second512=u.get('native_generated_tokens_per_second'),
                    native_generated_tokens_per_second1024=v.get('native_generated_tokens_per_second'),
                    timed_response_count512=u.get('timed_response_count'),
                    timed_response_count1024=v.get('timed_response_count'),
                    timing_scope512=u.get('timing_scope'), timing_scope1024=v.get('timing_scope'),
                    timing_metadata_status512=('available' if u.get('generation_seconds') is not None
                        else 'unavailable: generation timing was not recorded for all responses'),
                    timing_metadata_status1024=('available' if v.get('generation_seconds') is not None
                        else 'unavailable: generation timing was not recorded for all responses'),
                    comparison='same model checkpoint; same prompt IDs and seed policy; fresh generations'))
    return rows


def _display(value):
    if value is None or value == '':
        return 'unavailable'
    if isinstance(value, float):
        return f'{value:.6g}'
    return str(value).replace('|', '\\|').replace('\n', ' ')


def _markdown_table(title, rows, fields):
    lines = ['', '## ' + title, '',
        '|' + '|'.join(fields) + '|', '|' + '|'.join('---' for _ in fields) + '|']
    for row in rows:
        lines.append('|' + '|'.join(_display(row.get(field)) for field in fields) + '|')
    if not rows:
        lines += ['', 'Unavailable: required saved evidence is incomplete or invalid.']
    return lines


def _discussion(tables):
    lines = ['', '## Evidence and interpretation', '']
    training = next((row for row in tables[TABLE_NAMES[0]] if row['model'] == 'Full LoRA 10k'
        and row['Q'] == 100 and row['alpha'] == .01), None)
    if training and training['full_minus_pilot_TPR_pp'] is not None:
        lines.append('At Q100 and nominal alpha=0.01, full minus pilot owner-key detection at512 '
            f"is {training['full_minus_pilot_TPR_pp']:.3f} percentage points on the shared prompts. "
            'This is the comparison that isolates training size; group results remain descriptive.')
    else:
        lines.append('The matched512 pilot/full comparison is incomplete; no training-size benefit can be concluded.')
    for model in ('teacher', 'student'):
        original = next((r for r in tables['fingerprint1024'] if r['model'] == model
            and r['Q'] == 100 and r['alpha'] == .01), None)
        attack = next((r for r in tables['paraphrase1024'] if r['model'] == model
            and int(r['Q']) == 100 and float(r['alpha']) == .01), None)
        if original:
            lines.append(f"{model}: original1024 Q100 owner detections {original['detections']}/{original['groups']}; "
                f"observed clean FPR {_display(original['observed_clean_FPR'])}. "
                'The teacher and student are reported independently.')
        else:
            lines.append(model + ': original1024 evidence is incomplete; no black-box transfer conclusion is available.')
        if attack:
            lines.append(f"{model}: fixed400 Q100 paraphrase owner detections "
                f"{attack['original_detections']} before and {attack['paraphrased_detections']} after "
                f"across {attack['groups']} groups (drop {_display(attack['drop_pp'])} percentage points). "
                'Q20/Q50 and alpha=1e-4 remain separate rows; successful teacher detection does not establish student robustness.')
        else:
            lines.append(model + ': paraphrase evidence is incomplete; no robustness conclusion is available.')
    for row in tables['arc_c_utility']:
        lines.append(f"ARC-C {row['model']}: {row.get('correct')}/{row.get('total')} correct, "
            f"accuracy {_display(row.get('accuracy'))}, delta versus clean "
            f"{_display(row.get('delta_vs_clean_pp'))} percentage points. "
            'These measurements do not define an equivalence/noninferiority margin or select checkpoints.')
    arc_full = next((row for row in tables['arc_c_utility'] if row['model'] == 'full_student'), None)
    arc_pilot = next((row for row in tables['arc_c_utility'] if row['model'] == 'pilot_student'), None)
    if arc_full and arc_pilot:
        lines.append('Full minus pilot ARC-C accuracy is '
            f"{100*(float(arc_full['accuracy'])-float(arc_pilot['accuracy'])):.3f} percentage points. "
            'Use this together with the matched512 fingerprint result when assessing training-size effects.')
    if not tables['arc_c_utility']:
        lines.append('ARC-C utility is incomplete; accuracy preservation or fingerprint/utility tradeoffs cannot be concluded.')
    lines += [
        'Read the same-checkpoint512/1024 table to assess generation length. Pilot512 versus full1024 confounds training size and length.',
        'Nominal alpha comes from the unchanged analytical spherical random-key null. Observed clean FPR is a separate diagnostic, not empirical calibration or a universal fixed-key guarantee.',
        'Groups are disjoint within each Q; Q20/Q50/Q100 share nested prompt populations. Q1000 is one descriptive aggregate and has no independent-group confidence interval. Wrong keys share responses and are not independent deployments.',
        'One owner key and one training seed support a feasibility study, not multi-key reliability. Lower held-out KL and teacher detectability alone do not establish student transfer.',
        'A completed paraphrase condition retains all400 responses, including nonempty outputs marked incomplete by the API. Lexical, length, numeric and structural diagnostics cannot establish semantic fidelity; human review remains pending.',
        'The ARC-C adapter records the original harness/task/zero-shot protocol and matchedFP32 loading required by the exact teacher. This precision differs from the historicalBF16 loader; numerical equivalence is not asserted. Dataset fingerprints are recorded at runtime; the historical driver did not separately pin a dataset commit.',
        'Example outputs, when available, use ten IDs selected from frozen prompt ordering before scoring; they remain private and are excluded from the review ZIP.'
    ]
    return lines


def _safe_file(base, relative):
    base = Path(base).resolve(); candidate = (base/relative).resolve()
    if base not in candidate.parents:
        raise ValueError('Public allowlist path leaves its root: ' + relative)
    return candidate if candidate.is_file() else None


def package_review(state):
    root, out = state['root'], state['out']; files = {}
    source_names = ['main_exp3_full.py', 'main_exp3.py', 'run_experiment3.sh',
        'run_experiment3_pilot.sh', 'requirements-lora.txt', 'requirements-paraphrase.txt',
        'requirements-exp3-full.txt', 'requirements.txt', 'run_experiment3_full.pbs',
        'README_EXPERIMENT3_FULL.md',
        'docs/ARC_C_SOURCE_REUSE.md',
        'docs/superpowers/reviews/2026-11-04-e3-full-source-review.md',
        'docs/superpowers/plans/2026-11-04-e3-fp32-vllm-throughput.md',
        'tests/test_exp3_vllm_throughput.py',
        'tests/test_exp3_journal_recovery.py',
        'docs/superpowers/plans/2026-11-04-e3-journal-recovery.md',
        'gpt_guide/KLRD_RS_Experiment3_FULL_1024_ARCC_SPEC.md',
        'docs/superpowers/plans/2026-11-03-e3-full-1024-arcc.md']
    for directory in ('klrd_rs', 'klrd_scw', 'klrd_arc'):
        source_names += [path.relative_to(root).as_posix() for path in (root/directory).rglob('*.py')]
    source_names += [path.relative_to(root).as_posix() for path in (root/'tests').glob('test_exp3_full*.py')]
    source_names += [path.relative_to(root).as_posix() for path in (root/'pbs').glob('*exp3_full*') if path.is_file()]
    source_names += ['klrd_arc/copy_receipt.json', 'klrd_arc/SOURCE_RECEIPT.json']
    for name in source_names:
        path = _safe_file(root, name)
        if path is not None:
            files['source/' + name] = path
    public = list(ROOT_PUBLIC)
    public += ['report/tables/' + name + '.csv' for name in TABLE_NAMES]
    for length in (512, 1024):
        public += ['eval' + str(length) + '/' + name for name in PHASE_PUBLIC]
        public += ['eval' + str(length) + '/provenance/evaluation_plan.json']
    public += ['eval1024/' + name for name in PARAPHRASE_PUBLIC]
    for model in ('clean', 'teacher', 'pilot_student', 'full_student'):
        public += ['arc_c/' + model + '/' + name for name in ('complete.json', 'summary.json', 'protocol.json', 'result.json')]
    for name in public:
        path = _safe_file(out, name)
        if path is not None:
            files['run/' + name] = path
    manifest = dict(schema=1,
        policy='Literal public artifact allowlist and source code only; no recursive run-directory inclusion',
        excluded='Weights, adapters, optimizer/checkpoints, secret keys/basis, private prompt/response examples, API journals/receipts, environment credentials',
        included_sha256={name: sha256(path) for name, path in sorted(files.items())})
    write_json(out/'report/review_manifest.json', manifest)
    files['run/report/review_manifest.json'] = out/'report/review_manifest.json'
    destination = out/'KLRD_RS_experiment3_full_review.zip'
    temporary = destination.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(files.items()):
            archive.write(path, name)
    with zipfile.ZipFile(temporary) as archive:
        if len(archive.namelist()) != len(files) or set(archive.namelist()) != set(files) or archive.testzip():
            raise ValueError('Review ZIP membership/integrity mismatch')
    temporary.replace(destination)
    return dict(path=destination.name, sha256=sha256(destination), files=len(files))


def run(state, package=True):
    out = state['out']; errors = []
    full = _training(state, errors, 'full training')
    pilot = _training(state['pilot'], errors, 'pilot training')
    phases = {length: _phase(state, length, errors) for length in (512, 1024)}
    arc, arc_status = _receipt(state, 'arc_c/complete.json', errors, 'ARC-C')
    if arc is not None and arc.get('status') not in ('ARC_C_COMPLETE', 'ARC_COMPLETE', 'COMPLETE'):
        errors.append('ARC-C: unexpected completion status'); arc = None; arc_status = 'INVALID'
    arc_models = {}
    for model in ('clean', 'teacher', 'pilot_student', 'full_student'):
        receipt, status = _receipt(state, 'arc_c/' + model + '/complete.json', errors, 'ARC-C ' + model)
        metadata = _json(out/'arc_c'/model/'summary.json', {}) if receipt else {}
        started = (out/'arc_c'/model).exists()
        arc_models[model] = dict(status='PARTIAL' if status == 'MISSING' and started else status,
            correct=metadata.get('correct_count'), total=metadata.get('total_count'))
    phase1024 = phases[1024]['state']; attack = None; attack_status = 'MISSING'; condition_status = {}
    if phase1024 is not None:
        attack, attack_status = _receipt(phase1024, 'paraphrase/complete.json', errors, 'paraphrase1024')
        if attack is not None and (attack.get('status') != 'PARAPHRASE_COMPLETE'
                or attack.get('responses') != 1200 or attack.get('original_max_new_tokens') != 1024):
            errors.append('Paraphrase receipt population/status mismatch'); attack = None; attack_status = 'INVALID'
        if attack is not None:
            manifest = read_json(phase1024['out']/'paraphrase/manifest.json')
            if (phases[1024]['receipt'] is None or manifest.get('source_evaluation_receipt_sha256') !=
                    sha256(phase1024['out']/'provenance/evaluation_complete.json')):
                errors.append('Paraphrase source1024 evaluation receipt is missing/changed')
                attack = None; attack_status = 'INVALID'
        for model in ('clean', 'teacher', 'student'):
            receipt, status = _receipt(phase1024, 'paraphrase/' + model + '/complete.json', errors,
                                       'paraphrase1024 ' + model)
            if receipt and (receipt.get('status') != 'PARAPHRASE_CONDITION_COMPLETE'
                    or receipt.get('responses') != 400 or receipt.get('model') != model):
                errors.append('Paraphrase condition completion mismatch: ' + model)
                receipt = None; status = 'INVALID'
            started = (phase1024['out']/'paraphrase'/model/'private').exists()
            condition_status[model] = dict(status='PARTIAL' if status == 'MISSING' and started else status,
                responses=receipt.get('responses', 0) if receipt else 0)
    else:
        condition_status = {model: dict(status='MISSING', responses=0) for model in ('clean', 'teacher', 'student')}
    arc_rows = _rows(out/'arc_c/comparison.csv') if arc else []
    attack_rows = _rows(phase1024['out']/'paraphrase/metrics.csv') if attack else []
    tables = dict(zip(TABLE_NAMES, (
        _training_comparison(pilot, full, phases[512]), _fingerprint(phases[1024]), attack_rows,
        arc_rows, [dict(model=model, **value) for model, value in phases[1024]['utility'].items()],
        _length_comparison(phases[512], phases[1024]))))
    fields = (
        ['model', 'selected_training_teacher_KL_nats_per_token', 'validation_KL_nats_per_token',
         'selected_global_step', 'elapsed_training_seconds', 'Q', 'alpha', 'groups', 'owner_TPR', 'wrong_worst'],
        ['model', 'Q', 'alpha', 'groups', 'rate', 'observed_clean_FPR', 'cosine_median', 'wrong_worst'],
        ['model', 'Q', 'alpha', 'groups', 'original_rate', 'paraphrased_rate', 'drop_pp'],
        ['model', 'correct', 'total', 'accuracy', 'delta_vs_clean_pp', 'dtype'],
        ['model', 'responses', 'mean_generated_tokens', 'median_generated_tokens', 'eos_termination_rate',
         'max_length_truncation_rate', 'empty_response_rate', 'severe_repetition_rate', 'text_token_roundtrip_consistency_rate'],
        ['model', 'Q', 'alpha', 'rate512', 'rate1024', 'delta_rate_pp', 'truncation_rate512', 'truncation_rate1024',
         'generation_seconds512', 'generation_seconds1024',
         'native_generated_tokens_per_second512', 'native_generated_tokens_per_second1024',
         'timing_scope512', 'timing_scope1024', 'timing_metadata_status512', 'timing_metadata_status1024'])
    for name, columns in zip(TABLE_NAMES, fields):
        _write_table(out, name, tables[name], columns)
    statuses = dict(training=full['status'], pilot_training=pilot['status'],
        evaluation={str(length): dict(status=value['status'], models={model: _model_status(value, model)
            for model in (['clean', 'teacher', 'student', 'pilot512'] if length == 512 else ['clean', 'teacher', 'student'])})
            for length, value in phases.items()},
        arc_c=dict(status=arc_status, models=arc_models),
        paraphrase=dict(status=attack_status, models=condition_status))
    all_complete = (full['status'] == pilot['status'] == 'COMPLETE'
        and all(value['status'] == 'COMPLETE' for value in phases.values())
        and arc_status == attack_status == 'COMPLETE' and not errors)
    any_complete = full['status'] == 'COMPLETE' or any(value['receipt'] for value in phases.values())
    summary = dict(schema=1, identity=state['identity'], status='COMPLETE' if all_complete else
        ('PARTIAL' if any_complete else 'INCOMPLETE'), complete=all_complete,
        scientific_success=None, stages=statuses, errors=errors,
        effective_training_counts={'pilot': pilot['effective_counts'], 'full': full['effective_counts']},
        training_selection={'pilot_validation_KL': _teacher_kl(pilot), 'full_validation_KL': _teacher_kl(full)},
        tables={name: dict(path='report/tables/' + name + '.csv', rows=len(rows)) for name, rows in tables.items()},
        checkpoint_reporting_policy='Selection/export identity, file presence and JSON content hashes; evaluation performs full weight-content verification',
        checkpoint_selection='Minimum held-out validation teacher KL; no test/ARC/paraphrase selection',
        inference_or_api_invoked=False, all_counts_are_nested_by_phase_and_model=True)
    write_json(out/'report/summary.json', summary)
    lines = ['# KLRD-RS full10k experiment', '', 'Status: **' + summary['status'] + '**.', '',
        'Rates below are proportions; deltas marked pp are percentage points. No scientific SUCCESS flag is inferred.',
        'Only saved, identity-bound artifacts are read. No inference or API calls occur in report/export.', '',
        'Stage status: ' + _display(statuses)]
    titles = ('Pilot versus full training at512', 'Original fingerprint at1024',
        'High-fidelity paraphrase at1024', 'ARC-C utility', 'Generation utility at1024',
        'Generation length at the same checkpoints')
    for name, title, columns in zip(TABLE_NAMES, titles, fields):
        lines += _markdown_table(title, tables[name], columns)
    lines += _discussion(tables)
    if errors:
        lines += ['', '## Incomplete or invalid evidence', ''] + ['- ' + error for error in errors]
    _text(out/'report/full_experiment_report.md', '\n'.join(lines) + '\n')
    if package:
        summary['review_package'] = package_review(state)
    # The ZIP contains the pre-package summary to avoid a self-referential hash.
    return summary
