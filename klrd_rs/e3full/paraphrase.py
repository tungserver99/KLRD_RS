"""Explicit 1024-token, three-condition attack using the unchanged E2B transport.

Only this phase initializes the copied API transport. Originals and accepted
rewrites belong to this run; historical E2B supplies IDs/protocol/groups only.
"""
from importlib.metadata import version
import re
import statistics

from ..io import read_json, write_json, digest, sha256
from .common import checked, complete


MODELS = ('clean', 'teacher', 'student')
SCORE_FILES = ('per_key_group.csv', 'per_response.csv',
               'closed_set_identification.csv', 'clean_wrong_controls.csv',
               'group_summary.csv')


def _population(records, ids, model):
    index = {row['prompt_id']: row for row in records}
    if len(index) != len(records) or len(ids) != 400 or len(set(ids)) != 400:
        raise ValueError('Require unique original records and the frozen400 IDs')
    if not set(ids) <= set(index):
        raise ValueError('Missing fixed400 1024-token originals: ' + model)
    selected = [index[pid] for pid in ids]
    population = [dict(condition='0', model=model, prompt_id=row['prompt_id'],
        response_id=row['response_id'], prompt=row['prompt'],
        original_text=row['response'], original_text_sha256=digest(row['response']),
        selection_index=i) for i, row in enumerate(selected)]
    if len({row['response_id'] for row in population}) != 400:
        raise ValueError('Duplicated original response identity: ' + model)
    return selected, population


def _sum_usage(values):
    """Keep provider usage hierarchy, including reasoning/cache subcounts."""
    result = {}
    for value in values:
        for key, item in value.items():
            if isinstance(item, dict):
                result[key] = _sum_usage([result.get(key, {}), item])
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                result[key] = result.get(key, 0) + item
    return result


def _quality(pairs, tokenizer):
    from klrd_scw.paraphrase_report import pair_diagnostics
    rows = []
    for pair in pairs:
        a, b = pair['original_text'], pair['paraphrased_text']
        rows.append(dict(model=pair['model'], response_hash=digest(pair['response_id']),
            **pair_diagnostics(a, b, tokenizer),
            numeric_literal_sets_match=set(re.findall(r'\d+(?:[.,]\d+)*', a)) ==
                                      set(re.findall(r'\d+(?:[.,]\d+)*', b)),
            code_fence_count_changed=a.count('```') != b.count('```'),
            line_count_changed=len(a.splitlines()) != len(b.splitlines()),
            api_response_status=pair['api_metadata'].get('api_response_status', 'unknown')))
    ratios = [row['token_length_ratio'] for row in rows if row['token_length_ratio'] is not None]
    summary = dict(pairs=len(rows),
        token_length_ratio_mean=statistics.mean(ratios) if ratios else None,
        token_length_ratio_median=statistics.median(ratios) if ratios else None,
        word_jaccard_mean=statistics.mean(row['word_jaccard'] for row in rows),
        word_sequence_similarity_mean=statistics.mean(row['word_sequence_similarity'] for row in rows),
        numeric_change_flags=sum(not row['numeric_literal_sets_match'] for row in rows),
        code_structure_change_flags=sum(row['code_fence_count_changed'] for row in rows),
        line_structure_change_flags=sum(row['line_count_changed'] for row in rows),
        incomplete_response_status_count=sum(row['api_response_status'] == 'incomplete' for row in rows),
        api_usage=_sum_usage([pair['api_metadata'].get('api_usage', {}) for pair in pairs]),
        semantic_quality_status='PENDING_HUMAN_REVIEW; lexical diagnostics do not prove semantic fidelity',
        selection_policy='All400 retained; no quality/score filtering, resampling or response trimming')
    return rows, summary


def _comparisons(summaries, group_rows):
    rows = []
    for model in MODELS:
        for q in (20, 50, 100):
            original_cos = [float(row['cosine']) for row in group_rows
                if row['model'] == model + '_original' and int(row['Q']) == q and int(row['test_key']) == 0]
            paraphrased_cos = [float(row['cosine']) for row in group_rows
                if row['model'] == model + '_paraphrased' and int(row['Q']) == q and int(row['test_key']) == 0]
            if len(original_cos) != 400 // q or len(paraphrased_cos) != 400 // q:
                raise ValueError('Incomplete fixed400 group cosine coverage')
            for alpha in (.01, .0001):
                before = next(row for row in summaries if row['model'] == model + '_original'
                              and row['Q'] == q and row['alpha'] == alpha)
                after = next(row for row in summaries if row['model'] == model + '_paraphrased'
                             and row['Q'] == q and row['alpha'] == alpha)
                clean_before = next(row for row in summaries if row['model'] == 'clean_original'
                                    and row['Q'] == q and row['alpha'] == alpha)
                clean_after = next(row for row in summaries if row['model'] == 'clean_paraphrased'
                                   and row['Q'] == q and row['alpha'] == alpha)
                rows.append(dict(model=model, Q=q, alpha=alpha, groups=before['groups'],
                    rate_kind='observed_clean_FPR' if model == 'clean' else 'owner_TPR',
                    original_detections=before['detections'], paraphrased_detections=after['detections'],
                    original_rate=before['rate'], paraphrased_rate=after['rate'],
                    drop_pp=100 * (before['rate'] - after['rate']),
                    original_clean_FPR=clean_before['rate'], paraphrased_clean_FPR=clean_after['rate'],
                    wrong_mean_original=before['wrong_mean'], wrong_mean_paraphrased=after['wrong_mean'],
                    wrong_worst_original=before['wrong_worst'], wrong_worst_paraphrased=after['wrong_worst'],
                    cosine_mean_original=statistics.mean(original_cos),
                    cosine_mean_paraphrased=statistics.mean(paraphrased_cos),
                    cosine_median_original=statistics.median(original_cos),
                    cosine_median_paraphrased=statistics.median(paraphrased_cos),
                    cosine_mean_drop=statistics.mean(a-b for a,b in zip(original_cos, paraphrased_cos)),
                    cosine_median_drop=statistics.median(original_cos)-statistics.median(paraphrased_cos),
                    paired_cosine_drop_median=statistics.median(a-b for a,b in zip(original_cos, paraphrased_cos)),
                    p_median_original=before['p_median'], p_median_paraphrased=after['p_median']))
    return rows


def run(state):
    from .evaluation import load_observations
    from ..e3.evaluate import score_rows, cached_scores, tables, csv
    from ..e2b.source import load_prepared, immutable_rows
    from ..e2b.transport import (requests_for, prepare_condition_resume,
        run_copied_transport, receipts_path, checked_results, read_condition)
    from ..e2b.protocol import PARAPHRASE_PROMPT
    from ..e2b.config import SDK_VERSION
    from klrd_scw.artifacts import ensure_manifest

    phase, originals, _, source_e2 = load_observations(state, 1024)
    previous = checked(phase, 'paraphrase/complete.json')
    if previous is not None:
        return previous
    source = load_prepared(state['root'], state['root']/state['cfg'].e2b_run)
    protocol = read_json(source['out']/'provenance/protocol.json')
    if (protocol['model'] != 'gpt-5-mini-2025-08-07'
            or protocol['prompt'] != PARAPHRASE_PROMPT
            or protocol.get('protocol_variant') != 'high_fidelity_v1'
            or protocol != source['manifest']['protocol']
            or protocol.get('request_fields') != ['model', 'input']):
        raise ValueError('Require frozen run002 seven-rule high-fidelity prompt/model and API fields')
    ids = source['manifest']['prompt_ids']
    groups = source['manifest']['groups']
    if any(group['Q'] not in (20, 50, 100) for group in groups):
        raise ValueError('Unexpected frozen400 query budgets')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(state['cfg'].model, revision=state['cfg'].revision,
        trust_remote_code=False, local_files_only=True)
    generation = source_e2['environment']['generation']
    if (tokenizer.chat_template != generation.get('chat_template', tokenizer.chat_template)
            or digest(tokenizer.get_vocab()) != generation['tokenizer_vocab_sha256']):
        raise ValueError('Quality tokenizer differs from the frozen source')
    if version('openai') != SDK_VERSION:
        raise RuntimeError('Frozen server requirements-paraphrase.txt SDK required: ' + SDK_VERSION)

    out = phase['out']; observations = {}; attacks = {}; resumes = {}; products = []
    # Validate ALL namespaces, originals and accepted journals before any call.
    for model in MODELS:
        records, vectors, eligible = originals[model]
        selected, population = _population(records, ids, model)
        index = {row['prompt_id']: i for i, row in enumerate(records)}
        positions = [index[pid] for pid in ids]
        observations[model + '_original'] = (selected, vectors[positions], eligible[positions])
        namespace = out/'paraphrase'/model
        if out.resolve() not in namespace.resolve().parents:
            raise ValueError('Paraphrase namespace leaves the phase output')
        immutable_rows(namespace/'private/population.jsonl', population)
        attack = dict(phase, out=namespace, population=population, manifest={'protocol': protocol})
        attacks[model] = attack
        resumes[model] = prepare_condition_resume(attack, '0', requests_for(population, '0'))
        checked(phase, 'paraphrase/' + model + '/complete.json')
    ensure_manifest(out/'paraphrase/protocol.json', protocol)
    ensure_manifest(out/'paraphrase/manifest.json', dict(identity=phase['identity'],
        original_max_new_tokens=1024, models=list(MODELS), responses_per_model=400,
        prompt_ids_sha256=digest(ids), groups_sha256=digest(groups), protocol_sha256=digest(protocol),
        source_E2B_population_manifest_sha256=sha256(source['out']/'provenance/population_manifest.json'),
        source_evaluation_receipt_sha256=sha256(out/'provenance/evaluation_complete.json'),
        originals_sha256={model: digest(attacks[model]['population']) for model in MODELS},
        historical_paraphrase_texts_reused=0))
    write_json(out/'paraphrase/api_execution.json', dict(model=protocol['model'], sdk_version=SDK_VERSION,
        conditions=list(MODELS), expected_per_condition=400, expected_total=1200,
        original_max_new_tokens=1024, workers=state.get('api_workers', state['cfg'].workers),
        request_fields=['model', 'input'], output_token_cap='omitted, as in copied transport',
        input_policy='Full original response; no truncation, chunking or prompt change',
        acceptance='Unchanged transport: nonempty output_text, including incomplete status; reported separately',
        historical_paraphrase_texts_reused=0, conditions_sequential=True))
    write_json(out/'paraphrase/api_resume.json', resumes)
    all_diagnostics = []; quality = {}; condition_receipts = {}
    for model in MODELS:
        attack = attacks[model]; population = attack['population']
        requests = requests_for(population, '0')
        try:
            # No client is created for a fully accepted namespace on resume.
            if resumes[model]['kept'] == 400:
                returned = read_condition(attack, '0')
            else:
                returned = run_copied_transport(requests, receipts_path(attack, '0'),
                    prompt=protocol['prompt'], workers=state.get('api_workers', state['cfg'].workers),
                    max_attempts=5)
                checked_results(requests, returned, protocol['prompt'])
                returned = read_condition(attack, '0')
        except Exception:
            raise RuntimeError(model + ' 1024 paraphrases incomplete; durable accepted receipts retained. '
                'Retry explicit paraphrase mode; no reduced-cohort metrics are produced.') from None
        pairs = [dict(row, paraphrased_text=answer['paraphrased_text'],
            paraphrase_text_sha256=digest(answer['paraphrased_text']),
            api_metadata={key: value for key, value in answer.items()
                if key.startswith('api_') or key in ('status', 'attempts', 'paraphraser_model')})
            for row, answer in zip(population, returned)]
        pair_path = 'paraphrase/' + model + '/private/original_paraphrase_pairs.jsonl'
        immutable_rows(out/pair_path, pairs)
        texts = [dict(row, response=row['paraphrased_text']) for row in pairs]
        label = model + '_paraphrased400'
        condition_receipt = checked(phase, 'paraphrase/' + model + '/complete.json')
        if condition_receipt is None:
            vectors, eligible = score_rows(phase, texts, label, source_e2['environment'])
        else:
            vectors, eligible = cached_scores(phase, texts, label)
        observations[model + '_paraphrased'] = (texts, vectors, eligible)
        diagnostics, quality[model] = _quality(pairs, tokenizer)
        all_diagnostics.extend(diagnostics)
        score_products = ['provenance/' + label + '_scores_complete.json'] + [
            path.relative_to(out).as_posix() for path in sorted((out/'private/eval_scores'/label).glob('*'))
            if path.is_file()]
        condition_products = [pair_path, 'paraphrase/' + model + '/private/population.jsonl'] + score_products
        condition_receipts[model] = condition_receipt or complete(phase,
            'paraphrase/' + model + '/complete.json', condition_products,
            status='PARAPHRASE_CONDITION_COMPLETE', model=model, responses=400,
            original_max_new_tokens=1024, pairs_sha256=digest(pairs), quality=quality[model])
        products.extend(condition_products + ['paraphrase/' + model + '/complete.json'])

    summaries = tables(phase, observations, groups, out/'paraphrase/scores')
    import csv as csv_module
    with (out/'paraphrase/scores/per_key_group.csv').open(encoding='utf-8', newline='') as stream:
        group_rows = list(csv_module.DictReader(stream))
    comparisons = _comparisons(summaries, group_rows)
    csv(out/'paraphrase/metrics.csv', comparisons)
    csv(out/'paraphrase/quality.csv', all_diagnostics)
    write_json(out/'paraphrase/quality_summary.json', dict(models=quality, pairs=1200,
        semantic_quality_status='PENDING_HUMAN_REVIEW',
        cohort_policy='400 fixed prompt IDs per model, all retained; no quality/score filtering'))
    products.extend(['paraphrase/' + name for name in ('metrics.csv', 'quality.csv',
        'quality_summary.json', 'protocol.json', 'manifest.json', 'api_execution.json')])
    products.extend('paraphrase/scores/' + name for name in SCORE_FILES)
    return complete(phase, 'paraphrase/complete.json', products,
        status='PARAPHRASE_COMPLETE', responses=1200, responses_per_model=400,
        original_max_new_tokens=1024, model=protocol['model'], summary=summaries,
        conditions={model: {'status': receipt['status'], 'responses': 400}
                    for model, receipt in condition_receipts.items()},
        historical_paraphrase_texts_reused=0, source_E2B_protocol_run=state['cfg'].e2b_run)
