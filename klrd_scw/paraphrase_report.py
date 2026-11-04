"""Auditable before/after SCW report and text-only rewrite diagnostics."""
import csv
from difflib import SequenceMatcher
import math
from pathlib import Path
import random
import re
import statistics
from .artifacts import dump_json


def pair_diagnostics(original,paraphrased,tokenizer):
    before=len(tokenizer.encode(original,add_special_tokens=False))
    after=len(tokenizer.encode(paraphrased,add_special_tokens=False))
    a=set(re.findall(r'\b\w+\b',original.lower()))
    b=set(re.findall(r'\b\w+\b',paraphrased.lower()))
    union=a|b
    similarity=SequenceMatcher(None,original.split(),paraphrased.split(),autojunk=False).ratio()
    return {'original_characters':len(original),'paraphrased_characters':len(paraphrased),
            'original_tokens':before,'paraphrased_tokens':after,
            'token_length_ratio':after/before if before else None,
            'character_length_ratio':len(paraphrased)/len(original) if original else None,
            'word_jaccard':len(a&b)/len(union) if union else 1.,
            'word_sequence_similarity':similarity,'unchanged_exactly':original==paraphrased,
            'very_small_edit':similarity>=.95}


def write_text_diagnostics(state,terminal,tokenizer):
    root=state['root']; summaries={}; all_rows=[]
    examples=['# Deterministic SCW output-paraphrase examples','',
              'Unfiltered source/output pairs selected by the predeclared seed; exact text is inside tags.']
    for split in ('fingerprint','clean'):
        successful=[r for r in terminal if r['split']==split and r['status']=='ok']
        diagnostics=[]
        for row in successful:
            value={'split':split,'prompt_id':row['prompt_id'],'response_id':row['response_id'],
                   **pair_diagnostics(row['source_text'],row['paraphrased_text'],tokenizer)}
            all_rows.append(value); diagnostics.append(value)
        ratios=[r['token_length_ratio'] for r in diagnostics if r['token_length_ratio'] is not None]
        overlaps=[r['word_jaccard'] for r in diagnostics]
        n=len(diagnostics)
        summaries[split]={'successful_pairs':n,
            'mean_token_length_ratio':statistics.mean(ratios) if ratios else None,
            'median_token_length_ratio':statistics.median(ratios) if ratios else None,
            'mean_word_jaccard':statistics.mean(overlaps) if overlaps else None,
            'median_word_jaccard':statistics.median(overlaps) if overlaps else None,
            'fraction_unchanged_exactly':sum(r['unchanged_exactly'] for r in diagnostics)/n if n else None,
            'fraction_very_small_edits':sum(r['very_small_edit'] for r in diagnostics)/n if n else None,
            'very_small_edit_definition':state['manifest']['very_small_edit_definition']}
        count=min(state['manifest']['example_count_per_model'],n)
        selected=sorted(random.Random(state['manifest']['example_seed']).sample(range(n),count))
        for i in selected:
            row=successful[i]
            examples += ['',f"## {split}: {row['prompt_id']}",'','<original>',row['source_text'],
                         '</original>','','<paraphrased>',row['paraphrased_text'],'</paraphrased>']
    fields=['split','prompt_id','response_id','original_characters','paraphrased_characters',
            'original_tokens','paraphrased_tokens','token_length_ratio','character_length_ratio',
            'word_jaccard','word_sequence_similarity','unchanged_exactly','very_small_edit']
    with (root/'paraphrase_diagnostics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(all_rows)
    (root/'paraphrase_examples.md').write_text('\n'.join(examples)+'\n',encoding='utf-8')
    dump_json(root/'paraphrase_diagnostics.json',summaries)
    return summaries


def _pct(value):
    return 'N/A' if value is None else f'{value:.3%}'


def _number(value):
    return 'N/A' if value is None else f'{value:.7g}'


def classify(report,manifest):
    """Predeclared descriptive operating points, not tests of significance."""
    queries=report['views']['frozen']['queries']; policy=manifest['classification_policy']
    def strong(q,kind):
        entry=queries[str(q)]
        correct=entry['keys']['correct'][kind]['after']
        tpr=correct['fingerprint']['detection_rate']; fpr=correct['clean']['detection_rate']
        wrong=entry['specificity'][kind]['after']['worst_wrong_tpr']
        return (tpr is not None and fpr is not None and wrong is not None and
                tpr>=policy['strong_tpr'] and fpr<=policy['fpr_ceiling'] and wrong<=policy['wrong_tpr_ceiling'])
    def z_retained(q):
        entry=queries[str(q)]; c=entry['keys']['correct']['centered_z']
        before=c['before']; after=c['after']
        b_tpr=before['fingerprint']['detection_rate']; a_tpr=after['fingerprint']['detection_rate']
        b_fpr=before['clean']['detection_rate']; a_fpr=after['clean']['detection_rate']
        b_wrong=entry['specificity']['centered_z']['before']['worst_wrong_tpr']
        a_wrong=entry['specificity']['centered_z']['after']['worst_wrong_tpr']
        if any(x is None for x in [b_tpr,a_tpr,b_fpr,a_fpr,b_wrong,a_wrong]): return False
        return (a_tpr>=policy.get('centered_z_tpr_retention',.90)*b_tpr
            and a_fpr<=max(policy['fpr_ceiling'],b_fpr+policy.get('centered_z_fpr_increase_tolerance',.005))
            and a_wrong<=max(policy['wrong_tpr_ceiling'],b_wrong+policy.get('centered_z_wrong_increase_tolerance',.05))
            and (entry['separation']['centered_z']['after'] or 0)>0)
    primary=queries[str(policy['primary_Q'])]
    missing=[split for split in ('clean','fingerprint') if primary['keys']['correct']['raw']['after'][split]['detection_rate'] is None]
    if missing:
        return {'classification':'not_evaluable','conclusion':'SCW output-paraphrase robustness could not be estimated because no successful '+ ' and '.join(missing)+' outputs were available.'}
    if strong(policy['primary_Q'],'raw') and z_retained(policy['primary_Q']):
        return {'classification':'1_robust_at_current_Q',
                'conclusion':'KLRD remains strongly detectable by the frozen raw verifier after SCW output paraphrasing, while centered-Z retains its pre-attack diagnostic behavior.'}
    for q in manifest['query_budgets_extended']:
        if strong(q,'raw') and z_retained(q):
            return {'classification':'2_attenuated_recoverable_with_more_queries','recovery_Q':q,
                    'conclusion':'SCW-style output paraphrasing attenuates KLRD, but the primary raw signal recovers with larger query budgets.'}
    values=[e['keys']['correct'][kind]['after']['fingerprint']['detection_rate']
            for e in queries.values() for kind in ('raw','centered_z')]
    if values and all(value is not None and value<=policy.get('near_chance_tpr_ceiling',.05) for value in values):
        return {'classification':'3_most_measurable_signal_removed_at_tested_operating_points',
                'conclusion':'SCW-style output paraphrasing removes most detectable KLRD signal under both frozen verifiers at the tested operating points.'}
    return {'classification':'mixed_inconclusive',
            'conclusion':'SCW-style output paraphrasing gives mixed attenuation evidence; robust ownership detection or signal erasure is not established by the tested operating points.'}


def write_reports(state,report,terminal,diagnostics):
    root=state['root']; manifest=state['manifest']
    api_counts={split:{'source':sum(r['split']==split for r in terminal),
                       'ok':sum(r['split']==split and r['status']=='ok' for r in terminal),
                       'failed':sum(r['split']==split and r['status']!='ok' for r in terminal)}
                for split in ('clean','fingerprint')}
    interpretation=classify(report,manifest)
    if any(count['failed'] for count in api_counts.values()):
        interpretation['conclusion']+=' This estimate is conditional on successful API outputs; exclusions limit conclusions about the full source population.'
    report['manifest']=manifest;report['api_counts']=api_counts
    report['paraphrase_diagnostics']=diagnostics;report['interpretation']=interpretation
    dump_json(root/'results.json',report)
    dump_json(root/'signal_diagnostics.json',report['signal_diagnostics'])
    for kind,filename in [('raw','metrics_raw.json'),('centered_z','metrics_centered_z.json')]:
        value={'statistic':kind,'views':{},'cohorts':report['cohorts'],'api_counts':api_counts}
        for view_name,view in report['views'].items():
            value['views'][view_name]={**{k:v for k,v in view.items() if k!='queries'},'queries':{}}
            for q,entry in view['queries'].items():
                value['views'][view_name]['queries'][q]={
                    'keys':{k:v[kind] for k,v in entry['keys'].items()},
                    'specificity':entry['specificity'][kind],
                    'separation':entry['separation'][kind],'identification':entry['identification']}
        dump_json(root/filename,value)
    comparison=[];specificity=[]
    for view_name,view in report['views'].items():
        for q,entry in view['queries'].items():
            for kind in ('raw','centered_z'):
                c=entry['keys']['correct'][kind];spec=entry['specificity'][kind]
                before_tpr=c['before']['fingerprint']['detection_rate'];after_tpr=c['after']['fingerprint']['detection_rate']
                comparison.append({'view':view_name,'statistic':kind,'Q':q,'threshold':c['threshold'],
                    'original_correct_TPR':before_tpr,'paraphrased_correct_TPR':after_tpr,
                    'delta_TPR_percentage_points':100*(after_tpr-before_tpr) if before_tpr is not None and after_tpr is not None else None,
                    'original_clean_FPR':c['before']['clean']['detection_rate'],
                    'paraphrased_clean_FPR':c['after']['clean']['detection_rate'],
                    'average_wrong_TPR':spec['after']['average_wrong_tpr'],
                    'median_wrong_TPR':spec['after']['median_wrong_tpr'],
                    'worst_wrong_TPR':spec['after']['worst_wrong_tpr'],'inverse_TPR':spec['after']['inverse_tpr']})
                for stage in ('before','after'):
                    identity=entry['identification'][stage]
                    for key,value in entry['keys'].items():
                        evidence=value[kind][stage]
                        specificity.append({'view':view_name,'statistic':kind,'Q':q,'stage':stage,'key':key,
                            'threshold':value[kind]['threshold'],'TPR':evidence['fingerprint']['detection_rate'],
                            'FPR':evidence['clean']['detection_rate'],
                            'raw_strict_key_ID':identity['raw_strict_correct_rate'],
                            'raw_tie_rate':identity['raw_tie_rate'],
                            'Z_strict_key_ID':identity['z_strict_correct_rate'],'Z_tie_rate':identity['z_tie_rate']})
    for name,rows in [('comparison.csv',comparison),('key_specificity.csv',specificity)]:
        with (root/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=[interpretation['conclusion'],'','# A. Attack reproduction','',
        f"SCW repo: {manifest['scw_repo']}",f"SCW commit: {manifest['scw_commit']}",
        f"Model: {manifest['paraphraser_model']}",f"Prompt (JSON preserves trailing space): {__import__('json').dumps(manifest['paraphrase_prompt'])}",
        'API: OpenAI Responses API; only model/input request arguments; no custom system/decoding instructions.',
        f"Paraphrases/failures: {api_counts}",
        'No KLRD generation, training, key recalibration, fingerprint re-embedding or adaptive attack.',
        '', '# B. Main raw verifier — View A (primary frozen thresholds)', '']
    def table(kind,budgets,view='frozen'):
        output=['| Q | Original TPR | Paraphrased TPR | Delta TPR (pp) | Original FPR | Paraphrased FPR | Avg wrong TPR | Worst wrong TPR |',
                '|---:|---:|---:|---:|---:|---:|---:|---:|']
        for q in budgets:
            entry=report['views'][view]['queries'].get(str(q))
            if entry is None: continue
            c=entry['keys']['correct'][kind];s=entry['specificity'][kind]['after']
            b=c['before']['fingerprint']['detection_rate'];a=c['after']['fingerprint']['detection_rate']
            delta=100*(a-b) if a is not None and b is not None else None
            output.append(f"| {q} | {_pct(b)} | {_pct(a)} | {_number(delta)} | "
                f"{_pct(c['before']['clean']['detection_rate'])} | {_pct(c['after']['clean']['detection_rate'])} | "
                f"{_pct(s['average_wrong_tpr'])} | {_pct(s['worst_wrong_tpr'])} |")
        return output
    lines+=table('raw',manifest['query_budgets_primary'])
    lines+=['','# C. Centered-Z — View A (diagnostic unchanged detector)','']
    lines+=table('centered_z',manifest['query_budgets_primary'])
    lines+=['','# D. Supplemental query budgets','', 'Raw:','']+table('raw',manifest['query_budgets_extended'])
    lines+=['','Centered-Z:','']+table('centered_z',manifest['query_budgets_extended'])
    lines+=['','# E. Key specificity','',
        'All original 20 wrong keys and inverse-correct are retained. Baseline and attacked rows are shown; key_specificity.csv contains every key and both threshold views.',
        '| Q | Stage | Statistic | Avg wrong TPR | Median wrong TPR | Worst wrong TPR | Inverse TPR | Raw ID | Raw ties | Z ID | Z ties |',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for q,entry in report['views']['frozen']['queries'].items():
        for stage in ('before','after'):
            identity=entry['identification'][stage]
            for kind in ('raw','centered_z'):
                s=entry['specificity'][kind][stage]
                lines.append(f"| {q} | {stage} | {kind} | {_pct(s['average_wrong_tpr'])} | {_pct(s['median_wrong_tpr'])} | "
                    f"{_pct(s['worst_wrong_tpr'])} | {_pct(s['inverse_tpr'])} | "
                    f"{_pct(identity['raw_strict_correct_rate'])} | {_pct(identity['raw_tie_rate'])} | "
                    f"{_pct(identity['z_strict_correct_rate'])} | {_pct(identity['z_tie_rate'])} |")
    lines+=['','# F. Signal attenuation','', '| Population | Full aggregate raw LLR/token | Mean response LLR/token | Median response LLR/token | Mean centered contribution/token |',
            '|---|---:|---:|---:|---:|']
    signal=report['signal_diagnostics']['keys']['correct']
    for stage in ('original_full','before','after'):
        for split,value in signal[stage].items():
            lines.append(f"| {stage} {split} | {_number(value['full_raw_llr_per_token'])} | "
                f"{_number(value['mean_raw_llr_per_token'])} | {_number(value['median_raw_llr_per_token'])} | "
                f"{_number(value['mean_centered_contribution_per_token'])} |")
    lines += ['',f"Matched FP raw retention: {signal['fingerprint_raw_retention']}",
              '', '| Q | Centered-Z FP-clean separation before | After |','|---:|---:|---:|']
    for q,entry in report['views']['frozen']['queries'].items():
        lines.append(f"| {q} | {_number(entry['separation']['centered_z']['before'])} | {_number(entry['separation']['centered_z']['after'])} |")
    lines+=['','# G. Paraphrase diagnostics','',f'{diagnostics}',
        '', 'See paraphrase_diagnostics.csv for every successful pair and paraphrase_examples.md for the deterministic unfiltered sample.',
        '', '# H. Interpretation','',f"Classification: {interpretation['classification']}",
        f"Predeclared descriptive policy: {manifest['classification_policy']}",
        'View A is the primary result. Labels use predeclared descriptive operating points, not significance tests. Centered-Z retention is checked against its own before baseline (including its existing cross-key issue), not assumed perfect. Mixed cases are inconclusive rather than forced into an erasure claim.',
        'Zero exceedances do not imply zero population risk. Bootstrap draws reuse responses and are not independent endpoint experiments. No independent-binomial confidence intervals are fabricated.',
        'Observed-token LLR omits special/EOS targets and uses decoded/re-encoded text; it is not the complete sequence likelihood ratio or proof of ownership.',
        f"Cohort/attrition: clean {report['cohorts']['clean']['successful_responses']}/{report['cohorts']['clean']['source_responses']}; "
        f"FP {report['cohorts']['fingerprint']['successful_responses']}/{report['cohorts']['fingerprint']['source_responses']}. "
        'Main before/after tables use matching successful IDs within each model. Full original-population baselines are separately in results.json; every successful paraphrase is retained.',
        f"Cross-model successful prompt IDs match: {report['cohorts']['cross_model_successful_ids_match']}",
        'Query resampling uses replacement, including supplemental budgets; these curves assess aggregation of the retained fixed response population.',
        '', '## View B — secondary paraphrased-clean diagnostic recalibration','',
        'Thresholds use paraphrased-clean evidence only. Its clean FPR uses the same pool and is in-sample, not independent held-out calibration. No attacked fingerprint statistic sets these thresholds.',
        'Raw:','']
    if report['views']['paraphrased_clean_diagnostic']['available']:
        lines+=table('raw',manifest['query_budgets_primary']+manifest['query_budgets_extended'],'paraphrased_clean_diagnostic')
        lines+=['','Centered-Z:','']+table('centered_z',manifest['query_budgets_primary']+manifest['query_budgets_extended'],'paraphrased_clean_diagnostic')
    else:
        lines+=['Unavailable: no positive-token paraphrased-clean population.']
    (root/'summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(interpretation['conclusion'],flush=True)
