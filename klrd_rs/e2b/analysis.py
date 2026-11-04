"""Paired, disjoint-group spherical analysis; no fitting, calibration or new rules."""
from collections import defaultdict
import numpy as np
from ..io import write_csv,write_json,read_json,sha256,digest
from ..diagnostics import binomial_ci
from ..e2.analysis import wrong_indices,identify
from ..e2.verifier import SphericalBlackBoxVerifier
from .config import CONDITIONS,COUNT,QUERIES,ALPHAS,make_groups
from .scoring import load_vectors
from .quality import human_review_summary


def group_scores(state,pairs,before,after):
    groups=state['manifest']['groups']
    if groups!=make_groups(state['manifest']['prompt_ids']):raise ValueError('E2B group design changed')
    index={}
    for i,row in enumerate(pairs):
        key=(row['condition'],row['prompt_id'])
        if key in index:raise ValueError('Duplicate paired response identity')
        index[key]=i
    bank=state['frozen']['bank'];rows=[];identifications=[]
    for stage,data in [('original',before),('paraphrased',after)]:
        for group in groups:
            grouped={}
            for condition in CONDITIONS:
                selected=[index[(condition,pid)] for pid in group['prompt_ids']]
                v=data['vectors'][selected].sum(0,dtype=np.float64)
                grouped[condition]=(SphericalBlackBoxVerifier.key_scores(v,bank),int(data['eligible_tokens'][selected].sum()))
            for owner in range(8):
                candidates=[owner]+wrong_indices(owner)
                for condition,role,tests in [(str(owner),'fingerprint',candidates),('clean','clean',[owner])]:
                    scores,tokens=grouped[condition]
                    for key in tests:
                        rows.append({'stage':stage,'owner':owner,'Q':group['Q'],'group_index':group['group_index'],
                            'group_id':group['group_id'],'role':'clean' if role=='clean' else 'correct' if key==owner else 'wrong',
                            'source_condition':condition,'test_key':key,'responses':group['Q'],'eligible_tokens':tokens,
                            'vector_norm':scores['norm'],'zero_vector':scores['zero_vector'],
                            'cosine':float(scores['cosine'][key]),'p_value':float(scores['p'][key]),
                            'detect_at_0_01':bool(scores['p'][key]<=.01),'detect_at_0_0001':bool(scores['p'][key]<=.0001)})
                scores,_=grouped[str(owner)];cos=scores['cosine'][candidates];result=identify(cos)
                identifications.append({'stage':stage,'owner':owner,'Q':group['Q'],'group_index':group['group_index'],
                    'group_id':group['group_id'],'candidate_keys':','.join(map(str,candidates)),
                    'identified_key':None if result['tie'] else candidates[int(np.argmax(cos))],**result})
    return rows,identifications


def summarize_comparisons(rows):
    result=[]
    owner_budgets=sorted({(r['owner'],r['Q']) for r in rows if r['role']=='correct'})
    for owner,q in owner_budgets:
        block=[r for r in rows if r['owner']==owner and r['Q']==q]
        conditions={}
        for stage in ['original','paraphrased']:
            current=[r for r in block if r['stage']==stage]
            correct=[r for r in current if r['role']=='correct'];clean=[r for r in current if r['role']=='clean']
            expected=set(range(COUNT//q))
            if len(correct)!=len(expected) or {r['group_index'] for r in correct}!=expected or len(clean)!=len(expected) or {r['group_index'] for r in clean}!=expected:
                raise ValueError('Incomplete/duplicate paired group coverage')
            wrong_keys=sorted({r['test_key'] for r in current if r['role']=='wrong'})
            wrong={key:[r for r in current if r['role']=='wrong' and r['test_key']==key] for key in wrong_keys}
            if not wrong or any(len(v)!=len(expected) or {r['group_index'] for r in v}!=expected for v in wrong.values()):
                raise ValueError('Missing/duplicate wrong-key group coverage')
            conditions[stage]=(correct,clean,wrong)
        if set(conditions['original'][2])!=set(conditions['paraphrased'][2]):raise ValueError('Wrong-key controls changed across stages')
        n=COUNT//q
        for alpha in ALPHAS:
            row={'owner':owner,'Q':q,'alpha':alpha,'groups':n,'responses_per_condition':COUNT}
            detections={}
            for stage,(correct,clean,wrong) in conditions.items():
                detected={r['group_index']:r['p_value']<=alpha for r in correct};count=sum(detected.values())
                clean_count=sum(r['p_value']<=alpha for r in clean)
                rates=[sum(r['p_value']<=alpha for r in values)/n for values in wrong.values()]
                t_ci=binomial_ci(count,n);c_ci=binomial_ci(clean_count,n)
                row.update({f'owner_detected_{stage}':count,f'TPR_{stage}':count/n,
                    f'TPR_CI95_low_{stage}':t_ci[0],f'TPR_CI95_high_{stage}':t_ci[1],
                    f'clean_detected_{stage}':clean_count,f'clean_FPR_{stage}':clean_count/n,
                    f'clean_FPR_CI95_low_{stage}':c_ci[0],f'clean_FPR_CI95_high_{stage}':c_ci[1],
                    f'wrong_key_mean_{stage}':float(np.mean(rates)),f'wrong_key_worst_{stage}':max(rates),
                    f'any_wrong_group_rate_{stage}':sum(any(r['p_value']<=alpha for values in wrong.values() for r in values if r['group_index']==g) for g in range(n))/n,
                    f'correct_cosine_median_{stage}':float(np.median([r['cosine'] for r in correct])),
                    f'correct_p_median_{stage}':float(np.median([r['p_value'] for r in correct]))})
                detections[stage]=detected
            row['delta_TPR_percentage_points']=100*(row['TPR_paraphrased']-row['TPR_original'])
            row['TPR_drop_percentage_points']=-row['delta_TPR_percentage_points']
            row['paired_lost_detections']=sum(detections['original'][g] and not detections['paraphrased'][g] for g in range(n))
            row['paired_gained_detections']=sum(not detections['original'][g] and detections['paraphrased'][g] for g in range(n))
            result.append(row)
    return result


def analyze(state,pairs):
    before,after=load_vectors(state,pairs);out=state['out']
    rows,identifications=group_scores(state,pairs,before,after)
    summaries=summarize_comparisons(rows)
    if len(summaries)!=8*len(QUERIES)*len(ALPHAS):raise ValueError('Incomplete owner/budget/alpha comparison table')
    write_csv(out/'scores/group_scores.csv',rows,list(rows[0]))
    write_csv(out/'scores/per_key_before_after.csv',summaries,list(summaries[0]))
    write_csv(out/'scores/closed_set_identification.csv',identifications,list(identifications[0]))
    wrong=[]
    for owner in range(8):
        for q in QUERIES:
            for key in wrong_indices(owner):
                for alpha in ALPHAS:
                    entry={'owner':owner,'wrong_key':key,'Q':q,'alpha':alpha,'groups':COUNT//q}
                    for stage in ['original','paraphrased']:
                        selected=[r for r in rows if r['stage']==stage and r['owner']==owner and r['Q']==q and r['role']=='wrong' and r['test_key']==key]
                        if len(selected)!=COUNT//q:raise ValueError('Wrong-key result coverage changed')
                        entry['detections_'+stage]=sum(r['p_value']<=alpha for r in selected)
                        entry['rate_'+stage]=entry['detections_'+stage]/len(selected)
                    wrong.append(entry)
    write_csv(out/'scores/per_wrong_key_before_after.csv',wrong,list(wrong[0]))
    human=human_review_summary(out)
    quality=read_json(out/'quality/rewrite_summary.json')
    receipt=read_json(out/'provenance/quality_complete.json')
    if receipt['pairs_sha256']!=digest(pairs) or receipt['pairs']!=3600:raise ValueError('Quality population differs from scoring population')
    summary={'status':'E2B_DESCRIPTIVE_PILOT_COMPLETE','complete':True,'responses':3600,'shared_prompts':COUNT,
        'owners':list(range(8)),'clean_responses':COUNT,'model':state['manifest']['protocol']['model'],
        'protocol_variant':state['manifest']['protocol'].get('protocol_variant','historical_SCW'),
        'historical_protocol_equivalent':state['manifest']['protocol'].get('historical_protocol_equivalent',True),
        'queries':{str(q):COUNT//q for q in QUERIES},'alphas':list(ALPHAS),'per_key_before_after':summaries,
        'rewrite_quality':quality,'semantic_quality_review':human,
        'source_E2_concerns':state['manifest']['source_E2_concerns'],
        'scientific_success':None,'decision':'Descriptive robustness evidence only; no new GO/NO_GO or LoRA authorization',
        'limitations':['Groups disjoint within each Q only; budgets and owner/clean comparisons share prompts.',
            'No pooled independent-trial CI. Small group counts limit FPR/TPR precision.',
            'Meaning preservation cannot be established from lexical overlap/length alone.',
            'Original vectors reused from checksummed E2 canonical text-only caches.',
            'Nonempty incomplete API outputs are retained under the copied protocol and reported in quality diagnostics.']}
    write_json(out/'report/summary.json',summary)
    lines=['# KLRD-RS Experiment2B: GPT-5/SCW paraphrase robustness','',
        'Status: E2B_DESCRIPTIVE_PILOT_COMPLETE (no training or automatic scientific GO).',
        f"API model: `{summary['model']}`. 400 shared prompts, 8 owners + clean, 3600 paired responses.",
        f"Paraphrase instruction variant: `{summary['protocol_variant']}`; historical SCW protocol equivalent: {summary['historical_protocol_equivalent']}. Full instruction: provenance/protocol.json.",
        'Original vectors: checksummed E2 text-only cache. Paraphrase scoring: unchanged E2 verifier with original prompts.',
        'TPR drop is original minus paraphrased TPR, in percentage points. Rates in CSV are fractions.','',
        '|Owner|Q|Groups|alpha|Original TPR|Paraphrased TPR|Drop pp|Clean FPR after|Worst wrong after|',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"|{r['owner']}|{r['Q']}|{r['groups']}|{r['alpha']}|{r['TPR_original']:.1%}|{r['TPR_paraphrased']:.1%}|{r['TPR_drop_percentage_points']:.1f}|{r['clean_FPR_paraphrased']:.1%}|{r['wrong_key_worst_paraphrased']:.1%}|")
    lines+=['','## Audit and interpretation','',
        'Full cosine/p-values and decisions are in scores/group_scores.csv. Confidence intervals and paired lost/gained counts are in per_key_before_after.csv. All20 wrong controls per owner are retained.',
        'Clean comparisons reuse the same400 clean responses; they are not8 independent clean samples. Q20/Q50/Q100 partitions overlap across budgets.',
        'Source E2 concerns (retained): '+str(summary['source_E2_concerns']),
        'Human semantic-quality review: '+human['status']+f" ({human['fully_rated_cases']}/{human['cases']} fully rated).",
        'Inspect quality/rewrite_summary.json and the45 preselected pairs in private/blinded_quality_pairs.md. Fill quality/human_reviews.csv and rerun --stage report to incorporate review counts. A reviewed subset still cannot certify all3600 outputs.',
        'See numeric-literal/structure flags and API status counts, not just lexical averages. No paraphrase is filtered or regenerated based on quality/detection.',
        'No finite-sample uniform robustness/FPR guarantee or student/LoRA learning result is established by this experiment.']
    report=out/'report/experiment2b_report.md';report.parent.mkdir(parents=True,exist_ok=True)
    temporary=report.with_suffix('.md.tmp');temporary.write_text('\n'.join(lines)+'\n',encoding='utf-8');temporary.replace(report)
    products=['report/summary.json','report/experiment2b_report.md','scores/group_scores.csv',
        'scores/per_key_before_after.csv','scores/per_wrong_key_before_after.csv','scores/closed_set_identification.csv',
        'quality/per_response_quality.csv','quality/rewrite_summary.json','quality/human_review_summary.json']
    write_json(out/'provenance/result_manifest.json',{'status':summary['status'],'pairs_sha256':digest(pairs),
        'products_sha256':{p:sha256(out/p) for p in products},'no_original_artifacts_modified':True})
    return summary
