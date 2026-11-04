"""Copied KLRD rewrite metrics plus explicit, separate human semantic review."""
from collections import Counter
from pathlib import Path
import csv,re
import numpy as np
from klrd_scw.paraphrase_report import pair_diagnostics
from ..io import write_json,write_csv,read_json,sha256,digest
from .config import CONDITIONS
from .source import immutable_rows


def _atomic_text(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(text,encoding='utf-8');temporary.replace(path)


def _quoted(text):
    runs=re.findall(r'`+',text);fence='`'*max(4,1+max(map(len,runs),default=0))
    return fence+'text\n'+text+'\n'+fence


def write_quality(state,pairs,tokenizer):
    out=state['out'];rows=[];summaries={}
    for pair in pairs:
        before,after=pair['original_text'],pair['paraphrased_text']
        rows.append({'condition':pair['condition'],'prompt_id':pair['prompt_id'],'response_id':pair['response_id'],
            **pair_diagnostics(before,after,tokenizer),
            'numeric_literal_sets_match':set(re.findall(r'\d+(?:[.,]\d+)*',before))==set(re.findall(r'\d+(?:[.,]\d+)*',after)),
            'code_fence_count_changed':before.count('```')!=after.count('```'),
            'api_response_status':pair['api_metadata'].get('api_response_status','unknown')})
    for condition in CONDITIONS:
        values=[r for r in rows if r['condition']==condition];n=len(values)
        if n!=400:raise ValueError('Quality diagnostics require400 pairs per condition')
        ratios=[r['token_length_ratio'] for r in values if r['token_length_ratio'] is not None]
        summaries[condition]={'pairs':n,'token_length_ratio_mean':float(np.mean(ratios)) if ratios else None,
            'token_length_ratio_median':float(np.median(ratios)) if ratios else None,
            'word_jaccard_mean':float(np.mean([r['word_jaccard'] for r in values])),
            'word_sequence_similarity_mean':float(np.mean([r['word_sequence_similarity'] for r in values])),
            'fraction_unchanged_exactly':sum(r['unchanged_exactly'] for r in values)/n,
            'fraction_very_small_edits':sum(r['very_small_edit'] for r in values)/n,
            'fraction_numeric_literal_sets_changed':sum(not r['numeric_literal_sets_match'] for r in values)/n,
            'fraction_code_fence_count_changed':sum(r['code_fence_count_changed'] for r in values)/n,
            'API_status_counts':dict(Counter(r['api_response_status'] for r in values))}
    write_csv(out/'quality/per_response_quality.csv',rows,list(rows[0]))
    write_json(out/'quality/rewrite_summary.json',{'method':'Original copied klrd.paraphrase_report.pair_diagnostics',
        'very_small_edit_definition':'Whitespace-word SequenceMatcher similarity >=0.95, autojunk=False',
        'per_condition':summaries,'scope':'Lexical/length/structure diagnostics, not semantic-equivalence proof',
        'semantic_quality_status':'Requires separate human review; no extra judge API/model is called',
        'numeric_literal_caveat':'Literal formatting changes may be benign; unchanged literals do not prove preserved meaning'})
    cases=[];document=['# E2B blind semantic-quality review','',
        'Five preselected prompt IDs per condition (45 pairs); independent of paraphrase text/detection.',
        'Assess meaning/completeness and whether wording was substantially rewritten. No score/key is shown.',
        'Enter yes/no/uncertain in quality/human_reviews.csv; leave unreviewed cells blank.','']
    for condition in CONDITIONS:
        selected=[p for p in pairs if p['condition']==condition][:5]
        for pair in selected:
            case=digest({'purpose':'E2B_semantic_quality','response_id':pair['response_id']})[:20]
            cases.append({'case_id':case,'condition':condition,'prompt_id':pair['prompt_id'],'response_id':pair['response_id']})
    # Case order also has no score or quality dependence and hides owner blocks.
    cases.sort(key=lambda c:c['case_id']);by_response={p['response_id']:p for p in pairs}
    for case in cases:
        pair=by_response[case['response_id']]
        document += ['## '+case['case_id'],'','Original task:',_quoted(pair['prompt']),
            '','Original response:',_quoted(pair['original_text']),'','Paraphrase:',_quoted(pair['paraphrased_text']),'']
    immutable_rows(out/'private/quality_review_mapping.jsonl',cases)
    _atomic_text(out/'private/blinded_quality_pairs.md','\n'.join(document)+'\n')
    fields=['case_id','meaning_preserved','complete_content','substantially_rewritten','notes']
    template=[dict.fromkeys(fields,'')|{'case_id':c['case_id']} for c in cases]
    write_csv(out/'quality/human_review_template.csv',template,fields)
    review_path=out/'quality/human_reviews.csv'
    if not review_path.exists():write_csv(review_path,template,fields)
    files=['quality/per_response_quality.csv','quality/rewrite_summary.json','quality/human_review_template.csv',
        'private/quality_review_mapping.jsonl','private/blinded_quality_pairs.md']
    write_json(out/'provenance/quality_complete.json',{'pairs':3600,'semantic_review_cases':45,
        'pairs_sha256':digest(pairs),'files_sha256':{p:sha256(out/p) for p in files},
        'manual_ratings_are_editable':'quality/human_reviews.csv; never used for paraphrase selection/regeneration'})
    return summaries


def human_review_summary(out):
    out=Path(out);receipt=read_json(out/'provenance/quality_complete.json')
    for name,expected in receipt['files_sha256'].items():
        if sha256(out/name)!=expected:raise ValueError('Automatic quality artifact changed: '+name)
    from ..io import read_jsonl
    cases=read_jsonl(out/'private/quality_review_mapping.jsonl');known={c['case_id'] for c in cases}
    with (out/'quality/human_reviews.csv').open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=len(known) or len({r['case_id'] for r in rows})!=len(rows) or {r['case_id'] for r in rows}!=known:
        raise ValueError('Human review sheet IDs/coverage changed')
    ratings={};complete=0
    for field in ['meaning_preserved','complete_content','substantially_rewritten']:
        counts=Counter(r[field].strip().lower() for r in rows)
        if any(value not in ('','yes','no','uncertain') for value in counts):raise ValueError('Human ratings must be blank/yes/no/uncertain')
        ratings[field]=dict(counts)
    complete=sum(all(r[k].strip().lower() in ('yes','no','uncertain') for k in ratings) for r in rows)
    result={'status':'REVIEWED_SUBSET' if complete==len(rows) else 'PENDING_HUMAN_REVIEW',
        'cases':len(rows),'fully_rated_cases':complete,'ratings':ratings,
        'scope':'Deterministic45-pair subset only; neither a census nor automatic semantic-preservation certification'}
    write_json(out/'quality/human_review_summary.json',result)
    return result
