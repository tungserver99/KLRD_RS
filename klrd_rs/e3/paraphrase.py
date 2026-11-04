"""Explicit opt-in 400-response student attack using unchanged E2B API transport."""
from importlib.metadata import version
import re
import numpy as np
from ..io import read_json, read_jsonl, write_json, digest, sha256
from ..e2b.source import load_prepared, immutable_rows
from ..e2b.transport import (load_pairs, prepare_condition_resume, requests_for,
    receipts_path, run_copied_transport, checked_results, read_condition)
from ..e2b.scoring import load_vectors
from ..e2b.protocol import PARAPHRASE_PROMPT
from ..e2b.config import SDK_VERSION
from .evaluate import checked_receipt, score_rows, cached_scores, tables, csv, frozen_e2


def run(state):
    evaluation=checked_receipt(state,'provenance/evaluation_complete.json')
    primary=[r for r in evaluation['summary'] if r['model']=='student' and r['Q']==100 and r['alpha']==.01]
    if len(primary)!=1 or primary[0]['detections']<=0:
        raise ValueError('NO_ORIGINAL_STUDENT_SIGNAL: zero primary Q100 detections at frozen alpha=.01; no paraphrase API calls made. This is a necessary signal check, not a new calibrated pass criterion.')
    # All control validation precedes new API spending. Read only already accepted
    # E2B teacher and clean observations; never invoke its nine-condition runner.
    source=load_prepared(state['root'],state['root']/state['cfg'].e2b_run)
    protocol=read_json(source['out']/'provenance/protocol.json')
    if (protocol['model']!='gpt-5-mini-2025-08-07' or protocol['prompt']!=PARAPHRASE_PROMPT
            or protocol.get('protocol_variant')!='high_fidelity_v1'
            or source['manifest']['protocol']!=protocol):
        raise ValueError('Require frozen run002 high-fidelity prompt/model, not brief SCW')
    if not (source['out']/'private/original_paraphrase_pairs.jsonl').is_file():
        raise FileNotFoundError('MISSING_REQUIRED_ARTIFACT: sync historical run002 private/original_paraphrase_pairs.jsonl; E3 never publishes derived files into E2B history')
    old_pairs=load_pairs(source);before,after=load_vectors(source,old_pairs)
    ids=source['manifest']['prompt_ids']; source_e2=frozen_e2(state)
    student=read_jsonl(state['out']/'private/eval_responses.jsonl')
    prompts={r['prompt_id']:r['prompt'] for r in source_e2['prompts']}
    original=[dict(r,prompt=prompts[r['prompt_id']]) for r in student]
    v,t=cached_scores(state,original,'student');index={r['prompt_id']:i for i,r in enumerate(original)}
    if len(ids)!=400 or len(set(ids))!=400 or not set(ids)<=set(index):raise ValueError('Missing fixed400 student IDs')
    selected=[original[index[p]] for p in ids]
    population=[{'condition':'0','prompt_id':r['prompt_id'],'response_id':r['response_id'],
        'prompt':r['prompt'],'original_text':r['response'],'original_text_sha256':digest(r['response']),
        'selection_index':i} for i,r in enumerate(selected)]
    immutable_rows(state['out']/'private/paraphrase_population.jsonl',population)
    write_json(state['out']/'paraphrase/protocol.json',protocol)
    observations={'student_original':(selected,v[[index[p] for p in ids]],t[[index[p] for p in ids]])}
    for condition,label in [('0','teacher'),('clean','clean')]:
        indices=[i for i,r in enumerate(old_pairs) if r['condition']==condition]
        controls=[old_pairs[i] for i in indices]
        if [r['prompt_id'] for r in controls]!=ids or any(r['prompt']!=prompts[r['prompt_id']] for r in controls):
            raise ValueError('E2B control identity/prompt mismatch')
        # Cross-check raw original text AND original scores against authoritative E2.
        e2rows,e2v,e2t=source_e2['observations'][label]
        ei={r['prompt_id']:i for i,r in enumerate(e2rows)}
        for pos,i in enumerate(indices):
            r=old_pairs[i];j=ei[r['prompt_id']]
            if (r['response_id']!=e2rows[j]['response_id'] or r['original_text']!=e2rows[j]['response']
                    or not np.array_equal(before['vectors'][i],e2v[j]) or before['eligible_tokens'][i]!=e2t[j]):
                raise ValueError('E2B original control does not match E2 canonical cached record')
        observations[label+'_original']=([dict(r,response=r['original_text']) for r in controls],before['vectors'][indices],before['eligible_tokens'][indices])
        observations[label+'_paraphrased']=([dict(r,response=r['paraphrased_text']) for r in controls],after['vectors'][indices],after['eligible_tokens'][indices])
    if version('openai')!=SDK_VERSION:raise RuntimeError('Install server requirements-paraphrase.txt; frozen SDK '+SDK_VERSION+' required')
    attack=dict(state,population=population,manifest={'protocol':protocol})
    requests=requests_for(population,'0')
    resume=prepare_condition_resume(attack,'0',requests)
    write_json(state['out']/'paraphrase/api_resume.json',resume)
    # Exactly400 new student requests. run.lock is held by main_exp3; accepted
    # outputs remain immutable across retries/preemption under copied transport.
    try:
        returned=run_copied_transport(requests,receipts_path(attack,'0'),prompt=protocol['prompt'],
            workers=state.get('api_workers',state['cfg'].workers),max_attempts=5)
        checked_results(requests,returned,protocol['prompt'])
    except Exception:
        raise RuntimeError('Student paraphrase incomplete; durable API receipts retained. Retry explicit paraphrase mode; accepted outputs are preserved.') from None
    returned=read_condition(attack,'0')
    pairs=[dict(r,paraphrased_text=a['paraphrased_text'],api_metadata={k:x for k,x in a.items()
        if k.startswith('api_') or k in ('status','attempts','paraphraser_model')}) for r,a in zip(population,returned)]
    immutable_rows(state['out']/'private/student_paraphrase_pairs.jsonl',pairs)
    texts=[dict(r,response=r['paraphrased_text']) for r in pairs]
    pv,pt=score_rows(state,texts,'student_paraphrased400',source_e2['environment'])
    observations['student_paraphrased']=(texts,pv,pt)
    summaries=tables(state,observations,source['manifest']['groups'],state['out']/'paraphrase/scores')
    comparisons=[]
    for model in ['student','teacher','clean']:
        for q in [20,50,100]:
            for alpha in [.01,.0001]:
                a=next(r for r in summaries if r['model']==model+'_original' and r['Q']==q and r['alpha']==alpha)
                b=next(r for r in summaries if r['model']==model+'_paraphrased' and r['Q']==q and r['alpha']==alpha)
                comparisons.append({'model':model,'Q':q,'alpha':alpha,'groups':a['groups'],
                    'rate_kind':'clean_FPR' if model=='clean' else 'owner_TPR',
                    'original_rate':a['rate'],'paraphrased_rate':b['rate'],'drop_pp':100*(a['rate']-b['rate']),
                    'wrong_worst_original':a['wrong_worst'],'wrong_worst_paraphrased':b['wrong_worst']})
    csv(state['out']/'paraphrase/metrics.csv',comparisons)
    from transformers import AutoTokenizer
    from klrd_scw.paraphrase_report import pair_diagnostics
    tokenizer=AutoTokenizer.from_pretrained(state['cfg'].model,revision=state['cfg'].revision,
        trust_remote_code=False,local_files_only=True)
    if (tokenizer.chat_template!=source_e2['environment']['generation'].get('chat_template',tokenizer.chat_template)
            or digest(tokenizer.get_vocab())!=source_e2['environment']['generation']['tokenizer_vocab_sha256']):
        raise ValueError('Quality tokenizer differs from frozen source')
    diagnostics=[]
    for p in pairs:
        a,b=p['original_text'],p['paraphrased_text']
        diagnostics.append({'response_hash':digest(p['response_id']),**pair_diagnostics(a,b,tokenizer),
            'numeric_literal_sets_match':set(re.findall(r'\d+(?:[.,]\d+)*',a))==set(re.findall(r'\d+(?:[.,]\d+)*',b)),
            'code_fence_count_changed':a.count('```')!=b.count('```'),
            'api_response_status':p['api_metadata'].get('api_response_status','unknown')})
    csv(state['out']/'paraphrase/quality.csv',diagnostics)
    write_json(state['out']/'paraphrase/quality_summary.json',{'pairs':400,
        'word_jaccard_mean':float(np.mean([r['word_jaccard'] for r in diagnostics])),
        'word_sequence_similarity_mean':float(np.mean([r['word_sequence_similarity'] for r in diagnostics])),
        'token_length_ratio_mean':float(np.mean([r['token_length_ratio'] for r in diagnostics if r['token_length_ratio'] is not None])) if any(r['token_length_ratio'] is not None for r in diagnostics) else None,
        'numeric_change_flags':sum(not r['numeric_literal_sets_match'] for r in diagnostics),
        'structure_change_flags':sum(r['code_fence_count_changed'] for r in diagnostics),
        'semantic_quality_status':'PENDING_HUMAN_REVIEW; lexical diagnostics are not semantic proof',
        'selection_policy':'All400 retained; no quality/score filtering or resampling'})
    from ..e2b.quality import _atomic_text, _quoted
    blind=['# Private blinded student paraphrase review','',
        'First five frozen selected IDs; assess meaning/completeness. No detection score shown.','']
    for p in pairs[:5]:
        blind+=['## '+digest(p['response_id'])[:16],_quoted(p['prompt']),_quoted(p['original_text']),_quoted(p['paraphrased_text'])]
    _atomic_text(state['out']/'private/blinded_paraphrase_review.md','\n'.join(blind)+'\n')
    products=['paraphrase/metrics.csv','paraphrase/quality.csv','paraphrase/quality_summary.json',
        'private/student_paraphrase_pairs.jsonl','paraphrase/protocol.json',
        'provenance/student_paraphrased400_scores_complete.json']+[
        'paraphrase/scores/'+name for name in ['per_key_group.csv','per_response.csv','closed_set_identification.csv','clean_wrong_controls.csv','group_summary.csv']]
    receipt={'identity':state['identity'],'status':'PARAPHRASE_COMPLETE','responses':400,
        'model':protocol['model'],'control_source':state['cfg'].e2b_run,'new_control_API_calls':0,
        'summary':summaries,'pairs_sha256':digest(pairs),
        'products_sha256':{p:sha256(state['out']/p) for p in products}}
    write_json(state['out']/'paraphrase/complete.json',receipt)
    return receipt
