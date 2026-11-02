"""Optional historical eight-sign diagnostic. Never spherical legacy p-values."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from .io import read_jsonl, write_csv
from .math import sign_p
from .replay import response_scores


def run_legacy(root,inputs,adapter,cfg,store,out):
    root=Path(root)
    required=['private/legacy_key.pt','legacy_fingerprint.jsonl','legacy_scores.jsonl']
    if not all((root/name).is_file() for name in required):
        return {'status':'SKIPPED','reason':'Missing exact historical key or matched fingerprint/LLR records'}
    state=torch.load(root/'private/legacy_key.pt',map_location='cpu',weights_only=True)
    meta=state['metadata']
    if meta.get('model_revision')!=cfg.revision or meta.get('tau')!=cfg.tau:
        return {'status':'SKIPPED','reason':'Historical key/model identity mismatch'}
    key=state['key'];p=key['P'].double().numpy();q=key['Q'].double().numpy();signs=key['signs'].double().numpy()
    if p.shape[1]!=8 or q.shape!=p.shape or signs.shape!=(8,):
        return {'status':'SKIPPED','reason':'Legacy key does not have eight planes'}
    frame=np.concatenate([p,q],1)
    if not np.allclose(frame.T@frame,np.eye(16),atol=1e-6):
        return {'status':'SKIPPED','reason':'Legacy plane geometry invalid'}
    rows=read_jsonl(root/'legacy_fingerprint.jsonl');raw=read_jsonl(root/'legacy_scores.jsonl')
    scored={r['prompt_id']:r for r in raw}
    if len(scored)!=len(raw) or len(rows)!=len({r['prompt_id'] for r in rows}):
        return {'status':'SKIPPED','reason':'Duplicate legacy response IDs'}
    if {r['prompt_id'] for r in rows}!=set(scored) or any(
        r['response']!=scored[r['prompt_id']]['response'] or r['prompt']!=scored[r['prompt_id']]['prompt'] for r in rows):
        return {'status':'SKIPPED','reason':'Legacy raw LLR text/prompt matching failed'}
    engine=adapter.readout(p,q)
    vectors=response_scores(rows,adapter,engine,cfg,store,'legacy_original_phi')
    llr=np.array([scored[r['prompt_id']]['keys']['correct']['raw_llr'] for r in rows])
    linear=vectors@signs
    conditional=bool(inputs['index'].get('legacy_plane_sign_independence'))
    per_response=[]
    for i,r in enumerate(rows):
        per_response.append({'prompt_id':r['prompt_id'],'linear_tangent':linear[i],
            'historical_raw_llr':llr[i],'exact_sign_p':sign_p(vectors[i],signs) if conditional else None})
    write_csv(out/'legacy/per_response.csv',per_response,list(per_response[0]))
    populations={'original':{'responses':len(rows),'linear_score_sum':float(linear.sum()),
        'exact_sign_p_aggregate':sign_p(vectors.sum(0),signs) if conditional else None}}
    para_path=root/'paraphrases.jsonl'
    if para_path.exists():
        para={r['prompt_id']:r for r in read_jsonl(para_path) if r['split']=='fingerprint'}
        if len(para)==len(rows) and all(r['prompt_id'] in para and para[r['prompt_id']]['source_text']==r['response']
                and para[r['prompt_id']].get('status') in ('ok','success','completed') for r in rows):
            transformed=[dict(r,response=para[r['prompt_id']]['paraphrased_text']) for r in rows]
            pv=response_scores(transformed,adapter,engine,cfg,store,'legacy_paraphrase_phi')
            populations['paraphrase']={'responses':len(transformed),'linear_score_sum':float((pv@signs).sum()),
                'exact_sign_p_aggregate':sign_p(pv.sum(0),signs) if conditional else None}
        else:populations['paraphrase']={'status':'SKIPPED','reason':'Unmatched historical paraphrases'}
    correlations={}
    if np.std(linear)>0 and np.std(llr)>0:
        correlations={'Pearson':float(pearsonr(linear,llr)[0]),'Spearman':float(spearmanr(linear,llr)[0])}
    return {'status':'HISTORICAL_DIAGNOSTIC_ONLY','matching_raw_llr_correlations':correlations,
        'populations':populations,'plane_sign_independence_recorded':conditional,
        'sign_randomization_assignments':256,'minimum_possible_p':1/256,
        'scope':'Conditional sign-randomization given independent frozen legacy planes; not new-method TPR; cannot support alpha=1e-4',
        'theta_positive':float(key['theta'])>0}
