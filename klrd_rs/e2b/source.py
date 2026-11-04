"""Freeze selected E2 observations once; resume from the small local E2B snapshot."""
from pathlib import Path
import numpy as np
from klrd_scw.artifacts import ensure_manifest
from klrd_scw.scw_paraphrase import _write_jsonl
from ..io import read_json,read_jsonl,sha256,digest,save_npz
from .config import CONDITIONS,COUNT,SOURCE_RUN,OUTPUT_RUN,SELECTION_DOMAIN,select_prompt_ids,make_groups


def resolve_output(root,value):
    root=Path(root).resolve();allowed=(root/'experiment2_paraphrase').resolve()
    out=Path(value).resolve() if value is not None else root/OUTPUT_RUN
    if out==allowed or allowed not in out.parents:
        raise ValueError('E2B output must be a distinct run under experiment2_paraphrase/')
    return out


def immutable_rows(path,rows):
    path=Path(path)
    if path.exists():
        if read_jsonl(path)!=rows:raise ValueError('Immutable E2B rows changed: '+str(path))
    else:
        path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        _write_jsonl(path,rows)


def protocol_receipt(root):
    receipt=read_json(Path(root)/'docs/superpowers/references/e2b_scw_protocol.json')
    from klrd_scw.scw_paraphrase import SCW_MODEL,SCW_COMMIT,PARAPHRASE_PROMPT
    if (receipt['model']!=SCW_MODEL or receipt['scw_commit']!=SCW_COMMIT
        or receipt['prompt']!=PARAPHRASE_PROMPT or receipt['request_fields']!=['model','input']):
        raise ValueError('Copied SCW protocol differs from the audited original')
    for item in receipt['copied_files']:
        if sha256(Path(root)/item['copy'])!=item['sha256']:
            raise ValueError('Byte-for-byte SCW copy changed: '+item['copy'])
    from .protocol import PARAPHRASE_PROMPT as active_prompt,PROTOCOL_VARIANT
    return dict(receipt,prompt=active_prompt,historical_prompt=receipt['prompt'],
        protocol_variant=PROTOCOL_VARIANT,historical_protocol_equivalent=False,
        modification='User-requested high-fidelity instruction; same model and copied API transport')


def load_verifier_inputs(root):
    """Load only the frozen geometry/key bank needed by verification/reporting.

    No E1/E2 directory scan or historical source-code audit. The two geometry
    files are bound to the existing E1 receipt; the verifier itself is unchanged.
    Paraphrase-only runs never call this loader.
    """
    from ..e2.anchor import INPUT_FOLDER,RECEIPT_SHA256
    from ..e2.config import E2Config
    from ..math import keys,key_ids
    root=Path(root);cfg=E2Config();e1=root/'experiment1_rs_continuation'
    receipt_path=root/INPUT_FOLDER/'source_receipt.json'
    if sha256(receipt_path)!=RECEIPT_SHA256:raise ValueError('Frozen E1 receipt changed')
    receipt=read_json(receipt_path)
    for name in ['geometry/basis.npz','geometry/whitening.npz']:
        path=e1/name;relative=path.relative_to(root).as_posix()
        if sha256(path)!=receipt['files'][relative]['sha256']:
            raise ValueError('Frozen verifier geometry changed: '+name)
    with np.load(e1/'geometry/basis.npz',allow_pickle=False) as f:p,q=f['P'].copy(),f['Q'].copy()
    with np.load(e1/'geometry/whitening.npz',allow_pickle=False) as f:l=f['L'].copy()
    bank=keys(64,cfg.m,cfg.key_seed)
    expected=read_json(e1/'keys/validation_key_ids.json')
    if expected['seed']!=cfg.key_seed or key_ids(bank)!=expected['keys']:
        raise ValueError('Frozen verifier key bank changed')
    meta=read_json(root/'inputs/calibration_manifest.json')
    return {'P':p,'Q':q,'L':l,'bank':bank,'chat_template':meta['chat_template']}

def prepare(root,out):
    """Never write into run_012, E1 or original KLRD; no model/API calls."""
    from ..e2.config import E2Config
    root=Path(root).resolve();out=resolve_output(root,out);source=root/SOURCE_RUN
    protocol=protocol_receipt(root)
    print('E2B prepare: reading selected E2 jobs and original score caches once.',flush=True)
    manifest=read_json(source/'provenance/prompt_manifest.json')
    identity=read_json(source/'private/jobs/identity.json')
    environment=read_json(source/'provenance/engine_manifest.json')
    summary=read_json(source/'report/summary.json')
    cfg=E2Config()
    if identity['config']!=cfg.resolved() or manifest['E2_config']!=cfg.resolved():
        raise ValueError('Source E2 scientific/runtime configuration differs')
    receipt_sha=identity['source_receipt_sha256']
    from ..e2.anchor import RECEIPT_SHA256
    if receipt_sha!=RECEIPT_SHA256 or manifest['input_receipt_sha256']!=receipt_sha:
        raise ValueError('Source E2 and E2B do not share the frozen E1 receipt')
    design=digest({'cohort':manifest,'config':cfg.resolved(),'source':receipt_sha})
    if identity['design_sha256']!=design:raise ValueError('Frozen E2 design digest mismatch')
    expected_readout=read_json(root/'experiment1_rs_continuation/numerical/fp64_diagnostic.json')['readout_parameter_sha256']
    if (environment['readout_parameter_sha256']!=expected_readout
        or identity['environment']['readout_parameter_sha256']!=expected_readout
        or environment['generation']!=identity['environment']['generation']
        or environment.get('verification_linear_accumulation')!='float64'
        or environment.get('verification_attention_implementation')!='eager'):
        raise ValueError('Source E2 verifier/readout runtime receipt differs from its frozen identity')
    if summary.get('sampler_valid') is not True:
        raise ValueError('Source E2 does not declare a valid frozen runtime configuration')
    if any(summary['committed_generation_counts'].get(c,0)<1000 for c in CONDITIONS):
        raise ValueError('Complete1000-response pilot population required for all nine conditions')
    groups=[g for g in manifest['groups'] if g['Q']==1000 and g['group_index']==0]
    if len(groups)!=1:raise ValueError('Missing or duplicated frozen E2 group0')
    candidates=groups[0]['prompt_ids'];ids=select_prompt_ids(candidates);selected=set(ids)
    prompt_path=source/'private/prompts.jsonl'
    prompts=read_jsonl(prompt_path);lookup={r['prompt_id']:r for r in prompts}
    if len(lookup)!=len(prompts) or any(pid not in lookup for pid in candidates):
        raise ValueError('Frozen E2 prompt population missing or duplicated')
    population=[];vectors=[];eligible=[]
    for condition in CONDITIONS:
        print(f'E2B prepare: condition {condition}, collecting400 originals and cached vectors.',flush=True)
        records_by_id={};vectors_by_id={};tokens_by_id={}
        for start in range(0,len(candidates),cfg.batch_size):
            chunk=start//cfg.batch_size;batch_ids=candidates[start:start+cfg.batch_size]
            if not selected.intersection(batch_ids):continue
            name=f'primary_owner{condition}_g00_chunk{chunk:04}'
            checkpoint=read_json(source/'private/jobs'/(name+'.json'))
            expected=dict(owner=condition,group=0,chunk=chunk,prompt_ids=batch_ids,phase='primary',frozen_run_sha256=design)
            if checkpoint['identity']!=expected:raise ValueError('E2 job identity mismatch: '+name)
            records=checkpoint['records'];record_hash=digest(records)
            if record_hash!=checkpoint['records_sha256'] or [r['prompt_id'] for r in records]!=batch_ids:
                raise ValueError('E2 generation checksum/order mismatch: '+name)
            cache=source/'private/scores'/(name+'.npz')
            marker=read_json(cache.with_suffix('.sha256.json'))
            if marker['sha256']!=sha256(cache) or marker['records_sha256']!=record_hash or marker.get('primary_path')!='canonical_visible_text_only':
                raise ValueError('E2 canonical score checksum/record binding mismatch: '+name)
            with np.load(cache,allow_pickle=False) as loaded:
                v=loaded['vectors'].copy();t=loaded['eligible_tokens'].copy();cache_ids=loaded['prompt_ids'].tolist()
            if cache_ids!=batch_ids or v.shape!=(len(records),cfg.m) or t.shape!=(len(records),) or not np.isfinite(v).all() or (t<0).any():
                raise ValueError('Invalid E2 score arrays: '+name)
            for index,record in enumerate(records):
                pid=record['prompt_id']
                if pid not in selected:continue
                if (pid in records_by_id or record.get('owner_key_id')!=condition or record.get('phase')!='primary'
                    or record.get('group_index')!=0 or record.get('chunk_index')!=chunk or not isinstance(record.get('response'),str)):
                    raise ValueError('E2 response metadata mismatch: '+name)
                response_id=digest({'owner':condition,'prompt':pid,'seed':record['seed'],'tokens':record['generated_token_ids']})
                if record['response_id']!=response_id:raise ValueError('E2 response ID mismatch: '+name)
                records_by_id[pid]=(record,name);vectors_by_id[pid]=v[index];tokens_by_id[pid]=int(t[index])
        if set(records_by_id)!=selected:raise ValueError('Missing selected E2 observations: '+condition)
        for index,pid in enumerate(ids):
            record,name=records_by_id[pid]
            population.append({'condition':condition,'prompt_id':pid,'response_id':record['response_id'],
                'prompt':lookup[pid]['prompt'],'original_text':record['response'],
                'original_text_sha256':digest(record['response']),'source_job':name,'selection_index':index})
            vectors.append(vectors_by_id[pid]);eligible.append(tokens_by_id[pid])
    if len(population)!=3600 or len({r['response_id'] for r in population})!=3600:
        raise ValueError('E2B requires exactly3600 uniquely identified original responses')
    private=out/'private';private.mkdir(parents=True,exist_ok=True,mode=0o700)
    if out not in private.resolve().parents:raise ValueError('Private output resolves outside the E2B run')
    original_path=private/'original_vectors.npz'
    expected_arrays={'vectors':np.asarray(vectors,dtype=np.float64),'eligible_tokens':np.array(eligible,dtype=np.int64),
        'response_ids':np.array([r['response_id'] for r in population])}
    if original_path.exists():
        with np.load(original_path,allow_pickle=False) as existing:
            if set(existing.files)!=set(expected_arrays) or any(not np.array_equal(existing[k],v) for k,v in expected_arrays.items()):
                raise ValueError('Frozen E2B original vectors differ from their E2 source')
    else:save_npz(original_path,**expected_arrays)
    immutable_rows(private/'population.jsonl',population)
    prepared={'schema':1,'experiment':'KLRD_RS_E2B_GPT5_SCW_PARAPHRASE','source_run':SOURCE_RUN,
        'integrity_policy':'Selected job/text/vector binding at preparation; local E2B population on resume; no full E1/E2 scan',
        'protocol':protocol,'conditions':list(CONDITIONS),'selection_domain':SELECTION_DOMAIN,
        'selection_rule':'SHA256(domain + NUL + prompt_id), first400 of frozen group0; no text/score-dependent selection',
        'prompt_ids':ids,'groups':make_groups(ids),'population_sha256':digest(population),
        'original_vectors_sha256':sha256(original_path),'source_E2_environment':environment,
        'source_E2_concerns':summary.get('concerns',{}),'E1_receipt_sha256':receipt_sha,
        'scope':'Paired descriptive pilot; unchanged spherical verifier and thresholds; no new training or GO rule'}
    ensure_manifest(out/'provenance/population_manifest.json',prepared)
    ensure_manifest(out/'provenance/protocol.json',protocol)
    ensure_manifest(out/'provenance/prepare_complete.json',{'population_manifest_sha256':sha256(out/'provenance/population_manifest.json'),
        'population_sha256':digest(population),'original_vectors_sha256':sha256(original_path)})
    print('E2B prepared:400 shared prompts x9 conditions;3600 existing responses, no teacher generation.',flush=True)
    return {'root':root,'out':out,'manifest':prepared,'population':population,'frozen':None}


def load_prepared(root,out):
    root=Path(root).resolve();out=resolve_output(root,out)
    protocol_receipt(root) # Only the small vendored protocol files, never E1/E2 trees.
    manifest=read_json(out/'provenance/population_manifest.json')
    receipt=read_json(out/'provenance/prepare_complete.json')
    if receipt['population_manifest_sha256']!=sha256(out/'provenance/population_manifest.json'):
        raise ValueError('Prepared population manifest checksum changed')
    if manifest['source_run']!=SOURCE_RUN or manifest['conditions']!=list(CONDITIONS):
        raise ValueError('Prepared E2B source or conditions changed')
    ids=manifest['prompt_ids']
    if (len(ids)!=COUNT or len(set(ids))!=COUNT
        or manifest['selection_domain']!=SELECTION_DOMAIN or manifest['groups']!=make_groups(ids)):
        raise ValueError('Prepared deterministic selection/groups changed')
    # Old manifests may retain source/code hashes as historical provenance.
    # Do not recheck them: logging-only code updates must not invalidate paid API receipts.
    population=read_jsonl(out/'private/population.jsonl')
    expected_order=[(condition,pid) for condition in CONDITIONS for pid in manifest['prompt_ids']]
    if (len(population)!=3600 or digest(population)!=manifest['population_sha256']
        or len({r['response_id'] for r in population})!=3600
        or [(r['condition'],r['prompt_id']) for r in population]!=expected_order):
        raise ValueError('Prepared E2B population changed')
    if sha256(out/'private/original_vectors.npz')!=manifest['original_vectors_sha256']:
        raise ValueError('Prepared original vector cache changed')
    return {'root':root,'out':out,'manifest':manifest,'population':population,'frozen':None}
