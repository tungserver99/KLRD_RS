"""New full-run identities, read-only pilot access and phase-bound artifacts."""
from pathlib import Path
from dataclasses import replace
from ..e3.config import E3Config
from ..e3.provenance import initialize_run,bind_inputs
from ..e2b.source import immutable_rows
from ..io import read_json,read_jsonl,write_json,sha256,digest
from klrd_scw.artifacts import ensure_manifest


def project_path(root,value):
    value=Path(value)
    return (value if value.is_absolute() else Path(root)/value).resolve()


def readonly_run(root,value):
    """Inspect an existing run without updating its environment/source/report."""
    root=Path(root).resolve();out=project_path(root,value)
    raw=read_json(out/'provenance/resolved_config.json')
    cfg=E3Config(**{name:tuple(raw[name]) if name in ('alphas','target_modules') else raw[name]
        for name in E3Config.__dataclass_fields__ if name in raw})
    manifest=read_json(out/'provenance/input_manifest.json')
    if manifest['config']!=cfg.resolved():raise ValueError('Saved pilot scientific configuration changed')
    if manifest['identity']!=digest({'config':manifest['config'],'inputs':manifest['inputs']}):
        raise ValueError('Saved pilot run identity is inconsistent')
    state={'root':root,'out':out,'cfg':cfg,'identity':manifest['identity']}
    if bind_inputs(state)!=manifest['inputs']:raise ValueError('Saved pilot frozen input manifests changed')
    return state


def initialize(root,out=None,pilot_path='experiment3_rs/run_001',legacy_path=None,mode='train',workers=128):
    root=Path(root).resolve();out=project_path(root,out or 'experiment3_full/run_001')
    base=(root/'experiment3_full').resolve()
    if out==base or base not in out.parents:raise ValueError('Full study output must be a new run below experiment3_full/')
    pilot=readonly_run(root,pilot_path)
    if pilot['out']==out:raise ValueError('Full study must never overwrite the pilot')
    cfg=replace(pilot['cfg'],profile='full',max_new_tokens=512,
        legacy_root=legacy_path or pilot['cfg'].legacy_root)
    if cfg.train_count!=10000 or cfg.validation_count!=1000:raise ValueError('Full sizes must be10000/1000')
    state={'root':root,'out':out,'cfg':cfg,'requested_profile':'full','mode':mode,
        'pilot':pilot,'api_workers':workers}
    initialize_run(state,create=mode=='train')
    fields={
        'schema':1,'training_identity':state['identity'],
        'pilot_path':str(pilot['out'].relative_to(root)) if root in pilot['out'].parents else str(pilot['out']),
        'pilot_identity':pilot['identity'],
        'pilot_selection_sha256':sha256(pilot['out']/'private/full_selection.json'),
        'pilot_checkpoint_selection_sha256':sha256(pilot['out']/'training/selection.json'),
        'training_initialization':'clean pinned base + new adapter, never pilot weights',
        'training_max_new_tokens':512,'evaluation_lengths':[512,1024],
        'evaluation_prompts':'same1000 frozen E2 group0 IDs/order',
        'paraphrase_prompts':'same400 frozen E2B run002 IDs/order; new1024 texts for all3conditions',
        'paraphrase_conditions':['clean','teacher','student'],
        'arc_harness_version':'0.4.13','arc_seed':20261021,
        'arc_metric':'acc_norm,none','arc_precision':'float32 matched all4models; historical BF16 difference disclosed',
        'shell_orchestration':'user-authorized explicit train/eval512/eval1024/arc/paraphrase/report commands'}
    receipt=root/'klrd_arc/SOURCE_RECEIPT.json'
    if not receipt.is_file():raise FileNotFoundError('Sync the vendored ARC code/receipt: '+str(receipt))
    fields['arc_source_copy_receipt_sha256']=sha256(receipt)
    ensure_manifest(out/'provenance/full_protocol.json',fields)
    state['full_plan']=fields
    return state


def phase_state(state,length):
    if length not in (512,1024):raise ValueError('Only predeclared evaluation lengths512/1024 are allowed')
    out=state['out']/f'eval{length}'
    merged=read_json(state['out']/'provenance/merged_manifest.json')
    if merged['identity']!=state['identity']:raise ValueError('Full merged checkpoint identity mismatch')
    body={'training_identity':state['identity'],'length':length,
        'merged_manifest_sha256':sha256(state['out']/'provenance/merged_manifest.json'),
        'E2_cohort_sha256':sha256(state['root']/state['cfg'].e2_run/'provenance/prompt_manifest.json'),
        'sampling_seed_policy':'same per-model512/1024 seeds; clean/teacher preserve E2 seeds; student preserves E3 seed'}
    identity=digest(body);plan=dict(body,identity=identity)
    path=out/'provenance/evaluation_plan.json'
    if state.get('mode')=='report':
        if not path.exists():raise FileNotFoundError('Evaluation phase not started: '+str(path))
        if read_json(path)!=plan:raise ValueError('Evaluation plan changed')
    else:ensure_manifest(path,plan)
    result=dict(state,out=out,cfg=replace(state['cfg'],max_new_tokens=length),identity=identity,
        parent_out=state['out'],training_identity=state['identity'],root_state=state)
    if state.get('verified_merged_file_manifest'):
        result['verified_merged_file_manifest']=state['verified_merged_file_manifest']
    return result


def checked(state,relative):
    path=state['out']/relative
    if not path.exists():return None
    value=read_json(path)
    if value.get('identity')!=state['identity']:raise ValueError('Completed phase identity changed: '+relative)
    for name,expected in value['products_sha256'].items():
        target=(state['out']/name).resolve()
        if state['out'].resolve() not in target.parents or not target.is_file() or sha256(target)!=expected:
            raise ValueError('Completed phase artifact changed/missing: '+name)
    return value


def complete(state,relative,products,**fields):
    hashes={}
    for name in products:
        path=Path(name)
        if path.is_absolute():path=path.relative_to(state['out'])
        target=(state['out']/path).resolve()
        if state['out'].resolve() not in target.parents:raise ValueError('Phase product outside output run')
        hashes[path.as_posix()]=sha256(target)
    value=dict(fields,identity=state['identity'],products_sha256=hashes)
    write_json(state['out']/relative,value)
    return value


def prepare_training(state):
    """Reuse pilot clean contexts, never pilot LoRA weights or eval responses."""
    from ..e3 import data
    out=state['out'];pilot=state['pilot']
    selection=out/'private/full_selection.json'
    receipt_path=out/'provenance/data_selection.json'
    reuse_path=out/'provenance/pilot_clean_reuse.json'
    # Reconstruct the same immutable bundle after a crash between its writes.
    # Never treat a selection file alone as a committed three-file transaction.
    if not all(path.exists() for path in (selection,receipt_path,reuse_path)):
        source=pilot['out']/'private/full_selection.json'
        receipt=read_json(pilot['out']/'provenance/data_selection.json')
        if receipt['identity']!=pilot['identity'] or receipt['selection_sha256']!=sha256(source):
            raise ValueError('Pilot full prompt partitions changed')
        selected=read_json(source)
        if {s:len(v) for s,v in selected.items()}!={'train':10000,'validation':1000}:
            raise ValueError('Pilot does not contain the predeclared10000/1000 full prompt partitions')
        history,sources=data._exclusions(state)
        if any(data._overlap(row,history) for rows in selected.values() for row in rows):
            raise ValueError('Full frozen training cohort overlaps held-out data')
        all_ids=[r['prompt_id'] for rows in selected.values() for r in rows]
        all_texts=[data.text_hash(r['prompt'],True) for rows in selected.values() for r in rows]
        if len(set(all_ids))!=11000 or len(set(all_texts))!=11000:
            raise ValueError('Full train/validation prompt IDs/texts overlap')
        evidence=read_json(pilot['out']/'provenance/data_complete.json')
        if evidence['identity']!=pilot['identity']:raise ValueError('Pilot clean data identity changed')
        reused={};source_hashes={}
        for split,filename,count in [('train','train_manifest.jsonl',2000),('validation','val_manifest.jsonl',200)]:
            path=pilot['out']/'private'/filename;rows=read_jsonl(path)
            if len(rows)!=count or digest(rows)!=evidence['rows_sha256'][split]:
                raise ValueError('Pilot clean training corpus changed: '+split)
            if any(not isinstance(r.get('response'),str) or r.get('source')!='train_gen'
                or r.get('split')!=split or r['prompt_id']!=p['prompt_id'] or r['prompt']!=p['prompt']
                for r,p in zip(rows,selected[split][:count])):
                raise ValueError('Pilot clean corpus does not match the fixed full-prefix partition')
            authority=pilot['out']/'private/data'/f'{split}_clean.identity.json'
            if authority.exists():
                settings=read_json(authority)['settings']
                for key,value in {'model':state['cfg'].model,'revision':state['cfg'].revision,
                    'dtype':'float32','temperature':.8,'max_new_tokens':512,'kind':'clean_train'}.items():
                    if settings.get(key)!=value:raise ValueError('Pilot clean generation provenance mismatch: '+key)
            elif any(not r.get('legacy_file_sha256') for r in rows):
                raise ValueError('Missing clean-generation provenance for pilot corpus: '+split)
            file_hash=sha256(path);source_hashes[split]=file_hash
            for index,row in enumerate(rows):
                selected[split][index]['legacy_response']=dict(row,
                    legacy_path=str(path),legacy_file_sha256=file_hash)
            reused[split]=count
        ensure_manifest(selection,selected)
        current=dict(receipt,identity=state['identity'],selection_sha256=sha256(selection),
            heldout_sources=sources,pilot_clean_reused=reused,pilot_clean_source_sha256=source_hashes,
            training_initialization='clean base + new LoRA, not pilot adapter')
        ensure_manifest(receipt_path,current)
        ensure_manifest(reuse_path,{'identity':state['identity'],
            'reused':reused,'source_sha256':source_hashes,'new_clean_needed':{'train':8000,'validation':800},
            'policy':'Reuse clean train_gen contexts only; no student/eval response training'})
    return data.prepare(state)
