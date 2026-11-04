"""Frozen train_gen partitions and held-out-safe clean completion contexts."""
from pathlib import Path
import json
import os
from ..io import read_json,read_jsonl,write_json,digest,sha256
from ..e2.cohort import text_hash
from ..e2.exclusions import UsedHistoryIndex,source_aliases,legacy_text_hash
from .provenance import get_frozen


def _write_rows(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def load_eval_prompts(state):
    """Exactly the original frozen E2 group0 population, in frozen group order."""
    source=state['root']/state['cfg'].e2_run
    manifest=read_json(source/'provenance/prompt_manifest.json')
    if sha256(source/'private/prompts.jsonl')!=manifest['private_prompts_sha256']:
        raise ValueError('Frozen E2 private prompt text checksum differs from authority')
    groups=[g for g in manifest['groups'] if g['Q']==1000 and g['group_index']==0]
    if len(groups)!=1:raise ValueError('Missing unique frozen E2 group0')
    ids=[str(p) for p in groups[0]['prompt_ids']]
    rows=read_jsonl(source/'private/prompts.jsonl');lookup={str(r['prompt_id']):r for r in rows}
    if len(ids)!=1000 or len(set(ids))!=1000 or len(lookup)!=len(rows) or not set(ids)<=lookup.keys():
        raise ValueError('Frozen E2 group0 must contain 1000 unique original private prompts')
    return [dict(lookup[p],prompt_id=p) for p in ids]


HELD_OUT_ROLES=frozenset(('null','final','test','heldout','evaluation','eval',
    'geometry_eval','nat_eval','tilt_dev','final_id','stress_dev','stage2b_fresh_id',
    'implant_eval','implant_dev'))


def _heldout_rows(value,forced=False):
    """Select consumed/evaluation roles without ingesting train/reserve inventories."""
    if isinstance(value,list):
        return [r for item in value for r in _heldout_rows(item,forced)]
    if not isinstance(value,dict):
        return [{'prompt_id':str(value)}] if forced and isinstance(value,(str,int)) else []
    if any(name in value for name in ('prompt','prompt_text','instruction','prompt_id','source_id','replaced_prompt_id','replacement_prompt_id')):
        return [value] if forced or value.get('split',value.get('role')) in HELD_OUT_ROLES else []
    result=[]
    for name,child in value.items():
        if name in HELD_OUT_ROLES:result.extend(_heldout_rows(child,True))
        elif name in ('rows','prompts','splits','selected','cohorts'):
            result.extend(_heldout_rows(child,forced))
    return result


def historical_heldout(index,legacy):
    """Known historical artifact layouts, with explicit absence/coverage reporting.

    Enumerate family/run directories only, never every candidate receipt/cache.
    Unknown layouts are recorded as a limitation, not claimed fully audited.
    """
    legacy=Path(legacy);artifacts=legacy/'artifacts';sources={};audit=[]
    if not artifacts.is_dir():
        return sources,[{'path':str(artifacts),'status':'MISSING_OPTIONAL_HISTORY',
            'limitation':'Historical artifact cohorts not available; sync them for broader exclusion coverage'}]
    names=('canonical_pairs.jsonl','data/prompts.jsonl','data/initial_prompts.jsonl',
        'data/implant_eval_prompts.jsonl','data/initial_implant_eval_prompts.jsonl',
        'data/implant_dev_prompts.jsonl','data/final_id_manifest.jsonl',
        'data/prompt_replacements.jsonl','data/technical_failures.jsonl',
        'provenance/prompt_selection_manifest.json')
    for family in sorted(p for p in artifacts.iterdir() if p.is_dir()):
        runs=[family]+sorted(p for p in family.iterdir() if p.is_dir() and p.name.startswith(('run','pilot','seed','v0','v1','v2')))
        matched=0
        for run in runs:
            for name in names:
                path=run/name
                if not path.is_file():continue
                value=read_jsonl(path) if path.suffix=='.jsonl' else read_json(path)
                forced=('implant_eval' in name or 'implant_dev' in name or 'final_id_manifest' in name
                    or (family.name.startswith(('phase_b','stage2b')) and name in
                        ('data/prompts.jsonl','data/initial_prompts.jsonl','data/prompt_replacements.jsonl','data/technical_failures.jsonl')))
                rows=_heldout_rows(value,forced)
                # Replacements/failed attempts in known evaluation-only families
                # can have identity fields but no raw text. Preserve their IDs.
                if rows:index.consume(path,rows)
                sources[str(path.resolve())]=sha256(path);matched+=1
                audit.append({'path':str(path.resolve()),'selected_heldout_rows':len(rows),
                    'status':'ROLE_FILTERED_HELDOUT_AUTHORITY','sha256':sources[str(path.resolve())]})
        if not matched:audit.append({'path':str(family.resolve()),'status':'NO_KNOWN_COHORT_LAYOUT',
            'limitation':'No known consumed held-out manifest found; unknown layouts are not certified absent'})
    return sources,audit

def _exclusions(state):
    """Use actual held-out cohorts; calibration/train inventories are not tests."""
    root=state['root'];cfg=state['cfg'];index=UsedHistoryIndex();sources={}
    def consume(path,value=None):
        if not path.is_file():return
        value=(read_jsonl(path) if path.suffix=='.jsonl' else read_json(path)) if value is None else value
        index.consume(path,value);sources[str(path.resolve())]=sha256(path)
    # All prior E2/E2B cohorts, including runs whose source split was train_gen.
    for dirname in ('experiment2_rs','experiment2_paraphrase'):
        parent=root/dirname
        if parent.exists():
            for run in sorted(parent.iterdir()):
                if not run.is_dir():continue
                for name in ('private/prompts.jsonl','provenance/prompt_manifest.json',
                        'provenance/population_manifest.json','private/selection.json',
                        'private/selected_prompts.json','provenance/selection.json'):
                    consume(run/name)
    # Mandatory current sources cannot silently disappear on server.
    for name in (cfg.e2_run+'/private/prompts.jsonl',cfg.e2_run+'/provenance/prompt_manifest.json',
            cfg.e2b_run+'/provenance/population_manifest.json'):
        if not (root/name).is_file():raise ValueError('MISSING_HELD_OUT_AUTHORITY: '+name)
    # Imported original final-test outputs are historical exposure even when the
    # old selected-prompts manifest was not included in a server sync.
    consume(root/'inputs/clean_original.jsonl')
    legacy=Path(cfg.legacy_root)
    if not legacy.is_absolute():legacy=root/legacy
    # Original KLRD null/final and historical E1 held-outs. Never consume the
    # whole selected_prompts object: it also contains permitted calibration/train.
    candidates=list(root.glob('experiment1*/private/selected_prompts.json'))
    candidates+=list(root.glob('experiment1*/data/selected_prompts.json'))
    candidates += [root/'inputs/calibration_selection.json',root/'inputs/diagnostic_selection.json']
    if legacy.exists():candidates+=list(legacy.glob('outputs/*/private/selected_prompts.json'))
    for path in sorted(set(candidates)):
        if path.is_file():
            value=read_json(path)
            held={k:v for k,v in value.items() if k in ('null','final','test','heldout','evaluation','eval')}
            if held:consume(path,held)
    if (legacy/'outputs').exists():
        for pattern in ('*/private/clean_null.jsonl','*/private/clean_final.jsonl'):
            for path in sorted((legacy/'outputs').glob(pattern)):consume(path)
    # Any explicitly named historical held-out prompt files are also authority.
    for parent in [root/'experiment1_rs',root/'experiment1_rs_continuation',legacy/'outputs']:
        if parent.exists():
            for pattern in ('**/null_prompts.jsonl','**/final_prompts.jsonl','**/test_prompts.jsonl'):
                for path in sorted(parent.glob(pattern)):consume(path)
    history_sources,history_audit=historical_heldout(index,legacy)
    sources.update(history_sources)
    write_json(state['out']/'provenance/historical_heldout_audit.json',{'sources':history_audit,
        'policy':'Known role-aware cohort layouts only; unknown or missing histories remain explicit limitations'})
    # Normalize nested official-source aliases without modifying historical code.
    aliases=set(index.ids)
    for value in list(aliases):
        for prefix in ('ultrachat:train_gen:','ultrachat:test_gen:','train_gen:','test_gen:','ultrachat:','ultrachat_200k:'):
            if value.startswith(prefix):aliases.add(value[len(prefix):])
    index.ids.update(aliases)
    return index,sources


def _overlap(row,index):
    return bool(source_aliases(row['prompt_id'])&index.ids or text_hash(row['prompt']) in index.exact
        or text_hash(row['prompt'],True) in index.normalized or legacy_text_hash(row['prompt']) in index.legacy)


def _legacy(state,template):
    """Reuse only clean corpora with model/sampling/precision and hash authority.

    Historical BF16 clean corpora are reported but deliberately not treated as
    equivalent to the frozen FP32 clean generation protocol.
    """
    root=Path(state['cfg'].legacy_root)
    if not root.is_absolute():root=state['root']/root
    cfg=state['cfg'];result={'train':[],'validation':[]};audit=[]
    for corpus in sorted((root/'outputs').glob('*/private')) if (root/'outputs').exists() else []:
        item={'directory':str(corpus.resolve()),'eligible':False,'reasons':[],'files':{}}
        manifest=corpus/'manifest.json';engine=corpus/'generation_engine_clean.json'
        if not manifest.exists() or not engine.exists():
            item['reasons'].append('missing clean model/sampling provenance');audit.append(item);continue
        meta=read_json(manifest);eng=read_json(engine);settings=eng.get('settings',{})
        signature=meta.get('signature',{}).get('config',{})
        expected={'model_id':cfg.model,'model_revision':cfg.revision,'tokenizer_revision':cfg.revision,
            'dataset_revision':cfg.dataset_revision,'tau':cfg.tau,'chat_template':template}
        for key,value in expected.items():
            if meta.get(key)!=value:item['reasons'].append('metadata mismatch: '+key)
        if eng.get('model_kind')!='clean' or settings.get('revision')!=cfg.revision:
            item['reasons'].append('engine is not pinned ordinary clean model')
        if settings.get('dtype')!='float32':item['reasons'].append('clean generation is not FP32')
        if signature.get('max_new_tokens')!=cfg.max_new_tokens or signature.get('tau')!=cfg.tau:
            item['reasons'].append('sampling mismatch')
        # Require explicit unrestricted sampling provenance, rather than inferring
        # it from an old runner's defaults or its model's generation_config.
        sampling=eng.get('sampling',eng.get('sampling_params',{}))
        if sampling.get('top_p')!=1 or sampling.get('top_k') not in (0,-1):
            item['reasons'].append('missing unrestricted top_p/top_k authority')
        for split in result:
            path=corpus/f'clean_{split}.jsonl';marker=path.with_suffix('.sha256.json')
            if not path.exists():continue
            item['files'][path.name]=sha256(path)
            expected_hash=read_json(marker).get('sha256') if marker.exists() else meta.get('files_sha256',{}).get(path.name)
            if expected_hash!=item['files'][path.name]:item['reasons'].append('missing/changed trusted corpus hash: '+path.name)
        item['provenance_sha256']={str(p.name):sha256(p) for p in (manifest,engine)}
        item['eligible']=not item['reasons'];audit.append(item)
        if item['eligible']:
            for split in result:
                path=corpus/f'clean_{split}.jsonl'
                if path.exists():
                    for row in read_jsonl(path):
                        if row.get('source')=='train_gen' and row.get('split')==split and isinstance(row.get('response'),str):
                            result[split].append(dict(row,legacy_path=str(path.resolve()),legacy_file_sha256=item['files'][path.name]))
    return result,audit


def _selection(state):
    cfg=state['cfg'];out=state['out'];selection=out/'private/full_selection.json'
    receipt=out/'provenance/data_selection.json'
    index,sources=_exclusions(state)
    if selection.exists():
        print('E3 data: reusing frozen full train/validation prompt partitions.',flush=True)
        metadata=read_json(receipt)
        if metadata['identity']!=state['identity'] or metadata['heldout_sources']!=sources or metadata['selection_sha256']!=sha256(selection):
            raise ValueError('Training selection identity/held-out sources/hash changed')
        selected=read_json(selection)
        if any(_overlap(r,index) for rows in selected.values() for r in rows):raise ValueError('Committed training cohort overlaps held-out')
        return selected
    from datasets import load_dataset
    from transformers import AutoTokenizer
    frozen=get_frozen(state)
    tokenizer=AutoTokenizer.from_pretrained(cfg.model,revision=cfg.revision,trust_remote_code=False)
    if tokenizer.chat_template!=frozen['chat_template']:raise ValueError('Pinned tokenizer chat template differs from calibration')
    print('E3 data: loading pinned train_gen and selecting held-out-disjoint full partitions.',flush=True)
    dataset=load_dataset(cfg.dataset,revision=frozen['dataset_revision'],split='train_gen')
    # Only original prompt strings are read; UltraChat answers are never used.
    candidates={};duplicates=set()
    for n,row in enumerate(dataset):
        if n%10000==0:print(f'E3 data: scanned {n}/{len(dataset)} source prompts; {len(candidates)} eligible so far.',flush=True)
        if not isinstance(row.get('prompt'),str):continue
        pid=str(row['prompt_id']);text=row['prompt'];length=len(tokenizer.encode(text,add_special_tokens=False))
        if not 16<=length<=512:continue
        candidate={'prompt_id':pid,'prompt':text,'source':'train_gen','source_index':n,'prompt_length':length,
            'prompt_text_sha256':text_hash(text),'normalized_prompt_sha256':text_hash(text,True)}
        if _overlap(candidate,index):continue
        if pid in candidates:
            duplicates.add(pid);continue
        candidates[pid]=candidate
    for pid in duplicates:candidates.pop(pid,None)
    old,audit=_legacy(state,frozen['chat_template']);selected={'train':[],'validation':[]};used=UsedHistoryIndex()
    for split in selected:
        for row in old[split]:
            pid=str(row['prompt_id']);candidate=candidates.get(pid)
            if candidate is None or row['prompt']!=candidate['prompt'] or _overlap(candidate,used):continue
            selected[split].append(dict(candidate,split=split,legacy_response=row))
            used.consume('used',candidate)
            if len(selected[split])=={'train':10000,'validation':1000}[split]:break
    ordered=sorted(candidates.values(),key=lambda r:(digest({'seed':cfg.seed,'prompt':r['prompt_id']}),r['prompt_id']))
    position=0
    for split,required in (('train',10000),('validation',1000)):
        while len(selected[split])<required and position<len(ordered):
            row=ordered[position];position+=1
            if _overlap(row,used):continue
            selected[split].append(dict(row,split=split));used.consume('used',row)
        if len(selected[split])!=required:
            raise ValueError(f'INSUFFICIENT_ELIGIBLE_TRAIN_GEN: {split}={len(selected[split])}, required={required}')
    calibration=UsedHistoryIndex()
    path=state['root']/'inputs/calibration_selection.json'
    if path.exists():calibration.consume(path,read_json(path).get('calibration',[]))
    _write_rows(out/'private/heldout_sources.jsonl',[{'path':p,'sha256':h} for p,h in sorted(sources.items())])
    write_json(selection,selected)
    write_json(receipt,{'identity':state['identity'],'selection_sha256':sha256(selection),'heldout_sources':sources,
        'dataset':cfg.dataset,'dataset_revision':frozen['dataset_revision'],'chat_template_sha256':digest(frozen['chat_template']),
        'legacy_discovery':audit,'full_counts':{s:len(r) for s,r in selected.items()},
        'pilot_prefix_counts':{'train':2000,'validation':200},'seed':cfg.seed,
        'selection_rule':'legacy eligible split first; hash-ranked train_gen deterministic completion; no scores',
        'calibration_overlap_counts':{s:sum(_overlap(r,calibration) for r in rows) for s,rows in selected.items()},
        'calibration_overlap_policy':'permitted and declared; all held-out overlap excluded',
        'heldout_comparison':'source ID aliases, exact text SHA256, casefold+NFKC whitespace SHA256, legacy NFKC SHA256'})
    return selected


def _prepare(state,smoke=False,max_rows=2):
    from .generation import generate
    selected=_selection(state);cfg=state['cfg'];out=state['out'];result={}
    for split,count in (('train',cfg.train_count),('validation',cfg.validation_count)):
        if smoke:count=min(max_rows,count)
        rows=selected[split][:count]
        directory=out/'private'/('smoke_data' if smoke else 'data')
        missing=[{k:v for k,v in row.items() if k!='legacy_response'} for row in rows if 'legacy_response' not in row]
        print(f'E3 data: {split}: {len(rows)-len(missing)} clean responses reused; {len(missing)} to generate.',flush=True)
        fresh=generate(state,missing,directory/f'{split}_clean.jsonl',kind='smoke_clean' if smoke else 'clean_train') if missing else []
        lookup={r['prompt_id']:r for r in fresh};records=[]
        for row in rows:
            if 'legacy_response' in row:
                record=dict(row['legacy_response'],**{k:v for k,v in row.items() if k!='legacy_response'})
                record['response_id']=digest({'legacy_file':record['legacy_file_sha256'],'prompt':record['prompt_id'],'response':record['response']})
                record['native_token_metadata_available']='generated_token_ids' in record
            else:record=lookup[row['prompt_id']]
            record=dict(record,source='train_gen',split=split)
            record['eligible_context_status']='error' if record.get('error') else ('empty' if not record['response'] else 'returned_text')
            records.append(record)
        target=(out/'private/smoke_data'/f'{split}_manifest.jsonl') if smoke else out/'private'/('train_manifest.jsonl' if split=='train' else 'val_manifest.jsonl')
        if target.exists():
            if read_jsonl(target)!=records:raise ValueError('Committed clean training manifest changed')
        else:_write_rows(target,records)
        result[split]=records
    write_json(out/('provenance/smoke_data_complete.json' if smoke else 'provenance/data_complete.json'),
        {'identity':state['identity'],'counts':{s:len(r) for s,r in result.items()},
        'errors':{s:sum(bool(r.get('error')) for r in rows) for s,rows in result.items()},
        'empty':{s:sum(not r['response'] for r in rows) for s,rows in result.items()},
        'rows_sha256':{s:digest(r) for s,r in result.items()},'policy':'preselected rows retained including failures; no resampling'})
    return result


def prepare(state):
    return _prepare(state)


def prepare_smoke(state,max_rows=2):
    """Generate only tiny prefixes; never implicitly generate the whole pilot."""
    if not 1<=max_rows<=2:raise ValueError('Smoke clean data is limited to 1-2 rows per split')
    return _prepare(state,smoke=True,max_rows=max_rows)
