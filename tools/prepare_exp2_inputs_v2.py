"""Correct the pre-generation reserve/used-history bug without changing E1.

Old input bundle/run remain immutable; a separate v2 receipt is frozen before E2.
"""
from pathlib import Path
import sys,json,csv,io,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from klrd_rs.io import sha256,write_json,read_json
from klrd_rs.e2.exclusions import UsedHistoryIndex,unused_inventory_path


def prepare(root,klrd):
    root,klrd=Path(root),Path(klrd);dest=root/'e2_inputs_v2'
    if (dest/'source_receipt.json').exists():raise ValueError('V2 receipt already frozen; refuse overwrite')
    failed=root/'experiment2_rs/run_001'
    if (failed/'private/jobs/identity.json').exists():raise ValueError('E2 observations may exist; no automatic cohort change')
    index=UsedHistoryIndex();sources=[];unparsed=[]
    authority=klrd/'artifacts/stage3a_wmle_implantation/provenance/source_exclusions.json'
    value=read_json(authority);proof=read_json(authority.with_name(authority.name+'.sha256.json'))
    if sha256(authority)!=proof['sha256']:raise ValueError('Frozen KLRD exclusion authority checksum failed')
    if value['dataset']!='HuggingFaceH4/ultrachat_200k' or value['revision']!='8049631c405ae6576f93f445c6b8166f76f5505a':
        raise ValueError('Wrong historical dataset authority')
    if 'do not consume unused reserve inventories' not in value['used_cohort_policy']:
        raise ValueError('Cannot establish used-vs-inventory role in historical authority')
    index.add_authority(value)
    sources.append({'source':'KLRD/'+authority.relative_to(klrd).as_posix(),'sha256':sha256(authority),
        'role':'verified_frozen_used_history_authority','recorded_prior_artifacts':len(value['prior_artifacts'])})
    candidates=[]
    for base in [klrd/'outputs',klrd/'artifacts']:
        candidates.extend(p for p in base.rglob('*') if p.is_file() and 'prompt' in p.name.lower()
            and p.suffix in ('.json','.jsonl','.csv') and '.sha256.' not in p.name)
    candidates.extend(root/'inputs'/n for n in ['calibration_selection.json','diagnostic_selection.json'])
    for p in sorted(set(candidates)):
        label='KLRD/'+p.relative_to(klrd).as_posix() if klrd in p.parents else 'KLRD_RS/'+p.relative_to(root).as_posix()
        before=len(index.ids);h=sha256(p)
        sidecar=p.with_name(p.name+'.sha256.json')
        if sidecar.exists() and read_json(sidecar).get('sha256')!=h:raise ValueError('Historical source changed: '+label)
        role='unused_inventory_not_exposure' if unused_inventory_path(p) else 'selected_or_prior_used_evidence'
        try:
            if unused_inventory_path(p):index.consume(label,[])
            elif p.suffix=='.jsonl':
                with p.open(encoding='utf-8-sig') as f:
                    for line in f:
                        if line.strip():index.consume(label,json.loads(line))
            elif p.suffix=='.csv':
                with p.open(encoding='utf-8-sig') as f:
                    for row in csv.DictReader(f):index.consume(label,row)
            else:index.consume(label,read_json(p))
        except (ValueError,UnicodeError,csv.Error) as exc:unparsed.append({'source':label,'error_type':type(exc).__name__})
        sources.append({'source':label,'sha256':h,'role':role,'new_used_IDs':len(index.ids)-before})
        print(role,label,flush=True)
    if unparsed:raise ValueError('Unparseable prior exposure evidence; refuse a shortened exclusion history: '+str(unparsed))
    out=index.serialized();out.update(source_manifests=sources,unparsed_manifests=[],
        status='VERIFIED_LOADED_MANIFESTS',coverage='Frozen used-history authority plus all selected initial/final cohorts, replacements and prior-used IDs; reserve/filter inventories hashed only',
        authority={'source':sources[0]['source'],'sha256':sources[0]['sha256'],'normalization':'NFKC, collapse whitespace, preserve case'},
        correction='v1 incorrectly marked unused reserves and filter-removal metadata as consumed; v2 changes exposure classification only, before any E2 generation')
    write_json(dest/'exclusion_index.json',out)
    # Reuse exactly the v1 E1 receipt entries; only the new exclusion input changes.
    old=read_json(root/'e2_inputs/source_receipt.json')
    files={k:v for k,v in old['files'].items() if k!='e2_inputs/exclusion_index.json'}
    for name,item in files.items():
        if sha256(root/name)!=item['sha256']:raise ValueError('Frozen E1 input changed: '+name)
    files['e2_inputs_v2/exclusion_index.json']={'sha256':sha256(dest/'exclusion_index.json')}
    write_json(dest/'source_receipt.json',dict(old,files=files,schema=2,exclusion_source_count=len(sources),
        exclusion_policy_version=2,supersedes_receipt_sha256=sha256(root/'e2_inputs/source_receipt.json'),
        prior_failed_run='experiment2_rs/run_001'))
    h=sha256(dest/'source_receipt.json')
    (root/'klrd_rs/e2/anchor.py').write_text('"""V2 used-history correction, before any E2 generation."""\nINPUT_FOLDER = \'e2_inputs_v2\'\nRECEIPT_SHA256 = '+repr(h)+'\n',encoding='utf-8')
    print(json.dumps({'used_IDs':len(index.ids),'casefold_text_hashes':len(index.normalized),
        'legacy_normalized_hashes':len(index.legacy),'IDs_without_raw_text':len(index.ids-index.with_text),
        'inventory_files_not_counted_as_used':len(index.inventory_sources),'v2_receipt_sha256':h}),flush=True)


if __name__=='__main__':prepare(Path(__file__).resolve().parents[1],sys.argv[1])
