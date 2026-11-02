"""Maintainer preparation only; normal E2 launcher uses the frozen receipt."""
from pathlib import Path
import sys,json,csv,io,zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from klrd_rs.io import sha256,write_json,read_json
from klrd_rs.e2.cohort import text_hash


def prepare(root,klrd):
    root,klrd=Path(root),Path(klrd);dest=root/'e2_inputs'
    if (dest/'source_receipt.json').exists():raise ValueError('Frozen E2 receipt already exists; do not overwrite')
    ids=set();exact=set();normal=set();with_text=set();sources=[];unparsed=[]
    def walk(x):
        if isinstance(x,list):
            for value in x:walk(value)
        if not isinstance(x,dict):return
        pid=x.get('prompt_id');text=x.get('prompt',x.get('prompt_text'))
        if pid is not None:ids.add(str(pid))
        if isinstance(text,str):
            exact.add(text_hash(text));normal.add(text_hash(text,True))
            if pid is not None:with_text.add(str(pid))
        for k,v in x.items():
            if isinstance(v,list) and ('prompt_ids' in k or k in ('excluded_ids','selected_ids')):
                ids.update(str(a) for a in v if isinstance(a,(str,int)))
            if k in ('prompt_sha256','prompt_text_sha256') and isinstance(v,str) and len(v)==64:exact.add(v)
            if k in ('normalized_prompt_sha256','normalized_text_sha256') and isinstance(v,str) and len(v)==64:normal.add(v)
            if isinstance(v,(list,dict)):walk(v)
    candidates=[]
    for base in [klrd/'outputs',klrd/'artifacts']:
        candidates.extend(p for p in base.rglob('*') if p.is_file() and 'prompt' in p.name.lower()
            and p.suffix in ('.json','.jsonl','.csv') and '.sha256.' not in p.name)
    candidates.extend(root/'inputs'/n for n in ['calibration_selection.json','diagnostic_selection.json'])
    for p in sorted(set(candidates)):
        try:
            before=len(ids)+len(exact)+len(normal)
            if p.suffix=='.jsonl':
                with p.open(encoding='utf-8-sig') as f:
                    for line in f:
                        if line.strip():walk(json.loads(line))
            elif p.suffix=='.csv':
                with p.open(encoding='utf-8-sig') as f:
                    for row in csv.DictReader(f):walk(row)
            else:walk(read_json(p))
            sources.append({'source':('KLRD/'+p.relative_to(klrd).as_posix()) if klrd in p.parents else 'KLRD_RS/'+p.relative_to(root).as_posix(),
                'sha256':sha256(p),'new_exclusion_items':len(ids)+len(exact)+len(normal)-before})
        except (ValueError,UnicodeError) as exc:unparsed.append({'file':p.name,'error_type':type(exc).__name__})
    write_json(dest/'exclusion_index.json',{'prompt_ids':sorted(ids),'exact_text_hashes':sorted(exact),
        'normalized_text_hashes':sorted(normal),'source_manifests':sources,'unparsed_manifests':unparsed,
        'ids_without_recovered_text':sorted(ids-with_text),
        'coverage':'All prompt-named JSON/JSONL/CSV manifests under historical KLRD outputs/artifacts plus both bundled E1 selections; no response-based selection',
        'status':'VERIFIED_LOADED_MANIFESTS' if not unparsed else 'COHORT_PROVENANCE_LIMITATION'})
    archive=root/'experiment1_rs_continuation/KLRD_RS_experiment1_continuation_review.zip'
    with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as z:
        manifest=json.loads(z.read('SHA256_MANIFEST.json'))
    wanted=['geometry/basis.npz','geometry/fisher.npz','geometry/whitening.npz','geometry/fisher_checks.json',
        'keys/calibration_key_ids.json','keys/validation_key_ids.json','teacher/global_lambda.json',
        'teacher/KL_by_key.csv','teacher/local_capacity_by_key.csv','provenance/resolved_config.json',
        'provenance/input_manifest.json','provenance/sha256_manifest.json','report/summary.json',
        'null/fixed_key_pilot.json','numerical/fp64_diagnostic.json']
    files={}
    for rel in wanted:
        expected=manifest['continuation/'+rel]['sha256'];path=root/'experiment1_rs_continuation'/rel
        if sha256(path)!=expected:raise ValueError('E1 review hash mismatch: '+rel)
        files['experiment1_rs_continuation/'+rel]={'sha256':expected}
    for name in ['model.py','readout.py','math.py','data.py','io.py','config.py']:
        rel='klrd_rs/'+name;expected=manifest['code/'+rel]['sha256']
        if sha256(root/rel)!=expected:raise ValueError('E1 reused code changed: '+rel)
        files[rel]={'sha256':expected}
    for name in ['inputs/calibration_manifest.json','e2_inputs/exclusion_index.json']:
        files[name]={'sha256':sha256(root/name)}
    write_json(dest/'source_receipt.json',{'schema':1,'authority':'E1 continuation review ZIP SHA256_MANIFEST',
        'archive_sha256':sha256(archive),'files':files,'exclusion_source_count':len(sources),
        'scope':'Public frozen E1 geometry/strength/key IDs; no private key vectors persisted'})
    print('Exclusions',len(ids),'IDs',len(exact),'exact hashes',len(normal),'normalized hashes',flush=True)
    print('Manifest sources',len(sources),'unparsed',len(unparsed),'IDs without text',len(ids-with_text),flush=True)


if __name__=='__main__':prepare(Path(__file__).resolve().parents[1],sys.argv[1])
