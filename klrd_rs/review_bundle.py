"""Shareable E1 evidence bundle; never includes private cache or owner keys."""
from pathlib import Path
import hashlib
import json
import os
import zipfile
from .io import read_json


def create_review_bundle(root,out,anchor):
    root,out=Path(root),Path(out)
    target=out/'KLRD_RS_experiment1_continuation_review.zip'
    temp=target.with_name(target.name+'.tmp')
    files={}
    for name in ['main.py','README.md','README_CONTINUATION.md','requirements.txt','run_experiment1.sh',
                 'KLRD_RS_FINAL_METHOD_AND_EXPERIMENT_1_SPEC.md']:
        p=root/name
        if p.is_file():files['code/'+name]=p
    for folder in ['klrd_rs','tests','docs']:
        for p in (root/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix in ('.py','.md','.json'):
                files['code/'+p.relative_to(root).as_posix()]=p
    for source,prefix in [(out,'continuation'),(root/'experiment1_rs','prior_NO_GO')]:
        for p in source.rglob('*'):
            rel=p.relative_to(source)
            if p.is_file() and 'private' not in rel.parts and p.suffix not in ('.zip','.tmp','.pyc'):
                files[prefix+'/'+rel.as_posix()]=p
    for p in (root/'continuation_inputs').rglob('*'):
        if p.is_file():files['frozen_diagnostic_inputs/'+p.relative_to(root/'continuation_inputs').as_posix()]=p
    summary=read_json(out/'report/summary.json')
    cover=(f"# Experiment 1 continuation review\n\nStatus: **{summary['status']}**.\n\n"
        'Start with continuation/numerical/fp64_diagnostic.md and its JSON/CSV, then '
        'continuation/report/experiment1_report.md and per-key teacher tables if present.\n\n'
        'prior_NO_GO/ is preserved historical evidence. Missing later-stage tables mean '
        'the continuation did not reach those stages; never substitute synthetic results.\n\n'
        'No Experiment 2, training or new paraphrase generation is included. Private caches, '
        'raw datasets, model weights and legacy secret key are omitted. Frozen two-context '
        'diagnostic residuals and the public geometry are included for numerical reproduction.\n\n'
        'See code/README_CONTINUATION.md. SHA256_MANIFEST.json covers every other entry.\n')
    manifest={}
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,p in sorted(files.items()):
            data=p.read_bytes();z.writestr(name,data)
            manifest[name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        for name,data in [('START_HERE.md',cover.encode()),('ANCHOR_ID.json',json.dumps({'anchor_sha256':anchor['manifest_sha256']}).encode())]:
            z.writestr(name,data);manifest[name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        z.writestr('SHA256_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:raise ValueError('Review archive CRC check failed')
        for name,item in manifest.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=item['sha256']:raise ValueError('Review ZIP checksum mismatch')
    os.replace(temp,target)
    return target
