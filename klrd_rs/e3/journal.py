"""Backed-up recovery of an uncommitted clean-generation JSONL journal.

Only malformed JSON lines can be quarantined. Every parseable record must pass
the caller's full prompt/run/seed/hash validation before any file is replaced.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

from ..io import sha256,write_json


def load_generation_journal(path,prompts,identity,validate,*,allow_recovery=False):
    from klrd_scw.artifacts import read_jsonl
    path=Path(path)
    if not path.exists():return []
    try:
        records=read_jsonl(path,repair_tail=False)
    except ValueError:
        if path.with_suffix('.complete.json').exists():
            raise ValueError('Corrupt completed generation journal; restore its committed data instead of regenerating: '+str(path)) from None
        if not allow_recovery:raise
        return _recover(path,prompts,identity,validate)
    validate(records,prompts,identity)
    return records


def _recover(path,prompts,identity,validate):
    records=[];lines=[];invalid=[];source_hash=hashlib.sha256();offset=0
    with path.open('rb') as stream:
        for number,raw in enumerate(stream,1):
            source_hash.update(raw)
            try:
                row=json.loads(raw)
            except (ValueError,UnicodeDecodeError):
                invalid.append({'line':number,'byte_offset':offset,'bytes':len(raw)})
            else:
                records.append(row)
                # A complete final JSON object without LF is recoverable only
                # if its own record hash and all binding checks below pass.
                lines.append(raw if raw.endswith(b'\n') else raw+b'\n')
            offset+=len(raw)
    # A valid JSON record with a bad hash/run/ID/seed is a conflict, not a reason
    # to discard a response. Fail without touching the source in that case.
    validate(records,prompts,identity)
    original_hash=source_hash.hexdigest();token=uuid.uuid4().hex
    backup=path.with_name(path.name+'.corrupt.'+token+'.bak')
    temporary=path.with_name(path.name+'.recovered.'+token+'.tmp')
    receipt=path.with_name(path.name+'.recovery.'+token+'.json')
    # Preserve exact original bytes, including malformed lines, durably before
    # replacing anything. Only this one affected journal is read/hashed.
    with path.open('rb') as source,backup.open('xb') as target:
        shutil.copyfileobj(source,target)
        target.flush();os.fsync(target.fileno())
    if sha256(backup)!=original_hash or sha256(path)!=original_hash:
        raise ValueError('Generation journal changed during recovery; stop other writers and retry: '+str(path))
    with temporary.open('xb') as stream:
        for line in lines:stream.write(line)
        stream.flush();os.fsync(stream.fileno())
    metadata={'schema':1,'status':'PREPARED','journal':path.name,'backup':backup.name,
        'original_sha256':original_hash,'recovered_sha256':sha256(temporary),
        'kept_records':len(records),'missing_responses':len(prompts)-len(records),
        'invalid_lines':invalid,
        'policy':'All parseable rows passed original prompt/run/seed/record hash checks; '
            'malformed bytes remain in exact backup; regenerate missing IDs only; no cohort change'}
    write_json(receipt,metadata)
    os.replace(temporary,path)
    write_json(receipt,dict(metadata,status='RECOVERED'))
    print(f'E3 journal recovered: {len(records)}/{len(prompts)} verified responses kept; '
        f'{len(prompts)-len(records)} pending; {len(invalid)} malformed lines quarantined. '
        f'Backup: {backup.name}',flush=True)
    return records
