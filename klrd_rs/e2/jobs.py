"""Atomic whole-batch commits: crash recovery never regroups pending requests."""
from pathlib import Path
from ..io import write_json,read_json,digest
from filelock import FileLock


class JobStore:
    def __init__(self,root,identity):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True,mode=0o700)
        p=self.root/'identity.json'
        with FileLock(str(self.root/'identity.lock')):
            if p.exists():
                if read_json(p)!=identity:raise ValueError('Experiment2 runtime/code/input identity changed; refuse resume')
            else:write_json(p,identity)

    def path(self,name):
        if '/' in name or '\\' in name or '..' in name:raise ValueError('Invalid job name')
        return self.root/(name+'.json')

    def load(self,name,identity):
        p=self.path(name)
        if not p.exists():return None
        x=read_json(p)
        if x['identity']!=identity:raise ValueError('Job identity mismatch: '+name)
        if digest(x['records'])!=x['records_sha256']:raise ValueError('Job checksum mismatch: '+name)
        ids=[r['prompt_id'] for r in x['records']]
        if ids!=identity['prompt_ids'] or len(ids)!=len(set(ids)):raise ValueError('Job prompt identity/duplication mismatch')
        return x['records']

    def commit(self,name,identity,records):
        previous=self.load(name,identity)
        if previous is not None:
            if previous!=records:raise ValueError('Refuse replacement of committed observations')
            return
        if [r['prompt_id'] for r in records]!=identity['prompt_ids']:raise ValueError('Incomplete batch commit')
        write_json(self.path(name),{'identity':identity,'records':records,'records_sha256':digest(records)})
