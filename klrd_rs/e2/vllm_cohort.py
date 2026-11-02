"""Carry the exact previously frozen prompt/group design into a new engine run."""
from dataclasses import asdict
from pathlib import Path
import os,shutil
from ..io import read_json,read_jsonl,write_json,sha256
from .cohort import make_groups


def reuse_frozen_cohort(root,out,frozen,cfg):
    root,out=Path(root),Path(out)
    public=out/'provenance/prompt_manifest.json';private=out/'private/prompts.jsonl'
    if public.exists() or private.exists():return False
    for run in ['run_005','run_004','run_003','run_002']:
        source=root/'experiment2_rs'/run
        manifest_path=source/'provenance/prompt_manifest.json';pool=source/'private/prompts.jsonl'
        if not manifest_path.exists():continue
        manifest=read_json(manifest_path)
        if not pool.is_file() or sha256(pool)!=manifest['private_prompts_sha256']:
            raise ValueError('Previously frozen E2 prompt pool changed: '+run)
        if manifest['input_receipt_sha256']!=frozen['receipt_sha256']:raise ValueError('E2 input receipt changed')
        for field,value in asdict(cfg).items():
            if isinstance(value,tuple):value=list(value)
            if manifest['E2_config'].get(field)!=value:raise ValueError('Scientific E2 config changed: '+field)
        rows=read_jsonl(pool)
        if len(rows)!=cfg.prompt_count or make_groups(rows,cfg.group_seed)!=manifest['groups']:
            raise ValueError('Frozen E2 cohort/group design changed')
        private.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        temporary=private.with_suffix('.jsonl.tmp');shutil.copyfile(pool,temporary);os.replace(temporary,private)
        manifest=dict(manifest,E2_config=cfg.resolved(),generation_backend='vllm',
            cohort_reused_from={'run':run,'manifest_sha256':sha256(manifest_path),'private_pool_sha256':sha256(pool),
                'policy':'Same prompts, order, groups, quality IDs and seeds. No HF responses or score caches are imported.'})
        write_json(public,manifest)
        audit=source/'provenance/exclusion_audit.json'
        if audit.exists():shutil.copyfile(audit,out/'provenance/exclusion_audit.json')
        return True
    return False
