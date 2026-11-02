"""Fail-closed loading from an independently frozen E1 review receipt."""
from pathlib import Path
import numpy as np
from ..io import read_json,sha256
from ..math import keys,key_ids
from .config import E2Config


class ArtifactError(ValueError):pass


def load_frozen(root):
    from .anchor import INPUT_FOLDER,RECEIPT_SHA256
    root=Path(root);receipt_path=root/INPUT_FOLDER/'source_receipt.json'
    if not receipt_path.is_file():raise ArtifactError('BLOCKED_MISSING_OR_CHANGED_ARTIFACT: '+str(receipt_path))
    if sha256(receipt_path)!=RECEIPT_SHA256:raise ArtifactError('BLOCKED_MISSING_OR_CHANGED_ARTIFACT: source receipt expected SHA256='+RECEIPT_SHA256)
    receipt=read_json(receipt_path)
    for name,item in receipt['files'].items():
        p=root/name
        if not p.is_file() or sha256(p)!=item['sha256']:
            raise ArtifactError('BLOCKED_MISSING_OR_CHANGED_ARTIFACT: '+name+' expected SHA256='+item['sha256'])
    e1=root/'experiment1_rs_continuation';cfg=E2Config()
    summary=read_json(e1/'report/summary.json')
    if summary.get('status')!='VALID_IMPLEMENTATION' or summary.get('GO_NO_GO')!='GO':raise ArtifactError('E1 is not a frozen GO')
    old=read_json(e1/'provenance/resolved_config.json')
    for field in ['model','revision','tau','m','pca_dim']:
        if old[field]!=getattr(cfg,field):raise ArtifactError('Frozen E1 setting mismatch: '+field)
    bank=keys(64,cfg.m,cfg.key_seed)
    expected=read_json(e1/'keys/validation_key_ids.json')
    if expected['seed']!=cfg.key_seed or key_ids(bank)!=expected['keys']:
        raise ArtifactError('BLOCKED_MISSING_OR_CHANGED_ARTIFACT: all64 E1 validation key hashes must match')
    strength=read_json(e1/'teacher/global_lambda.json')
    if strength['status']!='FROZEN' or strength['lambda']!=cfg.strength:raise ArtifactError('Frozen lambda mismatch')
    with np.load(e1/'geometry/basis.npz',allow_pickle=False) as f:p,q=f['P'].copy(),f['Q'].copy()
    with np.load(e1/'geometry/whitening.npz',allow_pickle=False) as f:l=f['L'].copy()
    if p.shape!=(2048,32) or q.shape!=p.shape or l.shape!=(32,32):raise ArtifactError('Frozen geometry shape mismatch')
    if not all(np.isfinite(x).all() for x in [p,q,l]):raise ArtifactError('Nonfinite frozen geometry')
    frame=np.concatenate([p,q],axis=1)
    if not np.allclose(frame.T@frame,np.eye(64),atol=1e-6):raise ArtifactError('Frozen planes invalid')
    meta=read_json(root/'inputs/calibration_manifest.json')
    return {'P':p,'Q':q,'L':l,'bank':bank,'strength':cfg.strength,'receipt':receipt,
        'receipt_sha256':sha256(receipt_path),'e1_config':old,'e1_summary':summary,
        'dataset_revision':meta['dataset_revision'],'chat_template':meta['chat_template']}
