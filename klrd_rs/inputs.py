"""Read and verify the portable, hash-pinned historical input bundle."""
from pathlib import Path
import numpy as np
import torch
from .io import read_json, read_jsonl, sha256
from .data import validate_rows, match_paraphrases


def load_inputs(root, cfg):
    root = Path(root)
    index = read_json(root/'bundle_manifest.json')
    optional={'private/legacy_key.pt','legacy_fingerprint.jsonl','legacy_scores.jsonl',
              'paraphrases.jsonl','paraphrase_manifest.json'}
    unavailable=[]
    for name, item in index['files'].items():
        path = root/name
        if not path.is_file() and name in optional:
            unavailable.append(name);continue
        if not path.is_file() or sha256(path) != item['sha256']:
            raise ValueError('Missing/changed bundled artifact: '+name)
    index=dict(index,unavailable_optional=unavailable)
    base = read_json(root/'calibration_manifest.json')
    diag = read_json(root/'diagnostic_manifest.json')
    for meta in [base,diag]:
        if meta['model_id'] != cfg.model or meta['model_revision'] != cfg.revision or meta['tokenizer_revision'] != cfg.revision or meta['tau'] != cfg.tau:
            raise ValueError('Historical model/tokenizer/temperature mismatch')
    selected = read_json(root/'calibration_selection.json')
    diagnostic_selected = read_json(root/'diagnostic_selection.json')
    calibration = validate_rows(read_jsonl(root/'clean_calibration.jsonl'), selected, 'calibration')
    if any(not isinstance(r.get('text_roundtrip_equal'),bool) for r in calibration):
        raise ValueError('Calibration text/token recoverability is not recorded')
    original = validate_rows(read_jsonl(root/'clean_original.jsonl'), diagnostic_selected, 'final')
    cids = {r['prompt_id'] for r in calibration}; ctext = {r['prompt'] for r in calibration}
    if any(r['prompt_id'] in cids or r['prompt'] in ctext for r in original):
        raise ValueError('Calibration/diagnostic cohort leakage')
    para_path = root/'paraphrases.jsonl'
    paraphrase = match_paraphrases(original, read_jsonl(para_path) if para_path.exists() else None)
    state = torch.load(root/'active_basis.pt', map_location='cpu', weights_only=True)
    basis, mean = state['basis'].double().numpy(), state['mean'].double().numpy()
    if basis.shape[1] != cfg.pca_dim or mean.shape != (basis.shape[0],):
        raise ValueError('Historical PCA dimensionality mismatch')
    if not np.isfinite(basis).all() or not np.isfinite(mean).all(): raise ValueError('Nonfinite public geometry')
    if np.linalg.norm(basis.T@mean) > 1e-6*max(1.,np.linalg.norm(mean)):
        raise ValueError('Public geometry is not orthogonal to activation mean')
    if not index.get('basis_provenance',{}).get('clean_calibration_only'):
        raise ValueError('Unverified public basis calibration provenance')
    return {'index':index, 'calibration':calibration, 'original':original, 'paraphrase':paraphrase,
            'basis':basis, 'mean':mean, 'manifest':base, 'diagnostic_manifest':diag}
