"""Strict JSON, immutable input hashes and atomic, checksummed stage caches."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    def encode(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=encode)+'\n', encoding='utf-8')
    os.replace(temp, path)


def save_npz(path, **values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('wb') as f: np.savez(f, **values)
    os.replace(temp, path)


class StageStore:
    def __init__(self, root, identity):
        self.root = Path(root); self.root.mkdir(parents=True, exist_ok=True)
        path = self.root/'identity.json'
        if path.exists() and read_json(path) != identity:
            raise ValueError('Run inputs/code/config changed; incompatible resume. Use a new output directory.')
        write_json(path, identity)

    def load(self, name):
        marker = self.root/(name+'.complete.json')
        if not marker.exists(): return None
        path = self.root/(name+'.npz')
        if not path.exists() or sha256(path) != read_json(marker)['sha256']:
            raise ValueError('Stage cache checksum mismatch: '+name)
        with np.load(path, allow_pickle=False) as f: return dict(f)

    def save(self, name, arrays):
        path = self.root/(name+'.npz')
        save_npz(path, **arrays)
        write_json(self.root/(name+'.complete.json'), {'sha256': sha256(path)})


def write_csv(path, rows, fields):
    import csv
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    with temp.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    os.replace(temp, path)
