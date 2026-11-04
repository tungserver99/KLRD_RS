"""Atomic small artifacts and append-only, resumable prompt-response records."""
import hashlib
import json
import os
from pathlib import Path


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    def encode(x):
        return x.item() if hasattr(x, 'item') else str(x)
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               allow_nan=False, default=encode), encoding='utf-8')
    os.replace(temp, path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def ensure_manifest(path, expected):
    path = Path(path)
    if path.exists():
        if load_json(path) != expected:
            raise ValueError(f'Incompatible artifacts/settings: {path}; use a new output directory')
    else:
        dump_json(path, expected)


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    with path.open('ab') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl(path, repair_tail=False):
    path = Path(path)
    if not path.exists():
        return []
    records, valid_offset = [], 0
    with path.open('rb') as stream:
        while True:
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b'\n'):
                if repair_tail:
                    break
                raise ValueError(f'Incomplete JSONL tail: {path}')
            try:
                records.append(json.loads(line))
            except (ValueError, UnicodeDecodeError) as exc:
                raise ValueError(f'JSONL corrupt before final incomplete line: {path}') from exc
            valid_offset = stream.tell()
    if repair_tail and path.stat().st_size != valid_offset:
        with path.open('r+b') as stream:
            stream.truncate(valid_offset)
    return records


def resume_records(path, selected):
    expected = {r['prompt_id']: r for r in selected}
    if len(expected) != len(selected):
        raise ValueError('Duplicate selected prompt IDs')
    records = read_jsonl(path, repair_tail=True)
    seen = set()
    for row in records:
        pid = row['prompt_id']
        if pid in seen:
            raise ValueError(f'Duplicate persisted prompt ID: {pid}')
        seen.add(pid)
        if pid not in expected or any(row.get(k) != expected[pid].get(k)
            for k in ['prompt', 'index', 'split', 'response'] if k in expected[pid]):
            raise ValueError(f'Incompatible persisted prompt: {pid}')
    return records, [r for r in selected if r['prompt_id'] not in seen]


# Exact pre-fix tree shipped for the first feasibility run. Only this known
# version can migrate, and only before any generated/scored/training artifact.
_PROMPT_IDS_LEGACY_HASH = '9d68d95eab20ce331fb12d90425a3d58c5363de4e95815693cac721cbcd381b9'


def migrate_pre_generation_manifest(private, metadata, signature):
    """Retain selected prompts after the reported first-generation API failure.

    Never accepts arbitrary code/package/config changes or downstream artifacts.
    The original manifest is backed up, and its exact revision/template retained.
    """
    private = Path(private)
    old = metadata.get('signature', {})
    comparable = lambda s: {k: v for k, v in s.items() if k != 'code_sha256'}
    allowed = {'manifest.json', 'selected_prompts.json', 'reference_tokenizer',
               'manifest.before_prompt_ids_fix.json'}
    if (old.get('code_sha256') != _PROMPT_IDS_LEGACY_HASH
            or comparable(old) != comparable(signature)
            or any(p.name not in allowed for p in private.iterdir())
            or any((private.parent / name).exists() for name in ['released_model', 'reports'])):
        raise ValueError('Incompatible run configuration/code/packages or downstream artifacts; '
                         'use a new output directory')
    backup = private / 'manifest.before_prompt_ids_fix.json'
    if backup.exists() and load_json(backup) != metadata:
        raise ValueError('Incompatible existing pre-fix manifest backup')
    if not backup.exists():
        dump_json(backup, metadata)
    updated = dict(metadata, signature=signature)
    updated['compatibility_migration'] = {
        'reason': 'Normalize chat-template BatchEncoding input_ids before first generation',
        'previous_code_sha256': old['code_sha256'],
        'new_code_sha256': signature['code_sha256']}
    return updated


# Exact preceding tree; changes from it affect stdout/timing only, so already
# generated responses and later scientific artifacts remain compatible.
_PROGRESS_LEGACY_HASH = '4b4756b29bdd11863b04e299cacf0d27939877b462518a74e756bcb90e878d54'


def migrate_logging_manifest(private, metadata, signature):
    private = Path(private)
    old = metadata.get('signature', {})
    comparable = lambda s: {k: v for k, v in s.items() if k != 'code_sha256'}
    if old.get('code_sha256') != _PROGRESS_LEGACY_HASH or comparable(old) != comparable(signature):
        raise ValueError('Incompatible run configuration/code/packages; use a new output directory')
    backup = private / 'manifest.before_progress_logging.json'
    if backup.exists() and load_json(backup) != metadata:
        raise ValueError('Incompatible existing progress-logging manifest backup')
    if not backup.exists():
        dump_json(backup, metadata)
    updated = dict(metadata, signature=signature)
    updated['progress_logging_migration'] = {
        'reason': 'Progress output and elapsed-time reporting only; scientific computation unchanged',
        'previous_code_sha256': old['code_sha256'], 'new_code_sha256': signature['code_sha256']}
    return updated


def append_jsonl_batch(path, records):
    """Serialize before writing; a single durable flush per vLLM batch."""
    payload = ''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in records)
    if not payload:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('ab') as stream:
        stream.write(payload.encode('utf-8'))
        stream.flush()
        os.fsync(stream.fileno())
