"""Small immutable run identities, atomic artifacts and lazy authoritative inputs."""
from pathlib import Path
from importlib.metadata import version,PackageNotFoundError
import os,platform,uuid
_checked_journals=set()
from ..io import read_json,write_json,sha256,digest
from .config import E3Config


def resolve_output(root,value=None):
    root=Path(root).resolve()
    allowed=[(root/name).resolve() for name in ('experiment3_rs','experiment3_full')]
    output=(Path(value) if value is not None else allowed[0]/'run_001')
    if not output.is_absolute():output=root/output
    output=output.resolve()
    if not any(output!=base and base in output.parents for base in allowed):
        raise ValueError('E3 output must be a separate run below experiment3_rs/ or experiment3_full/')
    return output


def append_event(path,row):
    """Durable append; ignore/truncate only a non-newline crash tail on resume."""
    from klrd_scw.artifacts import append_jsonl,read_jsonl
    path=Path(path)
    if path not in _checked_journals:
        if path.exists():read_jsonl(path,repair_tail=True)
        _checked_journals.add(path)
    append_jsonl(path,row)


def append_events(path,rows):
    """Reuse SCW/KLRD batch serializer: one durable flush for prepared records."""
    from klrd_scw.artifacts import append_jsonl_batch,read_jsonl
    path=Path(path)
    if path not in _checked_journals:
        if path.exists():read_jsonl(path,repair_tail=True)
        _checked_journals.add(path)
    append_jsonl_batch(path,rows)


def atomic_directory(destination,writer):
    """Publish a new complete directory; never replace a committed checkpoint."""
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError('Committed directory already exists: '+str(destination))
    temporary=destination.with_name('.'+destination.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.mkdir(mode=0o700)
    try:
        writer(temporary)
        if destination.exists():raise FileExistsError('Concurrent artifact publication: '+str(destination))
        os.replace(temporary,destination)
    except BaseException:
        # The caller can resume using the last published checkpoint. Keep the
        # incomplete directory for diagnosis; it is never a committed artifact.
        raise
    return destination


def file_manifest(directory):
    directory=Path(directory)
    return {p.relative_to(directory).as_posix():sha256(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def environment():
    packages={}
    for name in ('torch','transformers','peft','accelerate','numpy','scipy','datasets','safetensors','vllm','openai'):
        try:packages[name]=version(name)
        except PackageNotFoundError:packages[name]=None
    return {'python':platform.python_version(),'platform':platform.platform(),'packages':packages}


def source_manifest(root):
    root=Path(root)
    files=[root/'main_exp3.py',root/'gpt_guide/KLRD_RS_Experiment3_LoRA_Codex_Spec_REVISED.md']
    files+=sorted((root/'klrd_rs/e3').glob('*.py'))
    files+=[root/'main_exp3_full.py',root/'gpt_guide/KLRD_RS_Experiment3_FULL_1024_ARCC_SPEC.md']
    files+=sorted((root/'klrd_rs/e3full').glob('*.py'))
    files += [root/p for p in ('klrd_rs/readout.py','klrd_rs/data.py','klrd_rs/math.py','klrd_rs/e2/verifier.py')]
    return {p.relative_to(root).as_posix():sha256(p) for p in files if p.is_file()}


def bind_inputs(state):
    """Bind actual required small manifests, not every historical E1/E2 file."""
    root=state['root'];cfg=state['cfg']
    names=['e2_inputs_v2/source_receipt.json','inputs/calibration_manifest.json',
        cfg.e2_run+'/provenance/prompt_manifest.json',cfg.e2_run+'/private/jobs/identity.json',
        cfg.e2_run+'/provenance/engine_manifest.json',cfg.e2b_run+'/provenance/population_manifest.json',
        cfg.e2b_run+'/provenance/protocol.json']
    result={}
    for name in names:
        path=root/name
        if not path.is_file():raise ValueError('MISSING_REQUIRED_ARTIFACT: sync full server workspace: '+name)
        result[name]=sha256(path)
    return result


def initialize_run(state,create=False):
    """Call under run.lock; existing profile/config must never be reinterpreted."""
    root=Path(state['root']).resolve();out=resolve_output(root,state['out'])
    state.update(root=root,out=out)
    path=out/'provenance/resolved_config.json'
    if path.exists():
        values=read_json(path)
        fields=E3Config.__dataclass_fields__
        cfg=E3Config(**{name:(tuple(values[name]) if name in ('target_modules','alphas') else values[name])
            for name in fields if name in values})
        if state.get('requested_profile') and cfg.profile!=state['requested_profile']:
            raise ValueError('Existing run has a different profile; full requires a separate --output')
        state['cfg']=cfg
    elif not create:raise ValueError('Missing E3 run; run training pilot first (evaluation never trains)')
    out.mkdir(parents=True,exist_ok=True,mode=0o700)
    cfg=state['cfg'];inputs=bind_inputs(state);code=source_manifest(root)
    manifest_path=out/'provenance/input_manifest.json'
    if manifest_path.exists():
        old=read_json(manifest_path)
        if old['config']!=cfg.resolved() or old['inputs']!=inputs:
            raise ValueError('E3 config or frozen input identity changed; preserve this run and use a new output')
        state['identity']=old['identity']
        state['source_changed']=old.get('source')!=code
    else:
        body={'config':cfg.resolved(),'inputs':inputs}
        state['identity']=digest(body)
        write_json(path,cfg.resolved())
        write_json(manifest_path,dict(body,identity=state['identity'],source=code))
        state['source_changed']=False
    state['source_hashes']=code
    write_json(out/'provenance/current_source_manifest.json',code)
    env=environment()
    initial_environment=out/'provenance/environment.json'
    if not initial_environment.exists():write_json(initial_environment,env)
    write_json(out/'provenance/environments'/(uuid.uuid4().hex+'.json'),
        dict(env,mode=state.get('mode','unspecified'),identity=state['identity']))
    return state


def get_frozen(state):
    if state.get('frozen') is None:
        from ..e2.frozen import load_frozen
        print('E3: loading authoritative frozen E1 geometry/key bank once.',flush=True)
        state['frozen']=load_frozen(state['root'])
        cfg=state['cfg'];frozen=state['frozen']
        if frozen['strength']!=cfg.strength:raise ValueError('Frozen strength differs from E3 config')
        if frozen['dataset_revision']!=cfg.dataset_revision:raise ValueError('Pinned dataset revision differs from frozen E1')
    return state['frozen']
