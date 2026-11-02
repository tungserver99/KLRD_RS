"""Immutable historical anchors and gated, separately rooted E1 continuation."""
from pathlib import Path
import hashlib
import os
import shutil
import numpy as np
import torch
from .io import read_json,write_json,write_csv,sha256,digest,save_npz
from .numerical import precision_diagnostic,policy


def load_anchor(root,cfg):
    root=Path(root)
    manifest=read_json(root/'manifest.json')
    for name,item in manifest['files'].items():
        p=root/name
        if not p.is_file() or sha256(p)!=item['sha256']:
            raise ValueError('Frozen continuation input changed/missing: '+name)
    frozen=read_json(root/'historical_resolved_config.json')
    if frozen!=cfg.resolved():raise ValueError('Scientific config differs from frozen historical config')
    if read_json(root/'historical_summary.json')['status']!='NO_GO':
        raise ValueError('Expected the preserved historical NO_GO')
    historical=read_json(root/'historical_geometry_checks.json')
    if historical['status']!='FAIL' or historical['all_directions_tested']!=cfg.m:
        raise ValueError('Historical Step C failure identity mismatch')
    with np.load(root/'step_c_exact_contexts.npz',allow_pickle=False) as f:arrays=dict(f)
    if arrays['residuals'].shape!=(2,arrays['P'].shape[0]) or arrays['P'].shape[1]!=cfg.m:
        raise ValueError('Exact historical context/plane dimensions mismatch')
    source=read_json(root/'historical_run_manifest.json')
    return {'root':root,'arrays':arrays,'manifest':manifest,'source':source,'config':frozen,
        'input_manifest':read_json(root/'historical_input_manifest.json'),
        'manifest_sha256':sha256(root/'manifest.json'),'historical_checks':historical}


def verify_source(source,anchor,include_cache=True):
    source=Path(source)
    expected=anchor['source']['files']
    actual={p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()}
    if actual!=set(expected):raise ValueError('Preserved historical source inventory changed')
    for name,item in expected.items():
        path=source/name
        if path.stat().st_size!=item['bytes']:raise ValueError('Preserved source size changed: '+name)
        if not include_cache and name.startswith('private/cache/') and name.endswith('.npz'):continue
        if sha256(path)!=item['sha256']:raise ValueError('Preserved source checksum changed: '+name)
    return {'status':'PASS','historical_files':len(expected),
        'large_cache_bytes_hashed':include_cache,'cache_payloads':'Verified against frozen checksums on inheritance; source is read-only'}


def inherit_contexts(source,anchor,store):
    source=Path(source)
    for name in ['contexts_covariance','contexts_fisher','contexts_strength','contexts_local']:
        if store.load(name) is not None:continue
        relative='private/cache/'+name+'.npz'
        expected=anchor['source']['files'][relative]['sha256']
        marker=read_json(source/'private/cache'/(name+'.complete.json'))
        if marker['sha256']!=expected:raise ValueError('Historical context completion marker changed')
        destination=store.root/(name+'.npz');temp=destination.with_name(destination.name+'.importing')
        shutil.copyfile(source/relative,temp)
        if sha256(temp)!=expected:raise ValueError('Historical context payload checksum mismatch: '+name)
        os.replace(temp,destination)
        write_json(store.root/(name+'.complete.json'),{'sha256':expected})
        print('Inherited exact historical context cache:',name,flush=True)


def check_continuation_contexts(data,p,q,anchor):
    a=anchor['arrays']
    for name in ['residuals','prompt_ids','positions','targets']:
        if not np.array_equal(data[name][:2],a[name]):raise ValueError('Step C historical contexts changed: '+name)
    for name,x in [('P',p),('Q',q)]:
        if not np.array_equal(np.asarray(x,dtype=np.float32),np.asarray(a[name],dtype=np.float32)):
            raise ValueError('Step C native rotation planes changed: '+name)


def frozen_planes(inputs,anchor):
    arrays=anchor['arrays']
    for name in ['basis','mean']:
        if not np.array_equal(inputs[name],arrays[name]):raise ValueError('Frozen public basis/geometry changed: '+name)
    return arrays['P'].copy(),arrays['Q'].copy()


def tensor_sha256(tensor):
    h=hashlib.sha256()
    h.update(str((tuple(tensor.shape),str(tensor.dtype))).encode())
    if tensor.ndim==0:chunks=[tensor.reshape(1)]
    else:chunks=tensor.split(2048)
    for x in chunks:h.update(x.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def write_precision_report(out,result,rows,anchor):
    out=Path(out)
    result['exact_context_anchor_sha256']=anchor['manifest_sha256']
    result['historical_failure']=anchor['historical_checks']
    write_json(out/'numerical/fp64_diagnostic.json',result)
    write_csv(out/'numerical/error_by_context_direction_method.csv',rows,list(rows[0]))
    write_csv(out/'numerical/error_by_direction_method_pooled.csv',
        [r for r in rows if r['context_index']==-1],list(rows[0]))
    save_npz(out/'numerical/exact_two_contexts.npz',**anchor['arrays'])
    lines=['# Step C FP64 diagnostic','',f"Status: **{result['status']}**",'',
        f"Cause classification: {result['root_cause']}",
        f"FP32 FD h=.002 maximum absolute error: {result['FP32_FD_h002_max_abs']}",
        f"FP64 FD h=.002 maximum absolute error: {result['FP64_FD_h002_max_abs']}",
        f"FP32 analytic versus FP64 maximum absolute error: {result['FP32_analytic_vs_FP64_max_abs']}",
        f"FD error improvement ratio: {result['FD_error_improvement_ratio']}",'',
        'The original FP32 tolerance is unchanged. Analytic/autodiff and two Richardson pairs also face stricter independent checks.',
        'No vocabulary token, context, direction, step size or validation key is selected by its outcome.',
        'Worst-token IDs refer to the full-vocabulary score, which need not equal the observed target token.','',
        'See error_by_context_direction_method.csv for max/p99/median absolute and original-scaled errors.',
        'The JSON contains policy, strict acceptance checks, the worst token/context and all direction-23 entries.','']
    (out/'numerical/fp64_diagnostic.md').write_text('\n'.join(lines),encoding='utf-8')


def run_continuation(cfg,inputs,adapter,root,out):
    from .pipeline import run_experiment,report
    root,out=Path(root),Path(out)
    source=root/'experiment1_rs';pack=root/'continuation_inputs'
    if source.resolve()==out.resolve() or source.resolve() in out.resolve().parents or out.resolve() in source.resolve().parents:
        raise ValueError('Continuation output must be separate from preserved source run')
    anchor=load_anchor(pack,cfg)
    protection=verify_source(source,anchor)
    if inputs['index']!=anchor['input_manifest']:raise ValueError('Historical input bundle/config identity changed')
    # Diagnostic is recomputed; a preexisting user-supplied PASS file is never trusted.
    a=anchor['arrays'];engine=adapter.readout(a['P'],a['Q'])
    result,rows=precision_diagnostic(engine,a['residuals'],cfg,adapter.tokenizer,a)
    result['readout_parameter_sha256']={'weight':tensor_sha256(engine.weight),
        'gain':tensor_sha256(engine.gain),'bias':None if engine.bias is None else tensor_sha256(engine.bias)}
    result['source_integrity_before']=protection
    # A new directory owns this code/policy, leaving the prior NO_GO untouched.
    out.mkdir(parents=True,exist_ok=True,mode=0o700)
    from .io import StageStore
    from .pipeline import source_hashes
    StageStore(out/'private/continuation_identity',{'anchor':anchor['manifest_sha256'],
        'policy':policy(),'code':source_hashes(),'config':cfg.resolved(),
        'readout_parameter_sha256':result['readout_parameter_sha256']})
    write_precision_report(out,result,rows,anchor)
    if not result['continuation_allowed']:
        summary={'status':'NO_GO','synthetic':False,'scientific_success':False,
            'GO_NO_GO':'NO_GO','reason':result['root_cause'],
            'numerical_continuation_allowed':False,'remaining_steps':'NOT_RUN'}
        report(out,summary)
    else:
        continuation={'anchor':anchor,'source':source,'diagnostic':result,
            'identity':{'anchor_sha256':anchor['manifest_sha256'],'policy':policy(),
                'readout_parameter_sha256':result['readout_parameter_sha256']}}
        # Release the temporary projection objects; the fixed model itself stays untouched.
        del engine
        summary=run_experiment(cfg,inputs,adapter,out,input_root=root/'inputs',continuation=continuation)
        summary['GO_NO_GO']='GO' if summary['status']=='VALID_IMPLEMENTATION' else 'NO_GO'
        summary['numerical_continuation_allowed']=True
        summary['numerical_root_cause']=result['root_cause']
        report(out,summary)
    summary['criteria']=continuation_criteria(out,result,cfg)
    report(out,summary)
    final=verify_source(source,anchor)
    write_json(out/'provenance/historical_source_integrity.json',final)
    from .review_bundle import create_review_bundle
    bundle=create_review_bundle(root,out,anchor)
    print('Review ZIP:',bundle,flush=True)
    return summary


def continuation_criteria(out,numerical,cfg):
    out=Path(out)
    def artifact(name):
        p=out/name
        return read_json(p) if p.is_file() else None
    fisher=artifact('geometry/fisher_checks.json')
    strength=artifact('teacher/global_lambda.json')
    sphere=artifact('null/sphere_math_tests.json')
    summary=artifact('report/summary.json') or {}
    legacy=artifact('legacy/compatibility_diagnostic.json')
    return {'FP64_numerical_gate':'PASS' if numerical['continuation_allowed'] else 'FAIL',
        'Fisher_rank':fisher.get('rank','FAIL') if fisher else 'NOT_RUN',
        'required_Fisher_rank':cfg.m,
        'global_strength':strength.get('status') if strength else 'NOT_RUN',
        'spherical_null':sphere.get('status') if sphere else 'NOT_RUN',
        'positive_local_key_fraction':summary.get('positive_local_signal_fraction','NOT_RUN'),
        'required_positive_fraction':cfg.local_positive_fraction,
        'validation_keys':summary.get('local_validation_keys','NOT_RUN'),
        'required_validation_keys':cfg.validation_keys,
        'KL_violations_reported':summary.get('KL_violation_key_indices','NOT_RUN'),
        'legacy_optional_status':legacy.get('status') if legacy else 'NOT_RUN',
        'interpretation':'Same original E1 criteria. Per-key KL violations are reported without filtering/rescaling keys. Optional legacy absence is not a scientific failure.'}
