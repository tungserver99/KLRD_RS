"""Experiment 1 stages A–H; all gates distinguish evidence from claims."""
from pathlib import Path
import hashlib
import time
import numpy as np
import torch
from .io import StageStore, sha256, digest, write_json, read_json, save_npz, write_csv
from .data import partition
from .math import planes, keys, key_ids, fit_fisher, choose_lambda
from .replay import contexts, sampled_scores, response_scores
from .diagnostics import geometry_checks, sphere_math, pilot, clustered_summary


class ScientificNoGo(RuntimeError): pass


def decision(math_ok,strength_ok,positive_fraction,diagnostics_complete,synthetic,required_fraction=.95):
    if not math_ok or not strength_ok or positive_fraction<required_fraction:return 'NO_GO'
    if synthetic:return 'SYNTHETIC_CHECKS_PASSED'
    if not diagnostics_complete:return 'INCONCLUSIVE'
    return 'VALID_IMPLEMENTATION'


def source_hashes():
    root=Path(__file__).resolve().parent
    result={p.name:sha256(p) for p in sorted(root.glob('*.py'))}
    result['main.py']=sha256(root.parent/'main.py') if (root.parent/'main.py').exists() else 'absent'
    return result


def report(out,result):
    write_json(out/'report/summary.json',result)
    lines=['# KLRD-RS Experiment 1', '', 'Status: **'+result['status']+'**', '',
        'This is development-data implementation and local-context feasibility evidence.',
        'It is not a student/training result, generated-text TPR, fixed-key FPR certification, or paraphrase/SaTML robustness result.', '']
    if result.get('synthetic'):lines += ['**Synthetic toy execution only. No real-model scientific result.**','']
    for k,v in result.items():
        if k not in ('status','synthetic'):lines += [f'- {k}: {v}']
    lines += ['', 'The spherical null is exact over independent random owner keys conditional on fixed nonzero V.',
        'It is not an arbitrary fixed-key deployment guarantee. Fixed-key pilot intervals use disjoint response groups.',
        'Historical clean/SCW cohorts have already influenced development and are not new confirmatory test data.',
        'See geometry/geometry_checks.json, null/sphere_math_tests.json, null/fixed_key_pilot.json,',
        'teacher/global_lambda.json and teacher/*_by_key.csv for numerical evidence and every key/violation.', '']
    (out/'report/experiment1_report.md').write_text('\n'.join(lines),encoding='utf-8')


def run_experiment(cfg,inputs,adapter,out,synthetic=False,input_root=None,continuation=None):
    import platform
    from importlib.metadata import version,PackageNotFoundError
    environment={'python':platform.python_version(),'torch':torch.__version__,
        'CUDA':torch.version.cuda,'GPU':None if synthetic or not torch.cuda.is_available() else torch.cuda.get_device_name(0),
        'float32_matmul_precision':torch.get_float32_matmul_precision(),
        'CUDA_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}
    for name in ['numpy','scipy','transformers','mpmath']:
        try:environment[name]=version(name)
        except PackageNotFoundError:environment[name]=None
    out=Path(out);out.mkdir(parents=True,exist_ok=True,mode=0o700)
    config=cfg.resolved()
    identity={'config':config,'inputs':inputs['index'],'source_sha256':source_hashes(),'synthetic':synthetic,
        'environment':environment}
    if continuation is not None:identity['continuation']=continuation['identity']
    # Check identity before touching any successful output from a previous run.
    store=StageStore(out/'private/cache',identity)
    if continuation is not None:
        from .continuation import inherit_contexts
        inherit_contexts(continuation['source'],continuation['anchor'],store)
    write_json(out/'provenance/resolved_config.json',config)
    write_json(out/'provenance/environment.json',environment)
    write_json(out/'provenance/input_manifest.json',inputs['index'])
    write_json(out/'provenance/sha256_manifest.json',{'inputs':inputs['index'].get('files',{}),'code':source_hashes()})
    ledger={'status':'RUNNING','stages':{}}
    def stage(name,status,**details):
        ledger['stages'][name]={'status':status,**details};write_json(out/'provenance/stage_status.json',ledger)
        print(name,status,flush=True)
    stage('A_provenance','PASS',synthetic=synthetic)
    try:
        result=_execute(cfg,inputs,adapter,out,store,stage,synthetic,input_root,continuation)
    except ScientificNoGo as exc:
        result={'status':'NO_GO','reason':str(exc),'scientific_success':False,'synthetic':synthetic}
    ledger['status']=result['status'];write_json(out/'provenance/stage_status.json',ledger)
    report(out,result)
    return result


def _execute(cfg,inputs,adapter,out,store,stage,synthetic,input_root,continuation=None):
    if continuation is None:p,q=planes(inputs['basis'],cfg.m,cfg.geometry_seed)
    else:
        from .continuation import frozen_planes
        p,q=frozen_planes(inputs,continuation['anchor'])
    generator_hashes=[hashlib.sha256(np.concatenate([p[:,j],q[:,j]]).astype('<f8').tobytes()).hexdigest() for j in range(cfg.m)]
    save_npz(out/'geometry/basis.npz',basis=inputs['basis'],mean=inputs['mean'],P=p,Q=q)
    write_json(out/'geometry/generator_hashes.json',{'orientation':'A_j=q_j p_j^T-p_j q_j^T',
        'seed':cfg.geometry_seed,'generator_factor_sha256':generator_hashes})
    parts=partition(inputs['calibration'],cfg.partition_seed)
    manifest={name:[r['prompt_id'] for r in rows] for name,rows in parts.items()}
    write_json(out/'provenance/calibration_partitions.json',manifest)
    caps={'fisher':cfg.fisher_context_sample_cap,'covariance':cfg.covariance_contexts,
        'strength':cfg.strength_contexts,'local':cfg.local_contexts}
    engine=adapter.readout(p,q)
    # Check the loaded native tokenizer template, not just its revision string.
    template=inputs.get('manifest',{}).get('chat_template')
    if template is not None and adapter.tokenizer.chat_template!=template:
        raise ValueError('Pinned tokenizer chat template differs from historical template')
    data={}
    for i,name in enumerate(['covariance','fisher','strength','local']):
        data[name]=contexts(parts[name],adapter,caps[name],cfg.partition_seed+i+1,store,'contexts_'+name)
        write_json(out/('provenance/'+name+'_contexts.json'),{'sample_count':len(data[name]['targets']),
            'cap':caps[name],'seed':cfg.partition_seed+i+1,'context_sha256':digest(
                list(zip(data[name]['prompt_ids'].tolist(),data[name]['positions'].tolist()))),
            'excluded_prompt_ids':data[name]['excluded_prompt_ids'].tolist(),
            'exclusion_reason':'Historical generated tokens cannot be recovered exactly from visible text',
            'eligible_prompts':len(data[name]['eligible_prompt_ids']),
            'original_partition_prompts':len(parts[name]),
            'sampled_prompts':len(np.unique(data[name]['prompt_ids']))})
        stage('B_contexts_'+name,'PASS',contexts=len(data[name]['targets']))
    checks=geometry_checks(engine,data['covariance']['residuals'],cfg,adapter)
    if continuation is not None:
        from .continuation import check_continuation_contexts
        check_continuation_contexts(data['covariance'],p,q,continuation['anchor'])
        diagnostic=continuation['diagnostic']
        if not diagnostic.get('continuation_allowed') or not diagnostic.get('fp64_checks_passed'):
            raise ScientificNoGo('FP64 diagnostic did not permit continuation')
        write_json(out/'numerical/current_FP32_geometry_checks.json',checks)
        checks['original_FP32_gate_status']=checks['status']
        checks['tangent_validation_method']='FP64 analytic/autograd/FD with strict Richardson validation on exact historical contexts'
        checks['FP64_diagnostic_anchor_sha256']=continuation['anchor']['manifest_sha256']
        checks['status']='PASS' if checks['geometry_and_readout_passed'] else 'FAIL'
    mean_error=float(np.linalg.norm(np.concatenate([p,q],1).T@inputs['mean']))
    checks['activation_mean_projection_norm']=mean_error
    if mean_error>cfg.rotation_atol*max(1.,float(np.linalg.norm(inputs['mean']))):checks['status']='FAIL'
    write_json(out/'geometry/geometry_checks.json',checks)
    stage('C_rotation_tangent',checks['status'])
    if checks['status']!='PASS':raise ScientificNoGo('Rotation/tangent/readout numerical checks failed')
    samples=store.load('fisher_scores')
    if samples is None:
        scores=sampled_scores(data['fisher'],engine,cfg,cfg.token_sample_seed)
        store.save('fisher_scores',{'scores':scores})
    else:scores=samples['scores']
    try:f,l,finfo=fit_fisher(scores,cfg.rank_rtol)
    except ValueError as exc:
        write_json(out/'geometry/fisher_checks.json',{'status':'NO_GO','reason':str(exc)})
        raise ScientificNoGo(str(exc)) from exc
    cov=store.load('covariance_scores')
    if cov is None:
        cov_scores=sampled_scores(data['covariance'],engine,cfg,cfg.token_sample_seed+1)
        store.save('covariance_scores',{'scores':cov_scores})
    else:cov_scores=cov['scores']
    psi=cov_scores@l.T
    validation_cov=psi.T@psi/len(psi)
    eig=np.linalg.eigvalsh(validation_cov)
    finfo.update(heldout_whitened_covariance_eigenvalues=eig.tolist(),
        heldout_covariance_Frobenius_error=float(np.linalg.norm(validation_cov-np.eye(cfg.m))),
        heldout_score_mean=psi.mean(0).tolist(),observations=len(scores),heldout_observations=len(psi),
        regularization_rule='eps=1e-3*trace(F_hat)/m',rank_relative_tolerance=cfg.rank_rtol)
    if not np.isfinite(eig).all() or eig[0]<=0:raise ScientificNoGo('Degenerate held-out score covariance')
    save_npz(out/'geometry/fisher.npz',F_hat=f,F_eff=f+finfo['ridge']*np.eye(cfg.m))
    save_npz(out/'geometry/whitening.npz',L=l,heldout_covariance=validation_cov)
    write_json(out/'geometry/fisher_checks.json',finfo);stage('D_Fisher','PASS')
    calibration_bank=keys(cfg.calibration_keys,cfg.m,cfg.key_calibration_seed)
    write_json(out/'keys/calibration_key_ids.json',{'seed':cfg.key_calibration_seed,'keys':key_ids(calibration_bank)})
    strength_path=out/'teacher/global_lambda.json'
    committed=store.load('global_strength')
    if committed is not None:
        if not strength_path.exists() or sha256(strength_path)!=str(committed['artifact_sha256']):
            raise ValueError('Frozen strength checksum mismatch; refusing stale local-key caches')
        frozen=read_json(strength_path)
        strength=float(committed['lambda'])
        if frozen.get('status')!='FROZEN' or frozen['lambda']!=strength:
            raise ValueError('Frozen strength artifact is incompatible')
    else:
        trace=[]
        @torch.no_grad()
        def evaluate(value):
            kl=[]
            for key in calibration_bank:
                angle=torch.as_tensor(value*(l@key),device=engine.weight.device,dtype=engine.weight.dtype)
                for start in range(0,len(data['strength']['residuals']),cfg.readout_chunk):
                    r=torch.as_tensor(data['strength']['residuals'][start:start+cfg.readout_chunk],device=engine.weight.device,dtype=engine.weight.dtype)
                    kl.extend(engine.kl(r,angle).double().cpu().numpy().tolist())
            trace.append({'lambda':float(value),'mean':float(np.mean(kl)),'p99':float(np.quantile(kl,.99))})
            write_json(strength_path,{'status':'CALIBRATING','pooled_search_trace':trace,
                'keys':cfg.calibration_keys,'contexts_per_key':len(data['strength']['residuals'])})
            print('Global strength trial',trace[-1],flush=True)
            return kl
        try:strength,_=choose_lambda(evaluate,cfg.mean_KL_target_nats,cfg.p99_KL_safeguard_nats,cfg.lambda_max)
        except ValueError as exc:
            write_json(strength_path,{'status':'NO_GO','reason':str(exc),'pooled_search_trace':trace})
            raise ScientificNoGo(str(exc)) from exc
        write_json(strength_path,{'status':'FROZEN','lambda':strength,'pooled_search_trace':trace,
            'calibration_keys':cfg.calibration_keys,'calibration_contexts':len(data['strength']['residuals']),
            'global_only':True,'evaluation_keys_sampled_after_freeze':True,
            'KL_direction':'KL(q||p0)','mean_target':cfg.mean_KL_target_nats,'p99_max':cfg.p99_KL_safeguard_nats})
        store.save('global_strength',{'lambda':np.array(strength),'artifact_sha256':np.array(sha256(strength_path))})
    stage('E_global_lambda','FROZEN',value=strength)
    # This line deliberately occurs after freezing lambda, never before.
    bank=keys(cfg.validation_keys,cfg.m,cfg.key_evaluation_seed)
    write_json(out/'keys/validation_key_ids.json',{'seed':cfg.key_evaluation_seed,'keys':key_ids(bank),
        'frozen_geometry_strength_sha256':digest({'generators':generator_hashes,'lambda':strength,'Fisher':f.tolist()})})
    local_rows,kl_rows=_local(data['local'],engine,l,bank,strength,cfg,store)
    write_csv(out/'teacher/local_capacity_by_key.csv',local_rows,list(local_rows[0]))
    write_csv(out/'teacher/KL_by_key.csv',kl_rows,list(kl_rows[0]))
    fraction=float(np.mean([r['Delta']>0 for r in local_rows]))
    stage('H_local_teacher','PASS' if fraction>=cfg.local_positive_fraction else 'NO_GO',positive_fraction=fraction)
    original=response_scores(inputs['original'],adapter,engine,cfg,store,'clean_original_phi')@l.T
    raw=response_scores(inputs['paraphrase'],adapter,engine,cfg,store,'clean_paraphrase_phi')
    para=None if raw is None else raw@l.T
    null_checks=sphere_math(cfg,{'synthetic':np.random.default_rng(112).normal(size=cfg.m),
        'clean_original':original[:20].sum(0),'clean_paraphrase':None if para is None else para[:20].sum(0)})
    write_json(out/'null/sphere_math_tests.json',null_checks)
    stage('F_spherical_null',null_checks['status'])
    populations={
        'clean_original':pilot(inputs['original'],original,bank,cfg,out/'null/clean_original_key_pvalues.csv'),
        'clean_paraphrase':pilot(inputs['paraphrase'],para,bank,cfg,out/'null/clean_paraphrase_key_pvalues.csv')}
    write_json(out/'null/fixed_key_pilot.json',{'populations':populations,
        'scope':'Descriptive only, development cohorts. No empirically fitted detection threshold.',
        'same_groups_across_keys':True,'groups_independent_within_each_budget_only':True})
    stage('G_fixed_key_pilot','DESCRIPTIVE_ONLY')
    if input_root is not None:
        from .legacy import run_legacy
        legacy=run_legacy(input_root,inputs,adapter,cfg,store,out)
    else:legacy={'status':'SKIPPED','reason':'Synthetic run has no historical checkpoint'}
    write_json(out/'legacy/compatibility_diagnostic.json',legacy)
    math_ok=checks['status']=='PASS' and null_checks['status']=='PASS'
    status=decision(math_ok,True,fraction,para is not None,synthetic,cfg.local_positive_fraction)
    violations=[r['key_index'] for r in kl_rows if r['mean_KL_violation'] or r['p99_KL_violation']]
    return {'status':status,'synthetic':synthetic,'scientific_success':False,
        'math_checks_passed':math_ok,'lambda':strength,'positive_local_signal_fraction':fraction,
        'local_validation_keys':len(bank),'KL_violation_key_indices':violations,
        'diagnostic_original_responses':len(original),'diagnostic_paraphrase_responses':0 if para is None else len(para),
        'legacy_status':legacy['status'],
        'scientific_scope':'Implementation/local expectation only; full generation and student ownership detection remain unestablished'}


@torch.no_grad()
def _local(data,engine,l,bank,strength,cfg,store):
    local_rows=[];kl_rows=[];last=time.monotonic()
    for i,key in enumerate(bank):
        saved=store.load(f'local_key_{i:03}')
        if saved is None:
            columns=[[],[],[],[],[]]
            direction=torch.as_tensor(l@key,device=engine.weight.device,dtype=engine.weight.dtype)
            for start in range(0,len(data['residuals']),cfg.readout_chunk):
                r=torch.as_tensor(data['residuals'][start:start+cfg.readout_chunk],device=engine.weight.device,dtype=engine.weight.dtype)
                values=engine.local(r,direction,strength,cfg.finite_difference_h)
                for col,value in zip(columns,values):col.extend(value.double().cpu().numpy().tolist())
            saved=dict(zip(['delta','kl','prediction','finite_difference','noise'],map(np.asarray,columns)))
            store.save(f'local_key_{i:03}',saved)
        summary=clustered_summary(saved['delta'],data['prompt_ids'],cfg)
        noise=float(np.sqrt(np.mean(saved['noise']**2)))
        prediction=float(saved['prediction'].mean())
        local_rows.append({'key_index':i,'Delta':summary['mean'],
            'CI95_low':summary['CI95_prompt_cluster_bootstrap'][0],'CI95_high':summary['CI95_prompt_cluster_bootstrap'][1],
            'cluster_bootstrap_SE':summary['bootstrap_SE'],'independent_prompts':summary['independent_prompts'],
            'contexts':summary['contexts'],'first_order_prediction':prediction,
            'Delta_over_first_order':summary['mean']/prediction if prediction else None,
            'first_order_slope':prediction/strength,'finite_difference_slope':float(saved['finite_difference'].mean()),
            'per_token_clean_noise_RMS':noise,'Delta_over_per_token_noise':summary['mean']/noise if noise else None,
            'Delta_over_cluster_SE':summary['mean']/summary['bootstrap_SE'] if summary['bootstrap_SE'] else None})
        mean=float(saved['kl'].mean());p99=float(np.quantile(saved['kl'],.99))
        kl_rows.append({'key_index':i,'mean_KL':mean,'p99_KL':p99,
            'mean_KL_violation':mean>cfg.mean_KL_target_nats,
            'p99_KL_violation':p99>cfg.p99_KL_safeguard_nats})
        if time.monotonic()-last>=cfg.log_interval_seconds:
            print('Local teacher:',i+1,'/',len(bank),'keys; no key filtering',flush=True);last=time.monotonic()
    return local_rows,kl_rows
