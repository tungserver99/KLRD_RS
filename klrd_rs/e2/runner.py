"""Predeclared smoke -> pilot -> full study, with atomic fixed-shape GPU jobs."""
from pathlib import Path
import json,os,time,platform,uuid
import numpy as np
import torch
from filelock import FileLock,Timeout
from ..io import read_json,write_json,read_jsonl,write_csv,save_npz,digest,sha256
from ..data import prompt_ids
from ..math import key_ids,sphere_p
from .config import E2Config
from .anchor import INPUT_FOLDER
from .cohort import select_cohort,make_groups,generation_seed,text_hash
from .jobs import JobStore
from .analysis import analyze_groups,wrong_indices,key_name,decide
from .utility import diagnostics,examples
from .verifier import SphericalBlackBoxVerifier


def code_hashes(root):
    names=[root/'main_exp2.py',root/'gpt_guide/KLRD_RS_EXPERIMENT_2_CODEX_SPEC.md',
        root/'gpt_guide/KLRD_RS_EXPERIMENT_2_VLLM_SPEC.md',
        *sorted((root/'klrd_rs/e2').glob('*.py'))]
    return {p.relative_to(root).as_posix():sha256(p) for p in names if p.is_file()}


def resume_logging_identity(out,current):
    """Allow only the recorded runner-only progress-display upgrade.

    Preserve the original job identity. All other code, runtime, input, design
    and configuration fields must still match exactly.
    """
    path=out/'private/jobs/identity.json'
    if not path.exists():return current
    previous=read_json(path)
    if previous==current:return previous
    if {k:v for k,v in previous.items() if k!='code'}!={k:v for k,v in current.items() if k!='code'}:
        raise ValueError('Experiment2 runtime/input/config identity changed; refuse resume')
    old=previous['code'];new=current['code'];name='klrd_rs/e2/runner.py'
    compatibility=read_json(Path(__file__).with_name('runner_progress_resume.json'))
    if (set(old)!=set(new) or any(old[k]!=new[k] for k in old if k!=name)
        or old.get(name)!=compatibility['previous_runner_sha256']
        or new.get(name)!=compatibility['progress_runner_sha256']):
        raise ValueError('Experiment2 code changed beyond the recorded progress-only upgrade; refuse resume')
    receipt={'scope':'Progress display only; original job identity and scientific protocol retained',
        'original_code':old,'current_code':new}
    write_json(out/'provenance/progress_resume_upgrade.json',receipt)
    return previous


class E2Progress:
    """Batch progress including saved work; ETA measures generation plus scoring."""
    def __init__(self,manifest,cfg,args,out):
        self.jobs={}
        for owner in ['clean',*map(str,cfg.owners),*map(str,cfg.stress)]:
            if args.owner is not None and args.owner!=owner:continue
            if owner in map(str,cfg.stress) and (args.pilot_only or args.group not in (None,0)):continue
            for group,chunk,ids in job_chunks(manifest,cfg,owner):
                if args.pilot_only and group!=0:continue
                if args.group is not None and args.group!=group:continue
                if args.chunk is not None and args.chunk!=chunk:continue
                self.jobs[job_name(owner,group,chunk)]=len(ids)
        generated={p.stem for p in (out/'private/jobs').glob('primary_*.json')}
        scores={p.stem for p in (out/'private/scores').glob('primary_*.npz')}
        markers={p.name.removesuffix('.sha256.json') for p in (out/'private/scores').glob('primary_*.sha256.json')}
        # Completion markers are for display only; process_job still verifies
        # the original record/score checksums before reusing each saved job.
        self.done=set(self.jobs)&generated&scores&markers
        self.resumed=len(self.done);self.times=[];self.last=0.
        self.total_responses=sum(self.jobs.values())
        self.show(force=True)

    @staticmethod
    def duration(seconds):
        seconds=max(0,round(seconds));hours,remainder=divmod(seconds,3600)
        minutes,seconds=divmod(remainder,60)
        return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

    def show(self,force=False,active=None):
        now=time.monotonic()
        if not force and now-self.last<30:return
        self.last=now;remaining=len(self.jobs)-len(self.done)
        eta=('00:00:00' if remaining==0 else
            self.duration(sum(self.times)/len(self.times)*remaining) if self.times else 'estimating')
        percent=100*len(self.done)/len(self.jobs) if self.jobs else 100.
        responses=sum(self.jobs[name] for name in self.done)
        label=f' | running {active}' if active else ''
        print(f'E2 progress: {len(self.done):,}/{len(self.jobs):,} batches ({percent:.2f}%)'
            f' | responses {responses:,}/{self.total_responses:,} | resumed {self.resumed:,}'
            f' | ETA ~{eta} (generation + scoring; final aggregation excluded){label}',flush=True)

    def completed(self,name,worked,seconds):
        self.done.add(name)
        if worked:
            self.times.append(seconds);self.times=self.times[-20:]
        self.show(force=worked or len(self.done)==len(self.jobs))


def validate_design(manifest,cfg,receipt,identity):
    expected=digest({'cohort':manifest,'config':cfg.resolved(),'source':receipt})
    if expected!=identity['design_sha256'] or manifest['E2_config']!=cfg.resolved() or manifest['input_receipt_sha256']!=receipt:
        raise ValueError('Frozen design/group manifest changed; refuse analysis')


def prepare_cohort(root,out,frozen,tokenizer,cfg):
    with FileLock(str(out/'cohort.lock')):
        from .vllm_cohort import reuse_frozen_cohort
        reuse_frozen_cohort(root,out,frozen,cfg)
        public=out/'provenance/prompt_manifest.json';private=out/'private/prompts.jsonl'
        if public.exists():
            manifest=read_json(public)
            if not private.exists() or sha256(private)!=manifest['private_prompts_sha256']:raise ValueError('Frozen E2 prompt cohort missing/changed')
            if manifest['input_receipt_sha256']!=frozen['receipt_sha256'] or manifest['E2_config']!=cfg.resolved():raise ValueError('Frozen E2 cohort input/config changed')
            return read_jsonl(private),manifest
        if private.exists():raise ValueError('Uncommitted prompt pool exists; use a new output directory')
        exclusion_path=root/INPUT_FOLDER/'exclusion_index.json'
        exclusion=read_json(exclusion_path)
        write_json(out/'provenance/exclusion_audit.json',{'source_manifests':exclusion['source_manifests'],
            'status':exclusion['status'],'unparsed_manifests':exclusion['unparsed_manifests'],
            'prior_ID_count':len(exclusion['prompt_ids']),'prior_exact_text_hash_count':len(exclusion['exact_text_hashes']),
            'prior_normalized_text_hash_count':len(exclusion['normalized_text_hashes']),
            'prior_IDs_without_text_count':len(exclusion['ids_without_recovered_text']),
            'policy_version':exclusion.get('policy_version',1),
            'inventory_sources_not_counted_as_used':exclusion.get('inventory_sources_not_counted_as_used',[]),
            'exclusion_index_sha256':sha256(exclusion_path)})
        ex={'ids':set(exclusion['prompt_ids']),'exact':set(exclusion['exact_text_hashes']),
            'normalized':set(exclusion['normalized_text_hashes']),
            'legacy_normalized':set(exclusion.get('legacy_normalized_text_hashes',[]))}
        from datasets import load_dataset
        datasets={s:load_dataset('HuggingFaceH4/ultrachat_200k',split=s,revision=frozen['dataset_revision']) for s in ['test_gen','train_gen']}
        rows=select_cohort(datasets,tokenizer,ex,cfg.prompt_count,cfg.prompt_seed)
        for row in rows:row['prompt_ids']=prompt_ids(tokenizer,row['prompt'])
        groups=make_groups(rows,cfg.group_seed)
        selection=np.random.default_rng(cfg.group_seed+1).permutation(len(rows))
        quality=[rows[int(i)]['prompt_id'] for i in selection[:cfg.quality_count]]
        private.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
        temp=private.with_name(private.name+'.tmp')
        with temp.open('w',encoding='utf-8',newline='\n') as f:
            for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
        os.replace(temp,private)
        manifest={'schema':1,'E2_config':cfg.resolved(),'input_receipt_sha256':frozen['receipt_sha256'],
            'dataset':'HuggingFaceH4/ultrachat_200k','dataset_revision':frozen['dataset_revision'],
            'source_split_counts':{s:sum(r['source_split']==s for r in rows) for s in ['test_gen','train_gen']},
            'cohort_provenance_status':exclusion['status'],'exclusion_manifest_count':len(exclusion['source_manifests']),
            'unparsed_exclusion_manifests':exclusion['unparsed_manifests'],
            'unknown_prior_prompt_text_count':len(exclusion['ids_without_recovered_text']),
            'private_prompts_sha256':sha256(private),'groups':groups,'quality_prompt_ids':quality,
            'smoke_prompt_ids':quality[:cfg.smoke_count],
            'smoke_policy':'Correctness outputs use separate phase seeds and never enter primary observations; primary pool was frozen before smoke',
            'rows':[{k:v for k,v in r.items() if k not in ('prompt','prompt_ids')} for r in rows]}
        write_json(public,manifest)
        return rows,manifest


def runtime_environment():
    from importlib.metadata import version
    return {'python':platform.python_version(),'torch':torch.__version__,
        **{name:version(name) for name in ['numpy','scipy','transformers','datasets','filelock']},
        'CUDA':torch.version.cuda,'GPU':torch.cuda.get_device_name(0),'dtype':'float32',
        'matmul_precision':torch.get_float32_matmul_precision(),'TF32':torch.backends.cuda.matmul.allow_tf32,
        'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
        'engine':'Custom vLLM exact pre-final-RMSNorm latent rotation; HF reference verification only'}


def job_identity(owner,group,chunk,ids,phase,base):
    return {'owner':owner,'group':group,'chunk':chunk,'prompt_ids':ids,'phase':phase,'frozen_run_sha256':base}


def write_engine_manifest(out,environment,peak=None):
    with FileLock(str(out/'initialization.lock')):
        path=out/'provenance/engine_manifest.json'
        previous=read_json(path) if path.exists() else {}
        row=dict(environment)
        row['peak_CUDA_allocated_bytes']=max(previous.get('peak_CUDA_allocated_bytes',0),peak or 0)
        write_json(path,row)


def job_name(owner,group,chunk,phase='primary'):return f'{phase}_owner{owner}_g{group:02}_chunk{chunk:04}'


def job_chunks(manifest,cfg,owner):
    for group in manifest['groups']:
        if group['Q']!=1000 or owner in map(str,cfg.stress) and group['group_index']!=0:continue
        for n,start in enumerate(range(0,1000,cfg.batch_size)):
            yield group['group_index'],n,group['prompt_ids'][start:start+cfg.batch_size]


def process_job(owner,group,chunk,ids,phase,lookup,sampler,verifier,store,cfg,base,out,quality_ids):
    name=job_name(owner,group,chunk,phase);identity=job_identity(owner,group,chunk,ids,phase,base)
    with FileLock(str(store.path(name))+'.lock',timeout=0):
        records=store.load(name,identity)
        if records is None:
            requests=[{'prompt_id':pid,'prompt_ids':lookup[pid]['prompt_ids'],
                'seed':generation_seed(cfg.generation_seed,owner,pid,phase)} for pid in ids]
            key=None if owner=='clean' else sampler.bank[int(owner)]
            records=sampler.generate(requests,key,max_new_tokens=cfg.max_new_tokens)
            for record in records:
                record.update(owner_key_id=owner,group_index=group,chunk_index=chunk,phase=phase,
                    response_id=digest({'owner':owner,'prompt':record['prompt_id'],'seed':record['seed'],
                        'tokens':record['generated_token_ids']}))
            store.commit(name,identity,records)
        if phase=='smoke':return False
        cache=out/'private/scores'/(name+'.npz');marker=cache.with_suffix('.sha256.json')
        expected=digest(records)
        if marker.exists():
            m=read_json(marker)
            if not cache.exists() or sha256(cache)!=m['sha256'] or m['records_sha256']!=expected:
                raise ValueError('Intermediate score cache changed: '+name)
            return False
        if sampler.readout.weight.device.type=='cuda':torch.cuda.synchronize()
        start=time.perf_counter()
        values,tokens=verifier.score_text_batch([{'prompt':lookup[r['prompt_id']]['prompt'],'response':r['response']} for r in records])
        native=np.zeros_like(values);native_present=np.zeros(len(records),dtype=bool)
        pick=[i for i,r in enumerate(records) if r['prompt_id'] in quality_ids[:cfg.native_control_per_owner]]
        if pick and owner!='clean':
            v,_=verifier.score_native_batch([{'prompt':lookup[records[i]['prompt_id']]['prompt'],
                'generated_token_ids':records[i]['generated_token_ids']} for i in pick])
            native[pick]=v;native_present[pick]=True
        if sampler.readout.weight.device.type=='cuda':torch.cuda.synchronize()
        save_npz(cache,vectors=values,eligible_tokens=tokens,native_vectors=native,native_present=native_present,
            prompt_ids=np.array(ids),scoring_seconds=np.array(time.perf_counter()-start))
        write_json(marker,{'sha256':sha256(cache),'records_sha256':expected,'primary_path':'canonical_visible_text_only'})
        return True


def collect(manifest,store,cfg,base,out):
    observations={};native_rows=[];receipts=[];membership=[];generation_counts={}
    for owner in ['clean',*map(str,cfg.owners),*map(str,cfg.stress)]:
        records={};vectors={};generation_counts[owner]=0
        for group,chunk,ids in job_chunks(manifest,cfg,owner):
            name=job_name(owner,group,chunk);identity=job_identity(owner,group,chunk,ids,'primary',base)
            batch=store.load(name,identity)
            if batch is None:continue
            generation_counts[owner]+=len(batch)
            path=out/'private/scores'/(name+'.npz');marker=path.with_suffix('.sha256.json')
            if not marker.exists():continue
            m=read_json(marker)
            if sha256(path)!=m['sha256'] or digest(batch)!=m['records_sha256']:raise ValueError('Merged generation/score checksum mismatch')
            with np.load(path,allow_pickle=False) as f:data=dict(f)
            if data['prompt_ids'].tolist()!=ids:raise ValueError('Score prompt ordering mismatch')
            for i,r in enumerate(batch):
                pid=r['prompt_id']
                if pid in records:raise ValueError('Duplicate response while merging shards')
                records[pid]=dict(r,eligible_tokens=int(data['eligible_tokens'][i]),
                    scoring_amortized_seconds=float(data['scoring_seconds'])/len(batch))
                vectors[pid]=data['vectors'][i]
                membership.append({'owner':owner,'prompt_id':pid,'group_index':group,
                    'response_id':r['response_id'],'job_id':name})
                if data['native_present'][i]:native_rows.append((owner,r,data['vectors'][i],data['native_vectors'][i]))
            receipts.append({'job_id':name,'private_generation_sha256':sha256(store.path(name)),
                'private_score_sha256':m['sha256'],'responses':len(batch)})
        observations[owner]=(records,vectors)
        merged=out/'private'/('clean_generations.jsonl' if owner=='clean' else f'teacher_generations_key_{int(owner):02}.jsonl')
        temporary=merged.with_name(merged.name+'.tmp')
        with temporary.open('w',encoding='utf-8',newline='\n') as f:
            for pid in sorted(records):f.write(json.dumps(records[pid],ensure_ascii=False)+'\n')
        os.replace(temporary,merged)
    write_json(out/'provenance/private_generation_manifest.json',receipts)
    write_json(out/'provenance/group_response_membership.json',{'prompt_groups':manifest['groups'],'response_membership':membership})
    return observations,native_rows,generation_counts


def write_controls(observations,native_rows,frozen,manifest,out,cfg):
    controls=[]
    for owner,r,text,native in native_rows:
        key=frozen['bank'][int(owner)]
        def score(v):
            norm=np.linalg.norm(v);return float(key@v/norm) if norm else 0.,float(sphere_p(v,key))
        t,p=score(text);nt,np_=score(native)
        controls.append({'owner_key_id':key_name(int(owner)),'prompt_id':r['prompt_id'],
            'roundtrip_equal':r['roundtrip_equal'],'text_cosine':t,'text_p':p,'native_cosine':nt,'native_p':np_,
            'absolute_cosine_shift':abs(t-nt),'nominal_flip':(p<=.01)!=(np_<=.01),
            'conservative_flip':(p<=.0001)!=(np_<=.0001),'scope':'Privileged auxiliary only; not primary ownership trials'})
    write_csv(out/'scores/text_only_vs_native_control.csv',controls,list(controls[0]) if controls else ['owner_key_id','data_status'])
    stress=[]
    from .analysis import _batch
    for i in cfg.stress:
        records,vectors=observations[str(i)]
        for group in manifest['groups']:
            if group['group_index']==0:
                row,_=_batch(i,group,frozen['bank'],records,vectors);row['scope']='E1-prelabelled supplementary stress only';stress.append(row)
    write_csv(out/'scores/fixed_key_stress_controls.csv',stress,list(stress[0]))
    write_json(out/'tests/text_roundtrip_checks.json',{'status':'REPORTED','auxiliary_outputs':len(controls),
        'mismatches':sum(not x['roundtrip_equal'] for x in controls),
        'note':'Primary uses text only; mismatches retained. No privileged-token substitution.'})


def analyze(root,out,frozen,manifest,store,cfg,base,pilot=False):
    from .report import write_report,package
    with FileLock(str(out/'analysis.lock')):
        print('E2 phase: aggregating '+('pilot' if pilot else 'current/final')+' results and writing reports...',flush=True)
        observations,native,generated=collect(manifest,store,cfg,base,out)
        statistical=analyze_groups(manifest['groups'],frozen['bank'],observations,out,cfg)
        quality=diagnostics(observations,out,cfg)
        import csv
        prior={r['key_index']:r for r in csv.DictReader((root/'experiment1_rs_continuation/teacher/KL_by_key.csv').open(encoding='utf-8'))}
        calibration=read_json(root/'experiment1_rs_continuation/teacher/global_lambda.json')['pooled_search_trace'][-1]
        for r in quality['KL']:
            old=prior.get(r['owner'])
            r.update(E1_local_mean_KL=float(old['mean_KL']) if old else 0. if r['owner']=='clean' else None,
                E1_local_p99_KL=float(old['p99_KL']) if old else 0. if r['owner']=='clean' else None,
                E1_pooled_calibration_mean=calibration['mean'],E1_pooled_calibration_p99=calibration['p99'])
        write_csv(out/'utility/free_context_KL_by_key.csv',quality['KL'],list(quality['KL'][0]) if quality['KL'] else ['owner'])
        perKL={r['owner']:r for r in quality['KL']};perQuality={r['owner']:r for r in quality['quality']}
        for r in statistical['per_key']:
            owner=str(int(r['owner_key_id'].rsplit('_',1)[1]));k=perKL.get(owner,{});q=perQuality.get(owner,{})
            for field in ['mean','p95','p99','max','fraction_above_0_05']:r['free_context_KL_'+field]=k.get(field)
            for field in ['tokens_mean','characters_mean','loop_rate_or_mean','EOS_rate_or_mean','truncated_rate_or_mean','roundtrip_rate_or_mean']:
                r['utility_'+field]=q.get(field)
        fields=list(dict.fromkeys(k for row in statistical['per_key'] for k in row))
        write_csv(out/'scores/per_key_summary.csv',statistical['per_key'],fields)
        write_controls(observations,native,frozen,manifest,out,cfg)
        examples(read_jsonl(out/'private/prompts.jsonl'),observations,manifest['quality_prompt_ids'],out,cfg)
        concerns={**statistical['concerns'],**quality['concerns'],
            'cohort_provenance_concern':manifest['cohort_provenance_status']!='VERIFIED_LOADED_MANIFESTS' or manifest['unknown_prior_prompt_text_count']>0}
        from .vllm_gates import runtime_configuration_valid
        valid=runtime_configuration_valid(out)
        decision=decide(statistical['complete'],valid,statistical['conservative_counts'],statistical['nominal_counts'],concerns)
        summary={'status':'PILOT_ONLY' if pilot else decision,'decision':decision,'complete':statistical['complete'],
            'sampler_valid':valid,'sampler_valid_scope':'Frozen model/readout configuration; numerical parity is not an acceptance condition',
            'numerical_diagnostics':'OPTIONAL_NON_BLOCKING','primary_expected_teacher_responses':cfg.prompt_count*len(cfg.owners),
            'primary_actual_teacher_responses':sum(len(observations[str(i)][0]) for i in cfg.owners),
            'clean_expected_responses':cfg.prompt_count,'clean_actual_responses':len(observations['clean'][0]),
            'committed_generation_counts':generated,'count_definition':'Actual responses above are committed AND canonically scored; raw committed but unscored outputs are counted separately here',
            'Q1000_conservative_counts_by_owner':statistical['conservative_counts'],
            'Q1000_nominal_counts_by_owner':statistical['nominal_counts'],'concerns':concerns,
            'descriptive_pooled':statistical['pooled'],'utility':quality,
            'source_split_counts':manifest['source_split_counts'],'scientific_success':False,
            'scope':'Exact-teacher feasibility only; stop for review. No LoRA authorization, fixed-key certification or paraphrase/SaTML claim.'}
        write_report(out,summary,statistical,cfg)
        package(root,out)
        return summary


def execute(root,out,frozen,cfg,args):
    from .vllm_backend import VLLMExactRotationTeacherSampler,require_vllm
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True,mode=0o700)
    (out/'private').mkdir(exist_ok=True,mode=0o700)
    if args.analyze_only:
        manifest=read_json(out/'provenance/prompt_manifest.json')
        identity=read_json(out/'private/jobs/identity.json')
        identity=resume_logging_identity(out,dict(identity,code=code_hashes(root)))
        if identity['source_receipt_sha256']!=frozen['receipt_sha256'] or identity['config']!=cfg.resolved():
            raise ValueError('Analysis resume code/source/config changed')
        validate_design(manifest,cfg,frozen['receipt_sha256'],identity)
        if sha256(out/'private/prompts.jsonl')!=manifest['private_prompts_sha256']:raise ValueError('Frozen private prompt pool changed')
        store=JobStore(out/'private/jobs',identity)
        return analyze(root,out,frozen,manifest,store,cfg,identity['design_sha256'])
    if not torch.cuda.is_available():raise RuntimeError('GPU_UNAVAILABLE: full E2 requires CUDA. No toy/CPU substitution for the45000-response study.')
    require_vllm() # Fail on missing/incompatible engine before loading reference weights.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    from .sampler import E2ModelAdapter
    from ..config import Config
    adapter=E2ModelAdapter(Config());tokenizer=adapter.tokenizer
    if tokenizer.chat_template!=frozen['chat_template']:raise ValueError('Frozen chat template changed')
    engine=adapter.readout(frozen['P'],frozen['Q'])
    from ..continuation import tensor_sha256
    expected=read_json(root/'experiment1_rs_continuation/numerical/fp64_diagnostic.json')['readout_parameter_sha256']
    actual={'weight':tensor_sha256(engine.weight),'gain':tensor_sha256(engine.gain),
        'bias':None if engine.bias is None else tensor_sha256(engine.bias)}
    if actual!=expected:raise ValueError('BLOCKED_MISSING_OR_CHANGED_ARTIFACT: model readout parameter hashes differ from E1')
    rows,manifest=prepare_cohort(root,out,frozen,tokenizer,cfg);lookup={r['prompt_id']:r for r in rows}
    sampler=VLLMExactRotationTeacherSampler(adapter,engine,frozen,cfg)
    if sampler.environment['readout_hashes']!=[expected]:
        sampler.close();raise ValueError('BLOCKED_MISSING_OR_CHANGED_ARTIFACT: vLLM weights/norm differ from frozen E1')
    environment=runtime_environment();environment.update(generation=sampler.environment,
        verification_attention_implementation='eager',
        verification_linear_accumulation=adapter.model._e2_linear_accumulation,
        linear_policy='Generation: vLLM FP32, no FP64 Linear. Existing HF verifier: FP32 boundaries/FP64 Linear accumulation.',
        native_EOS_ids=adapter.model.generation_config.eos_token_id,
        readout_parameter_sha256=actual,prior_E1_precision='FP32',backbone_precision='FP32',
        sampling='Full-vocabulary categorical from exact rotated teacher; no truncation or penalties')
    design=digest({'cohort':manifest,'config':cfg.resolved(),'source':frozen['receipt_sha256']})
    identity={'code':code_hashes(root),'config':cfg.resolved(),'source_receipt_sha256':frozen['receipt_sha256'],
        'environment':environment,'design_sha256':design}
    with FileLock(str(out/'initialization.lock')):
        identity=resume_logging_identity(out,identity)
        store=JobStore(out/'private/jobs',identity)
        write_json(out/'provenance/input_manifest.json',frozen['receipt'])
        write_json(out/'provenance/resolved_config.json',{'E2':cfg.resolved(),'E1':frozen['e1_config']})
        write_json(out/'provenance/key_manifest.json',{'bank_ids':key_ids(frozen['bank']),
            'primary_owners':list(cfg.owners),'wrong_keys':{str(i):wrong_indices(i) for i in cfg.owners},
            'supplementary_stress':list(cfg.stress),'scope':'IDs/hashes only; vectors are never persisted/publicly packaged'})
        write_json(out/'provenance/sha256_manifest.json',{'source':frozen['receipt'],'code':code_hashes(root),
            'private_prompt_pool':manifest['private_prompts_sha256']})
    write_engine_manifest(out,environment)
    verifier=SphericalBlackBoxVerifier(adapter.model,tokenizer,engine,frozen['L'],cfg.score_chunk)
    torch.cuda.reset_peak_memory_stats()
    with FileLock(str(out/'sampler_validation.lock')):
        smoke_ids=manifest['smoke_prompt_ids']
        for owner in ['clean','0']:
            for j,start in enumerate(range(0,len(smoke_ids),cfg.batch_size)):
                process_job(owner,0,j,smoke_ids[start:start+cfg.batch_size],'smoke',lookup,sampler,verifier,store,cfg,design,out,manifest['quality_prompt_ids'])
    print('Stage0 correctness smoke complete; its outputs are excluded from all primary statistics.',flush=True)
    jobs_done=0
    progress=E2Progress(manifest,cfg,args,out)
    groups_to_run=[0] if args.pilot_only else list(range(5))
    for group in groups_to_run:
        for owner in ['clean',*map(str,cfg.owners)]:
            if args.owner is not None and args.owner!=owner:continue
            if args.group is not None and args.group!=group:continue
            for g,chunk,ids in job_chunks(manifest,cfg,owner):
                if g!=group or args.chunk is not None and args.chunk!=chunk:continue
                if args.max_jobs is not None and jobs_done>=args.max_jobs:
                    write_engine_manifest(out,environment,torch.cuda.max_memory_allocated())
                    sampler.close()
                    return analyze(root,out,frozen,manifest,store,cfg,design)
                name=job_name(owner,g,chunk)
                if name not in progress.done:progress.show(force=True,active=name)
                begin=time.monotonic()
                try:worked=process_job(owner,g,chunk,ids,'primary',lookup,sampler,verifier,store,cfg,design,out,manifest['quality_prompt_ids'])
                except Timeout:continue
                jobs_done+=int(worked)
                progress.completed(name,worked,time.monotonic()-begin)
        if group==0:analyze(root,out,frozen,manifest,store,cfg,design,pilot=True)
    if not args.pilot_only and args.group in (None,0):
        for owner in map(str,cfg.stress):
            if args.owner is not None and args.owner!=owner:continue
            for g,chunk,ids in job_chunks(manifest,cfg,owner):
                if args.chunk is not None and args.chunk!=chunk:continue
                if args.max_jobs is not None and jobs_done>=args.max_jobs:break
                name=job_name(owner,g,chunk)
                if name not in progress.done:progress.show(force=True,active=name)
                begin=time.monotonic()
                try:worked=process_job(owner,g,chunk,ids,'primary',lookup,sampler,verifier,store,cfg,design,out,manifest['quality_prompt_ids'])
                except Timeout:continue
                jobs_done+=int(worked)
                progress.completed(name,worked,time.monotonic()-begin)
    environment['peak_vllm_worker_CUDA_allocated_bytes']=sampler.peak_worker_bytes
    write_engine_manifest(out,environment,torch.cuda.max_memory_allocated())
    sampler.close()
    with FileLock(str(out/'initialization.lock')):
        write_json(out/'provenance/e1_fixed_key_diagnostic.json',read_json(root/'experiment1_rs_continuation/null/fixed_key_pilot.json'))
    from .frozen import load_frozen
    load_frozen(root) # Reverify E1 inputs remained unchanged; never writes to them.
    return analyze(root,out,frozen,manifest,store,cfg,design,pilot=args.pilot_only)
