"""Reviewable feasibility report and private-data-free package."""
from pathlib import Path
import hashlib,json,zipfile,os
from ..io import write_json,read_json
from .anchor import INPUT_FOLDER
from .vllm_runtime import TARGET_VERSION


LIMITATIONS='''The spherical null is exact over independent random keys conditional on fixed nonzero V,
not a blanket fixed-key FPR guarantee. E2 measures fixed-key empirical behavior on this
specified pool. Q1000 has only five disjoint groups per owner (20% rate increments).
Q100 groups are nested in Q1000; budgets are not independent. Eight owners share
prompts and the same5000 clean responses. Wrong-key scores reuse owner outputs.
Neither key reuse, response/token bootstrap nor wrong-key scoring creates extra
independent ownership trials. Pooled counts are descriptive, with no pooled CI.
Privileged-token controls are secondary; only returned-text verification is primary.
E1 cohorts and prelabelled stress23/59 remain development diagnostics. Stress keys
are not pooled with the representative primary8. Occasional KL violations, including
E1 key2, are retained. Automatic utility diagnostics and limited task-hint heuristics
do not establish broad capability preservation. The key acts at offline teacher
sampling time; no student weights or LoRA have been trained. Even a promising label
requires review and does not authorize Experiment3, PSS, paraphrasing or SaTML.
'''


def write_report(out,summary,statistical,cfg):
    write_json(out/'report/summary.json',summary)
    lines=['# Experiment 2: exact-teacher free generation','',f"Status: **{summary['status']}**",'',
        f"Teacher outputs: {summary['primary_actual_teacher_responses']}/{cfg.prompt_count*len(cfg.owners)} canonically scored; clean: {summary['clean_actual_responses']}/{cfg.prompt_count}.",
        f'Generation: custom vLLM {TARGET_VERSION} V1 FP32 with exact pre-final-RMSNorm latent rotation and native KV cache. No FP64 Linear or HF generation fallback. Verification retains the existing HF FP32-boundary/FP64-Linear policy; readout/probability/KL stay FP32.','',
        '|Owner|Q|Disjoint groups|Owner nominal|Owner conservative|Clean conservative|Wrong conservative mean / worst|21-way correct / trials|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in statistical['per_key']:
        n=row['n_disjoint_groups'];wrong=row['wrong_conservative_mean'];worst=row['wrong_conservative_worst']
        lines.append(f"|{row['owner_key_id']}|{row['Q']}|{n}/{row['planned_disjoint_groups']}|{row['owner_nominal_count']}/{n}|{row['owner_conservative_count']}/{n}|{row['clean_conservative_count']}/{row['clean_disjoint_groups']}|{wrong} / {worst}|{row['identification_count']}/{row['identification_trials']}|")
    lines += ['', 'Full counts, binomial intervals and score quantiles are in scores/per_key_summary.csv; all20 wrong-key raw scores remain in batch_wrong_key_scores.csv.',
        '', '## Predeclared decision concerns','',json.dumps(summary['concerns'],indent=2),
        '', 'Primary promising rule: >=32/40 conservative Q1000 detections, at least6/8 owners with>=4/5, and no wrong/clean/quality/distortion/provenance concern.',
        'No incomplete or invalid-sampler run receives a scientific GO. Pilot is labelled PILOT_ONLY and never selects keys or changes lambda.',
        '', '## Prompt source','',str(summary['source_split_counts']),
        'Any train_gen fallback is explicitly mixed-split, not a pristine test_gen final set.',
        '', '## Utility and free-context KL','',
        'See utility/generation_quality_by_key.csv and free_context_KL_by_key.csv. KL samples are full-vocabulary on teacher-visited prefixes using a fixed key-independent hashed1/16 position mask.',
        'The frozen E1 p99 violation at key2 is retained; E2 never rescales this key. Selected quality pairs and examples were fixed before generation.',
        'Recorded GPU hours include generation prefill/decode and canonical verification; session wall times and peak allocation are in provenance. Discarded work from uncommitted interrupted jobs can make per-response costs a lower bound.',
        '', '## Interpretation and limits','',LIMITATIONS,'',
        'STOP: Experiment2 completed or stopped for review. No subsequent experiment is launched.','']
    (out/'report/experiment2_report.md').write_text('\n'.join(lines),encoding='utf-8')
    (out/'report/findings_and_limitations.md').write_text(LIMITATIONS+'\n\n'+json.dumps(summary['concerns'],indent=2)+'\n',encoding='utf-8')


def package(root,out):
    root,out=Path(root),Path(out);target=out/'KLRD_RS_experiment2_review.zip';temp=target.with_name(target.name+'.tmp')
    files={}
    for folder in ['provenance','tests','scores','utility','report']:
        for p in (out/folder).rglob('*'):
            if p.is_file() and p.suffix not in ('.tmp','.zip','.lock'):files['experiment2/'+p.relative_to(out).as_posix()]=p
    for folder in ['klrd_rs/e2']:
        for p in (root/folder).glob('*.py'):files['code/'+p.relative_to(root).as_posix()]=p
    if (root/'tests/test_exp2.py').exists():files['code/tests/test_exp2.py']=root/'tests/test_exp2.py'
    for name in ['main_exp2.py','run_experiment2.sh','README_EXPERIMENT2.md','README_VLLM_EXPERIMENT2.md','README_NUMERICAL_POLICY.md','requirements.txt','requirements-vllm.txt','gpt_guide/KLRD_RS_EXPERIMENT_2_CODEX_SPEC.md','gpt_guide/KLRD_RS_EXPERIMENT_2_VLLM_SPEC.md']:
        p=root/name
        if p.is_file():files['code/'+name]=p
    for name in ['__init__.py','math.py','data.py','readout.py','model.py','io.py','config.py','diagnostics.py']:
        files['code/klrd_rs/'+name]=root/'klrd_rs'/name
    if (root/INPUT_FOLDER/'source_receipt.json').exists():files['frozen_source_receipt.json']=root/INPUT_FOLDER/'source_receipt.json'
    for name in ['tests/test_exp2_exclusion_fix.py','tests/test_exp2_attention_fix.py',
        'tests/test_exp2_vllm_contract.py','tests/test_vllm_v1_migration.py','tests/test_e2_numerical_policy.py','tools/check_vllm_exp2.py','tools/prepare_exp2_inputs_v2.py']:
        if (root/name).exists():files['code/'+name]=root/name
    if (root/'tools/prepare_exp2_inputs.py').exists():files['code/tools/prepare_exp2_inputs.py']=root/'tools/prepare_exp2_inputs.py'
    manifest={}
    with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,p in sorted(files.items()):
            data=p.read_bytes();z.writestr(name,data);manifest[name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        cover=('Read experiment2/report/summary.json and experiment2_report.md first.\n'
            'Incomplete, pilot or invalid sampler results are not full-study scientific results.\n'
            'Primary text-only scores, all keys, wrong-key controls and selected examples are retained.\n'
            'Private raw responses, full prompt datasets, keys, score caches and machine credentials are excluded.\n'
            'For a full rerun retain the original frozen E1 workspace and '+INPUT_FOLDER+' separately.\n').encode()
        z.writestr('START_HERE.md',cover);manifest['START_HERE.md']={'sha256':hashlib.sha256(cover).hexdigest(),'bytes':len(cover)}
        z.writestr('SHA256_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:raise ValueError('E2 review ZIP failed CRC')
        for name,item in manifest.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=item['sha256']:raise ValueError('E2 ZIP SHA256 mismatch')
    os.replace(temp,target)
    return target
