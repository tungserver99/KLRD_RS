"""GPU/API-free reporting and explicit allowlist-only private-safe review ZIP."""
from pathlib import Path
import csv
import html
import json
import zipfile
from ..io import read_json, write_json, sha256


PUBLIC_TABLES=(
    'scores/per_key_group.csv','scores/per_response.csv','scores/closed_set_identification.csv',
    'scores/clean_wrong_controls.csv','scores/group_summary.csv','scores/teacher_student_gap.csv',
    'paraphrase/metrics.csv','paraphrase/quality.csv','paraphrase/quality_summary.json',
    'paraphrase/protocol.json','paraphrase/scores/per_key_group.csv',
    'paraphrase/scores/per_response.csv','paraphrase/scores/closed_set_identification.csv',
    'paraphrase/scores/clean_wrong_controls.csv','paraphrase/scores/group_summary.csv',
    'utility/generation_summary.json','utility/kl_summary.csv','training/validation.csv','training/epochs.csv')
PUBLIC_PROVENANCE=('provenance/resolved_config.json','provenance/input_manifest.json',
    'provenance/environment.json','provenance/current_source_manifest.json',
    'provenance/historical_heldout_audit.json',
    'provenance/evaluation_complete.json','provenance/merged_manifest.json',
    'provenance/verification_environment_comparison.json','paraphrase/complete.json')
TRAINING_SUMMARIES=('training/baseline.json','training/selection.json','training/epochs.json',
    'training/validation.jsonl','training/steps.jsonl')


def checked(state, relative):
    path=state['out']/relative
    if not path.is_file():return None
    value=read_json(path)
    if value['identity']!=state['identity']:raise ValueError('Run identity mismatch in '+relative)
    for name,expected in value['products_sha256'].items():
        # Treat receipts as untrusted paths. A receipt cannot authorize publishing
        # an outside file; packaging additionally uses a literal safe allowlist.
        target=(state['out']/name).resolve()
        if state['out'].resolve() not in target.parents or sha256(target)!=expected:
            raise ValueError('Changed/invalid completed product: '+name)
    return value


def _rows(path):
    if not Path(path).is_file():return []
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def _text(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp');temp.write_text(value,encoding='utf-8');temp.replace(path)


def chart(out,rows):
    selected=[r for r in rows if int(r['Q'])==100 and float(r['alpha'])==.01]
    if not selected:return None
    width=760;height=70+len(selected)*42
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" fill="#172033"><text x="20" y="25" font-size="17">Q100 owner-key detection at alpha=1% (10 groups)</text>']
    for i,r in enumerate(selected):
        y=55+i*42;rate=float(r['rate']);label=html.escape(r['model'])
        parts += [f'<text x="20" y="{y+15}" font-size="13">{label}</text>',
            f'<rect x="230" y="{y}" width="400" height="22" fill="#eef2f6"/>',
            f'<rect x="230" y="{y}" width="{400*rate:.1f}" height="22" fill="#4668bd"/>',
            f'<text x="645" y="{y+16}" font-size="13">{rate:.0%}</text>']
    parts+=['</g></svg>'];path=out/'report/q100_detection.svg';_text(path,'\n'.join(parts)+'\n')
    return 'report/q100_detection.svg'


def package_review(state):
    """No directory-level artifact inclusion; only named summaries and source code."""
    root,out=state['root'],state['out'];files={}
    code=['main_exp3.py','run_experiment3.sh','requirements-lora.txt','requirements-paraphrase.txt',
        'README_EXPERIMENT3.md','gpt_guide/KLRD_RS_Experiment3_LoRA_Codex_Spec_REVISED.md',
        'docs/superpowers/plans/2026-11-03-experiment3-lora.md']
    code += [p.relative_to(root).as_posix() for directory in ['klrd_rs','klrd_scw']
        for p in (root/directory).rglob('*.py')]
    code += [p.relative_to(root).as_posix() for p in (root/'tests').glob('test_exp3*.py')]
    for name in code:
        path=root/name
        if path.is_file():files['source/'+name]=path
    public=PUBLIC_TABLES+PUBLIC_PROVENANCE+TRAINING_SUMMARIES+(
        'report/summary.json','report/experiment3_report.md','report/q100_detection.svg','report/kl_curve.svg')
    for name in public:
        path=out/name
        if path.is_file():files['run/'+name]=path
    # This manifest names hashes, never copies the private artifacts themselves.
    private_manifest={}
    for name in ['private/eval_responses.jsonl','private/untrained_responses.jsonl',
            'private/train_manifest.jsonl','private/val_manifest.jsonl','private/student_paraphrase_pairs.jsonl']:
        if (out/name).is_file():private_manifest[name]=sha256(out/name)
    manifest={'schema':1,'privacy_policy':'Explicit file allowlist; no raw prompts/responses, weights, adapters, owner key vectors or API receipts/secrets',
        'included_sha256':{name:sha256(path) for name,path in sorted(files.items())},
        'private_artifact_hashes_not_included':private_manifest}
    write_json(out/'report/review_manifest.json',manifest)
    files['run/report/review_manifest.json']=out/'report/review_manifest.json'
    path=out/'KLRD_RS_experiment3_review.zip';temp=path.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name,source in sorted(files.items()):archive.write(source,name)
    # Verify central-directory exact membership before publishing. This executes
    # only in explicit server report mode; no inference, API or weight loading.
    with zipfile.ZipFile(temp) as archive:
        if set(archive.namelist())!=set(files) or archive.testzip() is not None:
            raise ValueError('Review ZIP integrity mismatch')
    temp.replace(path)
    return {'path':path.name,'sha256':sha256(path),'files':len(files)}


def run(state,package=True):
    out=state['out'];errors=[];evaluation=attack=None
    training_rows=training_tables(out)
    try:evaluation=checked(state,'provenance/evaluation_complete.json')
    except (ValueError,KeyError,FileNotFoundError) as error:errors.append(str(error))
    try:attack=checked(state,'paraphrase/complete.json')
    except (ValueError,KeyError,FileNotFoundError) as error:errors.append(str(error))
    selection=read_json(out/'training/selection.json') if (out/'training/selection.json').is_file() else None
    training_valid=False
    merged_path=out/'provenance/merged_manifest.json'
    if selection is not None:
        try:
            merged=read_json(merged_path)
            if selection.get('identity')!=state['identity'] or merged.get('identity')!=state['identity']:
                raise ValueError('Selected adapter/merged model identity differs')
            for directory,files in [(out/'checkpoints/best_adapter',selection['adapter_files']),
                    (out/'merged_student',merged['files'])]:
                for name,expected in files.items():
                    target=(directory/name).resolve()
                    if directory.resolve() not in target.parents or not target.is_file():
                        raise ValueError('Selected/merged checkpoint file missing: '+name)
                    # Reporting validates small configuration files; evaluate
                    # performs the full merged-weight content hash verification.
                    if target.suffix=='.json' and sha256(target)!=expected:
                        raise ValueError('Selected/merged checkpoint configuration changed: '+name)
            if 'config.json' not in merged['files'] or not selection['adapter_files']:
                raise ValueError('Incomplete selected/merged checkpoint manifest')
            training_valid=True
        except (ValueError,KeyError,FileNotFoundError) as error:errors.append(str(error))
    if evaluation and not training_valid:errors.append('Selected adapter/merged export evidence is incomplete despite existing evaluation artifacts')
    eligibility=read_json(out/'training/eligibility.json') if (out/'training/eligibility.json').is_file() else None
    counts=None if eligibility is None else {'eligible_sequences':eligibility['eligible_sequences'],
        'eligible_response_positions':eligibility['eligible_tokens'],'excluded_sequence_count':len(eligibility['excluded'])}
    status='INCOMPLETE'
    if not errors:
        if evaluation and evaluation.get('status')=='EVAL_COMPLETE':status='EVAL_COMPLETE'
        elif training_valid:status='TRAINING_ONLY'
        if attack and evaluation and attack.get('status')=='PARAPHRASE_COMPLETE':status='PARAPHRASE_COMPLETE'
    if (out/'paraphrase/protocol.json').exists() and attack is None:
        errors.append('Paraphrase was prepared/started but has no complete receipt')
        status='INCOMPLETE'
    if (out/'private/eval_responses.jsonl').exists() and evaluation is None:
        errors.append('Student evaluation started but has no complete receipt');status='INCOMPLETE'
    rows=_rows(out/'scores/group_summary.csv') if evaluation else []
    summary={'status':status,'complete':status in ('TRAINING_ONLY','EVAL_COMPLETE','PARAPHRASE_COMPLETE'),'identity':state['identity'],'training_completed':training_valid,
        'training_evidence_status':selection.get('status') if selection else 'TRAINING_INCOMPLETE',
        'effective_training_counts':counts,
        'report_checkpoint_policy':'Manifest identity, complete file presence and JSON content hashes; evaluate verifies all merged weight hashes',
        'checkpoint_selection':'Minimum held-out clean-context teacher KL; E2/E2B never select checkpoints',
        'evaluation_complete':bool(evaluation),'paraphrase_complete':bool(attack),'errors':errors,
        'scientific_success':None,'owner_key_id':0,'primary_Q':100,'primary_groups':10 if evaluation else 0,
        'Q1000_scope':'One descriptive aggregate, no independent-group confidence interval',
        'evaluation_summary':evaluation.get('summary') if evaluation else None,
        'paraphrase_summary':attack.get('summary') if attack else None,
        'untrained_control':evaluation.get('untrained_control') if evaluation else None,
        'limitations':['One predeclared owner key, feasibility only; no multi-key reliability claim.',
            'Historical data exclusion covers discovered known roles/layouts only; missing/unknown histories are listed in provenance/historical_heldout_audit.json.',
            'Groups independent/disjoint within each Q; Q20/Q50/Q100 share nested prompts.',
            'Wrong keys reuse the same responses; do not pool them as independent deployments.',
            'Clean detections are diagnostics; analytical random-key null is not a universal fixed-key FPR guarantee.',
            'Teacher detection and reduced training KL do not establish student black-box transfer.',
            'Paraphrase lexical/length metrics do not establish semantic fidelity; human review remains separate.']}
    write_json(out/'report/summary.json',summary)
    lines=['# KLRD-RS Experiment 3: exact-teacher LoRA','',f'Status: **{status}**.','',
        'All settings and the returned-text spherical verifier remain frozen. Student responses come from ordinary merged Qwen2 weights.',
        'Checkpoint selection uses validation teacher KL only. No inference/API is invoked by reporting.','',
        '|Model|Q|alpha|Groups|Detection rate|95% group CI|Worst wrong rate|',
        '|---|---:|---:|---:|---:|---|---:|']
    for row in rows:
        ci='descriptive only' if int(row['Q'])==1000 else f"[{float(row['CI95_low']):.3f}, {float(row['CI95_high']):.3f}]"
        lines.append(f"|{row['model']}|{row['Q']}|{row['alpha']}|{row['groups']}|{float(row['rate']):.1%}|{ci}|{float(row['wrong_worst']):.1%}|")
    if evaluation:
        generation=read_json(out/'utility/generation_summary.json')
        lines+=['','Primary Q100 comparison (10 disjoint groups).',
            '|Model|TPR @1%|TPR @1e-4|Clean FPR @1%|Worst wrong @1%|Mean visible tokens|Max512 truncation|',
            '|---|---:|---:|---:|---:|---:|---:|']
        clean=next(r for r in rows if r['model']=='clean' and int(r['Q'])==100 and float(r['alpha'])==.01)
        for model,label in [('clean','Clean M0'),('teacher','Exact rotation teacher k0'),('student','LoRA merged student k0'),('untrained_zero_lora','Fresh untrained zero-LoRA equivalent')]:
            nominal=next(r for r in rows if r['model']==model and int(r['Q'])==100 and float(r['alpha'])==.01)
            conservative=next(r for r in rows if r['model']==model and int(r['Q'])==100 and float(r['alpha'])==.0001)
            g=generation[model];tpr='n/a' if model=='clean' else f"{float(nominal['rate']):.1%}"
            ctpr='n/a' if model=='clean' else f"{float(conservative['rate']):.1%}"
            trunc=f"{g['max512_truncated_count']}/{g['responses']}" if g['eos_metadata_available']==g['responses'] else 'metadata incomplete'
            lines.append(f"|{label}|{tpr}|{ctpr}|{float(clean['rate']):.1%}|{float(nominal['wrong_worst']):.1%}|{g['mean_visible_tokens']:.1f}|{trunc}|")
    if not rows:lines+=['','Student black-box evaluation is not complete. No transfer conclusion is available.']
    if evaluation:lines+=['',evaluation['untrained_control'],
        'Clean rows report false detections under owner key0. Teacher and student share prompts but use different generated samples.',
        'Generation lengths, EOS/truncation, empty outputs and repetition: utility/generation_summary.json.',
        'Exact cosine/p-values, closed-set21 identification and controls: scores/. Teacher/student gaps: scores/teacher_student_gap.csv.',
        'The ten deterministic raw comparison examples remain private in private/examples.md and are excluded from the ZIP.']
    if attack:lines+=['','400 fresh student paraphrases completed with the frozen high-fidelity GPT-5 mini protocol.',
        'Teacher0 and clean controls reuse E2B caches; no new control API calls. Before/after metrics: paraphrase/metrics.csv.',
        'Numeric/structure flags and lexical length/similarity: paraphrase/quality.csv. No rows excluded using these diagnostics.',
        'Five private blinded pairs: private/blinded_paraphrase_review.md; semantic review is pending.']
    lines+=['','Training evidence: '+str(summary['training_evidence_status']),
        'Training metrics are nats per effective response prediction position, not a free-generation detection result.','']
    if counts:lines+=['Effective data counts: '+json.dumps(counts,sort_keys=True)]
    lines+=summary['limitations']
    if errors:lines+=['','Incomplete artifacts:']+errors
    curve=kl_chart(out,training_rows)
    if curve:lines+=['','![Teacher-forced KL](kl_curve.svg)']
    image=chart(out,rows)
    if image:lines+=['','![Q100 owner detection](q100_detection.svg)']
    _text(out/'report/experiment3_report.md','\n'.join(lines)+'\n')
    if package:summary['review_package']=package_review(state)
    # ZIP embeds the pre-package summary to avoid a self-referential ZIP hash.
    # Its hash is returned to the caller; the durable manifest records members.
    return summary


def training_tables(out):
    """Flatten committed epoch summaries; metrics stay in nats/token."""
    from ..io import write_csv
    rows=[]; kl=[]
    path=out/'training/epochs.json'
    epochs=read_json(path) if path.is_file() else []
    baseline=read_json(out/'training/baseline.json') if (out/'training/baseline.json').is_file() else None
    entries=([dict(baseline,epoch=0)] if baseline else [])+epochs
    for entry in entries:
        flat={'epoch':entry['epoch']}
        for name,value in entry.items():
            if isinstance(value,dict):
                for statistic,number in value.items():
                    if isinstance(number,(int,float)):flat[name+'_'+statistic]=number
                if 'mean' in value and ('kl' in name or 'KL' in name):
                    kl.append({'epoch':entry['epoch'],'metric':name,'units':'nats/token',
                        **{k:value.get(k) for k in ['mean','p95','p99','tokens']}})
            elif isinstance(value,(int,float,str)) and name!='identity':flat[name]=value
        rows.append(flat)
    if rows:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        write_csv(out/'training/validation.csv',rows,fields)
        write_csv(out/'training/epochs.csv',[r for r in rows if r['epoch']>0],fields)
    if kl:write_csv(out/'utility/kl_summary.csv',kl,list(kl[0]))
    return rows





def kl_chart(out,rows):
    series=[('training_teacher_kl_mean','Train teacher KL','#4668bd'),
        ('teacher_kl_mean','Validation teacher KL','#cf6741'),('clean_kl_mean','Validation utility KL','#3d8466')]
    values=[float(r[key]) for r in rows for key,_,_ in series if key in r]
    if not values:return None
    max_y=max(max(values),1e-12);max_x=max(max(r['epoch'] for r in rows),1)
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="330" viewBox="0 0 760 330">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="sans-serif" font-size="13" fill="#172033">',
        '<text x="35" y="25" font-size="17">Teacher-forced KL (nats per effective response position)</text>',
        '<path d="M65 55 V270 H705" fill="none" stroke="#677080"/>',
        f'<text x="10" y="60">{max_y:.3g}</text><text x="40" y="275">0</text>',
        '<text x="340" y="315">Epoch (0 = initialization)</text>']
    for j,(key,label,color) in enumerate(series):
        points=[(65+620*r['epoch']/max_x,270-205*float(r[key])/max_y) for r in rows if key in r]
        if not points:continue
        points.sort()
        parts.append(f'<polyline points="'+ ' '.join(f'{x:.2f},{y:.2f}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="2"/>')
        parts += [f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>' for x,y in points]
        parts.append(f'<text x="{65+j*210}" y="295" fill="{color}">{label}</text>')
    parts+=['</g></svg>'];_text(out/'report/kl_curve.svg','\n'.join(parts)+'\n')
    return 'report/kl_curve.svg'
