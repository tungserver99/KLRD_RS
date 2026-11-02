"""Frozen automatic diagnostics and preselected, visibly blinded examples."""
import json,re
import numpy as np
from ..io import write_csv,write_json
from .cohort import text_hash


def text_quality(record):
    text=record['response'];words=text.split();n=len(words)
    def repeat(k):
        grams=[tuple(words[i:i+k]) for i in range(max(0,n-k+1))]
        return 1-len(set(grams))/len(grams) if grams else 0.
    short=record['native_visible_tokens']<5;empty=not text.strip()
    loop=record['native_visible_tokens']>=64 and repeat(4)>=.4
    return {'tokens':record['native_visible_tokens'],'characters':len(text),'empty':empty,'very_short':short,
        'loop':loop,'degenerate':empty or short or loop,'repeated3':repeat(3),'repeated4':repeat(4),
        'distinct1':len(set(words))/n if n else 0.,
        'distinct2':len(set(zip(words,words[1:])))/(n-1) if n>1 else 0.,
        'EOS':record['eos_terminated'],'truncated':record['max_length_truncated'],'roundtrip':record['roundtrip_equal']}


def diagnostics(observations,out,cfg):
    quality=[];KL=[]
    for label,(records,_) in observations.items():
        if not records:continue
        values=[text_quality(r) for r in records.values()]
        row={'owner':label,'n_responses':len(values),'scope':'clean' if label=='clean' else 'stress' if int(label) in cfg.stress else 'primary'}
        for field in ['tokens','characters']:
            v=[x[field] for x in values]
            row.update({field+'_mean':float(np.mean(v)),field+'_median':float(np.median(v)),
                field+'_p05':float(np.quantile(v,.05)),field+'_p95':float(np.quantile(v,.95))})
        for field in ['empty','very_short','loop','degenerate','repeated3','repeated4','distinct1','distinct2','EOS','truncated','roundtrip']:
            row[field+'_rate_or_mean']=float(np.mean([x[field] for x in values]))
        seconds=sum(r['amortized_generation_seconds'] for r in records.values())
        row.update(generation_seconds=seconds,native_visible_tokens_per_second=sum(x['tokens'] for x in values)/seconds if seconds else None)
        row['verification_seconds']=sum(r.get('scoring_amortized_seconds',0.) for r in records.values())
        row['recorded_GPU_hours']=(seconds+row['verification_seconds'])/3600
        quality.append(row)
        v=[x['KL'] for r in records.values() for x in r['KL_samples']]
        KL.append({'owner':label,'scope':row['scope'],'sampled_prefixes':len(v),
            'visited_prefixes':sum(len(r['generated_token_ids']) for r in records.values()),'sampling_stride':cfg.kl_stride,
            'mean':float(np.mean(v)) if v else None,'p95':float(np.quantile(v,.95)) if v else None,
            'p99':float(np.quantile(v,.99)) if v else None,'max':float(np.max(v)) if v else None,
            'fraction_above_0_05':float(np.mean(np.asarray(v)>.05)) if v else None})
    for name,rows in [('generation_quality_by_key.csv',quality),('free_context_KL_by_key.csv',KL)]:
        write_csv(out/'utility'/name,rows,list(rows[0]) if rows else ['owner'])
    clean=next((r for r in quality if r['owner']=='clean'),None)
    primary=[r for r in quality if r['scope']=='primary'];primaryKL=[r for r in KL if r['scope']=='primary']
    concerns={'quality_concern':any(clean is not None and r['degenerate_rate_or_mean']>clean['degenerate_rate_or_mean']+cfg.degeneration_increase for r in primary),
        'distortion_concern':sum((r['p99'] or 0)>.05 for r in primaryKL)>=cfg.widespread_KL_key_count or
            any((r['fraction_above_0_05'] or 0)>cfg.KL_exceed_fraction_concern for r in primaryKL)}
    return {'quality':quality,'KL':KL,'concerns':concerns}


def redact(text):
    text=re.sub(r'\bsk-[A-Za-z0-9_-]{12,}\b','[REDACTED_TOKEN]',text)
    text=re.sub(r'(?:[A-Za-z]:[\\/]|/(?:home|content|Users)/)[^\s<>]+','[REDACTED_LOCAL_PATH]',text)
    return text


def examples(rows,observations,selected_ids,out,cfg):
    prompts={r['prompt_id']:r['prompt'] for r in rows};clean=observations.get('clean',({},{}))[0]
    lines=['# Preselected examples','', 'Preselected before scores; no output quality filtering. Keys are IDs, never private vectors.','']
    blinded=['# Blinded clean/teacher comparison','', 'A/B assignment is hidden here; assess task following, coherence and repetition. No automated utility proof.','']
    answers=[];following=[]
    for j,pid in enumerate(selected_ids):
        if pid not in clean:continue
        if j<cfg.example_count:lines += [f'## Example {j+1}',redact(prompts[pid]),'### Clean',redact(clean[pid]['response'])]
        for owner in cfg.owners:
            record=observations.get(str(owner),({},{}))[0].get(pid)
            if record is None:continue
            pair=f'pair_{j:03}_{owner:02}';swap=int(text_hash(pair)[:8],16)%2
            a,b=[clean[pid]['response'],record['response']][::(-1 if swap else 1)]
            blinded += [f'## {pair}',redact(prompts[pid]),'### A',redact(a),'### B',redact(b)]
            answers.append({'pair_id':pair,'A':'teacher' if swap else 'clean','owner_key_index':owner,'prompt_id':pid})
            text=prompts[pid].casefold();hint=None
            if 'valid json' in text or 'json format' in text:
                hint='json_parse'
                try:json.loads(record['response']);passed=True
                except ValueError:passed=False
            elif 'one word' in text or 'single word' in text:hint='single_word';passed=len(record['response'].split())==1
            elif 'bullet' in text:hint='bullet_marker';passed=bool(re.search(r'(?m)^\s*(?:[-*]|\d+[.)])\s',record['response']))
            if hint:following.append({'pair_id':pair,'heuristic':hint,'passed':passed,'warning':'Limited task-hint heuristic, not rigorous utility certification'})
            if j<cfg.example_count:lines += [f'### Teacher key {owner}',redact(record['response'])]
    (out/'utility').mkdir(exist_ok=True)
    (out/'utility/examples_redacted.md').write_text('\n\n'.join(lines),encoding='utf-8')
    (out/'utility/blinded_quality_pairs.md').write_text('\n\n'.join(blinded),encoding='utf-8')
    write_json(out/'private/blinded_answer_key.json',answers)
    write_csv(out/'utility/task_hint_diagnostics.csv',following,list(following[0]) if following else ['pair_id','heuristic','passed','warning'])
