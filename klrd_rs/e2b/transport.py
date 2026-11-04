"""E2B source mapping around the unchanged, directly copied KLRD transport."""
from collections import Counter
from importlib.metadata import version,PackageNotFoundError
from pathlib import Path
from klrd_scw import scw_paraphrase as scw
from klrd_scw.artifacts import digest as scw_digest,read_jsonl as journal_rows
from ..io import read_json,write_json,digest
from .config import CONDITIONS,COUNT,SDK_VERSION,condition_directory
from .source import immutable_rows
from .protocol import PARAPHRASE_PROMPT,prompt_for

SOURCE_FIELDS=('split','prompt_id','response_id','source_text','source_index')


class IncompleteParaphrases(RuntimeError):
    """No reduced-cohort metrics, fallback paraphraser or output resampling."""


def requests_for(population,condition):
    return [dict(split='clean' if condition=='clean' else 'fingerprint',
        prompt_id=row['prompt_id'],response_id=row['response_id'],source_text=row['original_text'],source_index=i)
        for i,row in enumerate(r for r in population if r['condition']==condition)]


def checked_results(requests,results,prompt=PARAPHRASE_PROMPT):
    if len(results)!=len(requests):raise IncompleteParaphrases('Incomplete fixed-protocol API coverage')
    for source,row in zip(requests,results):
        if any(row.get(field)!=source[field] for field in SOURCE_FIELDS):
            raise ValueError('Original/paraphrase source mapping changed')
        if (row.get('paraphraser_model')!=scw.SCW_MODEL or row.get('paraphrase_prompt')!=prompt
            or row.get('api_response_model',scw.SCW_MODEL)!=scw.SCW_MODEL):
            raise ValueError('Paraphrase model or original SCW instruction changed')
        if row.get('status')!='ok' or not isinstance(row.get('paraphrased_text'),str) or not row['paraphrased_text'].strip():
            raise IncompleteParaphrases('Fixed3600 cohort has a failed/empty paraphrase; no reduced-cohort analysis')
    return results


def receipts_path(state,condition):
    path=state['out']/'private/api'/condition_directory(condition)
    if state['out'] not in path.resolve().parents:raise ValueError('API receipt path leaves E2B output')
    return path


def validate_api_manifest(requests,path,prompt=PARAPHRASE_PROMPT):
    if not (path/'api_manifest.json').is_file():raise IncompleteParaphrases('Missing API receipt manifest')
    manifest=read_json(path/'api_manifest.json')
    expected=dict(scw_commit=scw.SCW_COMMIT,paraphraser_model=scw.SCW_MODEL,
        paraphrase_prompt=prompt,source_population_sha256=scw_digest([
            {field:r[field] for field in SOURCE_FIELDS} for r in requests]))
    if manifest!=expected:raise ValueError('Copied transport receipt identity changed')


def prepare_condition_resume(state,condition,requests):
    """Requeue failed calls once per launch; never replace an accepted rewrite.

    The outer run lock must be held. Archive the exact previous journal before
    atomically retaining only accepted rows. Crash before replacement leaves the
    old journal retryable; crash after replacement leaves successful rows intact.
    The unchanged original transport then requests only missing identities.
    """
    import hashlib,json,os,uuid
    prompt=prompt_for(state)
    path=receipts_path(state,condition);journal=path/'api_results.jsonl'
    if (path/'api_manifest.json').exists():validate_api_manifest(requests,path,prompt)
    elif journal.exists():raise ValueError('API journal exists without its source manifest')
    if not journal.exists():return dict(kept=0,retry_failed=0,missing=len(requests))
    raw=journal.read_bytes();rows=[];partial_tail=False
    for line in raw.splitlines(keepends=True):
        if not line.endswith(b'\n'):
            partial_tail=True;break
        rows.append(json.loads(line))
    expected={(r['split'],r['response_id']):r for r in requests}
    seen=set();accepted=[];failed=[]
    for row in rows:
        identity=(row.get('split'),row.get('response_id'));source=expected.get(identity)
        if source is None or identity in seen:raise ValueError('Unknown or duplicate API journal identity')
        seen.add(identity)
        if (any(row.get(field)!=source[field] for field in SOURCE_FIELDS)
            or row.get('paraphraser_model')!=scw.SCW_MODEL
            or row.get('paraphrase_prompt')!=prompt
            or row.get('api_response_model',scw.SCW_MODEL)!=scw.SCW_MODEL
            or row.get('status') not in ('ok','failed')
            or not isinstance(row.get('paraphrased_text'),str)):
            raise ValueError('Incompatible API journal source/protocol/status')
        if row['status']=='ok':
            checked_results([source],[row],prompt);accepted.append(row)
        else:
            if row['paraphrased_text'].strip():
                raise ValueError('Failed API receipt contains nonempty text; inspect before retrying')
            failed.append(row)
    result=dict(kept=len(accepted),retry_failed=len(failed),missing=len(requests)-len(seen))
    if failed or partial_tail:
        history=path/'retry_history';history.mkdir(parents=True,exist_ok=True,mode=0o700)
        archive=history/(hashlib.sha256(raw).hexdigest()+'.jsonl')
        if archive.exists():
            if archive.read_bytes()!=raw:raise ValueError('Retry archive contents changed')
        else:
            temporary=history/(uuid.uuid4().hex+'.tmp')
            with temporary.open('xb') as stream:
                stream.write(raw);stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,archive)
        # Publishing the kept rows occurs only after the backup is durable.
        scw._write_jsonl(journal,accepted)
        result.update(archive=archive.relative_to(state['out']).as_posix(),recovered_partial_tail=partial_tail)
    return result


def read_condition(state,condition):
    prompt=prompt_for(state)
    requests=requests_for(state['population'],condition);path=receipts_path(state,condition)
    if len(requests)!=COUNT:raise ValueError('Exactly400 requests required in each condition')
    validate_api_manifest(requests,path,prompt)
    terminal=journal_rows(path/'api_results.jsonl',repair_tail=False)
    by_id={}
    for row in terminal:
        identity=(row.get('split'),row.get('response_id'))
        if identity in by_id:raise ValueError('Duplicate terminal API receipt')
        by_id[identity]=row
    if len(by_id)!=len(requests) or set(by_id)!={(r['split'],r['response_id']) for r in requests}:
        raise IncompleteParaphrases('Incomplete or unexpected terminal API receipts: '+condition)
    return checked_results(requests,[by_id[(r['split'],r['response_id'])] for r in requests],prompt)


def coverage(state):
    conditions={}
    for condition in CONDITIONS:
        path=receipts_path(state,condition)/'api_results.jsonl'
        # Preserve any incomplete tail for the original transport's own recovery.
        counts=Counter()
        if path.exists():
            with path.open('rb') as stream:
                import json
                for line in stream:
                    if not line.endswith(b'\n'):break
                    row=json.loads(line);counts[row.get('status','unknown')]+=1
        conditions[condition]={'expected':COUNT,'terminal':sum(counts.values()),'statuses':dict(counts)}
    write_json(state['out']/'provenance/api_coverage.json',{'expected_total':3600,'conditions':conditions,
        'failure_policy':'Archive failed calls and requeue on next launch; accepted outputs never regenerated; no reduced cohort'})
    return conditions


def load_pairs(state):
    results={}
    for condition in CONDITIONS:
        for row in read_condition(state,condition):results[row['response_id']]=row
    pairs=[]
    for source in state['population']:
        row=results[source['response_id']]
        pairs.append(dict(source,paraphrased_text=row['paraphrased_text'],paraphrase_text_sha256=digest(row['paraphrased_text']),
            api_metadata={key:value for key,value in row.items() if key.startswith('api_') or key in
                ('attempts','status','paraphraser_model','paraphrase_prompt')}))
    immutable_rows(state['out']/'private/original_paraphrase_pairs.jsonl',pairs)
    return pairs


def run_copied_transport(requests,path,*,prompt=PARAPHRASE_PROMPT,**kwargs):
    """Use the original API/retry loop with the recorded run instruction.

    The run lock and sequential condition loop exclude competing calls here.
    Worker threads finish inside run_paraphrases before the constant is restored.
    The vendored source file is never modified.
    """
    historical_prompt=scw.PARAPHRASE_PROMPT
    try:
        scw.PARAPHRASE_PROMPT=prompt
        return scw.run_paraphrases(requests,path,**kwargs)
    finally:
        scw.PARAPHRASE_PROMPT=historical_prompt

def run(state,workers=128):
    if type(workers) is not int or not 1<=workers<=128:raise ValueError('workers must be1..128')
    try:sdk=version('openai')
    except PackageNotFoundError:raise RuntimeError('Install requirements-paraphrase.txt on the server first') from None
    if sdk!=SDK_VERSION:raise RuntimeError('SCW reproduction requires the historical OpenAI SDK '+SDK_VERSION)
    write_json(state['out']/'provenance/api_execution.json',{'sdk_version':sdk,'workers':workers,
        'transport':'Direct byte-for-byte klrd_scw.scw_paraphrase.run_paraphrases',
        'max_attempts':5,'conditions_sequential':True,'within_condition_parallel':True,
        'request_payload':'Recorded run instruction plus original response only; no original task/key/score supplied',
        'no_provider_or_model_fallback':True})
    # Preflight all cached conditions before spending on any new API requests.
    resume={}
    for condition in CONDITIONS:
        requests=requests_for(state['population'],condition)
        if len(requests)!=COUNT:raise ValueError('Each condition requires400 immutable originals')
        resume[condition]=prepare_condition_resume(state,condition,requests)
    write_json(state['out']/'provenance/api_resume.json',{'conditions':resume,
        'policy':'Retry failed/unreceived calls once per launch; preserve accepted outputs; original SCW transport unchanged'})
    try:
        for condition in CONDITIONS:
            requests=requests_for(state['population'],condition)
            if len(requests)!=COUNT:raise ValueError('Each condition requires400 immutable originals')
            counts=resume[condition]
            print(f'E2B SCW: condition {condition}: {counts["kept"]} already OK; '
                f'{counts["retry_failed"]} failed to retry; {counts["missing"]} missing.',flush=True)
            # This is the ORIGINAL copied KLRD function, not a reimplemented API loop.
            rows=run_copied_transport(requests,receipts_path(state,condition),prompt=prompt_for(state),workers=workers,max_attempts=5)
            coverage(state)
            checked_results(requests,rows,prompt_for(state))
    except (scw.SCWUnavailableError,IncompleteParaphrases,ValueError):
        coverage(state);raise
    except Exception:
        coverage(state)
        raise RuntimeError('Copied SCW transport failed; upstream details omitted to protect credentials. Durable receipts retained.') from None
    pairs=load_pairs(state)
    write_json(state['out']/'provenance/api_complete.json',{'status':'COMPLETE','pairs':len(pairs),
        'pairs_sha256':digest(pairs),'model':scw.SCW_MODEL,'one_accepted_output_per_original':True,
        'incomplete_response_status_count':sum(p['api_metadata'].get('api_response_status')=='incomplete' for p in pairs),
        'note':'Legacy acceptance preserves nonempty output_text, including incomplete response status; quality is reported separately.'})
    return pairs
