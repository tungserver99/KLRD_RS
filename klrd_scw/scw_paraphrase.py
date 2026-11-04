"""Exact pinned SCW output rewrite with durable, source-ordered resumption.

The upstream transformation is prompt + source text through Responses.create;
retry/cache machinery never uses a verifier or chooses among generated outputs.
"""
# Request and prompt reproduced from the pinned SCW paraphrasers.py.
# Upstream copyright (c) 2025 Thibaud Gloaguen and contributors;
# upstream license: Responsible AI SOURCE CODE License, Version 1.1.
# Full original source and LICENSE_CODE are retained under
# docs/superpowers/references/scw_15bc1929569357130f2dbc0b09f91bbf4f4bd947/.
import json
import math
import os
import re
import shlex
from pathlib import Path
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from .artifacts import append_jsonl, digest, dump_json, ensure_manifest, read_jsonl, load_json

SCW_COMMIT = '15bc1929569357130f2dbc0b09f91bbf4f4bd947'
SCW_MODEL = 'gpt-5-mini-2025-08-07'
PARAPHRASE_PROMPT = 'Please rewrite the following text and return only the rewritten text: '
PROJECT_ENV_PATH = Path(__file__).resolve().parent.parent / '.env'
_SOURCE_FIELDS = ('split', 'prompt_id', 'response_id', 'source_text', 'source_index')


class SCWUnavailableError(RuntimeError):
    """Exact-model reproduction cannot run with this account/configuration."""


def _identity(row):
    return row['split'], row['response_id']


def _status_code(exc):
    status = getattr(exc, 'status_code', None)
    if isinstance(status, int):
        return status
    return {'AuthenticationError': 401, 'PermissionDeniedError': 403,
            'NotFoundError': 404}.get(type(exc).__name__)


def _error_code(exc):
    code = getattr(exc, 'code', None)
    if isinstance(code, str):
        return code
    body = getattr(exc, 'body', None)
    if isinstance(body, dict):
        detail = body.get('error', body)
        if isinstance(detail, dict) and isinstance(detail.get('code'), str):
            return detail['code']
    return None


def _unavailable(exc):
    status = _status_code(exc)
    if status in (401, 403, 404):
        return True
    code = _error_code(exc)
    if code in ('model_not_found', 'unsupported_model', 'model_not_available', 'insufficient_quota'):
        return True
    # Inspect only to classify; never persist an upstream exception message.
    if status == 400:
        message = str(exc).lower()
        return ('model' in message and any(term in message for term in
                ('not found', 'not available', 'unsupported', 'not supported', 'access')))
    return False


def _error_description(exc):
    status = _status_code(exc)
    if status is not None:
        return f'OpenAI request failed (HTTP {status}); upstream detail omitted'
    if isinstance(exc, (TimeoutError, ConnectionError)) or type(exc).__name__ in (
            'APIConnectionError', 'APITimeoutError'):
        return 'OpenAI connection or timeout failure; upstream detail omitted'
    return 'OpenAI request failed; upstream detail omitted'


def _transient(exc):
    status = _status_code(exc)
    return (_error_code(exc) != 'insufficient_quota' and (status == 429 or status is not None and 500 <= status <= 599
            or isinstance(exc, (TimeoutError, ConnectionError))
            or type(exc).__name__ in ('APIConnectionError', 'APITimeoutError')))



def _read_api_key():
    """Read only OPENAI_API_KEY, without executing/interpolating the .env file.

    The explicitly requested project file wins; an absent entry falls back to
    process environment. No global environment mutation or credential logging.
    """
    path=PROJECT_ENV_PATH
    value=None
    if path.exists():
        try:
            content=path.read_text(encoding='utf-8-sig')
        except (OSError,UnicodeError):
            raise SCWUnavailableError('Cannot read the project .env file') from None
        found=False
        for line in content.splitlines():
            match=re.match(r'^\s*(?:export\s+)?OPENAI_API_KEY\s*=\s*(.*)$',line)
            if match is None: continue
            if found:
                raise SCWUnavailableError('Duplicate OPENAI_API_KEY entries in project .env')
            found=True
            try:
                lexer=shlex.shlex(match.group(1),posix=True)
                lexer.whitespace_split=True;lexer.commenters='#'
                parts=list(lexer)
            except ValueError:
                raise SCWUnavailableError('Malformed OPENAI_API_KEY entry in project .env') from None
            if len(parts)!=1 or not parts[0] or any(c.isspace() for c in parts[0]):
                raise SCWUnavailableError('OPENAI_API_KEY must be one nonempty value in project .env')
            value=parts[0]
            if '${' in value or '$(' in value or '`' in value:
                raise SCWUnavailableError('Project .env OPENAI_API_KEY must contain the literal key, not an expression')
    if value is None:
        value=os.environ.get('OPENAI_API_KEY')
    if not value:
        raise SCWUnavailableError('OPENAI_API_KEY is required in project .env or process environment')
    return value


def _create_client():
    api_key = _read_api_key()
    try:
        from openai import OpenAI
    except ImportError:
        raise SCWUnavailableError('The OpenAI SDK is required for exact SCW reproduction') from None
    # Explicit base URL prevents OPENAI_BASE_URL from silently changing provider.
    try:
        return OpenAI(api_key=api_key, base_url='https://api.openai.com/v1', max_retries=0, timeout=180)
    except Exception:
        raise SCWUnavailableError('OpenAI client initialization failed; configuration detail omitted') from None


def _numeric_usage(usage):
    if hasattr(usage, 'model_dump'):
        usage = usage.model_dump()
    if not isinstance(usage, dict):
        return {}
    counts = {}
    for name, value in usage.items():
        if isinstance(value, dict):
            nested = _numeric_usage(value)
            if nested:
                counts[name] = nested
        elif (isinstance(value, (int, float)) and not isinstance(value, bool)
              and (not isinstance(value, float) or math.isfinite(value))):
            counts[name] = value
    return counts


def _write_jsonl(path, rows):
    """Atomically publish a derived view; the append journal is authoritative."""
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def _validated_population(population):
    rows = [dict(row) for row in population]
    identities, prompts = set(), set()
    for row in rows:
        if any(field not in row for field in _SOURCE_FIELDS):
            raise ValueError('Population requires split, prompt_id, response_id, source_text, source_index')
        if row['split'] not in ('fingerprint', 'clean') or not isinstance(row['source_text'], str):
            raise ValueError('Population requires a known split and text source')
        try:
            identity = _identity(row)
            prompt = row['split'], row['prompt_id']
            if identity in identities or prompt in prompts:
                raise ValueError('Duplicate source response or prompt IDs')
            identities.add(identity)
            prompts.add(prompt)
        except TypeError:
            raise ValueError('Population IDs must be scalar hashable values') from None
    return rows


def run_paraphrases(population, output_dir, *, workers=128, client=None,
                    max_attempts=5, sleep_fn=time.sleep):
    """Return terminal rows in exact input order; resume without resampling.

    Accepted outputs and terminal failures are durably cached immediately, even
    if another worker raises a fatal account/model error or KeyboardInterrupt.
    An interrupted in-flight request without a received output has no cached
    result and may need another request when the run resumes.
    """
    if type(workers) is not int or not 1 <= workers <= 128:
        raise ValueError('workers must be an integer from1to128')
    if not isinstance(max_attempts, int) or not 1 <= max_attempts <= 5:
        raise ValueError('max_attempts must be between 1 and 5')
    sources = _validated_population(population)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_manifest(output_dir / 'api_manifest.json', {
        'scw_commit': SCW_COMMIT, 'paraphraser_model': SCW_MODEL,
        'paraphrase_prompt': PARAPHRASE_PROMPT,
        'source_population_sha256': digest([
            {field: row[field] for field in _SOURCE_FIELDS} for row in sources])})
    journal = output_dir / 'api_results.jsonl'
    expected = {_identity(row): row for row in sources}
    completed = {}
    for row in read_jsonl(journal, repair_tail=True):
        try:
            identity = _identity(row)
            source = expected[identity]
        except (KeyError, TypeError):
            raise ValueError('Unknown persisted SCW source identity') from None
        if identity in completed:
            raise ValueError('Duplicate persisted SCW response IDs')
        if (any(row.get(field) != source[field] for field in _SOURCE_FIELDS)
                or row.get('paraphraser_model') != SCW_MODEL
                or row.get('paraphrase_prompt') != PARAPHRASE_PROMPT
                or row.get('api_response_model', SCW_MODEL) != SCW_MODEL
                or row.get('status') not in ('ok', 'failed')
                or not isinstance(row.get('paraphrased_text'), str)
                or row['status'] == 'ok' and not row['paraphrased_text'].strip()):
            raise ValueError('Incompatible persisted SCW source, text, model, or prompt')
        completed[identity] = row
    failures_path = output_dir / 'failures.jsonl'
    # Repairs a crash between a terminal journal append and failure-view append.
    _write_jsonl(failures_path, [row for row in completed.values() if row['status'] == 'failed'])
    pending = [row for row in sources if _identity(row) not in completed]
    if pending and client is None:
        client = _create_client()
    lock = threading.Lock()
    stop = threading.Event()
    counts = {status: sum(row['status'] == status for row in completed.values())
              for status in ('ok', 'failed')}

    def unavailable(error):
        stop.set()
        with lock:
            dump_json(output_dir / 'api_unavailable.json', {'error': error,
                'paraphraser_model': SCW_MODEL, 'status': 'unavailable'})
        raise SCWUnavailableError(error) from None

    def process(source):
        if stop.is_set():
            return None
        row = dict(source, paraphraser_model=SCW_MODEL,
                   paraphrase_prompt=PARAPHRASE_PROMPT, paraphrased_text='')
        try:
            for attempt in range(1, max_attempts + 1):
                if stop.is_set():
                    return None
                row['attempts'] = attempt
                try:
                    result = client.responses.create(model=SCW_MODEL,
                        input=PARAPHRASE_PROMPT + source['source_text'])
                except Exception as exc:
                    if _unavailable(exc):
                        status = _status_code(exc)
                        reason = ('OpenAI quota is exhausted for this account' if
                                  _error_code(exc) == 'insufficient_quota' else
                                  'Exact SCW model ' + SCW_MODEL + ' is unavailable for this account')
                        error = (reason
                                 + (f' (HTTP {status})' if status is not None else '')
                                 + '; reproduction stopped without a substitute')
                        unavailable(error)
                    if _transient(exc) and attempt < max_attempts:
                        if stop.is_set():
                            return None
                        sleep_fn(2 ** (attempt - 1))
                        continue
                    row.update(status='failed', error=_error_description(exc))
                    break
                returned_model = getattr(result, 'model', None)
                if returned_model is not None and returned_model != SCW_MODEL:
                    unavailable('Returned response model differs from exact SCW model '
                                + SCW_MODEL + '; reproduction stopped without accepting a substitute')
                for attribute in ('id', 'model', 'status'):
                    value = getattr(result, attribute, None)
                    if isinstance(value, str):
                        row['api_response_' + attribute] = value
                usage = _numeric_usage(getattr(result, 'usage', None))
                if usage:
                    row['api_usage'] = usage
                text = result.output_text
                if not isinstance(text, str):
                    row.update(status='failed', error='OpenAI output_text is not text')
                else:
                    row['paraphrased_text'] = text
                    if text.strip():
                        row['status'] = 'ok'
                    else:
                        row.update(status='failed', error='OpenAI output_text is empty')
                break
            with lock:
                append_jsonl(journal, row)
                completed[_identity(source)] = row
                if row['status'] == 'failed':
                    append_jsonl(failures_path, row)
                counts[row['status']] += 1
                saved = len(completed)
                if saved % 25 == 0 or saved == len(sources):
                    print(f"SCW paraphrasing: {saved}/{len(sources)} saved "
                          f"({counts['ok']} ok, {counts['failed']} failed)", flush=True)
            return row
        except BaseException:
            stop.set()
            raise

    if pending:
        print(f'SCW paraphrasing: {len(completed)}/{len(sources)} saved; '
              f'{len(pending)} pending', flush=True)
        # Check exact-model/account availability before allowing any fanout.
        process(pending[0])
        if len(pending) > 1:
            executor = ThreadPoolExecutor(max_workers=workers)
            futures = []
            try:
                futures = [executor.submit(process, row) for row in pending[1:]]
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                stop.set()
                for future in futures:
                    future.cancel()
                raise
            finally:
                # Already received outputs must reach their durable journal.
                executor.shutdown(wait=True, cancel_futures=True)
    ordered = [completed[_identity(source)] for source in sources]
    _write_jsonl(output_dir / 'paraphrases.jsonl', ordered)
    unavailable_path=output_dir/'api_unavailable.json'
    if unavailable_path.exists() and load_json(unavailable_path).get('status')=='unavailable':
        dump_json(unavailable_path,{'status':'resolved','paraphraser_model':SCW_MODEL,
                                   'previous_event':load_json(unavailable_path)})
    return ordered
