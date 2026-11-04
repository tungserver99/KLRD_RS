"""SERVER ONLY. Copied KLRD tests; module import path adapted, not executed locally."""
import importlib
import json
import sys
import threading
import time
from types import SimpleNamespace

import pytest


def api_module():
    try:
        return importlib.import_module('klrd_scw.scw_paraphrase')
    except ModuleNotFoundError:
        pytest.fail('SCW paraphrase implementation is missing')


def population(n=1):
    return [dict(split='fingerprint', prompt_id=f'p{i}', response_id=f'r{i}',
                 source_text=f'source {i}', source_index=i) for i in range(n)]


def response(text=' rewritten\n '):
    return SimpleNamespace(output_text=text, id='resp_safe', model='gpt-5-mini-2025-08-07',
                           status='completed', usage=SimpleNamespace(
                               model_dump=lambda: {'input_tokens': 7, 'output_tokens': 3}))


class APIError(Exception):
    def __init__(self, status_code, message='upstream error'):
        super().__init__(message)
        self.status_code = status_code


class FakeClient:
    def __init__(self, actions):
        self.actions = iter(actions)
        self.calls = []
        self.responses = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        action = next(self.actions)
        if isinstance(action, BaseException):
            raise action
        return action


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def test_exact_request_preserves_text_and_source(tmp_path):
    api = api_module()
    rows = population()
    rows[0]['original_prompt'] = 'PRIVATE ORIGINAL PROMPT'
    client = FakeClient([response()])
    result = api.run_paraphrases(rows, tmp_path, client=client)
    assert client.calls == [{'model': 'gpt-5-mini-2025-08-07', 'input':
        'Please rewrite the following text and return only the rewritten text: source 0'}]
    assert result[0]['paraphrased_text'] == ' rewritten\n '
    assert result[0]['source_text'] == 'source 0'
    assert result[0]['status'] == 'ok'
    assert read_rows(tmp_path / 'api_results.jsonl') == result
    assert read_rows(tmp_path / 'paraphrases.jsonl') == result


def test_terminal_failure_bounded_and_cached(tmp_path):
    api = api_module()
    client = FakeClient([APIError(429)] * 5)
    delays = []
    result = api.run_paraphrases(population(), tmp_path, client=client, sleep_fn=delays.append)
    assert len(client.calls) == 5
    assert delays == [1, 2, 4, 8]
    assert result[0]['status'] == 'failed'
    assert result[0]['attempts'] == 5
    assert read_rows(tmp_path / 'failures.jsonl') == result
    assert api.run_paraphrases(population(), tmp_path, client=FakeClient([])) == result


def test_transient_retry_accepts_first_output(tmp_path):
    api = api_module()
    client = FakeClient([TimeoutError('timeout'), APIError(503), response('same source')])
    result = api.run_paraphrases(population(), tmp_path, client=client, sleep_fn=lambda _: None)
    assert len(client.calls) == 3
    assert result[0]['paraphrased_text'] == 'same source'
    assert result[0]['attempts'] == 3


@pytest.mark.parametrize('status', [401, 403, 404])
def test_unavailable_fails_before_fanout_and_redacts(tmp_path, monkeypatch, status):
    api = api_module()
    secret = 'sk-test-DO-NOT-LEAK'
    monkeypatch.setenv('OPENAI_API_KEY', secret)
    client = FakeClient([APIError(status, f'key {secret} unavailable')])
    with pytest.raises(api.SCWUnavailableError) as error:
        api.run_paraphrases(population(20), tmp_path, client=client)
    assert len(client.calls) == 1
    assert secret not in str(error.value)
    assert str(status) in str(error.value)
    for path in tmp_path.iterdir():
        assert secret not in path.read_text(encoding='utf-8')


def test_empty_output_is_terminal_without_resampling(tmp_path):
    api = api_module()
    client = FakeClient([response(' \n\t')])
    result = api.run_paraphrases(population(), tmp_path, client=client)
    assert len(client.calls) == 1
    assert result[0]['status'] == 'failed'
    assert result[0]['paraphrased_text'] == ' \n\t'


def test_partial_journal_restart_does_not_repeat_success(tmp_path):
    api = api_module()
    client = FakeClient([response('first'), KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        api.run_paraphrases(population(3), tmp_path, client=client, workers=1)
    journal = tmp_path / 'api_results.jsonl'
    assert read_rows(journal)[0]['paraphrased_text'] == 'first'
    with journal.open('ab') as stream:
        stream.write(b'{"unfinished":')
    resumed = FakeClient([response('second'), response('third')])
    result = api.run_paraphrases(population(3), tmp_path, client=resumed, workers=1)
    assert len(resumed.calls) == 2
    assert [r['paraphrased_text'] for r in result] == ['first', 'second', 'third']
    assert len(read_rows(journal)) == 3


def test_parallel_completion_durable_and_final_ordered(tmp_path):
    api = api_module()
    released = threading.Event()

    def release_when_durable():
        for _ in range(500):
            if any(r['response_id'] == 'r2' for r in read_rows(tmp_path / 'api_results.jsonl')):
                released.set()
                return
            time.sleep(.01)

    class ParallelClient:
        def __init__(self):
            self.responses = self

        def create(self, **kwargs):
            source = kwargs['input'][-1]
            if source == '1':
                assert released.wait(5)
                assert any(r['response_id'] == 'r2' for r in read_rows(tmp_path / 'api_results.jsonl'))
            if source == '2':
                threading.Thread(target=release_when_durable, daemon=True).start()
            return response(source)

    result = api.run_paraphrases(population(3), tmp_path, client=ParallelClient(), workers=2)
    assert [r['response_id'] for r in read_rows(tmp_path / 'api_results.jsonl')] == ['r0', 'r2', 'r1']
    assert [r['response_id'] for r in result] == ['r0', 'r1', 'r2']


@pytest.mark.parametrize('field,value', [('source_text', 'changed'), ('source_index', 99),
                                        ('prompt_id', 'changed')])
def test_resume_rejects_changed_source(tmp_path, field, value):
    api = api_module()
    api.run_paraphrases(population(), tmp_path, client=FakeClient([response()]))
    rows = population()
    rows[0][field] = value
    client = FakeClient([])
    with pytest.raises(ValueError):
        api.run_paraphrases(rows, tmp_path, client=client)
    assert not client.calls


def test_duplicate_ids_rejected(tmp_path):
    api = api_module()
    with pytest.raises(ValueError, match='[Dd]uplicate'):
        api.run_paraphrases(population() * 2, tmp_path, client=FakeClient([]))


def test_failure_does_not_copy_exception_secrets(tmp_path, monkeypatch):
    api = api_module()
    monkeypatch.setenv('OPENAI_API_KEY', 'private-nonstandard-key')
    result = api.run_paraphrases(population(), tmp_path,
        client=FakeClient([APIError(400, 'private-nonstandard-key sk-another-secret')]))
    assert result[0]['status'] == 'failed'
    assert '400' in result[0]['error']
    for path in tmp_path.iterdir():
        text = path.read_text(encoding='utf-8')
        assert 'private-nonstandard-key' not in text
        assert 'sk-another-secret' not in text


def test_default_client_official_base_env_key_no_sdk_retries(tmp_path, monkeypatch):
    api = api_module()
    constructed = []
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://other-provider.invalid')

    def factory(**kwargs):
        constructed.append(kwargs)
        return FakeClient([response()])

    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(OpenAI=factory))
    api.run_paraphrases(population(), tmp_path)
    assert constructed == [dict(api_key='test-key', base_url='https://api.openai.com/v1', max_retries=0, timeout=180)]

@pytest.mark.parametrize('mutation', ['unknown', 'duplicate', 'model', 'prompt'])
def test_cache_tampering_rejected_before_api(tmp_path, mutation):
    api = api_module()
    api.run_paraphrases(population(), tmp_path, client=FakeClient([response()]))
    path = tmp_path / 'api_results.jsonl'
    rows = read_rows(path)
    if mutation == 'unknown':
        rows[0]['response_id'] = 'unknown'
    elif mutation == 'duplicate':
        rows.append(dict(rows[0]))
    elif mutation == 'model':
        rows[0]['paraphraser_model'] = 'gpt-5-mini'
    else:
        rows[0]['paraphrase_prompt'] += ' stronger'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    with pytest.raises(ValueError):
        api.run_paraphrases(population(), tmp_path, client=FakeClient([]))


def test_resume_rejects_reordered_population(tmp_path):
    api = api_module()
    api.run_paraphrases(population(2), tmp_path, client=FakeClient([response(), response()]))
    with pytest.raises(ValueError):
        api.run_paraphrases(population(2)[::-1], tmp_path, client=FakeClient([]))


def test_corrupt_complete_jsonl_line_is_not_repaired(tmp_path):
    api = api_module()
    api.run_paraphrases(population(), tmp_path, client=FakeClient([response()]))
    with (tmp_path / 'api_results.jsonl').open('ab') as stream:
        stream.write(b'{invalid}\n')
    with pytest.raises(ValueError):
        api.run_paraphrases(population(), tmp_path, client=FakeClient([]))


def test_cached_run_does_not_need_key_or_sdk(tmp_path, monkeypatch):
    api = api_module()
    original = api.run_paraphrases(population(), tmp_path, client=FakeClient([response()]))
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    assert api.run_paraphrases(population(), tmp_path) == original


def test_model_unsupported_400_is_fatal(tmp_path):
    api = api_module()
    client = FakeClient([APIError(400, 'The model is not supported by this account')])
    with pytest.raises(api.SCWUnavailableError):
        api.run_paraphrases(population(20), tmp_path, client=client)
    assert len(client.calls) == 1


def test_parallel_fatal_preserves_other_received_successes(tmp_path):
    api = api_module()
    started = threading.Event()
    unavailable = threading.Event()

    class Client:
        def __init__(self):
            self.responses = self

        def create(self, **kwargs):
            source = kwargs['input'][-1]
            if source == '1':
                assert started.wait(5)
                unavailable.set()
                raise APIError(403, 'secret')
            if source == '2':
                started.set()
                assert unavailable.wait(5)
                time.sleep(.02)
            return response(source)

    with pytest.raises(api.SCWUnavailableError):
        api.run_paraphrases(population(3), tmp_path, client=Client(), workers=2)
    assert [r['response_id'] for r in read_rows(tmp_path / 'api_results.jsonl')] == ['r0', 'r2']
    resumed = FakeClient([response('recovered')])
    result = api.run_paraphrases(population(3), tmp_path, client=resumed)
    assert len(resumed.calls) == 1
    assert [r['paraphrased_text'] for r in result] == ['0', 'recovered', '2']


def test_client_initialization_error_is_sanitized(tmp_path, monkeypatch):
    api = api_module()
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-private-key')

    def broken_factory(**kwargs):
        raise RuntimeError('configuration contained sk-private-key')

    monkeypatch.setitem(sys.modules, 'openai', SimpleNamespace(OpenAI=broken_factory))
    with pytest.raises(api.SCWUnavailableError) as error:
        api.run_paraphrases(population(), tmp_path)
    assert 'sk-private-key' not in str(error.value)


def test_progress_is_flushed_before_calls_and_reports_saved_counts(tmp_path, monkeypatch):
    import io
    api = api_module()

    class Output(io.StringIO):
        flushed = False

        def flush(self):
            self.flushed = True
            super().flush()

    output = Output()
    monkeypatch.setattr(sys, 'stdout', output)

    class Client(FakeClient):
        def create(self, **kwargs):
            assert output.flushed
            assert '0/26 saved; 26 pending' in output.getvalue()
            return super().create(**kwargs)

    api.run_paraphrases(population(26), tmp_path, client=Client([response()] * 26), workers=1)
    assert '25/26 saved (25 ok, 0 failed)' in output.getvalue()
    assert '26/26 saved (26 ok, 0 failed)' in output.getvalue()
    assert 'source 0' not in output.getvalue()
    assert 'rewritten' not in output.getvalue()


def test_response_metadata_and_only_numeric_usage_are_saved(tmp_path):
    api = api_module()
    received = response(' exact\n')
    received.usage = SimpleNamespace(model_dump=lambda: {
        'input_tokens': 7, 'output_tokens': 3, 'total_tokens': 10,
        'output_tokens_details': {'reasoning_tokens': 2, 'note': 'do not persist'},
        'note': 'do not persist', 'cached': True, 'not_finite': float('nan')})
    result = api.run_paraphrases(population(), tmp_path, client=FakeClient([received]))
    assert result[0]['api_response_id'] == 'resp_safe'
    assert result[0]['api_response_model'] == 'gpt-5-mini-2025-08-07'
    assert result[0]['api_response_status'] == 'completed'
    assert result[0]['api_usage'] == {'input_tokens': 7, 'output_tokens': 3,
        'total_tokens': 10, 'output_tokens_details': {'reasoning_tokens': 2}}
    assert read_rows(tmp_path / 'api_results.jsonl') == result
    assert result[0]['paraphrased_text'] == ' exact\n'


def test_output_only_response_still_works(tmp_path):
    api = api_module()
    result = api.run_paraphrases(population(), tmp_path,
        client=FakeClient([SimpleNamespace(output_text='minimal')]))
    assert result[0]['paraphrased_text'] == 'minimal'
    assert result[0]['status'] == 'ok'
    assert 'api_response_model' not in result[0]


def test_explicit_returned_model_mismatch_stops_without_accepting(tmp_path):
    api = api_module()
    received = response()
    received.model = 'other-model'
    client = FakeClient([received])
    with pytest.raises(api.SCWUnavailableError, match='model'):
        api.run_paraphrases(population(20), tmp_path, client=client)
    assert len(client.calls) == 1
    journal = tmp_path / 'api_results.jsonl'
    assert not journal.exists() or not read_rows(journal)
    assert json.loads((tmp_path / 'api_unavailable.json').read_text())['status'] == 'unavailable'


@pytest.mark.parametrize('code_location', ['code', 'body'])
def test_insufficient_quota_stops_without_transient_retry_or_fanout(tmp_path, code_location):
    api = api_module()
    error = APIError(429, 'quota exhausted secret-key')
    if code_location == 'code':
        error.code = 'insufficient_quota'
    else:
        error.body = {'error': {'code': 'insufficient_quota'}}
    client = FakeClient([error] * 5)
    delays = []
    with pytest.raises(api.SCWUnavailableError, match='quota') as captured:
        api.run_paraphrases(population(20), tmp_path, client=client, sleep_fn=delays.append)
    assert len(client.calls) == 1
    assert delays == []
    assert 'secret-key' not in str(captured.value)

def test_more_than_upstream_worker_limit_is_rejected_before_client(tmp_path):
    from klrd.scw_paraphrase import run_paraphrases
    with pytest.raises(ValueError,match='128|worker'):
        run_paraphrases([],tmp_path,workers=129)

@pytest.fixture(autouse=True)
def isolate_project_env_file(monkeypatch,tmp_path):
    # Never inspect the real project .env in tests.
    api=api_module()
    monkeypatch.setattr(api,'PROJECT_ENV_PATH',tmp_path/'.env',raising=False)


@pytest.mark.parametrize('line',[
    'OPENAI_API_KEY=sk-fixture-from-dotenv\n',
    ' export OPENAI_API_KEY = "sk-fixture-from-dotenv" # comment\n',
    "OPENAI_API_KEY='sk-fixture-from-dotenv'\n",
    '\ufeffOPENAI_API_KEY=sk-fixture-from-dotenv\r\n'])
def test_client_reads_project_dotenv_with_priority_without_exporting_other_values(tmp_path,monkeypatch,line):
    api=api_module()
    (tmp_path/'.env').write_text(line+'UNRELATED=$(not-a-command)\n',encoding='utf-8')
    monkeypatch.setenv('OPENAI_API_KEY','sk-stale-env-fixture')
    captured={}
    import openai
    def create(**kwargs):captured.update(kwargs);return object()
    monkeypatch.setattr(openai,'OpenAI',create)
    api._create_client()
    assert captured['api_key']=='sk-fixture-from-dotenv'
    assert captured['base_url']=='https://api.openai.com/v1'
    assert 'UNRELATED' not in __import__('os').environ
    assert __import__('os').environ['OPENAI_API_KEY']=='sk-stale-env-fixture'


@pytest.mark.parametrize('content',[
    'OPENAI_API_KEY="sk-private-broken\n',
    'OPENAI_API_KEY=sk-private-one\nOPENAI_API_KEY=sk-private-two\n',
    'OPENAI_API_KEY=\n',
    'OPENAI_API_KEY=${OTHER_SECRET}\n'])
def test_invalid_dotenv_is_clear_and_never_exposes_file_secret(tmp_path,monkeypatch,content):
    api=api_module();(tmp_path/'.env').write_text(content)
    monkeypatch.setenv('OPENAI_API_KEY','sk-other-env-key')
    with pytest.raises(api.SCWUnavailableError) as caught:api._create_client()
    assert '.env' in str(caught.value)
    assert 'sk-private' not in str(caught.value)
    assert 'sk-other-env-key' not in str(caught.value)


def test_missing_dotenv_still_accepts_explicit_environment(monkeypatch):
    api=api_module();monkeypatch.setenv('OPENAI_API_KEY','sk-env-only-fixture')
    import openai
    captured={}
    monkeypatch.setattr(openai,'OpenAI',lambda **kw:captured.update(kw))
    api._create_client()
    assert captured['api_key']=='sk-env-only-fixture'


def test_successful_resume_resolves_historical_api_unavailable_marker(tmp_path):
    api=api_module()
    with pytest.raises(api.SCWUnavailableError):
        api.run_paraphrases(population(),tmp_path,client=FakeClient([APIError(403)]))
    api.run_paraphrases(population(),tmp_path,client=FakeClient([response()]))
    marker=json.loads((tmp_path/'api_unavailable.json').read_text())
    assert marker['status']=='resolved'

def test_retry_does_not_start_new_request_after_another_worker_fatal(tmp_path):
    api=api_module();sleeping=threading.Event();calls=[]
    class RacingClient:
        responses=None
        def __init__(self):self.responses=self;self.once=False
        def create(self,**kwargs):
            index=int(kwargs['input'].rsplit(' ',1)[-1])
            calls.append((index,(tmp_path/'api_unavailable.json').exists()))
            if index==0:return response()
            if index==1 and not self.once:
                self.once=True;raise TimeoutError()
            if index==2:
                assert sleeping.wait(3);raise APIError(403)
            return response()
    def wait_for_fatal(_):
        sleeping.set();deadline=time.monotonic()+3
        while not (tmp_path/'api_unavailable.json').exists():
            assert time.monotonic()<deadline
            time.sleep(.001)
    with pytest.raises(api.SCWUnavailableError):
        api.run_paraphrases(population(3),tmp_path,client=RacingClient(),workers=2,sleep_fn=wait_for_fatal)
    assert not any(after_fatal for _,after_fatal in calls)

def test_dotenv_is_project_scoped_and_key_never_persisted_in_receipt_cache(tmp_path,monkeypatch):
    api=api_module();secret='sk-only-a-fixture-secret'
    (tmp_path/'.env').write_text('OPENAI_API_KEY="'+secret+'"\n')
    other=tmp_path/'different-working-folder';other.mkdir();monkeypatch.chdir(other)
    monkeypatch.delenv('OPENAI_API_KEY',raising=False)
    import openai
    def create(**kwargs):
        assert kwargs['api_key']==secret
        return FakeClient([response()])
    monkeypatch.setattr(openai,'OpenAI',create)
    root=tmp_path/'cache'
    api.run_paraphrases(population(),root)
    assert all(secret not in p.read_text() for p in root.iterdir() if p.is_file())
    (tmp_path/'.env').unlink()
    monkeypatch.setattr(openai,'OpenAI',lambda **kw:pytest.fail('cached receipts must not need a key/client'))
    assert api.run_paraphrases(population(),root)[0]['status']=='ok'
