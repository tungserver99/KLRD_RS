"""SERVER ONLY. Recovery regression sources; never run on the local machine."""
import json
from pathlib import Path
import pytest
from klrd_rs.io import digest,read_json,read_jsonl


def cohort():
    prompts=[{'prompt_id':str(i),'prompt':'prompt '+str(i)} for i in range(4)]
    identity={'run':'frozen','settings':{'seed':17,'kind':'clean_train'}}
    from klrd_rs.e3.generation import _request_seed
    records=[]
    for prompt in prompts:
        row=dict(prompt,response='response '+prompt['prompt_id'],
            seed=_request_seed(identity['settings'],prompt),generation_identity=digest(identity))
        row['record_sha256']=digest(row);records.append(row)
    return prompts,identity,records


def encoded(row):
    return (json.dumps(row,ensure_ascii=False)+'\n').encode('utf-8')


def load(path,prompts,identity,allow=True):
    from klrd_rs.e3.journal import load_generation_journal
    from klrd_rs.e3.generation import _validate
    return load_generation_journal(path,prompts,identity,_validate,allow_recovery=allow)


def test_middle_corruption_preserves_all_verified_rows_after_hole_and_backup(tmp_path):
    prompts,identity,records=cohort();path=tmp_path/'train_clean.jsonl'
    original=encoded(records[0])+b'broken middle fragment}\n'+encoded(records[2])+encoded(records[3])
    path.write_bytes(original)
    kept=load(path,prompts,identity)
    assert kept==[records[0],records[2],records[3]]
    assert read_jsonl(path)==kept
    backups=list(tmp_path.glob('*.corrupt.*.bak'))
    assert len(backups)==1 and backups[0].read_bytes()==original
    receipt=read_json(next(tmp_path.glob('*.recovery.*.json')))
    assert receipt['status']=='RECOVERED' and receipt['kept_records']==3
    assert receipt['missing_responses']==1 and receipt['invalid_lines'][0]['line']==2
    # A second load preserves the repaired bytes/record hashes without another backup.
    assert load(path,prompts,identity)==kept
    assert len(list(tmp_path.glob('*.corrupt.*.bak')))==1


def test_missing_newline_keeps_a_complete_hash_verified_final_record(tmp_path):
    prompts,identity,records=cohort();path=tmp_path/'train_clean.jsonl'
    path.write_bytes(encoded(records[0])+encoded(records[1]).rstrip(b'\n'))
    assert load(path,prompts,identity)==records[:2]
    assert path.read_bytes().endswith(b'\n')


@pytest.mark.parametrize('change',['hash','identity','duplicate'])
def test_valid_json_mismatch_is_never_silently_discarded_or_repaired(tmp_path,change):
    prompts,identity,records=cohort();path=tmp_path/'train_clean.jsonl'
    wrong=dict(records[2])
    if change=='hash':wrong['response']='changed without matching hash'
    elif change=='identity':
        wrong['generation_identity']='other-run';wrong.pop('record_sha256')
        wrong['record_sha256']=digest(wrong)
    else:wrong=records[0]
    original=encoded(records[0])+b'broken}\n'+encoded(wrong)
    path.write_bytes(original)
    with pytest.raises(ValueError):load(path,prompts,identity)
    assert path.read_bytes()==original and not list(tmp_path.glob('*.bak'))


def test_complete_journal_is_not_automatically_replaced(tmp_path):
    prompts,identity,records=cohort();path=tmp_path/'train_clean.jsonl'
    original=encoded(records[0])+b'broken}\n';path.write_bytes(original)
    path.with_suffix('.complete.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='completed'):load(path,prompts,identity)
    assert path.read_bytes()==original


def test_recovery_requires_explicit_partial_clean_generation_scope(tmp_path):
    prompts,identity,records=cohort();path=tmp_path/'train_clean.jsonl'
    original=encoded(records[0])+b'broken}\n';path.write_bytes(original)
    with pytest.raises(ValueError):load(path,prompts,identity,allow=False)
    assert path.read_bytes()==original


def test_sparse_resume_generates_missing_ids_only_and_returns_original_order():
    from klrd_rs.e3.generation import _pending_rows,_ordered_records,_validate
    prompts,identity,records=cohort()
    sparse=[records[0],records[2],records[3]]
    _validate(sparse,prompts,identity)
    assert _pending_rows(prompts,sparse)==[prompts[1]]
    assert _ordered_records(sparse+[records[1]],prompts)==records


def test_sparse_resume_does_not_accept_wrong_seed_even_with_recomputed_hash():
    from klrd_rs.e3.generation import _validate
    prompts,identity,records=cohort();wrong=dict(records[2],seed=999)
    wrong.pop('record_sha256');wrong['record_sha256']=digest(wrong)
    with pytest.raises(ValueError):_validate([wrong],prompts,identity)
