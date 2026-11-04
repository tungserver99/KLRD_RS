"""Server-only contracts. Authored locally; never execute on the author's laptop."""
from types import SimpleNamespace
import zipfile
import numpy as np
import pytest
from klrd_rs.io import write_json, sha256
from klrd_rs.e3 import evaluate, report
from klrd_rs.math import keys


def state(tmp_path):
    root=tmp_path/'repository';out=root/'experiment3_rs/run_001'
    out.mkdir(parents=True)
    return {'root':root,'out':out,'identity':'test-run','cfg':SimpleNamespace()}


def test_zero_vectors_remain_trials_and_wrong_keys_are_not_pooled(tmp_path,monkeypatch):
    s=state(tmp_path)
    bank=keys(64,32,2026102803)
    monkeypatch.setattr(evaluate,'get_frozen',lambda _: {'bank':bank})
    rows=[{'prompt_id':str(i),'response_id':'r'+str(i),'response':''} for i in range(200)]
    groups=[{'Q':100,'group_id':'g0','prompt_ids':[str(i) for i in range(100)]},
        {'Q':100,'group_id':'g1','prompt_ids':[str(i) for i in range(100,200)]},
        {'Q':1000,'group_id':'descriptive','prompt_ids':[str(i) for i in range(200)]}]
    # The deliberate incomplete aggregate must fail, rather than produce a fake
    # Q1000 result from200 rows.
    with pytest.raises(ValueError,match='Incomplete frozen group'):
        evaluate.tables(s,{'student':(rows,np.zeros((200,32)),np.zeros(200,dtype=np.int64))},groups,s['out']/'scores')
    summaries=evaluate.tables(s,{'student':(rows,np.zeros((200,32)),np.zeros(200,dtype=np.int64))},groups[:2],s['out']/'scores')
    assert all(r['groups']==2 for r in summaries)
    assert all(r['detections']==0 for r in summaries)
    controls=evaluate_csv(s['out']/'scores/clean_wrong_controls.csv')
    assert len(controls)==40  #20keys x2alpha, each has2 independent groups
    assert all(int(r['groups'])==2 for r in controls)
    identity=evaluate_csv(s['out']/'scores/closed_set_identification.csv')
    assert all(r['tie']=='True' for r in identity)


def evaluate_csv(path):
    import csv
    with path.open(newline='',encoding='utf-8-sig') as stream:return list(csv.DictReader(stream))


def test_q1000_no_independent_group_ci(tmp_path,monkeypatch):
    s=state(tmp_path);bank=keys(64,32,2026102803)
    monkeypatch.setattr(evaluate,'get_frozen',lambda _: {'bank':bank})
    rows=[{'prompt_id':str(i),'response_id':'r'+str(i),'response':''} for i in range(1000)]
    groups=[{'Q':1000,'group_id':'all','prompt_ids':[r['prompt_id'] for r in rows]}]
    summaries=evaluate.tables(s,{'student':(rows,np.zeros((1000,32)),np.zeros(1000,dtype=np.int64))},groups,s['out']/'scores')
    assert all(r['CI95_low'] is None and r['CI95_high'] is None for r in summaries)
    assert all(r['groups']==1 for r in summaries)


def test_report_cannot_claim_eval_from_training_and_zip_excludes_private(tmp_path):
    s=state(tmp_path);out=s['out']
    adapter=out/'checkpoints/best_adapter/adapter_config.json'
    write_json(adapter,{'r':16})
    write_json(out/'training/selection.json',{'identity':s['identity'],'status':'TRAINING_ONLY','adapter_files':{'adapter_config.json':sha256(adapter)}})
    secret=out/'private/eval_secret.jsonl';secret.parent.mkdir();secret.write_text('PRIVATE SENTINEL',encoding='utf-8')
    weight=out/'merged_student/model.safetensors';weight.parent.mkdir();weight.write_bytes(b'WEIGHT SENTINEL')
    write_json(out/'merged_student/config.json',{'model_type':'qwen2'})
    write_json(out/'provenance/merged_manifest.json',{'identity':s['identity'],'files':evaluate.file_manifest(out/'merged_student')})
    key=out/'private/keys.npz';key.write_bytes(b'KEY SENTINEL')
    api=out/'private/api/owner_00/api_results.jsonl';api.parent.mkdir(parents=True);api.write_text('API SECRET',encoding='utf-8')
    value=report.run(s,package=True)
    assert value['status']=='TRAINING_ONLY'
    assert value['evaluation_complete'] is False
    with zipfile.ZipFile(out/'KLRD_RS_experiment3_review.zip') as archive:
        names=archive.namelist()
        assert not any('private/' in name or 'merged_student/' in name or 'checkpoints/' in name for name in names)
        assert all(b'SENTINEL' not in archive.read(name) and b'API SECRET' not in archive.read(name) for name in names)


def test_changed_completed_product_is_incomplete(tmp_path):
    s=state(tmp_path);out=s['out'];product=out/'scores/group_summary.csv'
    product.parent.mkdir(parents=True);product.write_text('initial',encoding='utf-8')
    write_json(out/'provenance/evaluation_complete.json',{'identity':s['identity'],'status':'EVAL_COMPLETE',
        'products_sha256':{'scores/group_summary.csv':sha256(product)}})
    product.write_text('changed',encoding='utf-8')
    value=report.run(s,package=False)
    assert value['status']=='INCOMPLETE'
    assert value['evaluation_complete'] is False
    assert value['errors']


def test_evaluation_only_checks_ordinary_merged_hashes(tmp_path):
    s=state(tmp_path);out=s['out'];model=out/'merged_student';model.mkdir()
    write_json(model/'config.json',{'model_type':'qwen2','architectures':['Qwen2ForCausalLM']})
    write_json(out/'provenance/merged_manifest.json',{'identity':s['identity'],'files':evaluate.file_manifest(model)})
    assert evaluate.verify_merged(s)['identity']==s['identity']
    (model/'config.json').write_text('{}',encoding='utf-8')
    with pytest.raises(ValueError,match='merged checkpoint'):evaluate.verify_merged(s)




def test_selected_adapter_without_merged_checkpoint_is_incomplete(tmp_path):
    s=state(tmp_path)
    write_json(s['out']/'training/selection.json',{'identity':s['identity'],'status':'TRAINING_ONLY',
        'adapter_files':{'adapter_config.json':'missing'}})
    value=report.run(s,package=False)
    assert value['status']=='INCOMPLETE'
    assert value['training_completed'] is False


def test_paraphrase_zero_original_signal_precedes_transport(tmp_path,monkeypatch):
    from klrd_rs.e3 import paraphrase
    s=state(tmp_path)
    monkeypatch.setattr(paraphrase,'checked_receipt',lambda *_: {'summary':[
        {'model':'student','Q':100,'alpha':.01,'detections':0}]})
    def forbidden(*args,**kwargs):
        pytest.fail('Zero-signal evaluation must not load attack source or call API')
    monkeypatch.setattr(paraphrase,'load_prepared',forbidden)
    monkeypatch.setattr(paraphrase,'run_copied_transport',forbidden)
    with pytest.raises(ValueError,match='NO_ORIGINAL_STUDENT_SIGNAL'):
        paraphrase.run(s)


def test_utility_reuses_frozen_e2_repetition_policy():
    row={'prompt_id':'p','response_id':'r','response':'a b c d '*30,
        'native_visible_tokens':120,'eos_terminated':False,'max_length_truncated':False,'roundtrip_equal':True}
    value=evaluate.utility({'student':([row],np.zeros((1,32)),np.array([120]))})['student']
    assert value['severe_repeated_4gram_count']==1
    assert value['very_short_under5_visible_tokens']==0
