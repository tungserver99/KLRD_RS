import json
from pathlib import Path
import numpy as np
import pytest


def test_response_first_prediction_and_special_exclusion():
    from klrd_rs.data import encode_response
    class Tok:
        all_special_ids = [99]
        def apply_chat_template(self, *a, **kw): return {'input_ids': [10, 20, 30]}
        def encode(self, text, **kw): return [7, 99, 8]
    encoded = encode_response(Tok(), 'prompt', 'response')
    assert encoded == {'input_ids': [10,20,30,7,99,8], 'positions': [2,4], 'targets': [7,8]}


def test_disjoint_groups_and_duplicate_refusal():
    from klrd_rs.data import groups
    rows = [{'prompt_id': str(i)} for i in range(43)]
    batches, left = groups(rows, 20)
    assert len(batches) == 2 and left == 3
    assert len({r['prompt_id'] for b in batches for r in b}) == 40
    with pytest.raises(ValueError, match='Duplicate'): groups(rows+rows[:1], 20)


def test_no_calibration_evaluation_leakage_or_prompt_mismatch():
    from klrd_rs.data import validate_rows
    selected = {'calibration': [{'prompt_id':'a','prompt':'X','source':'train_gen'}],
                'final': [{'prompt_id':'b','prompt':'Y','source':'test_gen'}]}
    rows = [{'prompt_id':'a','prompt':'X','source':'train_gen','response':'Z'}]
    validate_rows(rows, selected, 'calibration')
    with pytest.raises(ValueError): validate_rows(rows, selected, 'final')
    with pytest.raises(ValueError): validate_rows([dict(rows[0], prompt='changed')], selected, 'calibration')


def test_resume_refuses_changed_inputs_and_detects_corrupt_cache(tmp_path):
    from klrd_rs.io import StageStore
    store = StageStore(tmp_path, {'input':'one'})
    store.save('x', {'value': np.array([1,2,3])})
    assert store.load('x')['value'].tolist() == [1,2,3]
    with pytest.raises(ValueError, match='changed|incompatible'):
        StageStore(tmp_path, {'input':'two'})
    (tmp_path/'x.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='checksum'): store.load('x')


def test_missing_paraphrase_stays_missing_and_unmatched_is_rejected():
    from klrd_rs.data import match_paraphrases
    original = [{'prompt_id':'a','prompt':'P','response':'R'}]
    assert match_paraphrases(original, None) is None
    row = {'split':'clean','prompt_id':'a','source_text':'R','paraphrased_text':'S','status':'success'}
    assert match_paraphrases(original, [row])[0]['response'] == 'S'
    with pytest.raises(ValueError): match_paraphrases(original, [dict(row, source_text='different')])


def test_launcher_is_exact_single_command():
    assert (Path(__file__).resolve().parents[1]/'run_experiment1.sh').read_bytes() == b'python main.py --continue-experiment1\n'
