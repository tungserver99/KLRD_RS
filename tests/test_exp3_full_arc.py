"""Prepared server-side ARC integrity tests. Never executed on the local host."""
from pathlib import Path
import hashlib
import json

import pytest

from klrd_rs.e3full.arc import _counts, _immutable_json, _verify_vendor


ROOT = Path(__file__).resolve().parents[1]


def native_result(accuracy=.5, original=4, effective=4, shots=0):
    return {'results': {'arc_challenge': {'acc_norm,none': accuracy}},
        'n-samples': {'arc_challenge': {'original': original, 'effective': effective}},
        'n-shot': {'arc_challenge': shots}}


def test_vendored_original_bytes_match_receipt():
    receipt = json.loads((ROOT / 'klrd_arc/SOURCE_RECEIPT.json').read_text())
    assert _verify_vendor(ROOT) == receipt['files']
    for name, expected in receipt['files'].items():
        assert hashlib.sha256((ROOT / 'klrd_arc' / name).read_bytes()).hexdigest() == expected


def test_normalized_correct_count_uses_effective_native_samples():
    result = _counts(native_result(accuracy=.75), 4)
    assert (result['correct_count'], result['total_count']) == (3, 4)
    assert result['accuracy_percent'] == 75


@pytest.mark.parametrize('result', [native_result(effective=2),
    native_result(original=5), native_result(accuracy=.51),
    native_result(accuracy=float('nan')), native_result(accuracy=1.25),
    native_result(shots=5), native_result(effective=True)])
def test_partial_nonintegral_invalid_or_nonzero_shot_results_rejected(result):
    with pytest.raises(ValueError):
        _counts(result, 4)


def test_protocol_resume_rejects_changed_identity_without_overwrite(tmp_path):
    path = tmp_path / 'binding.json'
    original = {'identity': 'run-A', 'model': 'teacher', 'protocol_sha256': 'task-A'}
    _immutable_json(path, original)
    _immutable_json(path, original)
    with pytest.raises(ValueError):
        _immutable_json(path, dict(original, protocol_sha256='task-B'))
    assert json.loads(path.read_text()) == original
