import itertools
import numpy as np
import pytest
import torch
from scipy.linalg import expm
from scipy.stats import kstest


def test_rotation_matches_matrix_exponential_and_preserves_norm():
    from klrd_rs.math import planes, rotate
    basis = np.eye(12)[:, :10]
    p, q = planes(basis, 3, 91)
    x = np.random.default_rng(4).normal(size=(7, 12))
    a = np.array([.2, -.4, .13])
    matrix = sum(t * (np.outer(v, u) - np.outer(u, v)) for t, u, v in zip(a, p.T, q.T))
    y = rotate(torch.tensor(x), torch.tensor(p), torch.tensor(q), torch.tensor(a)).numpy()
    np.testing.assert_allclose(y, x @ expm(matrix).T, atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(y, axis=1), np.linalg.norm(x, axis=1), atol=1e-12)


def test_tangent_full_vocab_agrees_with_autodiff_and_finite_difference():
    from klrd_rs.math import planes, rotate, tangent
    rng = np.random.default_rng(8)
    r = torch.tensor(rng.normal(size=(2, 12)), dtype=torch.float64)
    w = torch.tensor(rng.normal(size=(23, 12)), dtype=torch.float64)
    gain = torch.tensor(rng.uniform(.8, 1.2, 12))
    def tail(x):
        return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * gain) @ w.T
    p, q = map(torch.tensor, planes(np.eye(12), 3, 9))
    logp, phi = tangent(r, p, q, tail, .8)
    np.testing.assert_allclose((logp.exp().unsqueeze(-1) * phi).sum(1), 0, atol=1e-11)
    for j in range(3):
        a = torch.zeros(3, dtype=r.dtype); a[j] = 1e-5
        plus = torch.log_softmax(tail(rotate(r, p, q, a))/.8, -1)
        minus = torch.log_softmax(tail(rotate(r, p, q, -a))/.8, -1)
        torch.testing.assert_close(phi[:, :, j], (plus-minus)/2e-5, atol=1e-7, rtol=1e-6)


def test_spherical_tail_boundaries_reference_and_randomized_null():
    from klrd_rs.math import sphere_p, keys
    assert sphere_p(np.zeros(32), keys(1, 32, 3)[0]) == 1
    assert sphere_p(np.eye(32)[0], np.eye(32)[0]) == 0
    assert sphere_p(np.eye(32)[0], -np.eye(32)[0]) == 1
    assert sphere_p(np.eye(32)[0], np.eye(32)[1]) == pytest.approx(.5)
    ks = keys(50000, 32, 2027)
    values = sphere_p(np.eye(32)[0], ks)
    assert kstest(values, 'uniform').statistic < .01
    with pytest.raises(ValueError): sphere_p(np.ones(32), np.ones(32))
    with pytest.raises(ValueError): sphere_p(np.full(32, np.nan), ks[0])


def test_fisher_whitening_and_rank_failure():
    from klrd_rs.math import fit_fisher
    rng = np.random.default_rng(8)
    x = rng.normal(size=(10000, 4)) @ np.diag([1, 2, 3, 4])
    f, l, info = fit_fisher(x)
    np.testing.assert_allclose(l @ (f + info['ridge']*np.eye(4)) @ l, np.eye(4), atol=1e-12)
    with pytest.raises(ValueError, match='rank'): fit_fisher(np.ones((50, 4)))


def test_legacy_sign_p_is_discrete_and_ties_are_included():
    from klrd_rs.math import sign_p
    assert sign_p(np.ones(8), np.ones(8)) == 1/256
    assert sign_p(np.zeros(8), np.ones(8)) == 1


def test_strength_is_one_pooled_lambda_and_reports_infeasibility():
    from klrd_rs.math import choose_lambda
    value, trace = choose_lambda(lambda a: np.array([a*a, 2*a*a]))
    assert np.mean([value*value, 2*value*value]) == pytest.approx(.005, rel=.01)
    with pytest.raises(ValueError, match='target|feasible'):
        choose_lambda(lambda a: np.zeros(10))
