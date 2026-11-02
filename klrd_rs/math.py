"""Low-rank exact rotations and key-randomization statistics."""
import itertools
import hashlib
import numpy as np
import torch
from scipy.special import betainc


def planes(basis, m, seed):
    basis = np.asarray(basis, dtype=np.float64)
    if basis.ndim != 2 or basis.shape[1] < 2*m or not np.isfinite(basis).all():
        raise ValueError('Insufficient finite public basis directions')
    if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), atol=1e-6):
        raise ValueError('Public basis must be orthonormal')
    raw = np.random.default_rng(seed).normal(size=(basis.shape[1], 2*m))
    u, rr = np.linalg.qr(raw, mode='reduced')
    u *= np.where(np.diag(rr) >= 0, 1., -1.)
    vectors = basis @ u
    return vectors[:, ::2].copy(), vectors[:, 1::2].copy()


def rotate(r, p, q, angles):
    p, q, angles = [x.to(device=r.device, dtype=r.dtype) for x in (p, q, angles)]
    a, b = r @ p, r @ q
    ap = angles.cos()*a - angles.sin()*b
    bp = angles.sin()*a + angles.cos()*b
    return r + (ap-a) @ p.T + (bp-b) @ q.T


def tangent(r, p, q, tail, tau):
    """Full-vocabulary score via exact directional derivatives of fixed readout.

    Shape: logp [contexts,vocabulary], phi [contexts,vocabulary,m].
    Only use on small chunks; callers never materialize the entire corpus logits.
    """
    p, q = p.to(r), q.to(r)
    logits = tail(r) / tau
    logp = torch.log_softmax(logits, -1)
    prob = logp.exp()
    rp, rq = r @ p, r @ q
    directions = []
    for j in range(p.shape[1]):
        dr = rp[:, j:j+1]*q[:, j] - rq[:, j:j+1]*p[:, j]
        _, dz = torch.func.jvp(tail, (r,), (dr,))
        dz = dz / tau
        directions.append(dz - (prob*dz).sum(-1, keepdim=True))
    return logp.detach(), torch.stack(directions, -1).detach()


def keys(n, m, seed):
    if n < 1 or m < 2:
        raise ValueError('Keys require n>=1 and dimension>=2')
    x = np.random.default_rng(seed).standard_normal((n, m))
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def key_ids(bank):
    return [{'index': i, 'sha256': hashlib.sha256(np.asarray(k, dtype='<f8').tobytes()).hexdigest()}
            for i, k in enumerate(bank)]


def sphere_p(v, k):
    v, k = np.asarray(v, dtype=float), np.asarray(k, dtype=float)
    if v.ndim != 1 or k.shape[-1] != v.size or v.size < 2:
        raise ValueError('Invalid score/key dimensions')
    if not np.isfinite(v).all() or not np.isfinite(k).all():
        raise ValueError('Nonfinite score or key')
    if not np.allclose(np.linalg.norm(k, axis=-1), 1., atol=1e-8, rtol=0):
        raise ValueError('Keys must have unit norm')
    norm = np.linalg.norm(v)
    if norm == 0:
        return np.ones(k.shape[:-1]) if k.ndim > 1 else 1.
    t = np.clip(k @ (v/norm), -1., 1.)
    b = (v.size-1)/2
    # Symmetry gives the survival directly, avoiding cancellation in 1-CDF.
    # betaincc is unavailable in older supported SciPy versions.
    return betainc(b, b, (1-t)/2)


def sign_p(v, signs):
    v, signs = np.asarray(v), np.asarray(signs)
    if v.shape != (8,) or signs.shape != (8,) or not np.isin(signs, [-1, 1]).all():
        raise ValueError('Legacy diagnostic requires exactly eight sign coordinates')
    all_signs = np.asarray(list(itertools.product([-1., 1.], repeat=8)))
    threshold = float(signs @ v)
    tol = 1e-12 * max(1., abs(threshold), np.linalg.norm(v))
    return float(np.mean(all_signs @ v >= threshold-tol))


def fit_fisher(scores, rank_rtol=1e-8):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or not np.isfinite(scores).all() or len(scores) <= scores.shape[1]:
        raise ValueError('Insufficient finite Fisher observations')
    f = scores.T @ scores / len(scores)
    eig, u = np.linalg.eigh(f)
    rank = int((eig > max(eig[-1]*rank_rtol, 1e-14)).sum())
    if rank != f.shape[0]:
        raise ValueError(f'Fisher rank {rank} < {f.shape[0]}')
    eps = 1e-3 * np.trace(f)/f.shape[0]
    leig = eig + eps
    l = (u * (1/np.sqrt(leig))) @ u.T
    return f, l, {'ridge': float(eps), 'rank': rank, 'eigenvalues': eig.tolist(),
                   'effective_eigenvalues': leig.tolist(),
                   'condition_number': float(eig[-1]/eig[0]),
                   'effective_condition_number': float(leig[-1]/leig[0])}


def choose_lambda(evaluate, target=.005, p99_max=.05, max_lambda=2., iterations=24):
    trace = []
    def measure(value):
        x = np.asarray(evaluate(value), dtype=float)
        if not x.size or not np.isfinite(x).all() or x.min() < -1e-6:
            raise ValueError('Invalid pooled teacher KL')
        x = np.maximum(x, 0)
        item = {'lambda': value, 'mean': float(x.mean()), 'p99': float(np.quantile(x, .99))}
        trace.append(item)
        return item, item['mean'] <= target and item['p99'] <= p99_max
    zero, ok = measure(0.)
    if not ok or zero['mean'] > 1e-7:
        raise ValueError('Zero-strength teacher differs from clean readout')
    low, high = 0., .01
    while True:
        item, ok = measure(high)
        if not ok or high >= max_lambda:
            break
        low, high = high, min(2*high, max_lambda)
    if ok:
        low = high
    else:
        for _ in range(iterations):
            mid = (low+high)/2
            item, ok = measure(mid)
            if ok: low = mid
            else: high = mid
    final, ok = measure(low)
    if not ok or low <= 0 or final['mean'] < .95*target:
        raise ValueError('No feasible common lambda approaching mean KL target within 5%')
    return low, trace
