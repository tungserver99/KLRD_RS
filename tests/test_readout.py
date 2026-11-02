import numpy as np
import torch


def test_fast_rms_scores_match_generic_derivative_and_exact_teacher():
    from klrd_rs.readout import RotationReadout
    from klrd_rs.math import planes, tangent, rotate
    g = torch.Generator().manual_seed(14)
    r = torch.randn(4, 12, generator=g, dtype=torch.float64)
    w = torch.randn(25,12,generator=g,dtype=torch.float64)
    gain = torch.randn(12,generator=g,dtype=torch.float64)
    p, q = map(torch.tensor, planes(np.eye(12), 3, 7))
    engine = RotationReadout(w, gain, 1e-6, p, q, .8)
    lp, phi = tangent(r, p, q, engine.tail, .8)
    targets = torch.tensor([2, 5, 10, 21])
    scores = engine.observed(r, targets)
    torch.testing.assert_close(scores, phi[torch.arange(4), targets], atol=1e-11, rtol=1e-11)
    a = torch.tensor([.2,-.4,.1],dtype=torch.float64)
    torch.testing.assert_close(engine.log_probs(r,a), torch.log_softmax(engine.tail(rotate(r,p,q,a))/.8,-1))
    delta, kl, prediction, fd, noise = engine.local(r, a/.1, .1, h=1e-5)
    expected = ((engine.log_probs(r,a).exp()-lp.exp())*(phi @ (a/.1))).sum(-1)
    torch.testing.assert_close(delta, expected, atol=1e-11, rtol=1e-10)
    assert torch.all(kl>=0) and torch.all(prediction>0)
    torch.testing.assert_close(prediction/.1, fd, atol=1e-5, rtol=1e-4)
