"""Exact RMSNorm/linear tangent readout without C x vocabulary x m tensors."""
import torch


class RotationReadout:
    def __init__(self, weight, gain, eps, p, q, tau, bias=None):
        self.weight, self.gain, self.eps, self.tau = weight.detach(), gain.detach(), eps, tau
        self.p, self.q = p.to(weight), q.to(weight)
        self.bias = None if bias is None else bias.detach()
        self.wp = self.weight @ (self.gain[:,None]*self.p)
        self.wq = self.weight @ (self.gain[:,None]*self.q)

    def tail(self, r):
        h = r * torch.rsqrt(r.square().mean(-1, keepdim=True)+self.eps) * self.gain
        return torch.nn.functional.linear(h, self.weight, self.bias)

    def components(self, r):
        r = r.to(self.weight)
        inv = torch.rsqrt(r.square().mean(-1, keepdim=True)+self.eps)/self.tau
        return self.tail(r)/self.tau, r@self.p, r@self.q, inv

    def log_probs(self, r, angles=None):
        logits, rp, rq, inv = self.components(r)
        if angles is not None:
            angles = angles.to(rp)
            ap = angles.cos()*rp - angles.sin()*rq
            aq = angles.sin()*rp + angles.cos()*rq
            logits = logits + inv*((ap-rp)@self.wp.T + (aq-rq)@self.wq.T)
        return torch.log_softmax(logits,-1)

    def observed(self, r, targets=None, generator=None):
        z, rp, rq, inv = self.components(r)
        probs = torch.softmax(z,-1)
        if targets is None:
            targets = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
        targets = targets.to(device=z.device,dtype=torch.long)
        ep, eq = probs@self.wp, probs@self.wq
        return inv*(rp*(self.wq[targets]-eq) - rq*(self.wp[targets]-ep))

    def full_scores(self, r):
        z, rp, rq, inv = self.components(r)
        logp = torch.log_softmax(z,-1)
        raw = inv[:,:,None]*(rp[:,None,:]*self.wq[None,:,:] - rq[:,None,:]*self.wp[None,:,:])
        phi = raw - (logp.exp()[:,:,None]*raw).sum(1,keepdim=True)
        return logp, phi

    def kl(self, r, angles):
        p, q = self.log_probs(r), self.log_probs(r,angles)
        return (q.exp()*(q-p)).sum(-1)

    def local(self, r, direction, strength, h=.002):
        z, rp, rq, inv = self.components(r)
        direction = direction.to(rp)
        g = inv*((rp*direction)@self.wq.T - (rq*direction)@self.wp.T)
        lp = torch.log_softmax(z,-1); p = lp.exp()
        g = g-(p*g).sum(-1,keepdim=True)
        lq = self.log_probs(r, strength*direction); q = lq.exp()
        delta = ((q-p)*g).sum(-1)
        kl = (q*(lq-lp)).sum(-1)
        variance = (p*g.square()).sum(-1)
        plus = self.log_probs(r, h*direction).exp()
        minus = self.log_probs(r, -h*direction).exp()
        fd = ((plus-minus)*g).sum(-1)/(2*h)
        return delta, kl, strength*variance, fd, variance.sqrt()
