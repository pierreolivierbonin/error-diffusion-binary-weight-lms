import torch

BLK = 256


def _blockify(x):
    flat = x.flatten()
    pad = (-flat.numel()) % BLK
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=x.device, dtype=x.dtype)])
    return flat.reshape(-1, BLK)


def _quant(x, signed):
    b = _blockify(x)
    top = 127.0 if signed else 255.0
    s = b.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)
    q = (b * (top / s)).round().to(torch.int8 if signed else torch.uint8)
    return q, s.squeeze(1)


def _dequant(q, s, signed, numel, shape):
    top = 127.0 if signed else 255.0
    b = q.to(torch.float32) * (s.unsqueeze(1) / top)
    return b.flatten()[:numel].reshape(shape)


class Adam8:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.95), eps=1e-8):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.t = 0
        self.state = {}
        for p in self.params:
            zm, sm = _quant(torch.zeros_like(p), True)
            zv, sv = _quant(torch.zeros_like(p), False)
            self.state[id(p)] = {"m8": zm, "ms": sm, "v8": zv, "vs": sv}

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def state_bytes(self):
        total = 0
        for st in self.state.values():
            total += st["m8"].numel() + st["v8"].numel()
            total += (st["ms"].numel() + st["vs"].numel()) * 4
        return total

    @torch.no_grad()
    def step(self):
        self.t += 1
        for p in self.params:
            if p.grad is None:
                continue
            st = self.state[id(p)]
            m = _dequant(st["m8"], st["ms"], True, p.numel(), p.shape)
            v = _dequant(st["v8"], st["vs"], False, p.numel(), p.shape).pow(2)
            m = self.b1 * m + (1 - self.b1) * p.grad
            v = self.b2 * v + (1 - self.b2) * p.grad.pow(2)
            mh = m / (1 - self.b1**self.t)
            vh = v / (1 - self.b2**self.t)
            upd = (self.lr * mh / (vh.sqrt() + self.eps)).clamp_(-0.01, 0.01)
            p.add_(-upd)
            st["m8"], st["ms"] = _quant(m, True)
            st["v8"], st["vs"] = _quant(v.sqrt(), False)


class SGDM:
    def __init__(self, params, lr=1e-2, mom=0.9):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.mom = mom
        self.buf = {id(p): torch.zeros_like(p) for p in self.params}

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def state_bytes(self):
        return sum(b.numel() * b.element_size() for b in self.buf.values())

    @torch.no_grad()
    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            b = self.buf[id(p)]
            b.mul_(self.mom).add_(p.grad)
            p.add_(-self.lr * b)
