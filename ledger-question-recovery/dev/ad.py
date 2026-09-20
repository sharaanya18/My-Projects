"""Minimal reverse-mode autodiff on numpy arrays."""
import numpy as np


class T:
    __slots__ = ("d", "g", "_bw", "_pr")

    def __init__(self, d, prev=(), bw=None):
        self.d = d
        self.g = None
        self._pr = prev
        self._bw = bw

    @property
    def shape(self):
        return self.d.shape

    def backward(self):
        topo, seen = [], set()
        stack = [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._pr:
                if id(p) not in seen:
                    stack.append((p, False))
        self.g = np.ones_like(self.d)
        for node in reversed(topo):
            if node._bw is not None and node.g is not None:
                node._bw(node.g)


def _acc(t, g):
    if g.dtype != t.d.dtype:
        g = g.astype(t.d.dtype)
    if t.g is None:
        t.g = g.copy()
    else:
        t.g += g


def _unbroadcast(g, shape):
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and g.shape[i] != 1:
            g = g.sum(axis=i, keepdims=True)
    return g


def add(a, b):
    out = T(a.d + b.d, (a, b))

    def bw(g):
        _acc(a, _unbroadcast(g, a.d.shape))
        _acc(b, _unbroadcast(g, b.d.shape))
    out._bw = bw
    return out


def sub(a, b):
    out = T(a.d - b.d, (a, b))

    def bw(g):
        _acc(a, _unbroadcast(g, a.d.shape))
        _acc(b, _unbroadcast(-g, b.d.shape))
    out._bw = bw
    return out


def mul(a, b):
    out = T(a.d * b.d, (a, b))

    def bw(g):
        _acc(a, _unbroadcast(g * b.d, a.d.shape))
        _acc(b, _unbroadcast(g * a.d, b.d.shape))
    out._bw = bw
    return out


def scale(a, k):
    out = T(a.d * k, (a,))

    def bw(g):
        _acc(a, g * k)
    out._bw = bw
    return out


def matmul(a, b):
    out = T(a.d @ b.d, (a, b))

    def bw(g):
        ga = g @ np.swapaxes(b.d, -1, -2)
        gb = np.swapaxes(a.d, -1, -2) @ g
        _acc(a, _unbroadcast(ga, a.d.shape))
        _acc(b, _unbroadcast(gb, b.d.shape))
    out._bw = bw
    return out


def gelu(a):
    x = a.d
    c = np.sqrt(2.0 / np.pi)
    inner = c * (x + 0.044715 * x ** 3)
    t = np.tanh(inner)
    out = T(0.5 * x * (1.0 + t), (a,))

    def bw(g):
        dinner = c * (1.0 + 3 * 0.044715 * x ** 2)
        _acc(a, g * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * dinner))
    out._bw = bw
    return out


def sigmoid(a):
    s = 1.0 / (1.0 + np.exp(-np.clip(a.d, -60, 60)))
    out = T(s, (a,))

    def bw(g):
        _acc(a, g * s * (1.0 - s))
    out._bw = bw
    return out


def softmax(a, axis=-1):
    z = a.d - a.d.max(axis=axis, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=axis, keepdims=True)
    out = T(p, (a,))

    def bw(g):
        _acc(a, p * (g - (g * p).sum(axis=axis, keepdims=True)))
    out._bw = bw
    return out


def layernorm(a, w, b, eps=1e-5):
    x = a.d
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc ** 2).mean(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = xc * inv
    out = T(xhat * w.d + b.d, (a, w, b))

    def bw(g):
        gw = g * w.d
        n = x.shape[-1]
        gx = inv * (gw - gw.mean(axis=-1, keepdims=True)
                    - xhat * (gw * xhat).mean(axis=-1, keepdims=True))
        _acc(a, gx)
        _acc(w, _unbroadcast((g * xhat).reshape(-1, n).sum(axis=0), w.d.shape))
        _acc(b, _unbroadcast(g.reshape(-1, n).sum(axis=0), b.d.shape))
    out._bw = bw
    return out


def embed(table, idx):
    out = T(table.d[idx], (table,))

    def bw(g):
        gt = np.zeros_like(table.d)
        np.add.at(gt, idx, g)
        _acc(table, gt)
    out._bw = bw
    return out


def reshape(a, shape):
    old = a.d.shape
    out = T(a.d.reshape(shape), (a,))

    def bw(g):
        _acc(a, g.reshape(old))
    out._bw = bw
    return out


def transpose(a, axes):
    inv = np.argsort(axes)
    out = T(np.transpose(a.d, axes), (a,))

    def bw(g):
        _acc(a, np.transpose(g, inv))
    out._bw = bw
    return out


def gather_rows(a, flat_idx):
    """a: (N, D) -> (len(idx), D)"""
    out = T(a.d[flat_idx], (a,))

    def bw(g):
        gt = np.zeros_like(a.d)
        np.add.at(gt, flat_idx, g)
        _acc(a, gt)
    out._bw = bw
    return out


def add_const(a, c):
    out = T(a.d + c, (a,))

    def bw(g):
        _acc(a, g)
    out._bw = bw
    return out


def mul_const(a, c):
    out = T(a.d * c, (a,))

    def bw(g):
        _acc(a, g * c)
    out._bw = bw
    return out


def mean_all(a):
    n = a.d.size
    out = T(np.array(a.d.mean(), dtype=a.d.dtype), (a,))

    def bw(g):
        _acc(a, np.full_like(a.d, g / n))
    out._bw = bw
    return out


def ce_loss(logits, targets, weights, smooth=0.0):
    """logits (N, V), targets (N,), weights (N,). Returns scalar = sum(w*ce)/sum(w)."""
    z = logits.d - logits.d.max(axis=-1, keepdims=True)
    e = np.exp(z)
    s = e.sum(axis=-1, keepdims=True)
    logp = z - np.log(s)
    p = e / s
    n, v = logits.d.shape
    wsum = weights.sum()
    if wsum <= 0:
        wsum = 1.0
    nll = -logp[np.arange(n), targets]
    if smooth > 0:
        loss_vec = (1 - smooth) * nll + smooth * (-logp.mean(axis=-1))
    else:
        loss_vec = nll
    out = T(np.array((weights * loss_vec).sum() / wsum, dtype=logits.d.dtype), (logits,))

    def bw(g):
        gg = np.zeros((n, v), dtype=logits.d.dtype)
        gg[np.arange(n), targets] = -1.0
        if smooth > 0:
            grad = (1 - smooth) * (p - (-gg)) + smooth * (p - 1.0 / v)
        else:
            grad = p + gg
        _acc(logits, grad * (weights[:, None] * (g / wsum)))
    out._bw = bw
    return out


def bce_loss(logits, targets, weights):
    """logits (N,), targets (N,), weights (N,)."""
    x = np.clip(logits.d, -60, 60)
    s = 1.0 / (1.0 + np.exp(-x))
    eps = 1e-9
    lv = -(targets * np.log(s + eps) + (1 - targets) * np.log(1 - s + eps))
    wsum = weights.sum()
    if wsum <= 0:
        wsum = 1.0
    out = T(np.array((weights * lv).sum() / wsum, dtype=logits.d.dtype), (logits,))

    def bw(g):
        _acc(logits, (s - targets) * weights * (g / wsum))
    out._bw = bw
    return out


def mul_arr(a, arr):
    out = T(a.d * arr, (a,))

    def bw(g):
        _acc(a, g * arr)
    out._bw = bw
    return out


def abs_(a):
    s = np.sign(a.d)
    out = T(np.abs(a.d), (a,))

    def bw(g):
        _acc(a, g * s)
    out._bw = bw
    return out


def concat(parts, axis=-1):
    sizes = [p.d.shape[axis] for p in parts]
    out = T(np.concatenate([p.d for p in parts], axis=axis), tuple(parts))

    def bw(g):
        off = 0
        for p, s in zip(parts, sizes):
            sl = [slice(None)] * g.ndim
            sl[axis] = slice(off, off + s)
            _acc(p, g[tuple(sl)])
            off += s
    out._bw = bw
    return out
