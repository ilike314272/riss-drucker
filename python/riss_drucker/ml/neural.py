"""Neural amortized posterior: heteroscedastic MLP ensemble (numpy; optional torch).

q(theta | features) = N(mu(x), diag sigma^2(x)), trained by Gaussian negative
log-likelihood on simulated (features, theta) pairs from rd.ml.dataset. A deep
ensemble adds the spread of member means to the variance (epistemic part).
Same predict() interface as BayesLinearPosterior, so rd.ml.amortized.calibration
and verify work unchanged. Missing features (NaN) are imputed with the training
mean and flagged by a mask input.

Backend "torch" (GPU on Kaggle when available) trains the same architecture; the
numpy backend is the reference and needs nothing beyond numpy.
"""
from __future__ import annotations

import math

import numpy as np


def _prep_fit(X):
    X = np.asarray(X, float)
    mask = np.isnan(X)
    cnt = (~mask).sum(axis=0)
    mu = np.where(cnt > 0, np.where(mask, 0.0, X).sum(axis=0) / np.maximum(cnt, 1), 0.0)
    Xf = np.where(mask, mu, X)
    sd = Xf.std(axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    has_mask = mask.any(axis=0)
    return mu, sd, has_mask


def _prep(X, mu, sd, has_mask):
    X = np.asarray(X, float)
    mask = np.isnan(X)
    Z = (np.where(mask, mu, X) - mu) / sd
    if has_mask.any():
        Z = np.hstack([Z, mask[:, has_mask].astype(float)])
    return Z


class _NumpyMLP:
    def __init__(self, sizes, rng, out_mode="gauss"):
        self.W = [rng.standard_normal((a, b)) * math.sqrt(2.0 / (a + b)) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.out_mode = out_mode

    def forward(self, X):
        hs = [X]
        h = X
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            h = np.tanh(z) if i < len(self.W) - 1 else z
            hs.append(h)
        return hs

    def params(self):
        return self.W + self.b

    def grads(self, hs, dout):
        gW, gb = [None] * len(self.W), [None] * len(self.b)
        d = dout
        for i in range(len(self.W) - 1, -1, -1):
            gW[i] = hs[i].T @ d
            gb[i] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.W[i].T) * (1 - hs[i] ** 2)
        return gW + gb


def _adam_train(net, loss_and_grad, Xtr, Ytr, Xva, Yva, *, epochs, lr, batch, wd, patience, rng):
    params = net.params()
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    best, best_params, bad, t = np.inf, [p.copy() for p in params], 0, 0
    n = len(Xtr)
    for ep in range(epochs):
        perm = rng.permutation(n)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            _, g = loss_and_grad(net, Xtr[idx], Ytr[idx])
            t += 1
            for k, (p, gk) in enumerate(zip(params, g)):
                gk = gk / len(idx) + wd * p
                m[k] = 0.9 * m[k] + 0.1 * gk
                v[k] = 0.999 * v[k] + 0.001 * gk * gk
                p -= lr * (m[k] / (1 - 0.9**t)) / (np.sqrt(v[k] / (1 - 0.999**t)) + 1e-8)
        lv, _ = loss_and_grad(net, Xva, Yva, need_grad=False)
        if lv < best - 1e-6:
            best, best_params, bad = lv, [p.copy() for p in params], 0
        else:
            bad += 1
            if bad > patience:
                break
    for p, bp in zip(params, best_params):
        p[...] = bp
    return best


def _gauss_loss(net, X, Y, need_grad=True):
    hs = net.forward(X)
    out = hs[-1]
    P = Y.shape[1]
    mu, s = out[:, :P], np.clip(out[:, P:], -12, 8)          # s = log variance
    iv = np.exp(-s)
    r = Y - mu
    loss = 0.5 * np.mean(np.sum(s + r * r * iv, axis=1))
    if not need_grad:
        return loss, None
    dmu = -r * iv
    ds = 0.5 * (1 - r * r * iv)
    return loss, net.grads(hs, np.hstack([dmu, ds]))


def _bce_loss(net, X, Y, need_grad=True):
    hs = net.forward(X)
    z = hs[-1]
    p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    loss = -np.mean(np.sum(Y * np.log(p + 1e-12) + (1 - Y) * np.log(1 - p + 1e-12), axis=1))
    if not need_grad:
        return loss, None
    return loss, net.grads(hs, p - Y)


class MLPPosterior:
    """Gaussian amortized posterior from a deep ensemble of MLPs."""

    def __init__(self, hidden=(64, 64), ensemble: int = 3, epochs: int = 400, lr: float = 3e-3, batch: int = 64,
                 weight_decay: float = 1e-6, patience: int = 40, val_fraction: float = 0.15, seed: int = 0,
                 backend: str = "numpy"):
        self.hidden, self.ensemble, self.epochs, self.lr, self.batch = hidden, ensemble, epochs, lr, batch
        self.wd, self.patience, self.val_fraction, self.seed, self.backend = weight_decay, patience, val_fraction, \
            seed, backend

    def fit(self, X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        self.xmu, self.xsd, self.xmask = _prep_fit(X)
        Z = _prep(X, self.xmu, self.xsd, self.xmask)
        self.ymu, self.ysd = Y.mean(axis=0), np.where(Y.std(axis=0) > 0, Y.std(axis=0), 1.0)
        T = (Y - self.ymu) / self.ysd
        rng = np.random.default_rng(self.seed)
        n = len(Z)
        nva = max(1, int(self.val_fraction * n))
        self.members, self.val_loss = [], []
        for e in range(self.ensemble):
            perm = rng.permutation(n)
            va, tr = perm[:nva], perm[nva:]
            sizes = [Z.shape[1], *self.hidden, 2 * T.shape[1]]
            if self.backend == "torch":
                net, lv = _torch_train(sizes, Z[tr], T[tr], Z[va], T[va], self, "gauss", seed=self.seed + e)
            else:
                net = _NumpyMLP(sizes, np.random.default_rng(self.seed + 1000 * e))
                lv = _adam_train(net, _gauss_loss, Z[tr], T[tr], Z[va], T[va], epochs=self.epochs, lr=self.lr,
                                 batch=self.batch, wd=self.wd, patience=self.patience, rng=rng)
            self.members.append(net)
            self.val_loss.append(float(lv))
        return self

    def _member_out(self, net, Z):
        if self.backend == "torch":
            return _torch_forward(net, Z)
        return net.forward(Z)[-1]

    def predict(self, X):
        Z = _prep(np.atleast_2d(X), self.xmu, self.xsd, self.xmask)
        P = len(self.ymu)
        mus, vs = [], []
        for net in self.members:
            out = self._member_out(net, Z)
            mus.append(out[:, :P])
            vs.append(np.exp(np.clip(out[:, P:], -12, 8)))
        mus, vs = np.array(mus), np.array(vs)
        mean = mus.mean(axis=0)
        var = (vs.mean(axis=0) + mus.var(axis=0)) * self.ysd**2
        cov = np.einsum("ki,ij->kij", var, np.eye(var.shape[1]))     # diagonal covariances
        return mean * self.ysd + self.ymu, cov


class MLPClassifier:
    """Multi-label classifier (sigmoid outputs, BCE): p(term k in the structure | features)."""

    def __init__(self, hidden=(64, 64), epochs: int = 300, lr: float = 3e-3, batch: int = 64, weight_decay=1e-5,
                 patience: int = 40, seed: int = 0):
        self.hidden, self.epochs, self.lr, self.batch, self.wd, self.patience, self.seed = \
            hidden, epochs, lr, batch, weight_decay, patience, seed

    def fit(self, X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        self.xmu, self.xsd, self.xmask = _prep_fit(X)
        Z = _prep(X, self.xmu, self.xsd, self.xmask)
        rng = np.random.default_rng(self.seed)
        n = len(Z)
        perm = rng.permutation(n)
        nva = max(1, n // 6)
        self.net = _NumpyMLP([Z.shape[1], *self.hidden, Y.shape[1]], rng)
        self.val_loss = _adam_train(self.net, _bce_loss, Z[perm[nva:]], Y[perm[nva:]], Z[perm[:nva]], Y[perm[:nva]],
                                    epochs=self.epochs, lr=self.lr, batch=self.batch, wd=self.wd,
                                    patience=self.patience, rng=rng)
        return self

    def predict_proba(self, X):
        Z = _prep(np.atleast_2d(X), self.xmu, self.xsd, self.xmask)
        return 1 / (1 + np.exp(-np.clip(self.net.forward(Z)[-1], -30, 30)))


# ---- optional torch backend -------------------------------------------------------------

def _torch_train(sizes, Xtr, Ytr, Xva, Yva, cfg, mode, seed=0):
    import torch
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    layers = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(torch.nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(torch.nn.Tanh())
    net = torch.nn.Sequential(*layers).to(dev).double()
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    Xt, Yt = torch.tensor(Xtr, device=dev), torch.tensor(Ytr, device=dev)
    Xv, Yv = torch.tensor(Xva, device=dev), torch.tensor(Yva, device=dev)
    P = Ytr.shape[1]

    def loss_fn(out, y):
        mu, s = out[:, :P], out[:, P:].clamp(-12, 8)
        return 0.5 * torch.mean(torch.sum(s + (y - mu) ** 2 * torch.exp(-s), dim=1))
    best, best_state, bad = float("inf"), None, 0
    for ep in range(cfg.epochs):
        perm = torch.randperm(len(Xt), device=dev)
        for s in range(0, len(Xt), cfg.batch):
            idx = perm[s:s + cfg.batch]
            opt.zero_grad()
            loss_fn(net(Xt[idx]), Yt[idx]).backward()
            opt.step()
        with torch.no_grad():
            lv = float(loss_fn(net(Xv), Yv))
        if lv < best - 1e-6:
            best, best_state, bad = lv, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            bad += 1
            if bad > cfg.patience:
                break
    net.load_state_dict(best_state)
    return net, best


def _torch_forward(net, Z):
    import torch
    dev = next(net.parameters()).device
    with torch.no_grad():
        return net(torch.tensor(Z, device=dev, dtype=torch.float64)).cpu().numpy()
