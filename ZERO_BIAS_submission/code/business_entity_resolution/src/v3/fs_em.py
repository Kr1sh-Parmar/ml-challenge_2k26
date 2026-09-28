"""L5 Fellegi-Sunter model: comparison vectors, supervised m/u initialisation, per-country EM on unlabeled pairs (GPU).

Outputs (design 4.3):
  - F-EM features: total log-likelihood-ratio match weight + one weight per field
  - a standalone level-1 score: posterior P(match | comparison vector) under the EM-adapted model
EM runs per country label on that country's own unlabeled pairs (France test included), starting from the supervised
m/u tables; iterations are bounded and m/u are floored, and the result is used as features, never as hard rules.
"""
import numpy as np
import torch

from . import v2_base as B

# field -> (source column, level edges) ; level 0 is always "missing"
def _bands(x, edges, missing):
    lv = np.digitize(x, edges) + 1
    return np.where(missing, 0, lv).astype(np.int64)


FIELDS = ["name_jw", "name_tset", "house", "postal", "unit", "street", "region", "loc", "legal", "nums", "addr_tset", "dense"]


def comparison_vectors(X, feats):
    col = lambda f: X[:, feats.index(f)]
    tri = lambda f: np.where(col(f) < 0, 0, np.where(col(f) > 0.5, 2, 1)).astype(np.int64)
    G = np.stack([
        _bands(col("n_jw_core"), [0.75, 0.85, 0.92, 0.97, 0.999], np.zeros(len(X), bool)),
        _bands(col("n_tset"), [60, 75, 88, 97, 100], np.zeros(len(X), bool)),
        tri("a_house"), tri("a_postal"), tri("a_unit"),
        _bands(col("a_street_jacc"), [1e-6, 0.5, 0.999], col("a_street_jacc") < 0),
        tri("a_region"), tri("a_loc"), tri("n_legal"),
        _bands(col("a_num_jacc"), [1e-6, 0.5, 0.999], col("a_num_jacc") < 0),
        _bands(col("a_tset"), [50, 70, 85, 95], col("a_tset") < 0),
        _bands(col("dcos"), [0.80, 0.85, 0.90, 0.95, 0.98], col("dcos") < -0.5),
    ], axis=1)
    return G


N_LEVELS = [7, 7, 3, 3, 3, 5, 3, 3, 3, 5, 6, 7]


class FellegiSunter:
    def __init__(self, eps=1e-4):
        self.eps = eps
        self.m = self.u = None
        self.pi = None

    def fit_supervised(self, G, y):
        y = y.astype(bool)
        self.m = [self._dist(G[y, j], L) for j, L in enumerate(N_LEVELS)]
        self.u = [self._dist(G[~y, j], L) for j, L in enumerate(N_LEVELS)]
        self.pi = float(y.mean())
        return self

    def _dist(self, g, L):
        c = np.bincount(g, minlength=L).astype(np.float64) + 1.0
        return np.clip(c / c.sum(), self.eps, 1 - self.eps)

    @torch.no_grad()
    def em(self, G, iters=25, tol=1e-6):
        """Unsupervised re-estimation of m, u, pi on unlabeled pairs G (one country) — GPU."""
        Gt = torch.from_numpy(G).cuda()
        m = [torch.tensor(x, device="cuda") for x in self.m]
        u = [torch.tensor(x, device="cuda") for x in self.u]
        pi = torch.tensor(self.pi, device="cuda", dtype=torch.float64)
        prev = None
        for it in range(iters):
            lm = sum(torch.log(m[j][Gt[:, j]]) for j in range(G.shape[1]))
            lu = sum(torch.log(u[j][Gt[:, j]]) for j in range(G.shape[1]))
            a, b = lm + torch.log(pi), lu + torch.log1p(-pi)
            ll = torch.logaddexp(a, b)
            w = torch.exp(a - ll)                                   # E-step: posterior of match
            pi = w.mean().clamp(1e-4, 0.9)
            for j, L in enumerate(N_LEVELS):
                cm = torch.bincount(Gt[:, j], weights=w, minlength=L) + 1.0
                cu = torch.bincount(Gt[:, j], weights=1 - w, minlength=L) + 1.0
                m[j] = (cm / cm.sum()).clamp(self.eps, 1 - self.eps)
                u[j] = (cu / cu.sum()).clamp(self.eps, 1 - self.eps)
            cur = ll.mean().item()
            if prev is not None and abs(cur - prev) < tol:
                break
            prev = cur
        self.m = [x.cpu().numpy() for x in m]
        self.u = [x.cpu().numpy() for x in u]
        self.pi = float(pi.item())
        return self

    def weights(self, G):
        """-> (per-field log2 LR weights [n, F], total weight [n], posterior [n])"""
        W = np.stack([np.log2(self.m[j][G[:, j]] / self.u[j][G[:, j]]) for j in range(G.shape[1])], axis=1)
        tot = W.sum(1)
        logit = tot * np.log(2) + np.log(self.pi / (1 - self.pi))
        post = 1 / (1 + np.exp(-np.clip(logit, -40, 40)))
        return W.astype(np.float32), tot.astype(np.float32), post.astype(np.float32)


EM_FEATS = [f"em_{f}" for f in FIELDS] + ["em_total", "em_post"]


def em_features_oof(G, y, cty, folds):
    """Train side: supervised init on the other folds, EM on this fold's pairs per country -> F-EM matrix (OOF)."""
    out = np.zeros((len(G), len(EM_FEATS)), np.float32)
    for k in range(int(folds.max()) + 1):
        tr, va = folds != k, folds == k
        for c in np.unique(cty[va]):
            sel = va & (cty == c)
            fs = FellegiSunter().fit_supervised(G[tr], y[tr])
            fs.em(G[sel])
            W, tot, post = fs.weights(G[sel])
            out[sel] = np.column_stack([W, tot, post])
    return out


def em_features_test(G, cty, G_train, y_train):
    """Test side: supervised init on all of train, EM per test country (France adapts with no labels)."""
    out = np.zeros((len(G), len(EM_FEATS)), np.float32)
    tables = {}
    for c in np.unique(cty):
        sel = cty == c
        fs = FellegiSunter().fit_supervised(G_train, y_train).em(G[sel])
        W, tot, post = fs.weights(G[sel])
        out[sel] = np.column_stack([W, tot, post])
        tables[c] = (fs.m, fs.u, fs.pi)
    return out, tables
