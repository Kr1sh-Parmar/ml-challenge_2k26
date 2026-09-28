"""L11 J2: set transformer v2 over each S1's candidate list with a NULL slot (GPU).

Tokens: one per candidate (every level-1 score, F-UNC, F-CLUS, F-EM, key pair features, list context), plus a NULL
token (learned embedding + list-level statistics). Soft duplicate links enter as a learned per-head attention bias
(the cluster structure of design 6.3). Loss = per-candidate BCE + NULL BCE + soft macro-F0.5 surrogate.
Trained with repeated CV on OOF level-1 inputs (valid because those are already OOF); predictions averaged.
"""
import math, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import CFG, log


class Standardizer:
    def fit(self, X):
        Z = np.where(np.isfinite(X), X, np.nan)
        self.lo = np.nanquantile(Z, 0.001, axis=0)
        self.hi = np.nanquantile(Z, 0.999, axis=0)
        Zc = np.clip(Z, self.lo, self.hi)
        self.mu = np.nanmean(Zc, axis=0)
        self.sd = np.nanstd(Zc, axis=0) + 1e-6
        self.mu = np.nan_to_num(self.mu); self.sd = np.nan_to_num(self.sd, nan=1.0)
        return self

    def transform(self, X):
        Z = (np.clip(X, self.lo, self.hi) - self.mu) / self.sd
        miss = ~np.isfinite(Z)
        return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), miss


def pack_lists(i1, order_score, T, links=None, y=None, maxlen=None):
    """Pairs -> padded list tensors. Candidates in each list are ordered by order_score (desc), truncated at maxlen.
    Returns dict with X [n, L, F], mask [n, L], pair_row [n, L] (-1 pad), list_i1 [n], y [n, L] and link [n, L, L]."""
    L = maxlen or CFG["st_maxlen"]
    o = np.lexsort((-order_score, i1))
    i1s = i1[o]
    starts = np.r_[0, np.where(np.diff(i1s) != 0)[0] + 1]
    n = len(starts)
    lens = np.diff(np.r_[starts, len(o)])
    pos = np.arange(len(o)) - np.repeat(starts, lens)
    keep = pos < L
    li = np.repeat(np.arange(n), lens)[keep]
    pi = pos[keep]
    rows = o[keep]
    Xp = np.zeros((n, L, T.shape[1]), np.float32)
    Xp[li, pi] = T[rows]
    mask = np.zeros((n, L), bool); mask[li, pi] = True
    prow = np.full((n, L), -1, np.int64); prow[li, pi] = rows
    out = dict(X=Xp, mask=mask, prow=prow, list_i1=i1s[starts])
    if y is not None:
        yp = np.zeros((n, L), np.float32); yp[li, pi] = y[rows]
        out["y"] = yp
    link = np.zeros((n, L, L), np.float16)
    if links is not None and len(links[0]):
        lx, ly, lp = links                          # pair rows of the two records in the same list, p_link
        where = np.full(len(i1), -1, np.int64); where[rows] = pi
        lst = np.full(len(i1), -1, np.int64); lst[rows] = li
        a, b = where[lx], where[ly]
        ok = (a >= 0) & (b >= 0) & (lst[lx] == lst[ly])
        link[lst[lx][ok], a[ok], b[ok]] = lp[ok]
        link[lst[lx][ok], b[ok], a[ok]] = lp[ok]
    out["link"] = link
    return out


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dk = h, d // h
        self.qkv = nn.Linear(d, 3 * d)
        self.o = nn.Linear(d, d)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.link_w = nn.Parameter(torch.zeros(h))

    def forward(self, x, keymask, link):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, self.dk).unbind(2)
        att = torch.einsum("blhd,bmhd->bhlm", q, k) / math.sqrt(self.dk)
        att = att + self.link_w.view(1, -1, 1, 1) * link.unsqueeze(1)
        att = att.masked_fill(~keymask.view(B, 1, 1, L), -1e4)
        a = att.softmax(-1)
        x = x + self.o(torch.einsum("bhlm,bmhd->blhd", a, v).reshape(B, L, D))
        return x + self.ff(self.ln2(x))


class SetTransformer(nn.Module):
    def __init__(self, n_in, n_list, d=None, h=None, layers=None):
        super().__init__()
        d, h, layers = d or CFG["st_dim"], h or CFG["st_heads"], layers or CFG["st_layers"]
        self.inp = nn.Sequential(nn.Linear(2 * n_in, d), nn.GELU(), nn.Linear(d, d))
        self.null = nn.Parameter(torch.randn(d) * 0.02)
        self.null_in = nn.Linear(n_list, d)
        self.pos = nn.Embedding(64, d)
        self.blocks = nn.ModuleList([Block(d, h) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.null_head = nn.Linear(d, 1)

    def forward(self, X, miss, mask, link, lstat):
        B, L, _ = X.shape
        t = self.inp(torch.cat([X, miss], -1)) + self.pos(torch.arange(L, device=X.device)).unsqueeze(0)
        nt = (self.null + self.null_in(lstat)).unsqueeze(1)
        x = torch.cat([nt, t], 1)
        km = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=X.device), mask], 1)
        lk = F.pad(link, (1, 0, 1, 0))
        for blk in self.blocks:
            x = blk(x, km, lk)
        x = self.ln(x)
        return self.head(x[:, 1:]).squeeze(-1), self.null_head(x[:, 0]).squeeze(-1)


def list_stats(Xs, mask, score_col):
    s = np.where(mask, Xs[:, :, score_col], -5.0)
    srt = -np.sort(-s, axis=1)
    return np.column_stack([mask.sum(1) / 20.0, srt[:, 0], srt[:, 1], srt[:, 2], (s > 0).sum(1) / 5.0]).astype(np.float32)


def soft_f05(p, y, mask, ntrue):
    p = p * mask
    tp = (p * y).sum(1)
    fp = (p * (1 - y)).sum(1)
    fn = ntrue - tp
    f = 5 * tp / (5 * tp + fn + 4 * fp + 1e-6)
    f_single = torch.exp(torch.log1p(-p.clamp(max=1 - 1e-6)).sum(1))
    return torch.where(ntrue > 0, f, f_single)


def _to(t, dev="cuda"):
    return torch.from_numpy(t).to(dev, non_blocking=True)


def train_st(pack, ntrue, score_col, tr, epochs=None, bs=None, lr=None, seed=0, lam=0.5):
    torch.manual_seed(seed)
    epochs, bs, lr = epochs or CFG["st_epochs"], bs or CFG["st_bs"], lr or CFG["st_lr"]
    Xs, miss = pack["Xs"], pack["miss"]
    lstat = list_stats(Xs, pack["mask"], score_col)
    model = SetTransformer(Xs.shape[2], lstat.shape[1]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    steps = epochs * math.ceil(len(tr) / bs)
    warm = max(1, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, steps)))))
    rng = np.random.default_rng(seed)
    model.train()
    for ep in range(epochs):
        perm = tr[rng.permutation(len(tr))]
        for lo in range(0, len(perm), bs):
            b = np.sort(perm[lo:lo + bs])
            X, M, m, lk = _to(Xs[b]), _to(miss[b].astype(np.float32)), _to(pack["mask"][b]), _to(pack["link"][b].astype(np.float32))
            y, nt, ls = _to(pack["y"][b]), _to(ntrue[b].astype(np.float32)), _to(lstat[b])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg, ln = model(X, M, m, lk, ls)
            lg, ln = lg.float(), ln.float()
            bce = (F.binary_cross_entropy_with_logits(lg, y, reduction="none") * m).sum() / m.sum()
            ynull = (y * m).sum(1) == 0
            nb = F.binary_cross_entropy_with_logits(ln, ynull.float())
            sf = soft_f05(torch.sigmoid(lg), y, m.float(), nt).mean()
            loss = bce + 0.5 * nb + lam * (1 - sf)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
    model.eval()
    return model


@torch.no_grad()
def predict_st(model, pack, score_col, idx, bs=4096):
    Xs, miss = pack["Xs"], pack["miss"]
    lstat = list_stats(Xs, pack["mask"], score_col)
    P = np.zeros((len(idx), Xs.shape[1]), np.float32)
    N = np.zeros(len(idx), np.float32)
    for lo in range(0, len(idx), bs):
        b = idx[lo:lo + bs]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg, ln = model(_to(Xs[b]), _to(miss[b].astype(np.float32)), _to(pack["mask"][b]),
                           _to(pack["link"][b].astype(np.float32)), _to(lstat[b]))
        P[lo:lo + len(b)] = torch.sigmoid(lg.float()).cpu().numpy()
        N[lo:lo + len(b)] = torch.sigmoid(ln.float()).cpu().numpy()
    return P, N


def st_repeated_cv(pack, ntrue, score_col, list_folds, seeds=(0,)):
    """list_folds: list of fold arrays over lists (repeated CV). Returns OOF (P [n,L], Pnull [n]) averaged over repeats."""
    n = pack["X"].shape[0]
    P = np.zeros(pack["mask"].shape, np.float32)
    N = np.zeros(n, np.float32)
    cnt = 0
    for r, folds in enumerate(list_folds):
        for k in range(int(folds.max()) + 1):
            tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
            t = time.time()
            for s in seeds:
                m = train_st(pack, ntrue, score_col, tr, seed=1000 * r + 10 * k + s)
                p, nn_ = predict_st(m, pack, score_col, va)
                P[va] += p / len(seeds); N[va] += nn_ / len(seeds)
            log(f"J2 set transformer repeat {r} fold {k}: {len(tr):,} train lists | {time.time() - t:.0f}s")
        cnt += 1
    return P / cnt, N / cnt
