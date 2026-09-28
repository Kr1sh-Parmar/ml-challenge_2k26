"""L12 calibration v3, L13 component decomposition + exact solve, L14 learned cost-sensitive decision policy.

L12: monotone GBDT calibrator on (score [monotone +], ensemble uncertainty, list-size / cluster descriptors), OOF under
     repeated CV; plain isotonic is the fallback. Calibration is fitted on real OOF pairs only.
L13: graph S1 -- S2/S3 with calibrated edges >= floor, split into connected components. A component with one S1 is
     solved exactly by the expected-F0.5 DP; components whose shared records admit <= exact_limit assignments are
     solved exactly by enumeration (each shared record -> one of its S1s or none, then each S1 takes its best
     expected-F prefix of what it may use); larger ones use greedy best response.
L14: per-S1 cost vectors c[k] = 1 - F0.5(top-k) (k = 0..K) learned by a GBDT from entity features that include the
     expected-F0.5 of each k; k* = argmin predicted cost, adopted only if it beats pure expected-F.
"""
import itertools, math, time

import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .common import CFG, log, f05_of_counts


# --------------------------------------------------------------------------------------------- L12
def fit_calibrator(score, aux, y, rounds=300):
    import xgboost as xgb
    X = np.column_stack([score, aux]).astype(np.float32)
    params = dict(tree_method="hist", device="cuda", objective="binary:logistic", eval_metric="logloss",
                  learning_rate=0.05, max_depth=4, min_child_weight=200, subsample=0.8, reg_lambda=5.0,
                  monotone_constraints="(" + ",".join(["1"] + ["0"] * aux.shape[1]) + ")")
    return xgb.train(params, xgb.DMatrix(X, y), rounds)


def apply_calibrator(m, score, aux):
    import xgboost as xgb
    return m.predict(xgb.DMatrix(np.column_stack([score, aux]).astype(np.float32))).astype(np.float32)


def calibrate_oof(score, aux, y, folds_pairs):
    out = np.zeros(len(score), np.float32)
    for k in range(int(folds_pairs.max()) + 1):
        tr, va = folds_pairs != k, folds_pairs == k
        m = fit_calibrator(score[tr], aux[tr], y[tr])
        out[va] = apply_calibrator(m, score[va], aux[va])
    return out


def ece(p, y, bins=15):
    e = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, e) - 1, 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        s = idx == b
        if s.any():
            tot += s.sum() * abs(p[s].mean() - y[s].mean())
    return tot / len(p)


# --------------------------------------------------------------------------------------------- expected F
def ef_prefix(q_sorted, rest, m_miss):
    """Expected F0.5 of selecting the top-k (k = 0..K) of sorted probabilities (one list; Poisson-binomial DP).
    rest = sum of probabilities of the list's items that are NOT selectable (assigned elsewhere / beyond K)."""
    K = len(q_sorted)
    tot = float(np.sum(q_sorted)) + rest
    ef = np.empty(K + 1)
    p_none = math.exp(sum(math.log1p(-min(float(q), 1 - 1e-9)) for q in q_sorted)) if K else 1.0
    # k = 0: F = 1 if the entity truly has no match (no selected, none elsewhere, no blocking miss)
    ef[0] = p_none * math.exp(-rest - m_miss)
    dist = np.zeros(K + 1); dist[0] = 1.0
    t = np.arange(K + 1)
    cum = 0.0
    for k in range(1, K + 1):
        q = q_sorted[k - 1]
        new = dist * (1 - q)
        new[1:] += dist[:-1] * q
        dist = new
        cum += q
        eu = max(tot - cum, 0.0) + m_miss
        denom = 5 * t + eu + 4 * (k - t)
        ef[k] = float((dist * np.where(t > 0, 5 * t / np.maximum(denom, 1e-9), 0.0)).sum())
    return ef


def ef_table(i1_list, p_lists, rest, m_miss, K):
    """Vectorised expected-F for many lists: p_lists [n, K] sorted desc (0-padded). -> EF [n, K+1]."""
    n = p_lists.shape[0]
    tot = p_lists.sum(1) + rest
    EF = np.zeros((n, K + 1))
    lp = np.log1p(-np.clip(p_lists, 0, 1 - 1e-9)).sum(1)
    EF[:, 0] = np.exp(lp - rest - m_miss)
    dist = np.zeros((n, K + 1)); dist[:, 0] = 1.0
    t = np.arange(K + 1)
    cum = np.zeros(n)
    for k in range(1, K + 1):
        q = p_lists[:, k - 1]
        new = dist * (1 - q)[:, None]
        new[:, 1:] += dist[:, :-1] * q[:, None]
        dist = new
        cum += q
        eu = np.maximum(tot - cum, 0) + m_miss
        denom = 5 * t[None, :] + eu[:, None] + 4 * (k - t[None, :])
        ef = (dist * np.where(t[None, :] > 0, 5 * t[None, :] / np.maximum(denom, 1e-9), 0)).sum(1)
        EF[:, k] = np.where(q > 0, ef, -1.0)
    return EF


# --------------------------------------------------------------------------------------------- L13
def component_sizes(i1, i23, p, floor=None):
    """Number of S1 entities in each pair's connected component (edges p >= floor) — no solving."""
    floor = floor or CFG["edge_floor"]
    live = np.asarray(p) >= floor
    u1, a1 = np.unique(i1, return_inverse=True)
    u2, a2 = np.unique(i23, return_inverse=True)
    n1, n2 = len(u1), len(u2)
    g = coo_matrix((np.ones(live.sum()), (a1[live], n1 + a2[live])), shape=(n1 + n2, n1 + n2))
    _, comp = connected_components(g, directed=False)
    return np.bincount(comp[:n1], minlength=comp.max() + 1)[comp[a1]]


def solve_components(i1, i23, p, m_miss, K=None, floor=None, limit=None):
    """-> allowed mask over pairs (exclusivity resolved) and component size per pair.
    Each S2/S3 record may be used by at most one S1; small components solved exactly."""
    K, floor, limit = K or CFG["policy_K"] + 2, floor or CFG["edge_floor"], limit or CFG["exact_limit"]
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    n = len(p)
    live = p >= floor
    allowed = live.copy()
    u1, a1 = np.unique(i1, return_inverse=True)
    u2, a2 = np.unique(i23, return_inverse=True)
    n1, n2 = len(u1), len(u2)
    g = coo_matrix((np.ones(live.sum()), (a1[live], n1 + a2[live])), shape=(n1 + n2, n1 + n2))
    _, comp = connected_components(g, directed=False)
    c_of_pair = comp[a1]
    deg = np.bincount(a2[live], minlength=n2)
    shared_pair = live & (deg[a2] >= 2)
    csize = np.bincount(comp[:n1], minlength=comp.max() + 1)[c_of_pair]
    # only components containing a shared record need joint solving
    comps = np.unique(c_of_pair[shared_pair])
    if len(comps) == 0:
        return allowed, csize
    idx_by_comp = {}
    sel = np.where(live & np.isin(c_of_pair, comps))[0]
    order = np.argsort(c_of_pair[sel], kind="stable")
    sel = sel[order]
    bounds = np.r_[0, np.where(np.diff(c_of_pair[sel]) != 0)[0] + 1, len(sel)]
    n_exact = n_greedy = 0
    t0 = time.time()
    for ci, (b0, b1) in enumerate(zip(bounds[:-1], bounds[1:])):
        if ci % 20000 == 0 and ci:
            log(f"L13: {ci:,}/{len(bounds) - 1:,} components solved ({time.time() - t0:.0f}s)")
        rows = sel[b0:b1]
        res = _solve_one(rows, a1[rows], a2[rows], p[rows], m_miss, K, limit)
        n_exact += res[1]; n_greedy += 1 - res[1]
        allowed[rows] = res[0]
    log(f"L13: {len(comps):,} multi-S1 components ({n_exact:,} exact, {n_greedy:,} best-response)")
    return allowed, csize


def _entity_value(rows_p, avail, m_miss, K):
    """best expected F of one S1 given which of its rows are available."""
    pa = np.sort(rows_p[avail])[::-1]
    return ef_prefix(pa[:K], rows_p[~avail].sum() + pa[K:].sum(), m_miss).max()


def _solve_one(rows, e, r, p, m_miss, K, limit):
    ents = np.unique(e)
    recs, rinv = np.unique(r, return_inverse=True)
    owners = [np.where(rinv == j)[0] for j in range(len(recs))]           # row positions per record
    shared = [j for j in range(len(recs)) if len(owners[j]) >= 2]
    ent_rows = {s: np.where(e == s)[0] for s in ents}
    n_conf = 1
    for j in shared:
        n_conf *= len(owners[j]) + 1
        if n_conf > limit:
            break

    def value(assign):                          # assign: dict record j -> row position allowed (or -1 none)
        avail = np.ones(len(rows), bool)
        for j, keep in assign.items():
            for q in owners[j]:
                avail[q] = q == keep
        return sum(_entity_value(p[ent_rows[s]], avail[ent_rows[s]], m_miss, K) for s in ents), avail

    if n_conf <= limit:
        best_v, best_a = -1.0, None
        for choice in itertools.product(*[list(owners[j]) + [-1] for j in shared]):
            v, avail = value(dict(zip(shared, choice)))
            if v > best_v:
                best_v, best_a = v, avail
        return best_a, 1
    # greedy best response: start from exclusivity (each record to its highest-p S1), then improve one record at a time
    assign = {j: owners[j][np.argmax(p[owners[j]])] for j in shared}
    cur, avail = value(assign)
    if len(shared) > CFG["br_max_shared"]:          # very large component: plain exclusivity (V2 rule)
        return avail, 0
    for _ in range(2):
        improved = False
        for j in shared:
            for opt in list(owners[j]) + [-1]:
                if opt == assign[j]:
                    continue
                trial = dict(assign); trial[j] = opt
                v, av = value(trial)
                if v > cur + 1e-12:
                    cur, assign, avail, improved = v, trial, av, True
        if not improved:
            break
    return avail, 0


# --------------------------------------------------------------------------------------------- lists after L13
def build_lists(i1, p, allowed, n1, K):
    """Sorted allowed probabilities per S1 (all n1 entities, empty lists included) -> (P [n1, K], rows [n1, K], rest)."""
    rows = np.where(allowed)[0]
    o = rows[np.lexsort((-p[rows], i1[rows]))]
    e = i1[o]
    starts = np.r_[0, np.where(np.diff(e) != 0)[0] + 1] if len(e) else np.array([], int)
    lens = np.diff(np.r_[starts, len(o)]) if len(e) else np.array([], int)
    pos = np.arange(len(o)) - np.repeat(starts, lens)
    P = np.zeros((n1, K)); R = np.full((n1, K), -1, np.int64)
    k = pos < K
    P[e[k], pos[k]] = p[o[k]]; R[e[k], pos[k]] = o[k]
    rest = np.bincount(i1, weights=np.where(allowed, 0, p), minlength=n1) + \
           np.bincount(e[~k], weights=p[o[~k]], minlength=n1)
    return P, R, rest


def select_topk(R, k):
    """rows selected when each S1 takes its top-k[i] allowed items."""
    K = R.shape[1]
    m = np.arange(K)[None, :] < k[:, None]
    r = R[m]
    return r[r >= 0]


def true_cost_table(R, y, ntrue, K):
    """c[i, k] = 1 - F0.5 of predicting top-k (truth incl. blocking misses)."""
    n = R.shape[0]
    yy = np.where(R >= 0, y[np.maximum(R, 0)], 0).astype(np.float64)
    tp = np.concatenate([np.zeros((n, 1)), np.cumsum(yy, 1)], 1)
    npred = np.arange(K + 1)[None, :].repeat(n, 0).astype(np.float64)
    valid = np.concatenate([np.ones((n, 1), bool), R >= 0], 1)
    f = f05_of_counts(tp, npred, ntrue[:, None])
    return 1 - f, valid


# --------------------------------------------------------------------------------------------- L14
POLICY_FEATS = ["k", "ef_k", "ef_gap", "ef_best_k", "p_k", "p_next", "cum_p", "rest_p", "n_items", "p_null",
                "unc_mean", "unc_top", "cl_size_top", "comp_size", "p1", "p2", "gap12", "is_best_ef"]


def policy_matrix(P, EF, rest, p_null, unc, cl_top, comp, K):
    n = P.shape[0]
    rows = []
    best = EF.argmax(1)
    n_items = (P > 0).sum(1)
    cum = np.concatenate([np.zeros((n, 1)), np.cumsum(P, 1)], 1)
    for k in range(K + 1):
        p_k = P[:, k - 1] if k > 0 else np.ones(n)
        p_next = P[:, k] if k < P.shape[1] else np.zeros(n)
        rows.append(np.column_stack([np.full(n, k), EF[:, k], EF.max(1) - EF[:, k], best, p_k, p_next, cum[:, k],
                                     rest, n_items, p_null, unc[:, 0], unc[:, 1], cl_top, comp, P[:, 0], P[:, 1],
                                     P[:, 0] - P[:, 1], (best == k).astype(float)]))
    return np.stack(rows, 1).astype(np.float32)          # [n, K+1, F]


def fit_policy(Xp, C, valid, rounds=400):
    import xgboost as xgb
    n, K1, Fdim = Xp.shape
    m = valid.ravel()
    params = dict(tree_method="hist", device="cuda", objective="reg:squarederror", learning_rate=0.05, max_depth=7,
                  min_child_weight=50, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0)
    d = xgb.DMatrix(Xp.reshape(-1, Fdim)[m], C.ravel()[m], feature_names=POLICY_FEATS)
    return xgb.train(params, d, rounds)


def policy_choose(model, Xp, valid):
    import xgboost as xgb
    n, K1, Fdim = Xp.shape
    pred = model.predict(xgb.DMatrix(Xp.reshape(-1, Fdim), feature_names=POLICY_FEATS)).reshape(n, K1)
    pred = np.where(valid, pred, np.inf)
    return pred.argmin(1)
