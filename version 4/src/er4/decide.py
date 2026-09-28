"""Decision layer (ported from V2, Part 6): calibration, exclusivity with ambiguity margin, singleton model,
expected-F0.5 list selection / global threshold. V4: fit() tunes on mirror TRAIN S1s using the scores of ALL
competing S1s; apply() runs the frozen rule on HOLDOUT or test."""
import numpy as np
import polars as pl
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

from .config import CFG
from .models import SING_FEATS, fit as xfit, params, predict, singleton_features
from .util import log, macro_f05


def iso_fit(p, y):
    return IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)


def iso_crossfit(p, y, folds):
    out = np.zeros_like(p, dtype=np.float32)
    for k in range(CFG["folds"]):
        tr, va = folds != k, folds == k
        out[va] = iso_fit(p[tr], y[tr]).predict(p[va])
    return out


def exclusivity(sc, eps=0.0, lam=1.0):
    """Each S2/S3 record keeps only its best S1; if the runner-up is within eps, the kept score is multiplied by lam."""
    top2 = sc.group_by("i23").agg(p1=pl.col("p").max(), p2=pl.col("p").top_k(2).min(), n=pl.len())
    s = sc.join(top2, on="i23").filter(pl.col("p") == pl.col("p1")).unique(["i23"], keep="first")
    amb = (pl.col("n") > 1) & ((pl.col("p1") - pl.col("p2")) < eps)
    return s.with_columns(p=pl.when(amb).then(pl.col("p") * lam).otherwise(pl.col("p"))).select("i1", "i23", "p")


def decide_global(sc_ex, thr, max_k=None):
    return (sc_ex.filter(pl.col("p") >= thr)
                 .with_columns(r=pl.col("p").rank("ordinal", descending=True).over("i1"))
                 .filter(pl.col("r") <= (max_k or CFG["max_list"])).select("i1", "i23"))


def expected_f_select(sc_excl, p_single, m_miss, floor, max_k):
    """Per S1: choose k in 0..max_k maximizing E[F0.5] of predicting its top-k (Poisson-binomial DP, vectorized)."""
    s = (sc_excl.filter(pl.col("p") >= floor).sort(["i1", "p"], descending=[False, True])
                .with_columns(r=pl.int_range(pl.len()).over("i1")))
    if s.height == 0:
        return pl.DataFrame(schema={"i1": pl.UInt32, "i23": pl.UInt32})
    rest = sc_excl.group_by("i1").agg(ptot=pl.col("p").sum())
    ids = s["i1"].unique().sort()
    n, K = len(ids), max_k
    row = ids.to_frame().with_row_index("row")
    s = s.join(row, on="i1").filter(pl.col("r") < K)
    Q = np.zeros((n, K), np.float64)
    Q[s["row"].to_numpy(), s["r"].to_numpy()] = s["p"].to_numpy()
    tot = row.join(rest, on="i1", how="left")["ptot"].fill_null(0).to_numpy()
    ps = row.join(p_single, on="i1", how="left")["p_single"].fill_null(0.5).to_numpy()
    dist = np.zeros((n, K + 1)); dist[:, 0] = 1.0
    best_ef, best_k = ps.copy(), np.zeros(n, dtype=np.int64)
    cum = np.zeros(n)
    t = np.arange(K + 1)
    for k in range(1, K + 1):
        q = Q[:, k - 1]
        new = dist * (1 - q)[:, None]
        new[:, 1:] += dist[:, :-1] * q[:, None]
        dist = new
        cum += q
        eu = np.maximum(tot - cum, 0) + m_miss
        denom = 5 * t[None, :] + eu[:, None] + 4 * (k - t[None, :])
        ef = (dist * np.where(t[None, :] > 0, 5 * t[None, :] / np.maximum(denom, 1e-9), 0)).sum(1)
        ef = np.where(q > 0, ef, -1)
        better = ef > best_ef
        best_ef, best_k = np.where(better, ef, best_ef), np.where(better, k, best_k)
    chosen = row.with_columns(k=pl.Series(best_k))
    return s.join(chosen.select("row", "k"), on="row").filter(pl.col("r") < pl.col("k")).select("i1", "i23")


def fit_decision(tab, y_edge, is_train, folds_edge, truth_T, i1_T, fold_of_i1, s1info, candinfo, m_miss, tag=""):
    """Tune the decision layer on mirror TRAIN S1s.
    tab: (i1, i23, p) for ALL mirror edges we have (TRAIN rows = OOF scores); y_edge / folds_edge align with the TRAIN
    rows (is_train mask). truth_T: TRAIN true pairs; i1_T: TRAIN S1 ids; fold_of_i1: (i1, fold) for TRAIN S1s."""
    st, ledger = {"m_miss": m_miss}, []
    p = tab["p"].to_numpy()
    pT, yT, fT = p[is_train], y_edge, folds_edge
    st["iso"] = iso_fit(pT, yT)
    p_cal = st["iso"].predict(p).astype(np.float32)
    p_cal[is_train] = iso_crossfit(pT, yT, fT)                  # TRAIN rows: cross-fitted (honest)
    sc = tab.select("i1", "i23").with_columns(p=pl.Series(p_cal))
    only_T = lambda d: d.join(pl.DataFrame({"i1": i1_T}), on="i1", how="semi")

    res = []
    for eps in CFG["eps_grid"]:
        for lam in CFG["lam_grid"]:
            if eps == 0 and lam != 1.0:
                continue
            ex = only_T(exclusivity(sc, eps, lam))
            for thr in CFG["thr_grid"]:
                res.append((eps, lam, thr, macro_f05(decide_global(ex, thr), truth_T, i1_T)))
    res = pl.DataFrame(res, schema=["eps", "lam", "thr", "f05"], orient="row").sort("f05", descending=True)
    st["eps"], st["lam"], st["thr"], f_glob = res.row(0)
    ledger.append((f"calibrated + exclusivity(eps={st['eps']}, lam={st['lam']}) + global thr {st['thr']}", f_glob))
    sc_ex = exclusivity(sc, st["eps"], st["lam"])

    # singleton model (entity level, OOF over TRAIN S1s)
    f = singleton_features(only_T(sc_ex), s1info.join(pl.DataFrame({"i1": i1_T}), on="i1", how="semi"), candinfo)
    f = f.join(fold_of_i1, on="i1")
    y = f.join(truth_T.select("i1").unique().with_columns(h=pl.lit(1)), on="i1", how="left")["h"].fill_null(0).to_numpy()
    X, fo = f.select(SING_FEATS).to_numpy().astype(np.float32), f["fold"].to_numpy()
    prm = params(eta=0.05, max_leaves=31)
    raw = np.zeros(len(y), np.float32)
    for k in range(CFG["folds"]):
        tr, va = fo != k, fo == k
        b = fit_(X[tr], y[tr], prm, X[va], y[va])
        raw[va] = predict(b, X[va])
    st["sing"] = fit_(X, y, prm)
    st["sing_iso"] = iso_fit(raw, y)
    p_has = iso_crossfit(raw, y, fo)
    log(f"[{tag}] singleton model OOF AUC {roc_auc_score(y, p_has):.4f}")
    p_single = f.select("i1").with_columns(p_single=pl.Series(1 - p_has))
    st["p_single_T"] = p_single                                  # OOF P(no match) per TRAIN S1 (V4.3 learned policy input)

    res_e = []
    ex_T = only_T(sc_ex)
    for floor in CFG["floor_grid"]:
        for mk in CFG["maxk_grid"]:
            res_e.append((floor, mk, macro_f05(expected_f_select(ex_T, p_single, m_miss, floor, mk), truth_T, i1_T)))
    res_e = pl.DataFrame(res_e, schema=["floor", "max_k", "f05"], orient="row").sort("f05", descending=True)
    st["floor"], st["max_k"], f_ef = res_e.row(0)
    ledger.append((f"+ singleton + expected-F0.5 (floor={st['floor']}, max_k={st['max_k']})", f_ef))
    st["rule"] = "expected_f" if f_ef > f_glob else "global"
    st["f_train"] = max(f_ef, f_glob)
    st["ledger"] = ledger
    log(f"[{tag}] decision rule: {st['rule']} -> TRAIN OOF macro F0.5 {st['f_train']:.4f}")
    return st


def fit_(X, y, prm, Xva=None, yva=None):
    return xfit(X, y, prm, 2000 if Xva is not None else 300, Xva, yva, 50 if Xva is not None else None, log_every=False)


def apply(st, tab, s1info, candinfo, i1_scope=None):
    """Frozen rule -> predicted (i1, i23). tab: (i1, i23, raw p) for all edges (competitors included).
    i1_scope: restrict the output to these S1s (e.g. HOLDOUT)."""
    sc = tab.select("i1", "i23").with_columns(p=pl.Series(st["iso"].predict(tab["p"].to_numpy()).astype(np.float32)))
    sc_ex = exclusivity(sc, st["eps"], st["lam"])
    if i1_scope is not None:
        sc_ex = sc_ex.join(pl.DataFrame({"i1": i1_scope}).cast({"i1": sc_ex.schema["i1"]}), on="i1", how="semi")
    if st["rule"] == "global":
        return decide_global(sc_ex, st["thr"])
    s1 = s1info if i1_scope is None else s1info.join(pl.DataFrame({"i1": i1_scope}).cast({"i1": s1info.schema["i1"]}), on="i1", how="semi")
    f = singleton_features(sc_ex, s1, candinfo)
    raw = predict(st["sing"], f.select(SING_FEATS).to_numpy().astype(np.float32))
    p_single = f.select("i1").with_columns(p_single=pl.Series(1 - st["sing_iso"].predict(raw)))
    return expected_f_select(sc_ex, p_single, st["m_miss"], st["floor"], st["max_k"])
