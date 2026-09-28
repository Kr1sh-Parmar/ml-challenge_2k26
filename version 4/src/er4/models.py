"""XGBoost models on the GPU (L1 pair model, L2 context re-ranker, singleton model) with automatic CPU fallback."""
import functools, gc, warnings

import numpy as np
import polars as pl
import xgboost as xgb
from tqdm.auto import tqdm

from .config import CFG
from .util import log

warnings.filterwarnings("ignore", message=".*Falling back to prediction using DMatrix.*")


@functools.cache
def device():
    """'cuda' if an XGBoost CUDA build and a GPU are usable, else 'cpu'."""
    try:
        X = np.random.rand(256, 4).astype(np.float32)
        xgb.train(dict(tree_method="hist", device="cuda", verbosity=0), xgb.QuantileDMatrix(X, X[:, 0] > 0.5), 2)
        return "cuda"
    except Exception as e:                         # no GPU / no CUDA build
        log(f"XGBoost CUDA unavailable ({str(e)[:80]}) -> CPU")
        return "cpu"


def params(monotone=None, **over):
    p = dict(CFG["xgb"], device=device(), **over)
    if monotone is not None:
        p["monotone_constraints"] = tuple(monotone)
    return p


def fit(X, y, prm, rounds, Xva=None, yva=None, early_stop=None, log_every=200):
    qtr = xgb.QuantileDMatrix(X, y, max_bin=prm["max_bin"])
    evals = [(xgb.QuantileDMatrix(Xva, yva, ref=qtr, max_bin=prm["max_bin"]), "valid")] if Xva is not None else []
    b = xgb.train(prm, qtr, rounds, evals=evals, early_stopping_rounds=early_stop if evals else None,
                  verbose_eval=log_every if evals else False)
    del qtr, evals; gc.collect()
    return b


def predict(b, X, chunk=2_000_000):
    it = (0, b.best_iteration + 1) if b.attr("best_iteration") is not None else (0, 0)
    out = np.empty(len(X), dtype=np.float32)
    for lo in range(0, len(X), chunk):
        out[lo:lo + chunk] = b.inplace_predict(X[lo:lo + chunk], iteration_range=it)
    return out


def train_oof(X, y, folds, prm, desc="L1"):
    """K-fold OOF grouped by S1 (folds are per-edge fold ids of the edge's S1) + final model on everything."""
    oof, iters = np.zeros(len(y), dtype=np.float32), []
    for k in tqdm(range(CFG["folds"]), desc=f"{desc} CV"):
        tr, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        b = fit(X[tr], y[tr], prm, CFG["rounds"], X[va], y[va], CFG["early_stop"])
        oof[va] = predict(b, X[va])
        iters.append(b.best_iteration + 1)
        log(f"{desc} fold {k}: best_iter {iters[-1]} | valid logloss {float(b.attr('best_score')):.5f}")
        del b; gc.collect()
    n_final = max(50, int(np.mean(iters) * 1.1))
    final = fit(X, y, prm, n_final)
    log(f"{desc} final model: {n_final} rounds on {len(y):,} rows")
    return oof, iters, final


# ------------------------------------------------------------------ L2 context features
L2_FEATS = ["p", "r_i1", "gap_i1", "n05_i1", "n09_i1", "psum_i1", "comp", "p_minus_comp", "r_i23", "n05_i23",
            "blk", "blk_rank", "blk_rev_rank", "rev_n", "blk_gap", "n_cand"]


def context_features(tab):
    """tab: edges (i1, i23, p, blk, blk_rank, blk_rev_rank, rev_n, blk_gap, n_cand) -> tab + L2_FEATS.
    List context within the S1, and competition for the S2/S3 record (best OTHER S1's score)."""
    t = tab.with_columns(
        r_i1=pl.col("p").rank("ordinal", descending=True).over("i1").cast(pl.Float32),
        gap_i1=(pl.col("p").max().over("i1") - pl.col("p")),
        n05_i1=(pl.col("p") >= 0.5).sum().over("i1").cast(pl.Float32),
        n09_i1=(pl.col("p") >= 0.9).sum().over("i1").cast(pl.Float32),
        psum_i1=pl.col("p").sum().over("i1"),
        m1=pl.col("p").max().over("i23"),
        m2=pl.col("p").top_k(2).min().over("i23"),
        n_i23=pl.len().over("i23"),
        r_i23=pl.col("p").rank("ordinal", descending=True).over("i23").cast(pl.Float32),
        n05_i23=(pl.col("p") >= 0.5).sum().over("i23").cast(pl.Float32))
    comp = pl.when(pl.col("n_i23") == 1).then(0.0).when(pl.col("r_i23") == 1).then(pl.col("m2")).otherwise(pl.col("m1"))
    return (t.with_columns(comp=comp).with_columns(p_minus_comp=pl.col("p") - pl.col("comp"))
             .drop("m1", "m2", "n_i23").with_columns(pl.col(L2_FEATS).cast(pl.Float32)))


# ------------------------------------------------------------------ singleton model
SING_FEATS = ["t1", "t2", "t3", "psum", "n05", "n08", "ncand", "blk_max", "npass_max", "f_core_s1_r", "f_core_s23_r",
              "f_addr_s1_r", "idf_sum", "a_len", "has_house"]


def singleton_features(sc_excl, s1info, candinfo):
    """sc_excl: post-exclusivity (i1, i23, p); s1info: per-S1 fields; candinfo: per-S1 blocking summary."""
    lst = sc_excl.sort("p", descending=True).group_by("i1").agg(
        top=pl.col("p").head(3), psum=pl.col("p").sum(), n05=(pl.col("p") >= 0.5).sum(), n08=(pl.col("p") >= 0.8).sum())
    f = (s1info.join(lst, on="i1", how="left").join(candinfo, on="i1", how="left")
               .with_columns(t1=pl.col("top").list.get(0, null_on_oob=True), t2=pl.col("top").list.get(1, null_on_oob=True),
                             t3=pl.col("top").list.get(2, null_on_oob=True)).drop("top").fill_null(0))
    return f.sort("i1")
