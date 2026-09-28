"""L7b pool (V2 passes union tri-modal dense pass) + pool features + L8 distilled student pruner with adaptive M."""
import gc, math, time

import numpy as np
import polars as pl

from .common import CFG, log, cached
from . import v2_base as B

DENSE_FEATS = ["dcos", "drank", "drev_rank", "in_v2"]
STUDENT_FEATS = B.FEATS + DENSE_FEATS


def build_pool(v2cand, dense, k=None):
    """V2 top-50 (in V2 row order, so cached V2 feature rows stay aligned) + dense-only pairs appended."""
    k = k or CFG["dense_topk"]
    dd = dense.filter((pl.col("drank") <= k) | pl.col("drev_rank").is_not_null())
    a = (v2cand.with_row_index("_r").join(dd, on=["i1", "i23"], how="left").sort("_r").drop("_r")
               .with_columns(in_v2=pl.lit(1, pl.UInt8)))
    b = dd.join(v2cand.select("i1", "i23"), on=["i1", "i23"], how="anti").with_columns(
        in_v2=pl.lit(0, pl.UInt8), n_passes=pl.lit(0, pl.UInt8), p6=pl.lit(0, pl.UInt8))
    pool = pl.concat([a, b], how="diagonal_relaxed")
    # list-level provenance recomputed on the union: list size; blocking-scorer columns stay null for dense-only pairs
    return pool.with_columns(n_cand=pl.len().over("i1").cast(pl.UInt16))


def dense_matrix(pool):
    return pool.select(dcos=pl.col("dcos").fill_null(-1.0), drank=pl.col("drank").fill_null(CFG["dense_topk"] + 1),
                       drev_rank=pl.col("drev_rank").fill_null(9), in_v2="in_v2").to_numpy().astype(np.float32)


def pool_features(pool, d, X_v2=None, chunk=3_000_000):
    """Feature matrix (STUDENT_FEATS order) for the whole pool.
    If X_v2 is given (train sample S) its rows are reused for the V2 part of the pool (same features, same code)."""
    n_v2 = int(pool["in_v2"].sum()) if X_v2 is not None else 0
    X = np.empty((pool.height, len(STUDENT_FEATS)), dtype=np.float32)
    if X_v2 is not None:
        assert X_v2.shape[0] == n_v2
        X[:n_v2, :len(B.FEATS)] = X_v2
    rest = pool.slice(n_v2)
    for lo in range(0, rest.height, chunk):
        part = rest.slice(lo, chunk)
        X[n_v2 + lo:n_v2 + lo + part.height, :len(B.FEATS)] = B.build_features(part, d["s1p"], d["s23p"],
                                                                               desc=f"features {d['name']} {lo // chunk}")
    # n_cand is recomputed on the union (the V2 value was the capped list size)
    X[:, B.FEATS.index("n_cand")] = pool["n_cand"].to_numpy().astype(np.float32)
    X[:, len(B.FEATS):] = dense_matrix(pool)
    return X


# --------------------------------------------------------------------------------------------- L8 student
def xgb_oof(X, y, folds, feats, params=None, rounds=None, desc="xgb", weight=None, base_margin=None, monotone=None):
    """GPU XGBoost OOF on the frozen split. Returns (oof, models, best_iters)."""
    import xgboost as xgb
    params = dict(params or CFG["xgb"])
    if monotone:
        params["monotone_constraints"] = "(" + ",".join(str(monotone.get(f, 0)) for f in feats) + ")"
    oof = np.zeros(len(y), np.float32)
    models, iters = [], []
    K = int(folds.max()) + 1
    for k in range(K):
        tr, va = np.where(folds != k)[0], np.where(folds == k)[0]
        dtr = xgb.QuantileDMatrix(X[tr], y[tr], weight=None if weight is None else weight[tr], feature_names=feats,
                                  max_bin=params.get("max_bin", 256))
        dva = xgb.QuantileDMatrix(X[va], y[va], ref=dtr, feature_names=feats)
        t = time.time()
        m = xgb.train(params, dtr, rounds or CFG["xgb_rounds"], evals=[(dva, "valid")],
                      early_stopping_rounds=CFG["xgb_early"], verbose_eval=False)
        oof[va] = m.predict(dva, iteration_range=(0, m.best_iteration + 1))
        models.append(m); iters.append(m.best_iteration)
        log(f"{desc} fold {k}: best_iter {m.best_iteration} | valid logloss {m.best_score:.5f} | {time.time() - t:.0f}s")
        del dtr, dva; gc.collect()
    return oof, models, iters


def xgb_fit_full(X, y, feats, rounds, params=None, monotone=None):
    import xgboost as xgb
    params = dict(params or CFG["xgb"])
    if monotone:
        params["monotone_constraints"] = "(" + ",".join(str(monotone.get(f, 0)) for f in feats) + ")"
    d = xgb.QuantileDMatrix(X, y, feature_names=feats, max_bin=params.get("max_bin", 256))
    return xgb.train(params, d, rounds)


def xgb_predict(model, X, feats, chunk=4_000_000):
    import xgboost as xgb
    out = np.empty(X.shape[0], np.float32)
    for lo in range(0, X.shape[0], chunk):
        out[lo:lo + chunk] = model.predict(xgb.DMatrix(X[lo:lo + chunk], feature_names=feats))
    return out


def adaptive_m(i1, p, base=None, lo=None, hi=None):
    """Per-S1 list size M (design 5.3): grows for uncertain lists (entropy / small gap at rank M), shrinks for decisive.
    Returns a boolean keep mask over pairs."""
    base, lo, hi = base or CFG["prune_base"], lo or CFG["prune_min"], hi or CFG["prune_max"]
    d = pl.DataFrame({"i1": i1, "p": p, "row": np.arange(len(p))})
    d = d.with_columns(r=pl.col("p").rank("ordinal", descending=True).over("i1"))
    ps = pl.col("p").clip(1e-6, 1 - 1e-6)
    ent = (-(ps * ps.log() + (1 - ps) * (1 - ps).log())).sum().over("i1")
    mass = (pl.col("p") >= 0.01).sum().over("i1")
    d = d.with_columns(ent=ent, mass=mass)
    m = (pl.when(pl.col("ent") < 0.05).then(lo)
           .when(pl.col("ent") > 1.0).then(hi)
           .otherwise(base)).cast(pl.Int32)
    m = pl.max_horizontal(m, pl.col("mass") + 2).clip(lo, hi)
    keep = d.with_columns(M=m).with_columns(k=(pl.col("r") <= pl.col("M")) & ((pl.col("p") >= CFG["prune_floor"]) | (pl.col("r") <= lo)))
    return keep.sort("row")["k"].to_numpy()
