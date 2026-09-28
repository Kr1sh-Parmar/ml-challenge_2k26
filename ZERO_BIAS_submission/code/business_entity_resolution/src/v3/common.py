"""V3 common layer: L0 config (with compute-tier switch), paths, stage cache, metric, folds, V2 state loading."""
import os, sys, gc, time, pickle, hashlib, json
from pathlib import Path

import numpy as np
import polars as pl

from . import v2_base as B

# --------------------------------------------------------------------------------------------- paths
from erpaths import DATA, VALIDATOR, WORK                  # noqa: E402  (src/ is on sys.path)
ROOT = WORK
V2CACHE = WORK / "v2" / "cache"                            # written by `python v2/prepare.py`
V3 = WORK / "v3"
CACHE = V3 / "cache"
OUT = V3 / "output"
MODELS = V3 / "models"
for d in (CACHE, OUT, MODELS):
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------------------------- L0 config
TIER = os.environ.get("V3_TIER", "M")                       # S / M / L  (design section 11)
CFG = dict(
    tier=TIER,
    seed=0,
    folds=5,                   # frozen split, heavy level-1 models (same split as V2)
    rcv=(3, 5),                # repeated CV (repeats, folds) for the cheap downstream layers L11-L14
    device="cuda",
    # L6 dense retrieval
    dense_model="intfloat/multilingual-e5-small",   # MIT; bge-m3 / Qwen3-Embedding selectable for tier L
    dense_dim_max_len=64, dense_topk=20, dense_ft_steps=3000, dense_ft_bs=384,
    # L8 pruning
    prune_base=10, prune_min=5, prune_max=20, prune_floor=0.002,
    # L7a clusters
    clus_strict=0.95, clus_soft=0.8, clus_cap=8, clus_top=8,
    # L10 cross-encoder (tier M+)
    ce_model="microsoft/mdeberta-v3-base", ce_folds=2, ce_max_len=128, ce_bs=64, ce_lr=3e-5,
    ce_epochs=1, ce_band=(0.003, 0.997), ce_train_pairs=900_000, ce_test_models=1,
    # L11 set transformer
    st_dim=128, st_heads=4, st_layers=3, st_epochs=12, st_bs=512, st_lr=2e-3, st_maxlen=20,
    # L13 / L14
    edge_floor=0.02, exact_limit=256, br_max_shared=40, policy_K=8,
    xgb=dict(tree_method="hist", device="cuda", objective="binary:logistic", eval_metric="logloss",
             learning_rate=0.06, max_depth=9, min_child_weight=20, subsample=0.8, colsample_bytree=0.7,
             reg_lambda=2.0, max_bin=256, seed=0),
    xgb_rounds=4000, xgb_early=80,
)
B.CFG["n_jobs"] = 16


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cached(name, fn, refresh=False):
    """Stage checkpoint in ER_WORK/v3/cache: *.parquet -> polars frame, *.npy -> array, anything else pickled."""
    p = CACHE / name
    if p.exists() and not refresh and name not in os.environ.get("V3_REFRESH", "").split(","):
        log(f"cache hit: {name}")
        if name.endswith(".parquet"):
            return pl.read_parquet(p)
        if name.endswith(".npy"):
            return np.load(p)
        with open(p, "rb") as f:
            return pickle.load(f)
    t = time.time()
    obj = fn()
    tmp = p.with_suffix(p.suffix + ".tmp")
    if name.endswith(".parquet"):
        obj.write_parquet(tmp)
    elif name.endswith(".npy"):
        with open(tmp, "wb") as f:
            np.save(f, obj)
    else:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=5)
    os.replace(tmp, p)
    log(f"computed + cached {name} in {time.time() - t:.0f}s")
    return obj


def v2_cached(name):
    p = V2CACHE / name
    log(f"V2 cache: {name}")
    if name.endswith(".parquet"):
        return pl.read_parquet(p)
    with open(p, "rb") as f:
        return pickle.load(f)


def file_hash(path, n=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------------------------- V2 runtime state
def load_v2_state():
    """Inject the V2 runtime tables (legal forms, mined substitutions) into the V2 library."""
    legal_found = v2_cached("legal_found.pkl")
    B.LEGAL = B.LEGAL_SEED | {t for v in legal_found.values() for t in v}
    train_tables, _ = v2_cached("tables_train.pkl")
    fr_tables, _ = v2_cached("tables_france.pkl")

    def subs_for(country):
        g = train_tables["*"]
        own = train_tables.get(country) or fr_tables.get(country) or ({}, {}, {})
        return tuple({**g[i], **own[i]} for i in range(3))

    subs = {c: subs_for(c) for c in ["us", "india", "france"]}
    subs["*"] = train_tables["*"]
    B.SUBS = subs
    return subs


# --------------------------------------------------------------------------------------------- metric
def f05_per_entity(i1, pred, y, ntrue, n1):
    """Per-S1 F0.5 over n1 entities.
    i1: pair -> S1 index; pred, y: 0/1 per pair; ntrue[n1]: true matches per S1 (incl. blocking misses)."""
    tp = np.bincount(i1, weights=(pred & y).astype(np.float64), minlength=n1)
    npred = np.bincount(i1, weights=pred.astype(np.float64), minlength=n1)
    fn = ntrue - tp
    fp = npred - tp
    den = 5 * tp + fn + 4 * fp
    return np.where(den == 0, 1.0, 5 * tp / np.maximum(den, 1e-12))


def macro_f05(i1, pred, y, ntrue, ents=None):
    f = f05_per_entity(i1, pred.astype(bool), y.astype(bool), ntrue, len(ntrue))
    return float(f.mean() if ents is None else f[ents].mean())


def f05_of_counts(tp, npred, ntrue):
    den = 5 * tp + (ntrue - tp) + 4 * (npred - tp)
    return np.where(den == 0, 1.0, 5 * tp / np.maximum(den, 1e-12))


def bootstrap_delta(fa, fb, n=1000, seed=0):
    """Paired bootstrap of mean(fa - fb) over entities -> (delta, lo95, hi95)."""
    d = np.asarray(fa) - np.asarray(fb)
    rng = np.random.default_rng(seed)
    m = len(d)
    bs = np.array([d[rng.integers(0, m, m)].mean() for _ in range(n)])
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


# --------------------------------------------------------------------------------------------- folds
def frozen_folds(n1, k=5, seed=0):
    """The V2 frozen split per S1 entity (identical draw), used by every heavy level-1 model."""
    return np.random.default_rng(seed).integers(0, k, n1).astype(np.int8)


def repeated_folds(n1, repeats=3, k=5, seed=100):
    """Fresh splits for the cheap layers L11-L14 (valid because their inputs are already OOF)."""
    return [np.random.default_rng(seed + r).integers(0, k, n1).astype(np.int8) for r in range(repeats)]


def gpu_info():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("V3 requires a CUDA GPU (torch.cuda.is_available() is False)")
    p = torch.cuda.get_device_properties(0)
    return f"{p.name} | {p.total_memory / 2**30:.1f} GB | torch {torch.__version__}"
