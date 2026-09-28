"""Logging, checkpoints, resource guards, the metric."""
import pickle, shutil, time

import numpy as np
import polars as pl

from .config import CACHE, CFG


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def disk_free_gb(path=CACHE):
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


def check_disk(stage=""):
    free = disk_free_gb()
    if free < CFG["disk_stop_gb"]:
        raise RuntimeError(f"only {free:.1f} GB free on the cache disk ({stage}); free space before continuing")
    if free < CFG["disk_warn_gb"]:
        log(f"WARNING: {free:.1f} GB free on the cache disk ({stage})")


def cached(name, fn, refresh=False):
    """Checkpoint: `*.parquet` holds a polars DataFrame (zstd); anything else is pickled (small objects only)."""
    p = CACHE / name
    if p.exists() and not refresh:
        log(f"cache hit: {name}")
        return pl.read_parquet(p) if name.endswith(".parquet") else pickle.load(open(p, "rb"))
    check_disk(name)
    t = time.time()
    obj = fn()
    tmp = p.with_suffix(p.suffix + ".tmp")                 # write-then-rename: a crash never leaves a truncated cache
    if name.endswith(".parquet"):
        obj.write_parquet(tmp, compression="zstd")
    else:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=5)
    tmp.replace(p)
    log(f"computed + cached {name} in {time.time() - t:.0f}s")
    return obj


def macro_f05(pred, truth, i1_all):
    """pred / truth: DataFrames (i1, i23). i1_all: every S1 index being scored (singletons included).
    Per S1: F0.5 = 5TP / (5TP + FN + 4FP); an S1 with no truth and no prediction scores 1.0."""
    tp = pred.join(truth, on=["i1", "i23"], how="semi").group_by("i1").len("tp")
    t = (pl.DataFrame({"i1": np.asarray(i1_all, dtype=np.uint32)})
         .join(tp, on="i1", how="left")
         .join(pred.group_by("i1").len("npred"), on="i1", how="left")
         .join(truth.group_by("i1").len("ntrue"), on="i1", how="left").fill_null(0)
         .with_columns(fp=pl.col("npred") - pl.col("tp"), fn=pl.col("ntrue") - pl.col("tp")))
    f = (pl.when((pl.col("npred") == 0) & (pl.col("ntrue") == 0)).then(1.0)
           .otherwise(5 * pl.col("tp") / (5 * pl.col("tp") + pl.col("fn") + 4 * pl.col("fp"))))
    return t.select(f.fill_nan(0.0)).to_series().mean()


def P(pairs):
    return pl.DataFrame(pairs, schema={"i1": pl.UInt32, "i23": pl.UInt32}, orient="row")


# README example: predict [47, 193, 812], truth [47, 812] -> 0.714; singleton rules
assert abs(macro_f05(P([(0, 47), (0, 193), (0, 812)]), P([(0, 47), (0, 812)]), [0]) - 0.7142857) < 1e-6
assert macro_f05(P([]), P([]), [5]) == 1.0 and macro_f05(P([(5, 1)]), P([]), [5]) == 0.0
assert macro_f05(P([]), P([(1, 2)]), [1]) == 0.0
