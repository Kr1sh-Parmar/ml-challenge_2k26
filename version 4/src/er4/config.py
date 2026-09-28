"""All knobs in one place. Paths resolve from this file, so the package works from any working directory."""
import os
from pathlib import Path

V4 = Path(__file__).resolve().parents[2]                     # .../version 4
ROOT = next(p for p in [V4, *V4.parents] if (p / "student_resource").exists())
DATA = ROOT / "student_resource" / "dataset"
SMOKE = os.environ.get("ER4_SMOKE") == "1"                 # tiny end-to-end run to catch bugs (~2.5% of S1)
# ER4_STRICT=1: optional strictly-train-only rerun (test records used only for inference and corpus statistics, never to
# fit a model, a table or an encoder) in its own cache_strict/. The default V4.1+ path keeps V4's unsupervised use of
# unlabeled test records (French self-mined tables, self-supervised encoder adaptation, per-country EM).
STRICT = os.environ.get("ER4_STRICT") == "1"
V5 = os.environ.get("ER4_V5") == "1"                       # Version 5: own cache_v5/, output_v5/, models_v5/
DENSE = STRICT or V5 or os.environ.get("ER4_DENSE") == "1"  # V4.1+: dense retrieval pass
_sfx = "_smoke" if SMOKE else ""
BASE_CACHE = V4 / (f"cache{_sfx}_dense" if V5 else ("cache_smoke" if SMOKE else "cache"))
if STRICT:
    CACHE, OUT = V4 / f"cache{_sfx}_strict", V4 / f"output{_sfx}_strict"
elif V5:
    CACHE, OUT = V4 / f"cache{_sfx}_v5", V4 / f"output{_sfx}_v5"
elif SMOKE:
    CACHE, OUT = (V4 / "cache_smoke_dense", V4 / "output_smoke_dense") if DENSE else (V4 / "cache_smoke", V4 / "output_smoke")
else:
    CACHE, OUT = (V4 / "cache_dense", V4 / "output_dense") if DENSE else (V4 / "cache", V4 / "output")
# stage outputs shared with the base cache (hard-linked). STRICT re-derives everything that touched test records;
# V5 links V4.1's stage outputs (+ the frozen V4.3 round-1 stacker for the like-for-like baseline) and re-derives the rest.
if STRICT:
    SHARED = ("roles.parquet", "truth_mirror.parquet")
elif V5:
    SHARED = ("roles.parquet", "truth_mirror.parquet", "tables.pkl", "blk_scorer.pkl", "parsed_*.parquet", "dense_*.parquet",
              "cand_*.parquet", "p_mirror_*.parquet", "p_test_*.parquet", "l1.json", "l1_oof.parquet", "l2.json",
              "l2_oof.parquet", "train_X_*.parquet", "l43a.json", "l43a_oof.parquet")
else:
    SHARED = ("roles.parquet", "truth_mirror.parquet", "tables.pkl", "blk_scorer.pkl", "parsed_*.parquet", "dense_*.parquet")
MODELS = V4 / (("models_smoke" if SMOKE else "models") + ("_v5" if V5 else ""))   # smoke/full and V5 never share weights

CFG = dict(
    seed=0,
    n_jobs=16,
    # ---- test mirror (u ~ U[0,1) per train S1, same draw as V2's make_samples)
    roles=dict(mine=(0.10, 0.20), train=(0.20, 0.32), holdout=(0.32, 0.42), drop=(0.79, 1.0)),
    folds=5,
    closure_rev_k=3,          # competitor edges kept per S2/S3 record for mirror exclusivity (by blocking rank)
    # ---- blocking (V2 values; n_cand re-chosen on the full-scale curve)
    top_k=dict(p1=30, p2=20, p3=20, p5=15, p7=20),
    cap_frac=dict(p1=2e-4, p2=2e-4, p3=5e-4, p5=2e-4, p7=2e-4, rev=1e-3),
    rev_k=5, p6_max=5, dup_group_max=6,
    n_cand_grid=(50, 60, 80), n_cand=50,
    chunk_rows=2e10,          # query chunk = chunk_rows / pool size (bounds join memory)
    # ---- mining
    mine_support=5, mine_lift=10.0, sim_lift=2.0,
    # ---- models (XGBoost; GPU when available)
    xgb=dict(objective="binary:logistic", eval_metric="logloss", tree_method="hist", eta=0.08, grow_policy="lossguide",
             max_leaves=63, max_depth=0, min_child_weight=1.0, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
             max_bin=128, seed=0),
    rounds=3000, early_stop=50,
    score_chunk=2_000_000,
    # ---- decision layer grids
    eps_grid=(0.0, 0.05, 0.1, 0.2), lam_grid=(1.0, 0.9, 0.75), thr_grid=tuple(round(0.3 + 0.05 * i, 2) for i in range(14)),
    floor_grid=(0.1, 0.2, 0.3), maxk_grid=(6, 8, 10), max_list=10,
    # ---- resources
    disk_warn_gb=25, disk_stop_gb=8,
)
