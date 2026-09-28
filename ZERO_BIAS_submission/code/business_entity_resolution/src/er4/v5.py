"""Version 5 = V4.3 + 2.5x labelled band data (context S1s) + CE-v5 + honest France proxy, gated on HOLDOUT / LOCO.

Run from version 4/src with ER4_V5=1 (own cache_v5/, output_v5/, models_v5/; V4.1 outputs hard-linked):
    python -m er4.v5 --stage ctx|loco|p5|bandx|ce5|stack5|decide|export|all

Leakage rules (every labelled row is out-of-fold / out-of-sample at every layer; HOLDOUT is never used to fit anything):
  * labelled = mirror TRAIN S1s + a context sample u in [0.42, 0.60) (never u < 0.10: V3's CE, the ce43 warm start, saw it)
  * labels come from truth_mirror, never from TRAIN-only OOF files (context rows would otherwise vanish silently)
  * folds: pipeline.fold_of(entity_id) for every labelled S1; CE halves h = fold % 2, the model predicting half h is
    warm-started from ce43_fold{h}, which was trained only on half 1-h
  * L1/L2 scores: TRAIN rows OOF (V4.1), context rows out-of-sample (V4.1 L1/L2 never saw them)
Stages:
  ctx    score the context sample's edges + their competitor closure with V4 features and V4.1 L1 (p_l1)
  loco   honest France proxy: US-only L1 + L2 + decision -> India TRAIN S1s; ranking vs calibration loss; tests a
         label-free prior-matching correction (adopted for France only if India F0.5 improves, bootstrap CI > 0)
  p5     union p_mirror + p_ctx at the p_l1 level (TRAIN rows first), add_l2 once -> p5_{split}_{c}.parquet
  bandx  band features for every band edge (0.003 <= p_l2 <= 0.997): pairx / ce43 reused where edge-local and computed
         otherwise, per-country EM (fold-wise init over all labelled rows), mates (strongest mate by p_l2)
  ce5    mDeBERTa cross-encoder re-trained on labelled band pairs (0.01 <= p_l2 <= 0.99), OOF by half
  stack5 GPU XGBoost stacker (2 seeds), OOF over labelled band rows; learning-curve check (TRAIN-only vs + context)
  decide V4 decision layer + learned policy fitted on labelled S1s; HOLDOUT report vs V4.1 and the frozen V4.3 stacker
  export test predictions (+ France prior-matched variant) -> output_v5/, official validator --check-ids
"""
import argparse, gc, json, pickle, subprocess, sys, time

import numpy as np
import polars as pl

from . import decide, models
from . import stack3 as S3
from . import final as F
from .config import BASE_CACHE, CACHE, CFG, DATA, MODELS, SMOKE, V4, V5, VALIDATOR
from .features import V4_FEATS, V4_IDX, build_features, REC, _feat_batch, PAIR_FEATS
from .util import log, macro_f05

assert V5, "run with ER4_V5=1"
OUT5 = V4 / ("output_smoke_v5" if SMOKE else "output_v5")
V4_MODELS = V4 / ("models_smoke" if SMOKE else "models")          # V4.3's ce43 folds (warm starts)
V3_DIR = F.V3_DIR
CTX_U = (0.42, 0.60)
CE5_BAND = (0.01, 0.99)
BLK = ["blk", "blk_rank", "blk_rev_rank", "rev_n", "blk_gap", "n_cand", "n_passes"]
CE5F = ["p_ce5", "ce5_present", "ce5_minus_p"]
F5 = models.L2_FEATS + ["p_l1"] + F.CEF + CE5F + S3.MATE_FEATS + F.PX + F.EM_F
F43A = F.F43A


def fpath(kind, split, c):
    return CACHE / f"{kind}_{split}_{c}.parquet"


def dense_path(kind, split, c):
    return BASE_CACHE / f"{kind}_{split}_{c}.parquet"


def countries(split):
    from .pipeline import countries as cs
    return cs(split)


# ------------------------------------------------------------------ roles, labels, folds
def ctx_idx(c):
    """per-country idx of the context sample: context S1s with u in CTX_U (u = V4's role draw, same order as roles)."""
    from .pipeline import load_parsed
    r = pl.read_parquet(CACHE / "roles.parquet")
    u = np.random.default_rng(CFG["seed"]).random(r.height)
    r = r.with_columns(u=pl.Series(u)).filter((pl.col("role") == "context") & (pl.col("u") >= CTX_U[0]) & (pl.col("u") < CTX_U[1]))
    a = load_parsed("mirror", c, "s1", ["idx", "entity_id"])
    return a.join(r.select("entity_id"), on="entity_id", how="semi")["idx"]


def labelled_global():
    """(i1, cty, role in {train, ctx}, fold) for every labelled mirror S1 (global ids)."""
    from .pipeline import MIRROR_C, load_parsed, offsets, role_idx, fold_of
    o1, _ = offsets("mirror")
    out = []
    for c in MIRROR_C:
        a = load_parsed("mirror", c, "s1", ["idx", "entity_id"]).with_columns(fold=fold_of(pl.col("entity_id")))
        for role, ids in [("train", role_idx(c, "train")), ("ctx", ctx_idx(c))]:
            d = a.join(pl.DataFrame({"idx": ids}).cast({"idx": a.schema["idx"]}), on="idx", how="semi")
            out.append(d.select(i1=(pl.col("idx") + o1[c]).cast(pl.UInt32), cty=pl.lit(c), role=pl.lit(role), fold="fold"))
    lab = pl.concat(out)
    assert lab["i1"].is_unique().all()
    return lab


def truth_edges():
    from .pipeline import mirror_truth_global
    return mirror_truth_global().select("i1", "i23").with_columns(y=pl.lit(1, pl.Int8))


# ------------------------------------------------------------------ stage: ctx (score the context sample + closure)
def stage_ctx():
    import xgboost as xgb
    from .pipeline import MIRROR_C, cand_path, load_parsed, offsets, p_path
    l1 = xgb.Booster(model_file=str(CACHE / "l1.json")); l1.set_param({"device": models.device()})
    o1, o23 = offsets("mirror")
    for c in MIRROR_C:
        if fpath("p_ctx", "mirror", c).exists():
            log(f"cache hit: p_ctx {c}"); continue
        t = time.time()
        C = pl.DataFrame({"i1": ctx_idx(c)})
        scan = pl.scan_parquet(cand_path("mirror", c))
        C = C.cast({"i1": scan.collect_schema()["i1"]})
        ce = scan.join(C.lazy(), on="i1", how="semi").collect()
        recs = ce.select("i23").unique()
        comp = (scan.join(recs.lazy(), on="i23", how="semi")
                    .filter(pl.col("blk_rev_rank") <= CFG["closure_rev_k"]).collect())
        edges = pl.concat([ce, comp]).unique(["i1", "i23"])
        have = pl.read_parquet(p_path("mirror", c), columns=["i1", "i23"]).select(
            i1=(pl.col("i1") - o1[c]).cast(edges.schema["i1"]), i23=(pl.col("i23") - o23[c]).cast(edges.schema["i23"]))
        new = edges.join(have, on=["i1", "i23"], how="anti")
        log(f"ctx {c}: {C.height:,} context S1 | {ce.height:,} own edges + closure -> {edges.height:,} | {new.height:,} not yet scored")
        a, b = load_parsed("mirror", c, "s1"), load_parsed("mirror", c, "s23")
        p = np.empty(new.height, np.float32)
        ch = CFG["score_chunk"]
        for k, lo in enumerate(range(0, new.height, ch)):
            X = build_features(new.slice(lo, ch), a, b, desc=f"ctx {c} {k + 1}/{-(-new.height // ch)}")
            p[lo:lo + len(X)] = models.predict(l1, X[:, V4_IDX])
            del X; gc.collect()
        out = new.select(i1=(pl.col("i1") + o1[c]).cast(pl.UInt32), i23=(pl.col("i23") + o23[c]).cast(pl.UInt32),
                         *[pl.col(x) for x in BLK]).with_columns(p_l1=pl.Series(p), is_train=pl.lit(False))
        out.write_parquet(fpath("p_ctx", "mirror", c), compression="zstd")
        log(f"ctx {c}: scored {out.height:,} edges in {time.time() - t:.0f}s")
        del a, b, ce, comp, edges, new, out; gc.collect()


# ------------------------------------------------------------------ stage: p5 (union + L2 once)
def stage_p5():
    from .pipeline import MIRROR_C, add_l2, p_path
    l2 = S3._l2()
    oof2 = pl.read_parquet(CACHE / "l2_oof.parquet")
    for split in ["mirror", "test"]:
        for c in countries(split):
            if fpath("p5", split, c).exists():
                log(f"cache hit: p5 {split} {c}"); continue
            base = pl.read_parquet(p_path(split, c))
            if split == "mirror":
                base = base.with_columns(pl.col("is_train").fill_null(False))
                ctxe = pl.read_parquet(fpath("p_ctx", "mirror", c))
                tab = pl.concat([base.select("i1", "i23", *BLK, "p_l1", "is_train"), ctxe.select("i1", "i23", *BLK, "p_l1", "is_train")])
                tab = tab.sort("is_train", descending=True).unique(["i1", "i23"], keep="first")       # TRAIN rows first
                out = add_l2(tab, l2, oof2)
            else:
                out = add_l2(base.select("i1", "i23", *BLK, "p_l1"), l2).with_columns(is_train=pl.lit(False))
            out.select("i1", "i23", *BLK, "p_l1", "p_l2", "is_train").write_parquet(fpath("p5", split, c), compression="zstd")
            log(f"p5 {split} {c}: {out.height:,} edges")
            del base, out; gc.collect()


def p5(split, c):
    return pl.read_parquet(fpath("p5", split, c))


# ------------------------------------------------------------------ stage: bandx (band features)
def stage_bandx():
    from .pipeline import cand_path, load_parsed, offsets
    lab = labelled_global()
    for split in ["mirror", "test"]:
        o1, o23 = offsets(split)
        for c in countries(split):
            if fpath("pairx", split, c).exists() and fpath("ce43", split, c).exists():
                log(f"cache hit: bandx {split} {c}"); continue
            t = time.time()
            band = p5(split, c).filter(F.in_band()).select("i1", "i23")
            # ---- pairx: edge-local -> reuse V4.3 rows, compute the rest
            old = pl.read_parquet(dense_path("pairx", split, c)) if dense_path("pairx", split, c).exists() else None
            reuse = band.join(old, on=["i1", "i23"], how="inner") if old is not None else None
            miss = band.join(old.select("i1", "i23"), on=["i1", "i23"], how="anti") if old is not None else band
            parts = [reuse] if reuse is not None else []
            if miss.height:
                cand = pl.read_parquet(cand_path(split, c))
                loc = miss.select(i1=(pl.col("i1") - o1[c]).cast(cand.schema["i1"]), i23=(pl.col("i23") - o23[c]).cast(cand.schema["i23"]))
                bc = loc.join(cand, on=["i1", "i23"], how="left", maintain_order="left")
                X = build_features(bc, load_parsed(split, c, "s1"), load_parsed(split, c, "s23"), desc=f"pairx5 {split} {c}")
                parts.append(pl.DataFrame(X[:, V4_IDX], schema=F.PX).with_columns(i1=miss["i1"], i23=miss["i23"]))
                del cand, bc, X
            px = pl.concat([p.select("i1", "i23", *F.PX) for p in parts])
            assert px.height == band.height, (px.height, band.height)
            px.write_parquet(fpath("pairx", split, c), compression="zstd")
            # ---- ce43 scores: edge-local; OOF for TRAIN rows (model k scores fold % 2 == k), fold-0 model otherwise
            oldc = pl.read_parquet(dense_path("ce43", split, c)) if dense_path("ce43", split, c).exists() else None
            reuse_c = band.join(oldc, on=["i1", "i23"], how="inner") if oldc is not None else band.clear().with_columns(p_ce=pl.lit(0.0, pl.Float32))
            miss_c = band.join(reuse_c.select("i1", "i23"), on=["i1", "i23"], how="anti")
            if split == "mirror":        # ctx rows were never seen by ce43 -> any model is out-of-sample; TRAIN rows need their fold
                miss_c = miss_c.join(lab.filter(pl.col("role") == "train").select("i1", "fold"), on="i1", how="left")
                grp = (miss_c["fold"].fill_null(0) % 2).to_numpy() if miss_c.height else np.zeros(0, np.int64)
            else:
                grp = np.zeros(miss_c.height, np.int64)
            newc = []
            if miss_c.height:
                raw = _raw(split)
                A, B = F._texts(split, c, miss_c.select("i1", "i23"), raw)
                p = np.empty(miss_c.height, np.float32)
                for k in (0, 1):
                    sel = np.where(grp == k)[0]
                    if len(sel):
                        p[sel] = _ce_predict(_ce43_path(k), [A[i] for i in sel], [B[i] for i in sel], f"ce43 {split} {c} k{k}")
                newc.append(miss_c.select("i1", "i23").with_columns(p_ce=pl.Series(p)))
            pl.concat([reuse_c.select("i1", "i23", "p_ce")] + newc).write_parquet(fpath("ce43", split, c), compression="zstd")
            log(f"bandx {split} {c}: {band.height:,} band edges | pairx computed {miss.height:,} | ce43 computed {miss_c.height:,} "
                f"| {time.time() - t:.0f}s")
            del band, px; gc.collect()
    stage_em5()
    stage_mates5()


_RAW = {}


def _raw(split):
    from .pipeline import load_raw
    if split not in _RAW:
        _RAW.clear(); gc.collect()
        _RAW[split] = load_raw("train" if split == "mirror" else "test")[:2]
    return _RAW[split]


def _ce43_path(k):
    p = V4_MODELS / f"ce43_fold{k}"
    return p if (p / "config.json").exists() else V3_DIR / "models" / "ce_fold0"


def _ce_predict(path, A, B, desc):
    import torch
    from v3.ce import CrossEncoder
    ce = CrossEncoder(str(path)); ce.model.to(torch.bfloat16)
    out = ce.predict(A, B, desc=desc)
    del ce; torch.cuda.empty_cache()
    return out


def stage_em5():
    """Per-country Fellegi-Sunter EM; init supervised on labelled rows of the other folds (OOF) / all labelled rows."""
    from .pipeline import MIRROR_C
    if all(fpath("em", s, c).exists() for s in ["mirror", "test"] for c in countries(s)):
        log("cache hit: em5"); return
    lab = labelled_global()
    tr = truth_edges()
    ids, Gs, ctys = [], [], []
    for c in MIRROR_C:
        d = pl.read_parquet(fpath("pairx", "mirror", c))
        ids.append(d.select("i1", "i23")); Gs.append(F.EMM.comparison_vectors(d.select(F.PX).to_numpy().astype(np.float32), V4_FEATS))
        ctys.append(np.full(d.height, c)); del d
    ids = pl.concat(ids)
    G, cty = np.vstack(Gs), np.concatenate(ctys)
    j = ids.join(lab.select("i1", "fold"), on="i1", how="left", maintain_order="left").join(tr, on=["i1", "i23"], how="left", maintain_order="left")
    is_lab = j["fold"].is_not_null().to_numpy()
    y = j["y"].fill_null(0).to_numpy()
    fold = j["fold"].fill_null(-1).to_numpy()
    out = np.zeros((len(G), len(F.EM_F)), np.float32)

    def em_block(init, target):
        fs = F.EMM.FellegiSunter().fit_supervised(G[init], y[init]).em(G[target])
        W, tot, post = fs.weights(G[target]); out[target] = np.column_stack([W, tot, post])

    for k in range(CFG["folds"]):
        for c in MIRROR_C:
            sel = is_lab & (fold == k) & (cty == c)
            if sel.any():
                em_block(is_lab & (fold != k), sel)
    for c in MIRROR_C:
        sel = (~is_lab) & (cty == c)
        if sel.any():
            em_block(is_lab, sel)
    for c in MIRROR_C:
        m = cty == c
        ids.filter(pl.Series(m)).with_columns(pl.DataFrame(out[m], schema=F.EM_F)).write_parquet(fpath("em", "mirror", c), compression="zstd")
    Gl, yl = G[is_lab], y[is_lab]
    del G, Gs; gc.collect()
    for c in countries("test"):
        d = pl.read_parquet(fpath("pairx", "test", c))
        Gt = F.EMM.comparison_vectors(d.select(F.PX).to_numpy().astype(np.float32), V4_FEATS)
        fs = F.EMM.FellegiSunter().fit_supervised(Gl, yl).em(Gt)
        W, tot, post = fs.weights(Gt)
        d.select("i1", "i23").with_columns(pl.DataFrame(np.column_stack([W, tot, post]).astype(np.float32), schema=F.EM_F)).write_parquet(
            fpath("em", "test", c), compression="zstd")
        log(f"EM5 test {c}: pi {fs.pi:.3f}")


def stage_mates5():
    """Round-1 evidence pooling on p5 (strongest other candidate by p_l2), for every band edge."""
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm
    from .pipeline import load_parsed, offsets
    idx_src = [PAIR_FEATS.index(f) for f in S3.MATE_SRC]
    for split in ["mirror", "test"]:
        o1, o23 = offsets(split)
        for c in countries(split):
            if S3.mates_path(split, c).exists():
                continue
            if split == "test" and dense_path("mates", split, c).exists():   # test p_l2 is unchanged -> identical to V4.3
                import os; os.link(dense_path("mates", split, c), S3.mates_path(split, c)); continue
            tab = p5(split, c).select("i1", "i23", "p_l2")
            band = tab.filter(F.in_band()).select("i1", "i23")
            strong = (tab.filter(pl.col("p_l2") >= S3.STRONG)
                         .with_columns(r=pl.col("p_l2").rank("ordinal", descending=True).over("i1"), n_strong=pl.len().over("i1"))
                         .filter(pl.col("r") <= 2).select("i1", mate="i23", m_p="p_l2", r="r", n_strong="n_strong"))
            m = (band.join(strong, on="i1").filter(pl.col("mate") != pl.col("i23")).sort("r").unique(["i1", "i23"], keep="first")
                     .with_columns(m_n_strong=pl.col("n_strong").cast(pl.Float32)))
            R = load_parsed(split, c, "s23").select(REC)
            x = m["i23"].to_numpy().astype(np.int64) - o23[c]
            yy = m["mate"].to_numpy().astype(np.int64) - o23[c]
            Bn = 40_000
            jobs = (delayed(_feat_batch)(R[x[lo:lo + Bn]].rows(), R[yy[lo:lo + Bn]].rows()) for lo in range(0, len(x), Bn))
            Fm = np.empty((len(x), len(S3.MATE_SRC)), np.float32); pos = 0
            for part in tqdm(Parallel(n_jobs=12, return_as="generator", pre_dispatch="2*n_jobs")(jobs),
                             total=-(-len(x) // Bn), desc=f"mates5 {split} {c}", mininterval=15):
                Fm[pos:pos + len(part)] = part[:, idx_src]; pos += len(part)
            (m.select("i1", "i23", m_p=pl.col("m_p").cast(pl.Float32), m_n_strong="m_n_strong")
              .with_columns(pl.DataFrame(Fm, schema=[f"m_{f}" for f in S3.MATE_SRC]))
              .write_parquet(S3.mates_path(split, c), compression="zstd"))
            log(f"mates5 {split} {c}: {m.height:,} band edges with a strong mate")
            del tab, band, strong, m, R, Fm; gc.collect()


# ------------------------------------------------------------------ stage: ce5 (cross-encoder re-trained on labelled band pairs)
def stage_ce5():
    import torch
    from v3.ce import CrossEncoder
    from .pipeline import MIRROR_C
    if all(fpath("ce5", s, c).exists() for s in ["mirror", "test"] for c in countries(s)):
        log("cache hit: ce5"); return
    if all(fpath("ce5", "mirror", c).exists() for c in MIRROR_C):
        ce0 = CrossEncoder(str(MODELS / "ce5_half0")); ce0.model.to(torch.bfloat16)
        log("CE5 mirror scores cached; resuming the test pass with ce5_half0")
        return _ce5_test(ce0)
    lab = labelled_global()
    tr = truth_edges()
    E, A, B = [], [], []
    raw = _raw("mirror")
    for c in MIRROR_C:
        e = (p5("mirror", c).filter((pl.col("p_l2") >= CE5_BAND[0]) & (pl.col("p_l2") <= CE5_BAND[1])).select("i1", "i23")
                            .with_columns(cty=pl.lit(c)))
        a, b = F._texts("mirror", c, e, raw)
        E.append(e); A += a; B += b
    E = (pl.concat(E).join(lab.select("i1", "fold"), on="i1", how="left", maintain_order="left")
                     .join(tr, on=["i1", "i23"], how="left", maintain_order="left"))
    is_lab = E["fold"].is_not_null().to_numpy()
    y = E["y"].fill_null(0).to_numpy().astype(np.float32)
    half = np.where(is_lab, E["fold"].fill_null(0).to_numpy() % 2, -1)
    log(f"CE5 band: {E.height:,} mirror edges | labelled {is_lab.sum():,} (pos {y[is_lab].mean():.3f})")
    p = np.full(E.height, np.nan, np.float32)
    ce0 = None
    for h in (0, 1):                    # model h predicts half h, trained on half 1-h, warm-started from ce43_fold{h}
        path = MODELS / f"ce5_half{h}"
        if (path / "config.json").exists():
            ce = CrossEncoder(str(path)); log(f"CE5 half {h}: loaded")
        else:
            trn = np.where(is_lab & (half == 1 - h))[0]
            ce = CrossEncoder(str(_ce43_path(h)))
            t = time.time()
            loss = ce.fit([A[i] for i in trn], [B[i] for i in trn], y[trn], lr=2e-5, desc=f"CE5 train half {h}")
            path.parent.mkdir(parents=True, exist_ok=True); ce.save(path)
            log(f"CE5 half {h}: trained on {len(trn):,} pairs (pos {y[trn].mean():.3f}) loss {loss:.4f} in {time.time() - t:.0f}s")
        ce.model.to(torch.bfloat16)
        va = np.where(is_lab & (half == h))[0]
        p[va] = ce.predict([A[i] for i in va], [B[i] for i in va], desc=f"CE5 OOF half {h}")
        if h == 0:
            rest = np.where(~is_lab)[0]
            p[rest] = ce.predict([A[i] for i in rest], [B[i] for i in rest], desc="CE5 mirror unlabelled")
            ce0 = ce
        else:
            del ce; torch.cuda.empty_cache()
    from sklearn.metrics import roc_auc_score
    log(f"CE5 OOF AUC on labelled band {roc_auc_score(y[is_lab], p[is_lab]):.5f}")
    assert np.isfinite(p).all()
    for c in MIRROR_C:
        m = (E["cty"] == c).to_numpy()
        E.filter(pl.Series(m)).select("i1", "i23").with_columns(p_ce5=pl.Series(p[m])).write_parquet(fpath("ce5", "mirror", c), compression="zstd")
    del A, B, E; gc.collect()
    _ce5_test(ce0)


def _ce5_test(ce0):
    raw = _raw("test")
    for c in countries("test"):
        if fpath("ce5", "test", c).exists():
            continue
        e = p5("test", c).filter((pl.col("p_l2") >= CE5_BAND[0]) & (pl.col("p_l2") <= CE5_BAND[1])).select("i1", "i23")
        a, b = F._texts("test", c, e, raw)
        t = time.time()
        e.with_columns(p_ce5=pl.Series(ce0.predict(a, b, desc=f"CE5 test {c}"))).write_parquet(fpath("ce5", "test", c), compression="zstd")
        log(f"CE5 test {c}: {e.height:,} edges in {time.time() - t:.0f}s")


# ------------------------------------------------------------------ stacker v5
def frame5(split, c):
    """all band rows of one split/country with every stacker input (context over all edges, band features inner-joined)."""
    tab = p5(split, c)
    bf = F.band_features(split, c, "mates")
    fr = F.frame(tab, bf, "p_l2")
    fr = fr.join(pl.read_parquet(fpath("ce5", split, c)), on=["i1", "i23"], how="left").with_columns(
        ce5_present=pl.col("p_ce5").is_not_null().cast(pl.Float32))
    fr = fr.with_columns(ce5_minus_p=pl.when(pl.col("ce5_present") > 0).then(pl.col("p_ce5") - pl.col("p")).otherwise(0.0),
                         p_ce5=pl.col("p_ce5").fill_null(-1.0))
    band_n = tab.filter(F.in_band()).height
    assert fr.height == band_n, f"band rows lost in frame5 {split} {c}: {fr.height} vs {band_n}"
    return fr


SEEDS = (0, 1)


def stage_stack5():
    import xgboost as xgb
    from .pipeline import MIRROR_C
    from sklearn.metrics import roc_auc_score
    lab = labelled_global()
    tr = truth_edges()
    if not (CACHE / "stack5_oof.parquet").exists():
        fr = pl.concat([frame5("mirror", c).with_columns(cty=pl.lit(c)) for c in MIRROR_C], how="diagonal_relaxed")
        trn = (fr.join(lab.select("i1", "fold", "role"), on="i1", how="inner")
                 .join(tr, on=["i1", "i23"], how="left").with_columns(pl.col("y").fill_null(0)))
        n_lab_band = trn.height
        log(f"stack5: {n_lab_band:,} labelled band rows ({(trn['role'] == 'ctx').sum():,} from context S1s)")
        X = trn.select(F5).to_numpy().astype(np.float32)
        y, folds, role = trn["y"].to_numpy(), trn["fold"].to_numpy(), trn["role"].to_numpy()
        mono = [1 if f in ("p", "p_ce", "p_ce5", "em_total") else 0 for f in F5]
        oof = np.zeros(len(y), np.float32); iters_all = []
        for s in SEEDS:
            prm = models.params(monotone=mono, seed=s, colsample_bytree=0.7 if s == 0 else 0.5)
            o, iters, final = models.train_oof(X, y, folds, prm, f"stack5 s{s}")
            oof += o / len(SEEDS); iters_all.append(iters)
            final.save_model(CACHE / f"stack5_s{s}.json")
        # learning-curve check: TRAIN-only model vs TRAIN + context, both evaluated OOF on TRAIN rows
        is_tr = role == "train"
        o_tr, _, _ = models.train_oof(X[is_tr], y[is_tr], folds[is_tr], models.params(monotone=mono, seed=0), "stack5 TRAIN-only")
        log(f"stack5 OOF AUC (labelled band) base p_l2 {roc_auc_score(y, trn['p'].to_numpy()):.5f} -> {roc_auc_score(y, oof):.5f} | "
            f"on TRAIN rows: TRAIN-only model {roc_auc_score(y[is_tr], o_tr):.5f} vs +context {roc_auc_score(y[is_tr], oof[is_tr]):.5f}")
        trn.select("i1", "i23").with_columns(p_v5=pl.Series(oof)).write_parquet(CACHE / "stack5_oof.parquet", compression="zstd")
        del fr, trn, X; gc.collect()
    boosters = []
    for s in SEEDS:
        b = xgb.Booster(model_file=str(CACHE / f"stack5_s{s}.json")); b.set_param({"device": models.device()}); boosters.append(b)
    oof = pl.read_parquet(CACHE / "stack5_oof.parquet")
    for split in ["mirror", "test"]:
        for c in countries(split):
            if fpath("pv5", split, c).exists():
                continue
            fr = frame5(split, c)
            X = fr.select(F5).to_numpy().astype(np.float32)
            p = np.mean([models.predict(b, X) for b in boosters], axis=0).astype(np.float32)
            # frozen V4.3 round-1 stacker on the same frame (like-for-like baseline): out-of-sample for context S1s,
            # V4.3's own OOF score for TRAIN rows, so no in-sample score ever competes in exclusivity
            b43 = xgb.Booster(model_file=str(CACHE / "l43a.json")); b43.set_param({"device": models.device()})
            out = fr.select("i1", "i23").with_columns(
                p_new=pl.Series(p), p_43a=pl.Series(models.predict(b43, fr.select(F43A).to_numpy().astype(np.float32))))
            if split == "mirror":
                out = (out.join(oof.rename({"p_v5": "p_oof"}), on=["i1", "i23"], how="left")
                          .with_columns(p_new=pl.coalesce("p_oof", "p_new")).drop("p_oof"))
                o43 = pl.read_parquet(CACHE / "l43a_oof.parquet").rename({"p": "p43_oof"})
                out = (out.join(o43, on=["i1", "i23"], how="left")
                          .with_columns(p_43a=pl.coalesce("p43_oof", "p_43a")).drop("p43_oof"))
            out.rename({"p_new": "p_v5"}).write_parquet(fpath("pv5", split, c), compression="zstd")
            log(f"scored stack5 {split} {c}: {out.height:,} band edges")
            del fr, X, out; gc.collect()


def scored5(split, c):
    """all edges with p_l2, p_v5 (band: stacker, else p_l2) and p_43a (frozen V4.3 round 1)."""
    t = p5(split, c)
    b = pl.read_parquet(fpath("pv5", split, c))
    return t.join(b, on=["i1", "i23"], how="left").with_columns(p_v5=pl.coalesce("p_v5", "p_l2"), p_43a=pl.coalesce("p_43a", "p_l2"))


# ------------------------------------------------------------------ per-entity F0.5 + bootstrap + prior matching
def f_entities(pred, truth, ids):
    """per-S1 F0.5 (aligned to sorted unique ids) for predictions / truth restricted to ids."""
    ids = np.sort(np.unique(np.asarray(ids)))
    base = pl.DataFrame({"i1": ids}).cast({"i1": pl.UInt32})
    pr = pred.select("i1", "i23").cast({"i1": pl.UInt32, "i23": pl.UInt32}).join(base, on="i1", how="semi")
    tt = truth.select("i1", "i23").cast({"i1": pl.UInt32, "i23": pl.UInt32}).join(base, on="i1", how="semi")
    tp = pr.join(tt, on=["i1", "i23"], how="semi").group_by("i1").len("tp")
    t = (base.join(tp, on="i1", how="left", maintain_order="left")
             .join(pr.group_by("i1").len("np"), on="i1", how="left", maintain_order="left")
             .join(tt.group_by("i1").len("nt"), on="i1", how="left", maintain_order="left").fill_null(0))
    tp_, np_, nt_ = (t[k].to_numpy().astype(float) for k in ("tp", "np", "nt"))
    den = 5 * tp_ + (nt_ - tp_) + 4 * (np_ - tp_)
    return np.where(den == 0, 1.0, 5 * tp_ / np.maximum(den, 1e-12))


def boot(fa, fb, n=1000, seed=0):
    d = np.asarray(fa) - np.asarray(fb)
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def shift(p, d):
    return (1 / (1 + np.exp(-(_logit(p) + d)))).astype(np.float32)


def match_rate(apply_fn, tab_p, target, lo=-4.0, hi=4.0, it=12):
    """label-free prior matching: bisection on a logit shift so predicted matches per S1 hit `target`.
    apply_fn(tab_p) -> (pred, matches_per_s1)."""
    p0 = tab_p["p"].to_numpy()
    for _ in range(it):
        mid = (lo + hi) / 2
        rate = apply_fn(tab_p.with_columns(p=pl.Series(shift(p0, mid))))[1]
        lo, hi = (mid, hi) if rate < target else (lo, mid)
    d = (lo + hi) / 2
    return d, apply_fn(tab_p.with_columns(p=pl.Series(shift(p0, d))))


# ------------------------------------------------------------------ stage: loco (honest France proxy)
def stage_loco():
    from .features import MONOTONE
    from .pipeline import mirror_truth_global, role_global, s1_info, cand_info
    if (CACHE / "loco_v5.json").exists():
        log(f"cache hit: loco_v5 -> {json.load(open(CACHE / 'loco_v5.json'))}"); return
    s1i, ci = s1_info("mirror"), cand_info("mirror")
    truth = mirror_truth_global()
    T = role_global("train")
    D = {c: pl.read_parquet(CACHE / f"train_X_{c}.parquet") for c in ("us", "india")}
    prm1 = models.params(monotone=[MONOTONE.get(f, 0) for f in V4_FEATS])
    Xu = D["us"].select(V4_FEATS).to_numpy().astype(np.float32)
    yu, fu = D["us"]["y"].to_numpy(), D["us"]["fold"].to_numpy()
    oof1u, _, l1u = models.train_oof(Xu, yu, fu, prm1, "LOCO L1 US-only")
    del Xu; gc.collect()
    p1i = models.predict(l1u, D["india"].select(V4_FEATS).to_numpy().astype(np.float32))
    ctxc = ["i1", "i23", "blk", "blk_rank", "blk_rev_rank", "rev_n", "blk_gap", "n_cand"]
    L2u = models.context_features(D["us"].select(ctxc).with_columns(p=pl.Series(oof1u)))
    prm2 = models.params(monotone=[1 if f == "p" else 0 for f in models.L2_FEATS])
    oof2u, _, l2u = models.train_oof(L2u.select(models.L2_FEATS).to_numpy(), yu, fu, prm2, "LOCO L2 US-only")
    L2i = models.context_features(D["india"].select(ctxc).with_columns(p=pl.Series(p1i)))
    p2i = models.predict(l2u, L2i.select(models.L2_FEATS).to_numpy())

    def setup(c, p):
        d = D[c]
        tab = d.select("i1", "i23").with_columns(p=pl.Series(p, dtype=pl.Float32))
        ids = T.filter(pl.col("cty") == c)["i1"]
        tt = truth.filter(pl.col("cty") == c).join(pl.DataFrame({"i1": ids}), on="i1", how="semi").select("i1", "i23")
        hit = tt.join(tab.select("i1", "i23"), on=["i1", "i23"], how="semi").height
        return (tab, d["y"].to_numpy(), d["fold"].to_numpy(), tt, ids, d.group_by("i1").agg(pl.col("fold").first()),
                (tt.height - hit) / len(ids))

    def run(st, tab, ids):
        pred = decide.apply(st, tab, s1i, ci, ids)
        return pred, pred.height / len(ids)

    tab_u, y_u, f_u, tt_u, ids_u, fo_u, mm_u = setup("us", oof2u)
    st_u = decide.fit_decision(tab_u, y_u, np.ones(len(y_u), bool), f_u, tt_u, ids_u, fo_u, s1i, ci, mm_u, tag="LOCO US rule")
    tab_i, y_i, f_i, tt_i, ids_i, fo_i, mm_i = setup("india", p2i)
    pred_loco, r_loco = run(st_u, tab_i, ids_i)
    F_loco = f_entities(pred_loco, tt_i, ids_i)
    st_or = decide.fit_decision(tab_i, y_i, np.ones(len(y_i), bool), f_i, tt_i, ids_i, fo_i, s1i, ci, mm_i, tag="LOCO India oracle rule")
    F_or = f_entities(run(st_or, tab_i, ids_i)[0], tt_i, ids_i)
    oof2 = pl.read_parquet(CACHE / "l2_oof.parquet")
    tab_in = tab_i.drop("p").join(oof2.rename({"p_l2": "p"}), on=["i1", "i23"], how="left", maintain_order="left").with_columns(
        pl.col("p").fill_null(0.0))
    st_in = decide.fit_decision(tab_in, y_i, np.ones(len(y_i), bool), f_i, tt_i, ids_i, fo_i, s1i, ci, mm_i, tag="India in-country")
    F_in = f_entities(run(st_in, tab_in, ids_i)[0], tt_i, ids_i)
    r_us = run(st_u, tab_u, ids_u)[1]
    d_pm, (pred_pm, _) = match_rate(lambda tp: run(st_u, tp, ids_i), tab_i, r_us)
    F_pm = f_entities(pred_pm, tt_i, ids_i)
    dlt, lo, hi = boot(F_pm, F_loco)
    res = dict(F_in_country=float(F_in.mean()), F_oracle_rule=float(F_or.mean()), F_loco=float(F_loco.mean()),
               ranking_loss=float(F_in.mean() - F_or.mean()), calibration_loss=float(F_or.mean() - F_loco.mean()),
               pred_per_s1_us=r_us, pred_per_s1_india_loco=r_loco, true_per_s1_us=tt_u.height / len(ids_u),
               true_per_s1_india=tt_i.height / len(ids_i), prior_match_shift=d_pm, F_prior_matched=float(F_pm.mean()),
               prior_match_delta=dlt, prior_match_ci=[lo, hi], adopt_prior_matching=bool(lo > 0))
    json.dump(res, open(CACHE / "loco_v5.json", "w"), indent=1)
    log("LOCO (US-only models -> India TRAIN S1s):\n" + json.dumps(res, indent=1))


# ------------------------------------------------------------------ stage: decide (+ policy, HOLDOUT report)
def stage_decide5():
    from .pipeline import MIRROR_C, mirror_truth_global, role_global, s1_info, cand_info
    lab = labelled_global()
    tab = pl.concat([scored5("mirror", c) for c in MIRROR_C], how="diagonal_relaxed")
    tab = (tab.join(lab.select("i1", "fold", "role"), on="i1", how="left")
              .join(truth_edges(), on=["i1", "i23"], how="left").with_columns(pl.col("y").fill_null(0)))
    truth = mirror_truth_global()
    H = role_global("holdout")
    truth_H = truth.join(H, on="i1", how="semi").select("i1", "i23")
    s1i, ci = s1_info("mirror"), cand_info("mirror")

    def fit(var, mask_expr, tag):
        m = tab.select(mask_expr.fill_null(False).alias("m"))["m"].to_numpy()
        sub = tab.filter(pl.Series(m))
        ids = lab.join(pl.DataFrame({"i1": sub["i1"].unique()}).cast({"i1": lab.schema["i1"]}), on="i1", how="semi")
        tT = truth.join(ids.select("i1"), on="i1", how="semi").select("i1", "i23")
        mm = (tT.height - tT.join(sub.select("i1", "i23"), on=["i1", "i23"], how="semi").height) / ids.height
        st_ = decide.fit_decision(tab.select("i1", "i23", p=var), sub["y"].to_numpy(), m, sub["fold"].to_numpy(), tT, ids["i1"],
                                  ids.select("i1", "fold"), s1i, ci, mm, tag=tag)
        return st_, m, sub, ids, tT

    st = {}
    st["p_l2"] = fit("p_l2", pl.col("fold").is_not_null(), "p_l2 (V4.1 scores)")[0]
    st["p_v5"], m_L, sub_L, ids_L, tT_L = fit("p_v5", pl.col("fold").is_not_null(), "p_v5")
    st["p_43a"] = fit("p_43a", pl.col("role") == "ctx", "p_43a (frozen V4.3 round 1, fitted on context S1s)")[0]
    # ---- learned policy on the V5 rule (cross-fitted calibration + OOF singleton probabilities for labelled S1s)
    sb = st["p_v5"]
    pv = tab["p_v5"].to_numpy()
    p_cal = sb["iso"].predict(pv).astype(np.float32)
    p_cal[m_L] = decide.iso_crossfit(pv[m_L], sub_L["y"].to_numpy(), sub_L["fold"].to_numpy())
    sc_ex = decide.exclusivity(tab.select("i1", "i23").with_columns(p=pl.Series(p_cal)), sb["eps"], sb["lam"]).join(
        ids_L.select("i1"), on="i1", how="semi")
    ids, P, R, rest, EF, Xpol = F.lists_from(sc_ex, sb, s1i, ci, ids_L["i1"], sb["p_single_T"])
    C, valid, Fk = F.cost_table(ids, R, tT_L)
    valid &= EF >= 0
    fo = (pl.DataFrame({"i1": ids}).cast({"i1": pl.UInt32})
            .join(ids_L.select("i1", "fold").cast({"i1": pl.UInt32}), on="i1", how="left", maintain_order="left")["fold"]
            .fill_null(0).to_numpy())
    kk = np.zeros(len(ids), np.int64)
    for k in range(CFG["folds"]):
        trm, va = fo != k, fo == k
        kk[va] = F.DC3.policy_choose(F.DC3.fit_policy(Xpol[trm], C[trm], valid[trm]), Xpol[va], valid[va])
    f_pol = Fk[np.arange(len(ids)), kk]
    if sb["rule"] == "expected_f":
        pr_rule = decide.expected_f_select(sc_ex, sb["p_single_T"], sb["m_miss"], sb["floor"], sb["max_k"])
    else:
        pr_rule = decide.decide_global(sc_ex, sb["thr"])
    f_ef = f_entities(pr_rule, tT_L, ids)                     # the fitted V4 rule on the same honest inputs
    dlt, lo, hi = boot(f_pol, f_ef)
    policy = F.DC3.fit_policy(Xpol, C, valid)
    use_policy = bool(lo > 0)
    log(f"learned policy (p_v5, {len(ids):,} labelled S1): {f_pol.mean():.5f} vs fitted rule ({sb['rule']}) {f_ef.mean():.5f} | "
        f"delta {dlt:+.5f} CI [{lo:+.5f}, {hi:+.5f}] -> {'ADOPTED' if use_policy else 'rejected'}")
    # ---- HOLDOUT report (never used above)
    Hs = np.sort(H["i1"].to_numpy())
    preds = {v: decide.apply(st[v], tab.select("i1", "i23", p=v), s1i, ci, H["i1"]) for v in st}
    idsH, _, RH, _, EFH, XH = F.policy_inputs(sb, tab.select("i1", "i23", p="p_v5"), s1i, ci, H["i1"])
    validH = np.concatenate([np.ones((len(idsH), 1), bool), RH >= 0], 1) & (EFH >= 0)
    preds["p_v5+policy"] = F.pred_from_k(idsH, RH, F.DC3.policy_choose(policy, XH, validH))
    Fh = {k: f_entities(v, truth_H, Hs) for k, v in preds.items()}
    Hdf = pl.DataFrame({"i1": Hs}).cast({"i1": pl.UInt32})
    cty = Hdf.join(H.select("i1", "cty").cast({"i1": pl.UInt32}), on="i1", how="left", maintain_order="left")["cty"].to_numpy()
    lab_recs = tab.filter(pl.col("fold").is_not_null() & F.in_band()).select("i23").unique()
    hb = (tab.filter(F.in_band()).join(Hdf, on="i1", how="semi").select("i1", "i23")
             .join(lab_recs.with_columns(seen=pl.lit(True)), on="i23", how="left").with_columns(pl.col("seen").fill_null(False)))
    seen = (Hdf.join(hb.group_by("i1").agg(pl.col("seen").any()).cast({"i1": pl.UInt32}), on="i1", how="left",
                     maintain_order="left")["seen"].fill_null(False).to_numpy())
    best = "p_v5+policy" if use_policy else "p_v5"
    rows = []
    for k, f in Fh.items():
        d43 = boot(f, Fh["p_43a"])
        rows.append((k + (" (chosen)" if k == best else ""), float(f.mean()), float(f[cty == "us"].mean()), float(f[cty == "india"].mean()),
                     float(f[seen].mean()), float(f[~seen].mean()), d43[0], d43[1], d43[2]))
    rep = pl.DataFrame(rows, schema=["model", "holdout_f05", "us", "india", "rec_seen", "rec_unseen", "d_vs_V4.3a", "ci_lo",
                                     "ci_hi"], orient="row")
    log(f"V5 HOLDOUT ({len(Hs):,} S1; {seen.mean():.1%} have a band record also in labelled training rows):\n{rep}")
    pickle.dump(dict(st=sb, policy=policy, use_policy=use_policy, best=best, report=rep), open(CACHE / "decision_v5.pkl", "wb"))


# ------------------------------------------------------------------ stage: export (+ France prior-matched variant)
def stage_export5():
    from .pipeline import load_parsed, load_raw, offsets, s1_info, cand_info
    d = pickle.load(open(CACHE / "decision_v5.pkl", "rb"))
    st = d["st"]
    loco = json.load(open(CACHE / "loco_v5.json")) if (CACHE / "loco_v5.json").exists() else {}
    s1i, ci = s1_info("test"), cand_info("test")
    o1, o23 = offsets("test")

    def predict(tp, ids):
        if d["use_policy"]:
            I, _, R, _, EF, X = F.policy_inputs(st, tp, s1i, ci, ids)
            valid = np.concatenate([np.ones((len(I), 1), bool), R >= 0], 1) & (EF >= 0)
            pr = F.pred_from_k(I, R, F.DC3.policy_choose(d["policy"], X, valid))
        else:
            pr = decide.apply(st, tp, s1i, ci, ids)
        return pr, pr.height / len(ids)

    preds, cands, stats, fr_in = {}, [], [], None
    for c in countries("test"):
        tab = scored5("test", c)
        ids = load_parsed("test", c, "s1", ["idx"]).select(i1=(pl.col("idx") + o1[c]).cast(pl.UInt32))["i1"]
        tp = tab.select("i1", "i23", p="p_v5")
        preds[c], rate = predict(tp, ids)
        cands.append((c, tab.select("i1", "i23")))
        stats.append((c, len(ids), rate, 1 - preds[c]["i1"].n_unique() / len(ids)))
        if c == "france":
            fr_in = (tp, ids)
        del tab; gc.collect()
    target = float(np.mean([s[2] for s in stats if s[0] != "france"]))
    dfr, (pred_fr_pm, rate_pm) = match_rate(lambda t: predict(t, fr_in[1]), fr_in[0], target)
    log(f"test predictions (p_v5{' + policy' if d['use_policy'] else ''}):\n"
        f"{pl.DataFrame(stats, schema=['cty', 'n_s1', 'pred_per_s1', 'pred_empty'], orient='row')}\n"
        f"France prior-matched variant: logit shift {dfr:+.3f} -> {rate_pm:.3f} matches/S1 (target {target:.3f}); "
        f"LOCO adopt={loco.get('adopt_prior_matching')}")
    order = load_raw("test")[0]["entity_id"]
    idmaps = {}
    for c in countries("test"):
        a = load_parsed("test", c, "s1", ["idx", "entity_id"]).with_columns(i1=(pl.col("idx") + o1[c]).cast(pl.UInt32))
        b = load_parsed("test", c, "s23", ["idx", "entity_id"]).with_columns(i23=(pl.col("idx") + o23[c]).cast(pl.UInt32))
        idmaps[c] = (a.select("i1", s1_id="entity_id"), b.select("i23", s23_id="entity_id"))

    def to_ids(c, e):
        return (e.select("i1", "i23").cast({"i1": pl.UInt32, "i23": pl.UInt32}).join(idmaps[c][0], on="i1")
                 .join(idmaps[c][1], on="i23").select("s1_id", "s23_id"))

    all_cand = pl.concat([to_ids(c, e) for c, e in cands])

    def write_variant(pr_by_c, folder):
        allp = pl.concat([to_ids(c, p) for c, p in pr_by_c.items()])
        assert allp.join(all_cand, on=["s1_id", "s23_id"], how="anti").height == 0, "matches must be a subset of candidates"
        assert allp["s23_id"].is_unique().all(), "an S2/S3 record may belong to at most one S1"
        lists = allp.sort("s1_id", "s23_id").group_by("s1_id", maintain_order=True).agg(pl.col("s23_id").str.join(",").alias("matched_entity_ids"))
        out = order.to_frame("source1_entity_id").join(lists.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id",
                                                       how="left").fill_null("")
        assert out.height == order.len() and out["source1_entity_id"].is_unique().all()
        folder.mkdir(parents=True, exist_ok=True)
        out.write_csv(folder / "matching_results.tsv", separator="\t", quote_style="never")
        log(f"wrote {folder / 'matching_results.tsv'}: {out.height:,} rows, {allp.height:,} matches")

    adopt = bool(loco.get("adopt_prior_matching"))
    main_v, alt_v = ({**preds, "france": pred_fr_pm}, preds) if adopt else (preds, {**preds, "france": pred_fr_pm})
    alt_name = "alt_france_unshifted" if adopt else "alt_france_prior_matched"
    log(f"primary submission: France {'prior-matched' if adopt else 'unshifted'} (LOCO adopt={adopt}); alternative -> {alt_name}/")
    write_variant(main_v, OUT5)
    write_variant(alt_v, OUT5 / alt_name)
    lists = all_cand.sort("s1_id", "s23_id").group_by("s1_id", maintain_order=True).agg(pl.col("s23_id").str.join(",").alias("candidate_entity_ids"))
    (order.to_frame("source1_entity_id").join(lists.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id", how="left")
          .fill_null("").write_csv(OUT5 / "candidate_pairs.tsv", separator="\t", quote_style="never"))
    if SMOKE:
        log("smoke run: official validator skipped"); return
    for folder in (OUT5, OUT5 / alt_name):
        r = subprocess.run([sys.executable, str(VALIDATOR),
                            "--matching", str(folder / "matching_results.tsv"), "--candidate", str(OUT5 / "candidate_pairs.tsv"),
                            "--test-dir", str(DATA / "test"), "--check-ids"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        print(r.stdout[-1500:], r.stderr[-800:])
        log(f"official validator [{folder.name}] exit code {r.returncode} -> {'PASS' if r.returncode == 0 else 'FIX ISSUES'}")


STAGES = dict(ctx=stage_ctx, loco=stage_loco, p5=stage_p5, bandx=stage_bandx, ce5=stage_ce5, stack5=stage_stack5,
              decide=stage_decide5, export=stage_export5)

if __name__ == "__main__":
    from .pipeline import link_shared
    link_shared()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    s = ap.parse_args().stage
    t0 = time.time()
    for name, fn in STAGES.items():
        if s in ("all", name):
            log(f"===== V5 stage {name} =====")
            fn()
    log(f"done in {(time.time() - t0) / 60:.1f} min")
