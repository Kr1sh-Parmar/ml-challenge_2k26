"""V4.2 = V4.1 + V3's text cross-encoder + an L3 stacker, evaluated on the same test-mirror HOLDOUT.

Run from version 4/src with ER4_DENSE=1 (reads V4.1's cache_dense/, writes cache_dense/ + output_v42/):
    python -m er4.stack3 --stage mates|ce|l3|decide|export|all

- mates:  evidence pooling (V3 L7a, scale-free form): every uncertain edge (i1, x) is compared record-to-record with
          the S1's strongest OTHER candidate ("mate", p_l2 >= 0.5): V4's 75 content pair features x-vs-mate, the mate's
          score, and the number of strong mates. A weak duplicate can then inherit evidence from its strong sibling.
- ce:     V3's mDeBERTa-v3 cross-encoder (fold-0 model, bf16, GPU) scores every edge whose V4.1 L2 score is uncertain
          (0.003 <= p_l2 <= 0.997), for the mirror (TRAIN + closure) and the test set. The CE reads only the two records'
          raw text (scale-free), and it was trained on train S1 with u < 0.10, disjoint from mirror TRAIN / HOLDOUT.
- l3:     XGBoost (GPU) on V4's context features computed from p_l2 + p_l1 + CE score, 5-fold OOF on mirror TRAIN
          (same S1 folds as L1/L2), monotone in p_l2 and p_ce.
- decide: V4's decision layer fitted on TRAIN for p_l2 (= V4.1) and p_l3; HOLDOUT report; the variant with the best
          TRAIN OOF is kept (V4's own adoption rule).
- export: test predictions with the kept variant -> output_v42/ + official validator --check-ids.
"""
import argparse, gc, pickle, subprocess, sys, time

import numpy as np
import polars as pl

from . import decide, models
from .config import CACHE, CFG, DATA, DENSE, ROOT, SMOKE, V4
from .util import log, macro_f05, cached

assert DENSE, "V4.2 builds on V4.1: run with ER4_DENSE=1"
OUT42 = V4 / ("output_v42_smoke" if SMOKE else "output_v42")
V3_DIR = V4.parent / "version 3"
BAND = (0.003, 0.997)
from .features import FAM, PAIR_FEATS, REC, _feat_batch
MATE_SRC = [f for f in PAIR_FEATS if f not in FAM["freq"]]                     # 75 content features
MATE_FEATS = [f"m_{f}" for f in MATE_SRC] + ["m_p", "m_n_strong", "m_present"]
L3_FEATS = models.L2_FEATS + ["p_l1", "p_ce", "ce_present", "ce_minus_p"] + MATE_FEATS
STRONG = 0.5


def mates_path(split, c):
    return CACHE / f"mates_{split}_{c}.parquet"


def stage_mates():
    """Band edges x strongest other candidate of the same S1 -> record-record content features."""
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm
    from .pipeline import countries, load_parsed, offsets
    l2, oof2 = _l2(), pl.read_parquet(CACHE / "l2_oof.parquet")
    idx_src = [PAIR_FEATS.index(f) for f in MATE_SRC]
    for split in ["mirror", "test"]:
        o1, o23 = offsets(split)
        for c in countries(split):
            if mates_path(split, c).exists():
                log(f"cache hit: {mates_path(split, c).name}"); continue
            t = time.time()
            tab = edges_with_l2(split, c, l2, oof2 if split == "mirror" else None).select("i1", "i23", "p_l2")
            band = tab.filter((pl.col("p_l2") >= BAND[0]) & (pl.col("p_l2") <= BAND[1])).select("i1", "i23")
            strong = (tab.filter(pl.col("p_l2") >= STRONG)
                         .with_columns(r=pl.col("p_l2").rank("ordinal", descending=True).over("i1"),
                                       n_strong=pl.len().over("i1"))
                         .filter(pl.col("r") <= 2).select("i1", mate="i23", m_p="p_l2", r="r", n_strong="n_strong"))
            # best mate that is not the edge's own record
            m = (band.join(strong, on="i1", how="inner").filter(pl.col("mate") != pl.col("i23"))
                     .sort("r").unique(["i1", "i23"], keep="first"))
            m = m.with_columns(m_n_strong=pl.col("n_strong").cast(pl.Float32))
            R = load_parsed(split, c, "s23").select(REC)
            x = m["i23"].to_numpy().astype(np.int64) - o23[c]
            y = m["mate"].to_numpy().astype(np.int64) - o23[c]
            B = 40_000
            jobs = (delayed(_feat_batch)(R[x[lo:lo + B]].rows(), R[y[lo:lo + B]].rows()) for lo in range(0, len(x), B))
            F = np.empty((len(x), len(MATE_SRC)), np.float32)
            pos = 0
            for part in tqdm(Parallel(n_jobs=CFG["n_jobs"], return_as="generator", pre_dispatch="2*n_jobs")(jobs),
                             total=-(-len(x) // B), desc=f"mates {split} {c}", mininterval=15):
                F[pos:pos + len(part)] = part[:, idx_src]
                pos += len(part)
            out = m.select("i1", "i23", m_p=pl.col("m_p").cast(pl.Float32), m_n_strong="m_n_strong").with_columns(
                pl.DataFrame(F, schema=[f"m_{f}" for f in MATE_SRC]))
            out.write_parquet(mates_path(split, c), compression="zstd")
            log(f"mates {split} {c}: {band.height:,} band edges, {out.height:,} with a strong mate | {time.time() - t:.0f}s")
            del tab, band, strong, m, R, F, out; gc.collect()


def _l2():
    import xgboost as xgb
    b = xgb.Booster(model_file=str(CACHE / "l2.json")); b.set_param({"device": models.device()})
    return b


def edges_with_l2(split, c, l2, oof2=None):
    from .pipeline import add_l2, p_path
    return add_l2(pl.read_parquet(p_path(split, c)), l2, oof2)


def ce_path(split, c):
    return CACHE / f"ce_{split}_{c}.parquet"


# ------------------------------------------------------------------ stage: cross-encoder on the uncertainty band
def stage_ce():
    from .pipeline import countries, load_parsed, load_raw, offsets
    sys.path.insert(0, str(V3_DIR))
    from v3.ce import CrossEncoder
    import torch
    l2 = _l2()
    oof2 = pl.read_parquet(CACHE / "l2_oof.parquet")
    ce = None
    for split in ["mirror", "test"]:
        todo = [c for c in countries(split) if not ce_path(split, c).exists()]
        if not todo:
            log(f"cache hit: ce {split}"); continue
        s1, s23, _ = load_raw("train" if split == "mirror" else "test")
        o1, o23 = offsets(split)
        for c in todo:
            tab = edges_with_l2(split, c, l2, oof2 if split == "mirror" else None)
            band = tab.filter((pl.col("p_l2") >= BAND[0]) & (pl.col("p_l2") <= BAND[1])).select("i1", "i23")
            a = load_parsed(split, c, "s1", ["idx", "entity_id"]).join(
                s1.select("entity_id", n="business_name", a="business_address"), on="entity_id", how="left").sort("idx")
            b = load_parsed(split, c, "s23", ["idx", "entity_id"]).join(
                s23.select("entity_id", n="business_name", a="business_address"), on="entity_id", how="left").sort("idx")
            i1 = band["i1"].to_numpy().astype(np.int64) - o1[c]
            i23 = band["i23"].to_numpy().astype(np.int64) - o23[c]
            clean = lambda s: "" if s.strip().lower() in {"", "none", "null", "<null>", "n/a", "na", "nan", "-", "--"} else s.strip()
            n1, a1, n2, a2 = a["n"].to_list(), a["a"].to_list(), b["n"].to_list(), b["a"].to_list()
            A = [f"{n1[i]} ; {clean(a1[i])}" for i in i1]
            B = [f"{n2[j]} ; {clean(a2[j])} ; dup 1" for j in i23]
            if ce is None:
                ce = CrossEncoder(str(V3_DIR / "models" / "ce_fold0"))
                ce.model.to(torch.bfloat16)
            t = time.time()
            p = ce.predict(A, B, desc=f"CE {split} {c}")
            band.with_columns(p_ce=pl.Series(p, dtype=pl.Float32)).write_parquet(ce_path(split, c), compression="zstd")
            log(f"CE {split} {c}: {band.height:,} band edges of {tab.height:,} in {time.time() - t:.0f}s")
            del tab, band, a, b, A, B; gc.collect()
        del s1, s23; gc.collect()


# ------------------------------------------------------------------ stage: L3 stacker
def l3_frame(tab, cep, mat):
    ctx = models.context_features(tab.drop("p_l1").rename({"p_l2": "p"}).with_columns(p_l1=tab["p_l1"]))
    ctx = ctx.join(mat, on=["i1", "i23"], how="left").with_columns(
        m_present=pl.col("m_p").is_not_null().cast(pl.Float32))
    ctx = ctx.join(cep, on=["i1", "i23"], how="left").with_columns(
        ce_present=pl.col("p_ce").is_not_null().cast(pl.Float32),
        p_ce=pl.col("p_ce").fill_null(-1.0))
    return ctx.with_columns(ce_minus_p=pl.when(pl.col("ce_present") > 0).then(pl.col("p_ce") - pl.col("p")).otherwise(0.0))


def stage_l3():
    from .pipeline import MIRROR_C
    if (CACHE / "l3.json").exists():
        log("cache hit: l3"); return
    l2, oof2 = _l2(), pl.read_parquet(CACHE / "l2_oof.parquet")
    tab = pl.concat([edges_with_l2("mirror", c, l2, oof2) for c in MIRROR_C], how="diagonal_relaxed")
    cep = pl.concat([pl.read_parquet(ce_path("mirror", c)) for c in MIRROR_C])
    mat = pl.concat([pl.read_parquet(mates_path("mirror", c)) for c in MIRROR_C])
    ctx = l3_frame(tab, cep, mat)
    oof1 = pl.read_parquet(CACHE / "l1_oof.parquet").select("i1", "i23", "y", "fold")
    trn = ctx.filter(pl.col("is_train") & (pl.col("ce_present") > 0)).join(oof1, on=["i1", "i23"])   # band only
    X = trn.select(L3_FEATS).to_numpy().astype(np.float32)
    y, folds = trn["y"].to_numpy(), trn["fold"].to_numpy()
    prm = models.params(monotone=[1 if f in ("p", "p_ce") else 0 for f in L3_FEATS])
    log(f"L3 on {models.device()}: {X.shape}")
    oof, iters, final = models.train_oof(X, y, folds, prm, "L3")
    final.save_model(CACHE / "l3.json")
    trn.select("i1", "i23").with_columns(p_l3=pl.Series(oof)).write_parquet(CACHE / "l3_oof.parquet", compression="zstd")
    b = np.ones(len(y), bool)
    from sklearn.metrics import roc_auc_score
    log(f"L3 iters {iters} | band rows {b.sum():,}: AUC p_l2 {roc_auc_score(y[b], trn['p'].to_numpy()[b]):.5f} "
        f"| p_ce {roc_auc_score(y[b], trn['p_ce'].to_numpy()[b]):.5f} | p_l3 OOF {roc_auc_score(y[b], oof[b]):.5f}")


def add_l3(tab, cep, mat, l3, oof3=None):
    """p_l3 on band edges (those the CE scored); every other edge keeps its V4.1 score p_l2."""
    ctx = l3_frame(tab, cep, mat)
    band = ctx["ce_present"].to_numpy() > 0
    p = ctx["p"].to_numpy().astype(np.float32).copy()
    if band.any():
        p[band] = models.predict(l3, ctx.filter(pl.Series(band)).select(L3_FEATS).to_numpy().astype(np.float32))
    out = ctx.select("i1", "i23").with_columns(p_l3=pl.Series(p))
    out = tab.join(out, on=["i1", "i23"], how="left", maintain_order="left")
    if oof3 is not None:
        out = (out.join(oof3.rename({"p_l3": "p_oof"}), on=["i1", "i23"], how="left")
                  .with_columns(p_l3=pl.coalesce("p_oof", "p_l3")).drop("p_oof"))
    return out


# ------------------------------------------------------------------ stage: decision layer + HOLDOUT report
def stage_decide():
    import xgboost as xgb
    from .pipeline import MIRROR_C, mirror_truth_global, role_global, s1_info, cand_info
    l2, oof2 = _l2(), pl.read_parquet(CACHE / "l2_oof.parquet")
    l3 = xgb.Booster(model_file=str(CACHE / "l3.json")); l3.set_param({"device": models.device()})
    oof3 = pl.read_parquet(CACHE / "l3_oof.parquet")
    tab = pl.concat([add_l3(edges_with_l2("mirror", c, l2, oof2), pl.read_parquet(ce_path("mirror", c)),
                            pl.read_parquet(mates_path("mirror", c)), l3, oof3) for c in MIRROR_C], how="diagonal_relaxed")
    truth, T, H = mirror_truth_global(), role_global("train"), role_global("holdout")
    truth_T = truth.join(T, on="i1", how="semi").select("i1", "i23")
    truth_H = truth.join(H, on="i1", how="semi").select("i1", "i23", "cty")
    oof1 = pl.read_parquet(CACHE / "l1_oof.parquet").select("i1", "i23", "y", "fold")
    tab = tab.join(oof1, on=["i1", "i23"], how="left")
    is_train = tab["is_train"].to_numpy()
    y_edge, f_edge = tab.filter(pl.col("is_train"))["y"].to_numpy(), tab.filter(pl.col("is_train"))["fold"].to_numpy()
    fold_of_i1 = tab.filter(pl.col("is_train")).group_by("i1").agg(pl.col("fold").first())
    hitT = truth_T.join(tab.filter(pl.col("is_train")).select("i1", "i23"), on=["i1", "i23"], how="semi").height
    m_miss = (truth_T.height - hitT) / T.height
    s1i, ci = s1_info("mirror"), cand_info("mirror")
    st = {}
    for var in ["p_l2", "p_l3"]:
        st[var] = decide.fit_decision(tab.select("i1", "i23", p=var), y_edge, is_train, f_edge, truth_T, T["i1"],
                                      fold_of_i1, s1i, ci, m_miss, tag=var)
    best = max(st, key=lambda k: st[k]["f_train"])
    rows = []
    for v in st:
        pred = decide.apply(st[v], tab.select("i1", "i23", p=v), s1i, ci, H["i1"])
        name = ("V4.1 (p_l2)" if v == "p_l2" else "V4.2 (p_l3: + cross-encoder L3)") + (" (chosen)" if v == best else "")
        r = [name, st[v]["f_train"], macro_f05(pred, truth_H.select("i1", "i23"), H["i1"])]
        for c in MIRROR_C:
            hc = H.filter(pl.col("cty") == c)
            r.append(macro_f05(pred.join(hc, on="i1", how="semi"), truth_H.filter(pl.col("cty") == c).select("i1", "i23"), hc["i1"]))
        hp = pred.join(H, on="i1", how="semi")
        r += [hp.height / H.height, 1 - hp["i1"].n_unique() / H.height]
        rows.append(r)
    rep = pl.DataFrame(rows, schema=["model", "train_oof_f05", "holdout_f05", *[f"f05_{c}" for c in MIRROR_C],
                                     "pred_per_s1", "pred_empty"], orient="row")
    log(f"V4.2 HOLDOUT (true matches/S1 {truth_H.height / H.height:.3f}, true empty "
        f"{1 - truth_H['i1'].n_unique() / H.height:.4f}):\n{rep}")
    pickle.dump(dict(st=st[best], variant=best, report=rep), open(CACHE / "decision_v42.pkl", "wb"))


# ------------------------------------------------------------------ stage: export
def stage_export():
    import xgboost as xgb
    from .pipeline import countries, load_parsed, load_raw, offsets, s1_info, cand_info
    d = pickle.load(open(CACHE / "decision_v42.pkl", "rb"))
    st, var = d["st"], d["variant"]
    l2 = _l2()
    l3 = xgb.Booster(model_file=str(CACHE / "l3.json")); l3.set_param({"device": models.device()})
    s1i, ci = s1_info("test"), cand_info("test")
    o1, o23 = offsets("test")
    preds, cands, stats = [], [], []
    for c in countries("test"):
        tab = edges_with_l2("test", c, l2)
        if var == "p_l3":
            tab = add_l3(tab, pl.read_parquet(ce_path("test", c)), pl.read_parquet(mates_path("test", c)), l3)
        pred = decide.apply(st, tab.select("i1", "i23", p=var), s1i, ci)
        a = load_parsed("test", c, "s1", ["idx", "entity_id"]).with_columns(i1=(pl.col("idx") + o1[c]).cast(pl.UInt32))
        b = load_parsed("test", c, "s23", ["idx", "entity_id"]).with_columns(i23=(pl.col("idx") + o23[c]).cast(pl.UInt32))
        ids = lambda e: (e.join(a.select("i1", s1_id="entity_id"), on="i1").join(b.select("i23", s23_id="entity_id"), on="i23")
                          .select("s1_id", "s23_id"))
        preds.append(ids(pred)); cands.append(ids(tab.select("i1", "i23")))
        stats.append((c, a.height, pred.height / a.height, 1 - pred["i1"].n_unique() / a.height))
        del tab; gc.collect()
    log(f"test predictions ({var}):\n{pl.DataFrame(stats, schema=['cty', 'n_s1', 'pred_per_s1', 'pred_empty'], orient='row')}")
    all_pred, all_cand = pl.concat(preds), pl.concat(cands)
    assert all_pred.join(all_cand, on=["s1_id", "s23_id"], how="anti").height == 0
    assert all_pred["s23_id"].is_unique().all()
    order = load_raw("test")[0]["entity_id"]

    def write(pairs, col, path):
        lists = (pairs.unique().sort("s1_id", "s23_id").group_by("s1_id", maintain_order=True)
                      .agg(pl.col("s23_id").str.join(",").alias(col)))
        out = order.to_frame("source1_entity_id").join(lists.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id",
                                                       how="left").fill_null("")
        assert out.height == order.len() and out["source1_entity_id"].is_unique().all()
        path.parent.mkdir(parents=True, exist_ok=True)
        out.write_csv(path, separator="\t", quote_style="never")
        return out

    m = write(all_pred, "matched_entity_ids", OUT42 / "matching_results.tsv")
    write(all_cand, "candidate_entity_ids", OUT42 / "candidate_pairs.tsv")
    log(f"wrote {OUT42 / 'matching_results.tsv'}: {m.height:,} rows, {(m['matched_entity_ids'] != '').sum():,} non-empty, "
        f"{all_pred.height:,} matches")
    if SMOKE:
        log("smoke run: official validator skipped (the smoke test set is a sample)"); return
    r = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"),
                        "--matching", str(OUT42 / "matching_results.tsv"), "--candidate", str(OUT42 / "candidate_pairs.tsv"),
                        "--test-dir", str(DATA / "test"), "--check-ids"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-2000:], r.stderr[-1000:])
    log(f"official validator exit code {r.returncode} -> {'PASS' if r.returncode == 0 else 'FIX ISSUES'}")


STAGES = dict(mates=stage_mates, ce=stage_ce, l3=stage_l3, decide=stage_decide, export=stage_export)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    s = ap.parse_args().stage
    t0 = time.time()
    for name, fn in STAGES.items():
        if s in ("all", name):
            log(f"===== V4.2 stage {name} =====")
            fn()
    log(f"done in {(time.time() - t0) / 60:.1f} min")
