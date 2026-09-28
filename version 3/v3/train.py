"""V3 training pipeline on the modelling sample S (all OOF), plus the refits used at test time.

    python -m v3.train            (V3_TIER=S|M|L, default M)

Stages (each cached in version 3/cache; delete a file or set V3_REFRESH=name to recompute):
  T1  L6/L7b  dense bi-encoder (fine-tuned on sample M) + dense pass on S
  T2  L7b     pool = V2 top-50 U dense top-K (+ reverse); pool features
  T3  L8      distilled student (GPU XGBoost) over the full pool, OOF; adaptive-M pruning -> pruned set
  T4  L7a     companion duplicate model, record clusters, F-CLUS
  T5  L5/L9   Fellegi-Sunter comparison vectors, supervised init + per-country EM (OOF) -> F-EM
  T6  L10     level-1 scorers on the pruned set: XGBoost (GPU), LightGBM, Fellegi-Sunter, cross-encoder (tier M+)
  T7  L9      F-UNC (ensemble uncertainty)
  T8  L11     J2 set transformer (GPU, repeated CV 3x5) vs J3 GBDT stacker, greedy selection
  T9  L12     calibration (monotone GBDT calibrator vs isotonic)
  T10 L13/L14 component decomposition + exact solve; expected-F vs learned policy (repeated CV, paired bootstrap)
"""
import gc, json, pickle, time

import numpy as np
import polars as pl
import torch

from .common import (CFG, CACHE, MODELS, log, cached, v2_cached, frozen_folds, repeated_folds, macro_f05,
                     f05_per_entity, bootstrap_delta, gpu_info)
from . import v2_base as B
from . import pool as PL
from . import clusters as CL
from . import fs_em as EM
from . import decision as DC
from . import stack as SK

LEDGER = []
LEDGER_PATH = CACHE.parent / "ledger_train.tsv"


def ledger(step, value, note=""):
    LEDGER.append((step, float(value), note))
    log(f"LEDGER | {step}: {value:.5f} {note}")
    pl.DataFrame(LEDGER, schema=["step", "value", "note"], orient="row").write_csv(LEDGER_PATH, separator="\t")


def simple_decision_f(i1, i23, p, y, ntrue, thr):
    """exclusivity + global threshold (the V2-M1 rule): a common yardstick for comparing score columns."""
    d = pl.DataFrame({"i1": i1, "i23": i23, "p": p, "row": np.arange(len(p))})
    best = d.filter(pl.col("p") == pl.col("p").max().over("i23")).unique("i23")
    pred = np.zeros(len(p), bool)
    pred[best.filter(pl.col("p") >= thr)["row"].to_numpy()] = True
    return macro_f05(i1, pred, y, ntrue)


def best_thr(i1, i23, p, y, ntrue, grid=np.arange(0.3, 0.91, 0.05)):
    return max(((float(t), simple_decision_f(i1, i23, p, y, ntrue, t)) for t in grid), key=lambda r: r[1])


def main():
    t_all = time.time()
    log(f"V3 train | tier {CFG['tier']} | GPU {gpu_info()}")
    from .data import load_train_sample
    from .stage_retrieval import get_encoder, dense_candidates
    from sklearn.metrics import roc_auc_score, log_loss
    S = load_train_sample("S")
    s1p, s23p, truth = S["s1p"], S["s23p"], S["truth"]
    n1 = s1p.height
    ntrue = np.bincount(truth["i1"].to_numpy(), minlength=n1).astype(np.float64)
    fold1 = frozen_folds(n1, CFG["folds"], CFG["seed"])
    cty1 = s1p["cty"].to_numpy()
    rfolds = repeated_folds(n1, *CFG["rcv"])

    # ------------------------------------------------------------------ T1 dense retrieval
    enc = get_encoder()
    dense = dense_candidates(S, enc)
    e23 = np.load(CACHE / "S_emb23.npy")
    del enc; gc.collect(); torch.cuda.empty_cache()

    # ------------------------------------------------------------------ T2 pool
    v2cand = v2_cached("S_cand.parquet")
    pool = cached("S_pool.parquet", lambda: PL.build_pool(v2cand, dense))
    del v2cand
    y_pool = pool.select("i1", "i23").join(truth.with_columns(y=pl.lit(1, pl.Int8)), on=["i1", "i23"],
                                           how="left")["y"].fill_null(0).to_numpy()

    def _poolX():
        X_v2, _ = v2_cached("S_features.pkl")
        return PL.pool_features(pool, S, X_v2)
    Xpool = cached("S_poolX.npy", _poolX)
    gc.collect()
    i1_pool, i23_pool = pool["i1"].to_numpy(), pool["i23"].to_numpy()
    n_v2 = int(pool["in_v2"].sum())
    ledger("V2 blocking pair completeness (top-50)", y_pool[:n_v2].sum() / truth.height)
    ledger("L7b pool pair completeness (V2 + dense)", y_pool.sum() / truth.height, f"{pool.height / n1:.1f} pairs/S1")
    ledger("oracle macro F0.5 on the pool", macro_f05(i1_pool, y_pool.astype(bool), y_pool, ntrue))
    fold_pool = fold1[i1_pool]

    # ------------------------------------------------------------------ T3 student + adaptive-M pruning
    feats_s = PL.STUDENT_FEATS

    def _student():
        oof, _, iters = PL.xgb_oof(Xpool, y_pool, fold_pool, feats_s, desc="L8 student", monotone=SK.MONO)
        final = PL.xgb_fit_full(Xpool, y_pool, feats_s, int(np.mean(iters) * 1.1) + 1, monotone=SK.MONO)
        final.save_model(str(MODELS / "student.json"))
        return oof
    p_stu = cached("S_student.npy", _student)
    thr, f = best_thr(i1_pool, i23_pool, p_stu, y_pool, ntrue)
    ledger("L8 student (full pool) + exclusivity + global thr", f, f"thr={thr:.2f}")
    keep = PL.adaptive_m(i1_pool, p_stu)
    ledger("L8 pruning recall vs pool", y_pool[keep].sum() / max(1, y_pool.sum()), f"avg M {keep.sum() / n1:.2f}")
    ledger("pruned-set pair completeness", y_pool[keep].sum() / truth.height)
    rows = np.where(keep)[0]
    pp = pool[rows].select("i1", "i23")
    Xp = Xpool[rows]
    del Xpool; gc.collect()
    y = y_pool[rows]
    i1, i23 = pp["i1"].to_numpy(), pp["i23"].to_numpy()
    ps = p_stu[rows]
    fold_p = fold1[i1]
    cty_p = cty1[i1]
    m_miss = (truth.height - y.sum()) / n1
    ledger("oracle macro F0.5 on the pruned set", macro_f05(i1, y.astype(bool), y, ntrue), f"m_miss {m_miss:.4f}")

    # ------------------------------------------------------------------ T4 companion + clusters + F-CLUS
    sc = pl.DataFrame({"i1": i1, "i23": i23, "p": ps})

    def _companion():
        rp = CL.record_pairs(sc)
        uniq = rp.select("x", "y").unique()
        Xc = CL.companion_matrix(uniq, s23p, e23)
        lab_u = CL.companion_labels(uniq, truth)
        fu = uniq.join(rp.group_by("x", "y").agg(pl.col("i1").min()), on=["x", "y"], how="left")["i1"].to_numpy()
        oof, _, iters = PL.xgb_oof(Xc, lab_u, fold1[fu], CL.COMP_FEATS, desc="L7a companion")
        final = PL.xgb_fit_full(Xc, lab_u, CL.COMP_FEATS, int(np.mean(iters) * 1.1) + 1)
        final.save_model(str(MODELS / "companion.json"))
        links = rp.join(uniq.with_columns(p_link=pl.Series(oof)), on=["x", "y"]).select("i1", "x", "y", "p_link")
        return links, oof, lab_u
    links, comp_oof, comp_lab = cached("S_companion.pkl", _companion)
    ledger("L7a companion OOF AUC", roc_auc_score(comp_lab, comp_oof), f"{len(comp_lab):,} record pairs, pos {comp_lab.mean():.3f}")
    lab = CL.clusters(links.select("x", "y", "p_link"), s23p.height)
    own = np.full(s23p.height, -1, np.int64)
    own[truth["i23"].to_numpy()] = truth["i1"].to_numpy()
    _, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
    mm = cnt[inv] > 1
    multi = pl.DataFrame({"cl": inv[mm], "own": own[mm]}).group_by("cl").agg(
        pure=(pl.col("own").min() >= 0) & (pl.col("own").n_unique() == 1))
    tr_cl = pl.DataFrame({"i1": truth["i1"], "cl": inv[truth["i23"].to_numpy()]})
    compl = tr_cl.group_by("i1").agg(nc=pl.col("cl").n_unique(), n=pl.len()).filter(pl.col("n") >= 2)
    ledger("L7a cluster purity (multi-record clusters)", multi["pure"].mean() if multi.height else 1.0,
           f"{multi.height:,} clusters, {mm.sum():,} records")
    ledger("L7a cluster completeness (S1 whose matches share one cluster)", (compl["nc"] == 1).mean())
    Fclus = CL.clus_features(pp, ps, Xp, feats_s, lab, links, s23p)

    # ------------------------------------------------------------------ T5 Fellegi-Sunter (EM per country)
    G = EM.comparison_vectors(Xp, feats_s)
    Fem = cached("S_em.npy", lambda: EM.em_features_oof(G, y, cty_p, fold_p))
    with open(MODELS / "fs_train.pkl", "wb") as fh:
        pickle.dump((G, y), fh)
    ledger("L10 Fellegi-Sunter standalone (EM per country) + excl + thr", best_thr(i1, i23, Fem[:, -1], y, ntrue)[1])

    # ------------------------------------------------------------------ T6 level-1 scorers
    X1 = SK.level1_matrix(Xp, Fem, Fclus, ps, i1)
    del Xp; gc.collect()
    L1F = SK.L1F

    def _l1_xgb():
        oof, _, iters = PL.xgb_oof(X1, y, fold_p, L1F, desc="L10 xgb", monotone=SK.MONO1)
        PL.xgb_fit_full(X1, y, L1F, int(np.mean(iters) * 1.1) + 1, monotone=SK.MONO1).save_model(str(MODELS / "l1_xgb.json"))
        return oof
    p_xgb = cached("S_l1_xgb.npy", _l1_xgb)
    ledger("L10 XGBoost (GPU) + excl + thr", best_thr(i1, i23, p_xgb, y, ntrue)[1])

    def _l1_lgb():
        import lightgbm as lgb
        params = dict(B.CFG["lgb"], learning_rate=0.05, num_leaves=127, feature_fraction=0.5, seed=1)
        params["monotone_constraints"] = [SK.MONO1.get(f, 0) for f in L1F]
        full = lgb.Dataset(X1, y, feature_name=L1F, free_raw_data=False).construct()
        oof = np.zeros(len(y), np.float32); iters = []
        for k in range(CFG["folds"]):
            tr, va = np.where(fold_p != k)[0], np.where(fold_p == k)[0]
            m = lgb.train(params, full.subset(tr), 3000, valid_sets=[full.subset(va)],
                          callbacks=[lgb.early_stopping(60, verbose=False)])
            oof[va] = m.predict(X1[va], num_iteration=m.best_iteration); iters.append(m.best_iteration)
            log(f"L10 lgb fold {k}: best_iter {m.best_iteration}")
        lgb.train(params, full, int(np.mean(iters) * 1.1) + 1).save_model(str(MODELS / "l1_lgb.txt"))
        return oof
    p_lgb = cached("S_l1_lgb.npy", _l1_lgb)
    ledger("L10 LightGBM + excl + thr", best_thr(i1, i23, p_lgb, y, ntrue)[1])

    p_ce = np.full(len(y), np.nan, np.float32)
    if CFG["tier"] in ("M", "L"):
        from . import ce as CE
        band = CE.ce_band_mask(ps)
        log(f"CE band: {band.sum():,} of {len(y):,} pruned pairs ({band.mean():.1%})")

        def _ce():
            A, Bt = CE.serialize(S["raw1"], S["raw23"], i1, i23, Fclus[:, CL.CLUS_FEATS.index("cl_size")])
            oof, _ = CE.ce_oof(A, Bt, y, fold_p % CFG["ce_folds"], band, p_xgb)
            return oof
        p_ce = cached("S_l1_ce.npy", _ce)
        b = np.isfinite(p_ce)
        if b.any():
            ledger("L10 cross-encoder AUC inside band", roc_auc_score(y[b], p_ce[b]),
                   f"(xgb AUC in band {roc_auc_score(y[b], p_xgb[b]):.4f})")

    # ------------------------------------------------------------------ T7 F-UNC
    Func = SK.unc_feats(p_xgb, p_lgb, Fem[:, -1], ps, p_ce)

    # ------------------------------------------------------------------ T8 joint layer
    from . import settx as ST
    T = SK.token_matrix(p_xgb, p_lgb, ps, p_ce, Func, Fem, Fclus, X1)
    lrows = SK.link_rows(links, i1, i23)

    def _j2():
        std = ST.Standardizer().fit(T)
        with open(MODELS / "st_std.pkl", "wb") as fh:
            pickle.dump(std, fh)
        pack = SK.st_pack(T, std, i1, p_xgb, lrows, y)
        nt = ntrue[pack["list_i1"]]
        P, N = ST.st_repeated_cv(pack, nt, 0, [f[pack["list_i1"]] for f in rfolds])
        p_j2 = np.zeros(len(y), np.float32)
        p_j2[pack["prow"][pack["mask"]]] = P[pack["mask"]]
        pnull = np.ones(n1, np.float32)
        pnull[pack["list_i1"]] = N
        for s in range(2):                                    # full refits for test (2 seeds, averaged)
            m = ST.train_st(pack, nt, 0, np.arange(len(pack["list_i1"])), seed=77 + s)
            torch.save(m.state_dict(), MODELS / f"st_full{s}.pt")
        return p_j2, pnull
    p_j2, pnull = cached("S_j2.pkl", _j2)
    f2 = best_thr(i1, i23, p_j2, y, ntrue)[1]
    ledger("L11 J2 set transformer (repeated CV 3x5) + excl + thr", f2)

    ctx = SK.list_rank_feats(i1, p_xgb)
    J3F = SK.STK_FEATS + ["x_rank", "x_gap", "x_lsum", "x_n"]
    j3p = dict(CFG["xgb"], max_depth=6, learning_rate=0.05)

    def _j3():
        Xs = np.column_stack([T, ctx]).astype(np.float32)
        out = np.zeros(len(y), np.float32)
        for r, f in enumerate(rfolds):
            o, _, iters = PL.xgb_oof(Xs, y, f[i1], J3F, params=j3p, desc=f"J3 stacker rep {r}")
            out += o / len(rfolds)
        PL.xgb_fit_full(Xs, y, J3F, int(np.mean(iters) * 1.1) + 1, params=j3p).save_model(str(MODELS / "j3_stacker.json"))
        return out
    p_j3 = cached("S_j3.npy", _j3)
    f3 = best_thr(i1, i23, p_j3, y, ntrue)[1]
    ledger("L11 J3 GBDT stacker (repeated CV) + excl + thr", f3)
    # greedy ensemble selection over the joint outputs (with replacement, weights 0..1 in steps of 1/4)
    best = None
    for w in (0.0, 0.25, 0.5, 0.75, 1.0):
        pj = w * p_j2 + (1 - w) * p_j3
        fj = best_thr(i1, i23, pj, y, ntrue)[1]
        if best is None or fj > best[1] + 1e-5:
            best = (w, fj)
    w_j2 = best[0]
    p_joint = w_j2 * p_j2 + (1 - w_j2) * p_j3
    ledger("L11 joint layer: ensemble weight on J2", best[1], f"w_J2={w_j2} (logloss J2 "
           f"{log_loss(y, np.clip(p_j2, 1e-6, 1 - 1e-6)):.4f}, J3 {log_loss(y, np.clip(p_j3, 1e-6, 1 - 1e-6)):.4f})")

    # ------------------------------------------------------------------ T9 calibration
    cl_size = Fclus[:, CL.CLUS_FEATS.index("cl_size")]
    aux = np.column_stack([Func[:, 1], ctx[:, 3], cl_size, pnull[i1]]).astype(np.float32)
    p_cal = np.zeros(len(y), np.float32)
    for f in rfolds:
        p_cal += DC.calibrate_oof(p_joint, aux, y, f[i1]) / len(rfolds)
    from sklearn.isotonic import IsotonicRegression
    p_iso = np.zeros(len(y), np.float32)
    for k in range(CFG["rcv"][1]):
        tr, va = rfolds[0][i1] != k, rfolds[0][i1] == k
        p_iso[va] = IsotonicRegression(out_of_bounds="clip").fit(p_joint[tr], y[tr]).predict(p_joint[va])
    e_cal, e_iso, e_raw = DC.ece(p_cal, y), DC.ece(p_iso, y), DC.ece(p_joint, y)
    use_gbdt_cal = bool(e_cal <= e_iso)
    p_final = p_cal if use_gbdt_cal else p_iso
    p_final = np.clip(p_final, 1e-6, 1 - 1e-6)
    ledger("L12 ECE of the monotone GBDT calibrator", e_cal,
           f"raw {e_raw:.4f} isotonic {e_iso:.4f} -> {'GBDT' if use_gbdt_cal else 'isotonic'}")
    DC.fit_calibrator(p_joint, aux, y).save_model(str(MODELS / "calibrator.json"))
    with open(MODELS / "iso.pkl", "wb") as fh:
        pickle.dump(IsotonicRegression(out_of_bounds="clip").fit(p_joint, y), fh)

    # ------------------------------------------------------------------ T10 L13 + L14
    K = CFG["policy_K"]
    allowed, csize = DC.solve_components(i1, i23, p_final, m_miss)
    Plist, Rlist, rest = DC.build_lists(i1, p_final, allowed, n1, K)
    EF = DC.ef_table(None, Plist, rest, m_miss, K)
    sel_rows = lambda R, k: np.isin(np.arange(len(y)), DC.select_topk(R, k))
    f_ef = f05_per_entity(i1, sel_rows(Rlist, EF.argmax(1)), y.astype(bool), ntrue, n1)
    ledger("L13 exact components + expected-F0.5", f_ef.mean())
    ex = pl.DataFrame({"i1": i1, "i23": i23, "p": p_final, "row": np.arange(len(y))})
    exr = ex.filter(pl.col("p") == pl.col("p").max().over("i23")).unique("i23")["row"].to_numpy()
    allowed_x = np.zeros(len(y), bool); allowed_x[exr] = True
    Px, Rx, rx = DC.build_lists(i1, p_final, allowed_x & (p_final >= CFG["edge_floor"]), n1, K)
    f_efx = f05_per_entity(i1, sel_rows(Rx, DC.ef_table(None, Px, rx, m_miss, K).argmax(1)), y.astype(bool), ntrue, n1)
    d, lo, hi = bootstrap_delta(f_ef, f_efx)
    use_l13 = bool(lo > 0 or (hi >= 0 and d >= 0))             # adoption rule: exact solve kept only if not worse
    ledger("  (reference) V2 exclusivity + expected-F0.5", f_efx.mean(),
           f"L13 delta {d:+.5f} CI [{lo:+.5f}, {hi:+.5f}] -> {'exact L13' if use_l13 else 'plain exclusivity'}")
    if not use_l13:
        allowed = allowed_x & (p_final >= CFG["edge_floor"])
        Plist, Rlist, rest = Px, Rx, rx
        EF = DC.ef_table(None, Plist, rest, m_miss, K)
        f_ef = f_efx

    top = Rlist[:, 0]
    has = top >= 0
    unc_top = np.zeros((n1, 2), np.float32)
    unc_top[has, 0], unc_top[has, 1] = Func[top[has], 1], Func[top[has], 0]
    cl_top = np.where(has, cl_size[np.maximum(top, 0)], 0)
    comp_top = np.where(has, csize[np.maximum(top, 0)], 1)
    Xpol = DC.policy_matrix(Plist, EF, rest, pnull, unc_top, cl_top, comp_top, K)
    C, valid = DC.true_cost_table(Rlist, y, ntrue, K)
    valid = valid & (EF >= 0)
    fs_pol = []
    for fo in repeated_folds(n1, *CFG["rcv"], seed=500):
        kk = np.zeros(n1, np.int64)
        for k in range(CFG["rcv"][1]):
            tr, va = fo != k, fo == k
            kk[va] = DC.policy_choose(DC.fit_policy(Xpol[tr], C[tr], valid[tr]), Xpol[va], valid[va])
        fs_pol.append(f05_per_entity(i1, sel_rows(Rlist, kk), y.astype(bool), ntrue, n1))
    f_pol = np.mean(fs_pol, 0)
    d, lo, hi = bootstrap_delta(f_pol, f_ef)
    ledger("L14 learned cost-sensitive policy (repeated CV)", f_pol.mean(),
           f"delta vs expected-F {d:+.5f} CI [{lo:+.5f}, {hi:+.5f}] repeats {[round(float(x.mean()), 5) for x in fs_pol]}")
    use_policy = bool(lo > 0)
    DC.fit_policy(Xpol, C, valid).save_model(str(MODELS / "policy.json"))
    final_f = f_pol if use_policy else f_ef
    ledger(f"FINAL V3 OOF macro F0.5 ({'policy' if use_policy else 'expected-F'})", final_f.mean())
    for c in np.unique(cty1):
        ledger(f"  country={c}", final_f[cty1 == c].mean())
    sing = ntrue == 0
    ledger("  singletons", final_f[sing].mean())
    ledger("  non-singletons", final_f[~sing].mean())

    meta = dict(tier=CFG["tier"], w_j2=w_j2, use_l13=use_l13, use_gbdt_cal=use_gbdt_cal, use_policy=use_policy, m_miss=float(m_miss),
                ce_used=bool(np.isfinite(p_ce).any()), minutes=(time.time() - t_all) / 60)
    with open(MODELS / "meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    log(f"V3 train done in {(time.time() - t_all) / 60:.1f} min | meta {meta}")


if __name__ == "__main__":
    main()
