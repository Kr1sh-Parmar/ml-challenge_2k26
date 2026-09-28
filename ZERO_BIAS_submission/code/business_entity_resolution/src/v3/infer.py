"""V3 test inference (L15 output), one country label at a time, with the models saved by v3.train.

    python -m v3.infer            (V3_TIER=S|M; tier must not exceed the trained tier)

Per country: V2 parse + blocking -> dense pass -> pool -> (chunked by S1) pool features + student + adaptive-M pruning
-> companion links + clusters + F-CLUS -> Fellegi-Sunter EM on this country's own pairs -> level-1 (XGB, LGB, CE band)
-> F-UNC -> J2 + J3 -> calibration -> L13 components -> L14 decision -> matches.
candidate_pairs.tsv = the pruned set the matching models score; matches are always a subset of it.
"""
import gc, json, pickle, subprocess, sys, time

import numpy as np
import polars as pl
import torch

from .common import CFG, CACHE, MODELS, OUT, DATA, VALIDATOR, log, cached, gpu_info
from . import v2_base as B
from . import pool as PL
from . import clusters as CL
from . import fs_em as EM
from . import decision as DC
from . import stack as SK


def load_models():
    import xgboost as xgb
    import lightgbm as lgb
    M = {}
    for name in ["student", "companion", "l1_xgb", "j3_stacker", "calibrator", "policy"]:
        b = xgb.Booster(); b.load_model(str(MODELS / f"{name}.json")); M[name] = b
    M["l1_lgb"] = lgb.Booster(model_file=str(MODELS / "l1_lgb.txt"))
    with open(MODELS / "meta.json") as f:
        M["meta"] = json.load(f)
    with open(MODELS / "fs_train.pkl", "rb") as f:
        M["fs_train"] = pickle.load(f)
    with open(MODELS / "st_std.pkl", "rb") as f:
        M["st_std"] = pickle.load(f)
    with open(MODELS / "iso.pkl", "rb") as f:
        M["iso"] = pickle.load(f)
    return M


def xpred(booster, X, feats):
    import xgboost as xgb
    out = np.empty(X.shape[0], np.float32)
    for lo in range(0, X.shape[0], 2_000_000):
        out[lo:lo + 2_000_000] = booster.predict(xgb.DMatrix(X[lo:lo + 2_000_000], feature_names=feats))
    return out


def pruned_pool(d, pool, M, chunk_pairs=8_000_000):
    """Features for the whole pool in S1-range chunks, student scores, adaptive-M pruning. Keeps only pruned rows."""
    pool = pool.sort("i1")
    i1_all = pool["i1"].to_numpy()
    bounds = [0]
    while bounds[-1] < len(i1_all):
        j = min(len(i1_all), bounds[-1] + chunk_pairs)
        if j < len(i1_all):                       # never split one S1's list across chunks
            j = int(np.searchsorted(i1_all, i1_all[j], side="left"))
        bounds.append(j)
    keep_rows, Xk, pk = [], [], []
    for b0, b1 in zip(bounds[:-1], bounds[1:]):
        part = pool.slice(b0, b1 - b0)
        X = PL.pool_features(part, d)
        p = xpred(M["student"], X, PL.STUDENT_FEATS)
        k = PL.adaptive_m(part["i1"].to_numpy(), p)
        keep_rows.append(np.arange(b0, b1)[k]); Xk.append(X[k]); pk.append(p[k])
        log(f"[{d['name']}] pool chunk {b0:,}-{b1:,}: kept {k.sum():,} of {len(k):,}")
        del X; gc.collect()
    rows = np.concatenate(keep_rows)
    return pool[rows].select("i1", "i23"), np.concatenate(Xk), np.concatenate(pk)


def run_country(c, M, enc):
    from .data import load_test_country
    from .stage_retrieval import dense_candidates
    meta = M["meta"]
    t0 = time.time()
    d = load_test_country(c)
    s1p, s23p = d["s1p"], d["s23p"]
    n1 = s1p.height
    tag = d["name"]
    if s23p.height == 0:
        log(f"{c}: no S2/S3 records -> every S1 predicted empty")
        empty = pl.DataFrame(schema={"i1": pl.UInt32, "i23": pl.UInt32})
        return d, empty, empty, {}
    dense = dense_candidates(d, enc)
    pool = PL.build_pool(d["cand"], dense)
    log(f"[{tag}] pool {pool.height:,} pairs ({pool.height / n1:.1f}/S1)")

    while not (CACHE / f"{tag}_pruned.pkl").exists() and (CACHE.parent / "logs" / "test_prune.log").exists()             and "test prune done" not in (CACHE.parent / "logs" / "test_prune.log").read_text(errors="ignore"):
        time.sleep(30)                                      # pruned set is being produced by stage_test_prune

    def _pruned():
        pp, Xp, ps = pruned_pool(d, pool, M)
        return pp, Xp, ps
    pp, Xp, ps = cached(f"{tag}_pruned.pkl", _pruned)
    i1, i23 = pp["i1"].to_numpy(), pp["i23"].to_numpy()
    log(f"[{tag}] pruned set {len(i1):,} pairs ({len(i1) / n1:.2f}/S1)")

    # L7a companion + clusters
    e23 = np.load(CACHE / f"{tag}_emb23.npy", mmap_mode="r")
    sc = pl.DataFrame({"i1": i1, "i23": i23, "p": ps})

    def _links():
        rp = CL.record_pairs(sc)
        uniq = rp.select("x", "y").unique()
        Xc = CL.companion_matrix(uniq, s23p, e23)
        pu = xpred(M["companion"], Xc, CL.COMP_FEATS)
        return rp.join(uniq.with_columns(p_link=pl.Series(pu)), on=["x", "y"]).select("i1", "x", "y", "p_link")
    links = cached(f"{tag}_links.parquet", _links)
    lab = CL.clusters(links.select("x", "y", "p_link"), s23p.height)
    Fclus = CL.clus_features(pp, ps, Xp, PL.STUDENT_FEATS, lab, links, s23p)

    # L5 Fellegi-Sunter: supervised init from train, EM on this country's own unlabeled pairs
    G = EM.comparison_vectors(Xp, PL.STUDENT_FEATS)
    Gtr, ytr = M["fs_train"]
    Fem, fs_tables = EM.em_features_test(G, s1p["cty"].to_numpy()[i1], Gtr, ytr)

    # L10 level-1
    X1 = SK.level1_matrix(Xp, Fem, Fclus, ps, i1)
    del Xp; gc.collect()
    p_xgb = xpred(M["l1_xgb"], X1, SK.L1F)
    p_lgb = M["l1_lgb"].predict(X1).astype(np.float32)
    p_ce = np.full(len(i1), np.nan, np.float32)
    if meta.get("ce_used") and CFG["tier"] in ("M", "L"):
        from . import ce as CE
        band = CE.ce_band_mask(ps)
        paths = sorted(MODELS.glob("ce_fold*"))[:CFG["ce_test_models"]]

        def _ce():
            out = np.full(len(i1), np.nan, np.float32)
            bi = np.where(band)[0]
            A, Bt = CE.serialize(d["raw1"], d["raw23"], i1[bi], i23[bi], Fclus[bi, CL.CLUS_FEATS.index("cl_size")])
            out[bi] = CE.ce_predict_bag(A, Bt, paths)
            return out
        log(f"[{tag}] CE band {band.sum():,} pairs")
        p_ce = cached(f"{tag}_ce.npy", _ce)

    # L9 F-UNC, L11 joint
    from . import settx as ST
    Func = SK.unc_feats(p_xgb, p_lgb, Fem[:, -1], ps, p_ce)
    T = SK.token_matrix(p_xgb, p_lgb, ps, p_ce, Func, Fem, Fclus, X1)
    lrows = SK.link_rows(links, i1, i23)
    pack = SK.st_pack(T, M["st_std"], i1, p_xgb, lrows)
    idx = np.arange(len(pack["list_i1"]))
    P = np.zeros(pack["mask"].shape, np.float32)
    N = np.zeros(len(idx), np.float32)
    seeds = sorted(MODELS.glob("st_full*.pt"))
    lstat_dim = 5
    for pth in seeds:
        m = ST.SetTransformer(T.shape[1], lstat_dim).cuda()
        m.load_state_dict(torch.load(pth, map_location="cuda"))
        m.eval()
        p_, n_ = ST.predict_st(m, pack, 0, idx)
        P += p_ / len(seeds); N += n_ / len(seeds)
    p_j2 = np.zeros(len(i1), np.float32)
    p_j2[pack["prow"][pack["mask"]]] = P[pack["mask"]]
    pnull = np.ones(n1, np.float32)
    pnull[pack["list_i1"]] = N
    ctx = SK.list_rank_feats(i1, p_xgb)
    p_j3 = xpred(M["j3_stacker"], np.column_stack([T, ctx]).astype(np.float32), SK.STK_FEATS + ["x_rank", "x_gap", "x_lsum", "x_n"])
    w = meta["w_j2"]
    p_joint = w * p_j2 + (1 - w) * p_j3

    # L12 calibration (+ France prior-shift monitor, logged)
    cl_size = Fclus[:, CL.CLUS_FEATS.index("cl_size")]
    aux = np.column_stack([Func[:, 1], ctx[:, 3], cl_size, pnull[i1]]).astype(np.float32)
    p_final = DC.apply_calibrator(M["calibrator"], p_joint, aux) if meta["use_gbdt_cal"] else \
        M["iso"].predict(p_joint).astype(np.float32)
    p_final = np.clip(p_final, 1e-6, 1 - 1e-6)

    # L13 + L14
    K = CFG["policy_K"]
    m_miss = meta["m_miss"]
    if meta.get("use_l13", True):
        allowed, csize = DC.solve_components(i1, i23, p_final, m_miss)
    else:                                                # adoption: plain exclusivity (each record to its best S1)
        csize = DC.component_sizes(i1, i23, p_final)
        ex = pl.DataFrame({"i23": i23, "p": p_final, "row": np.arange(len(i1))})
        exr = ex.filter(pl.col("p") == pl.col("p").max().over("i23")).unique("i23")["row"].to_numpy()
        allowed = np.zeros(len(i1), bool); allowed[exr] = True
        allowed &= p_final >= CFG["edge_floor"]
    Plist, Rlist, rest = DC.build_lists(i1, p_final, allowed, n1, K)
    EF = DC.ef_table(None, Plist, rest, m_miss, K)
    if meta["use_policy"]:
        top = Rlist[:, 0]; has = top >= 0
        unc_top = np.zeros((n1, 2), np.float32)
        unc_top[has, 0], unc_top[has, 1] = Func[top[has], 1], Func[top[has], 0]
        cl_top = np.where(has, cl_size[np.maximum(top, 0)], 0)
        comp_top = np.where(has, csize[np.maximum(top, 0)], 1)
        Xpol = DC.policy_matrix(Plist, EF, rest, pnull, unc_top, cl_top, comp_top, K)
        valid = np.concatenate([np.ones((n1, 1), bool), Rlist >= 0], 1) & (EF >= 0)
        kstar = DC.policy_choose(M["policy"], Xpol, valid)
    else:
        kstar = EF.argmax(1)
    sel = DC.select_topk(Rlist, kstar)
    pred = pl.DataFrame({"i1": i1[sel], "i23": i23[sel]})
    cand = pp.select("i1", "i23")
    stats = dict(country=c, n_s1=n1, n_s23=s23p.height, pool_per_s1=pool.height / n1, cands_per_s1=len(i1) / n1,
                 predicted_empty=float(1 - pred["i1"].n_unique() / n1), mean_matches=pred.height / n1,
                 best_p_median=float(np.median(Plist[:, 0])), fs_pi=float(list(fs_tables.values())[0][2]),
                 minutes=(time.time() - t0) / 60)
    log(f"[{tag}] done: {stats}")
    return d, cand, pred, stats


def write_tsv(pairs, col, path, s1_order):
    lists = (pairs.unique(["s1_id", "s23_id"]).sort("s1_id", "s23_id").group_by("s1_id", maintain_order=True)
                  .agg(pl.col("s23_id").str.join(",").alias(col)))
    out = s1_order.to_frame("source1_entity_id").join(lists.rename({"s1_id": "source1_entity_id"}),
                                                       on="source1_entity_id", how="left").fill_null("")
    assert out.height == s1_order.len() and out["source1_entity_id"].is_unique().all()
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never")
    return out


def main():
    log(f"V3 inference | tier {CFG['tier']} | GPU {gpu_info()}")
    from .data import test_raw, test_countries
    from .stage_retrieval import get_encoder
    M = load_models()
    enc = get_encoder()
    te1, _ = test_raw()
    allc, allp, mon = [], [], []
    for c in test_countries():
        log(f"===== test country: {c} =====")
        d, cand, pred, stats = run_country(c, M, enc)
        ids1, ids23 = d["s1p"]["entity_id"], d["s23p"]["entity_id"]
        to_ids = lambda x: x.select(s1_id=ids1[x["i1"].to_numpy()], s23_id=ids23[x["i23"].to_numpy()]) \
            if x.height else pl.DataFrame(schema={"s1_id": pl.String, "s23_id": pl.String})
        allc.append(to_ids(cand)); allp.append(to_ids(pred)); mon.append(stats)
        del d; gc.collect(); torch.cuda.empty_cache()
    all_cand, all_pred = pl.concat(allc), pl.concat(allp)
    assert all_pred.join(all_cand, on=["s1_id", "s23_id"], how="anti").height == 0, "matches must be a subset of candidates"
    assert all_pred["s23_id"].is_unique().all(), "an S2/S3 record may belong to at most one S1"
    order = te1["entity_id"]
    m_out = write_tsv(all_pred, "matched_entity_ids", OUT / "matching_results.tsv", order)
    write_tsv(all_cand, "candidate_entity_ids", OUT / "candidate_pairs.tsv", order)
    print(pl.DataFrame(mon))
    pl.DataFrame(mon).write_csv(OUT / "shift_monitor.tsv", separator="\t")
    log(f"wrote {OUT / 'matching_results.tsv'}: {m_out.height:,} rows, {(m_out['matched_entity_ids'] != '').sum():,} non-empty")
    r = subprocess.run([sys.executable, str(VALIDATOR),
                        "--matching", str(OUT / "matching_results.tsv"), "--candidate", str(OUT / "candidate_pairs.tsv"),
                        "--test-dir", str(DATA / "test")], capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-3000:], r.stderr[-1500:])
    log(f"official validator exit code {r.returncode} -> {'PASS' if r.returncode == 0 else 'FIX ISSUES'}")


if __name__ == "__main__":
    main()
