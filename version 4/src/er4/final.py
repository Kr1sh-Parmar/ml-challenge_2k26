"""V4.3 FINAL = V4.1 + V4.2 + everything planned for V4.3 / V4.4, measured on the same test-mirror HOLDOUT.

Run from version 4/src with ER4_DENSE=1 (reads V4.1's cache_dense/ and V4.2's mates files, writes output_final/):
    python -m er4.final --stage pairx|em|ce43|l43a|mates2|l43b|decide|export|all

Layers added on top of V4.2 (all on the uncertain band 0.003 <= p_l2 <= 0.997; every other edge keeps V4.1's score):
  pairx   full V4 pair features (x vs S1) for every band edge, so the stacker sees raw evidence, not only scores.
  em      [V4.4 France] Fellegi-Sunter comparison vectors; m/u initialised supervised on mirror TRAIN, then re-estimated by
          EM separately per country on that country's own unlabeled band pairs (France adapts without labels).
          Mirror TRAIN rows use fold-wise init (OOF).
  ce43    [V4.3] the mDeBERTa cross-encoder is RE-TRAINED on mirror TRAIN band pairs (test-like hard pairs), warm-started
          from V3's model; 2 folds by S1 (OOF on TRAIN); fold-0 model scores every other band edge.
  l43a    L3 stacker (GPU XGBoost): L2 context + p_l1 + CE + evidence-pooling mates + pair features + EM; OOF on TRAIN;
          LOCO report (US <-> India, the France proxy).
  mates2  [V4.3] second evidence-pooling round: strongest mate chosen with the improved p_43a.
  l43b    second stacker round on p_43a context + mates2.
  decide  V4 decision layer fitted on TRAIN for p_l2 (V4.1), p_43a, p_43b; best by TRAIN OOF; then [V4.3] V3's learned
          cost-sensitive policy (k* = argmin predicted 1 - F0.5) stacked on the chosen rule, adopted only if it beats the
          rule on TRAIN (paired bootstrap CI > 0). HOLDOUT report for everything.
  export  test predictions -> output_final/ + official validator --check-ids.
"""
import argparse, gc, pickle, subprocess, sys, time

import numpy as np
import polars as pl

from . import decide, models
from . import stack3 as S3
from .config import CACHE, CFG, DATA, MODELS, ROOT, SMOKE, V4
from .features import V4_FEATS, V4_IDX, build_features
from .util import log, macro_f05

V3_DIR = V4.parent / "version 3"
if str(V3_DIR) not in sys.path:
    sys.path.insert(0, str(V3_DIR))
from v3 import fs_em as EMM                                   # noqa: E402
from v3 import decision as DC3                                # noqa: E402

OUTF = V4 / ("output_smoke_final" if SMOKE else "output_final")
PX = [f"x_{f}" for f in V4_FEATS]
EM_F = EMM.EM_FEATS
CEF = ["p_ce", "ce_present", "ce_minus_p"]
F43A = models.L2_FEATS + ["p_l1"] + CEF + S3.MATE_FEATS + PX + EM_F
F43B = models.L2_FEATS + ["p_l1", "p_l2"] + CEF + S3.MATE_FEATS + PX + EM_F
POLICY_K = 8


def fpath(kind, split, c):
    return CACHE / f"{kind}_{split}_{c}.parquet"


def in_band(col="p_l2"):
    return (pl.col(col) >= S3.BAND[0]) & (pl.col(col) <= S3.BAND[1])


def oof1():
    return pl.read_parquet(CACHE / "l1_oof.parquet").select("i1", "i23", "y", "fold")


# ------------------------------------------------------------------ stage: pair features for band edges
def stage_pairx():
    from .pipeline import cand_path, countries, load_parsed, offsets
    l2, oof2 = S3._l2(), pl.read_parquet(CACHE / "l2_oof.parquet")
    for split in ["mirror", "test"]:
        o1, o23 = offsets(split)
        for c in countries(split):
            if fpath("pairx", split, c).exists():
                log(f"cache hit: pairx {split} {c}"); continue
            t = time.time()
            band = S3.edges_with_l2(split, c, l2, oof2 if split == "mirror" else None).filter(in_band()).select("i1", "i23")
            cand = pl.read_parquet(cand_path(split, c))
            loc = band.select(i1=(pl.col("i1") - o1[c]).cast(cand.schema["i1"]), i23=(pl.col("i23") - o23[c]).cast(cand.schema["i23"]))
            bc = loc.join(cand, on=["i1", "i23"], how="left", maintain_order="left")
            X = build_features(bc, load_parsed(split, c, "s1"), load_parsed(split, c, "s23"), desc=f"pairx {split} {c}")
            pl.DataFrame(X[:, V4_IDX], schema=PX).with_columns(i1=band["i1"], i23=band["i23"]).write_parquet(
                fpath("pairx", split, c), compression="zstd")
            log(f"pairx {split} {c}: {band.height:,} band edges in {time.time() - t:.0f}s")
            del band, cand, loc, bc, X; gc.collect()


def load_pairx(split, c):
    d = pl.read_parquet(fpath("pairx", split, c))
    return d.select("i1", "i23"), d.select(PX).to_numpy().astype(np.float32)


# ------------------------------------------------------------------ stage: Fellegi-Sunter EM per country (V4.4)
def stage_em():
    from .pipeline import MIRROR_C, countries
    if all(fpath("em", s, c).exists() for s in ["mirror", "test"] for c in countries(s)):
        log("cache hit: em"); return
    lab = oof1()
    ids, Gs, ctys = [], [], []
    for c in MIRROR_C:
        i, X = load_pairx("mirror", c)
        ids.append(i); Gs.append(EMM.comparison_vectors(X, V4_FEATS)); ctys.append(np.full(len(i), c))
        del X
    ids = pl.concat(ids)
    G, cty = np.vstack(Gs), np.concatenate(ctys)
    j = ids.join(lab, on=["i1", "i23"], how="left", maintain_order="left")
    is_tr = j["y"].is_not_null().to_numpy()
    y = j["y"].fill_null(0).to_numpy()
    fold = j["fold"].fill_null(-1).to_numpy()
    out = np.zeros((len(G), len(EM_F)), np.float32)

    def em_block(init_mask, target_mask):
        fs = EMM.FellegiSunter().fit_supervised(G[init_mask], y[init_mask]).em(G[target_mask])
        W, tot, post = fs.weights(G[target_mask])
        out[target_mask] = np.column_stack([W, tot, post])
        return fs

    for k in range(CFG["folds"]):                              # TRAIN rows: fold-wise supervised init (OOF), EM per country
        for c in MIRROR_C:
            sel = is_tr & (fold == k) & (cty == c)
            if sel.any():
                em_block(is_tr & (fold != k), sel)
    for c in MIRROR_C:                                          # other mirror rows (HOLDOUT + closure): init on all TRAIN
        sel = (~is_tr) & (cty == c)
        if sel.any():
            em_block(is_tr, sel)
    for c in MIRROR_C:
        m = cty == c
        ids.filter(pl.Series(m)).with_columns(pl.DataFrame(out[m], schema=EM_F)).write_parquet(fpath("em", "mirror", c), compression="zstd")
    Gtr, ytr = G[is_tr], y[is_tr]
    del G, Gs; gc.collect()
    for c in countries("test"):                                 # test: init on all TRAIN, EM on this country's own pairs
        i, X = load_pairx("test", c)
        Gt = EMM.comparison_vectors(X, V4_FEATS)
        fs = EMM.FellegiSunter().fit_supervised(Gtr, ytr).em(Gt)
        W, tot, post = fs.weights(Gt)
        i.with_columns(pl.DataFrame(np.column_stack([W, tot, post]).astype(np.float32), schema=EM_F)).write_parquet(
            fpath("em", "test", c), compression="zstd")
        log(f"EM test {c}: {len(Gt):,} band pairs | estimated match share pi {fs.pi:.3f}")
        del X, Gt; gc.collect()


# ------------------------------------------------------------------ stage: cross-encoder re-trained on mirror TRAIN band
def _texts(split, c, edges, raw):
    """(i1, i23) global ids -> CE text pairs (same serialisation V3 trained with)."""
    from .pipeline import load_parsed, offsets
    s1, s23 = raw
    o1, o23 = offsets(split)
    a = load_parsed(split, c, "s1", ["idx", "entity_id"]).join(
        s1.select("entity_id", n="business_name", a="business_address"), on="entity_id", how="left").sort("idx")
    b = load_parsed(split, c, "s23", ["idx", "entity_id"]).join(
        s23.select("entity_id", n="business_name", a="business_address"), on="entity_id", how="left").sort("idx")
    clean = lambda s: "" if s.strip().lower() in {"", "none", "null", "<null>", "n/a", "na", "nan", "-", "--"} else s.strip()
    n1, a1, n2, a2 = a["n"].to_list(), a["a"].to_list(), b["n"].to_list(), b["a"].to_list()
    i1 = edges["i1"].to_numpy().astype(np.int64) - o1[c]
    i23 = edges["i23"].to_numpy().astype(np.int64) - o23[c]
    return [f"{n1[i]} ; {clean(a1[i])}" for i in i1], [f"{n2[j]} ; {clean(a2[j])} ; dup 1" for j in i23]


def stage_ce43():
    import torch
    from v3.ce import CrossEncoder
    from .pipeline import MIRROR_C, countries, load_raw
    if all(fpath("ce43", s, c).exists() for s in ["mirror", "test"] for c in countries(s)):
        log("cache hit: ce43"); return
    lab = oof1()
    raw = load_raw("train")[:2]
    E, A, B = [], [], []
    for c in MIRROR_C:
        e = pl.read_parquet(fpath("pairx", "mirror", c), columns=["i1", "i23"]).with_columns(cty=pl.lit(c))
        a, b = _texts("mirror", c, e, raw)
        E.append(e); A += a; B += b
    E = pl.concat(E).join(lab, on=["i1", "i23"], how="left", maintain_order="left")
    y = E["y"].fill_null(0).to_numpy().astype(np.float32)
    is_tr = E["y"].is_not_null().to_numpy()
    grp = np.where(is_tr, E["fold"].fill_null(0).to_numpy() % 2, -1)
    p = np.full(len(E), np.nan, np.float32)
    for k in (0, 1):
        path = MODELS / f"ce43_fold{k}"
        if (path / "config.json").exists():
            ce = CrossEncoder(str(path)); log(f"CE43 fold {k}: loaded")
        else:
            tr = np.where(is_tr & (grp != k))[0]
            ce = CrossEncoder(str(V3_DIR / "models" / "ce_fold0"))           # warm start from V3's cross-encoder
            t = time.time()
            loss = ce.fit([A[i] for i in tr], [B[i] for i in tr], y[tr], lr=2e-5, desc=f"CE43 train fold {k}")
            path.parent.mkdir(parents=True, exist_ok=True); ce.save(path)
            log(f"CE43 fold {k}: trained on {len(tr):,} mirror TRAIN band pairs (pos {y[tr].mean():.3f}) loss {loss:.4f} in {time.time() - t:.0f}s")
        ce.model.to(torch.bfloat16)
        va = np.where(is_tr & (grp == k))[0]
        p[va] = ce.predict([A[i] for i in va], [B[i] for i in va], desc=f"CE43 OOF fold {k}")
        if k == 0:
            rest = np.where(~is_tr)[0]
            p[rest] = ce.predict([A[i] for i in rest], [B[i] for i in rest], desc="CE43 mirror non-TRAIN")
            ce0 = ce
        else:
            del ce; torch.cuda.empty_cache()
    from sklearn.metrics import roc_auc_score
    log(f"CE43 OOF AUC on TRAIN band {roc_auc_score(y[is_tr], p[is_tr]):.5f}")
    for c in MIRROR_C:
        m = (E["cty"] == c).to_numpy()
        E.filter(pl.Series(m)).select("i1", "i23").with_columns(p_ce=pl.Series(p[m])).write_parquet(fpath("ce43", "mirror", c), compression="zstd")
    del A, B, E, raw; gc.collect()
    raw = load_raw("test")[:2]
    for c in countries("test"):
        e = pl.read_parquet(fpath("pairx", "test", c), columns=["i1", "i23"])
        a, b = _texts("test", c, e, raw)
        t = time.time()
        pt = ce0.predict(a, b, desc=f"CE43 test {c}")
        e.with_columns(p_ce=pl.Series(pt, dtype=pl.Float32)).write_parquet(fpath("ce43", "test", c), compression="zstd")
        log(f"CE43 test {c}: {e.height:,} band edges in {time.time() - t:.0f}s")


# ------------------------------------------------------------------ stacker frames
def band_features(split, c, mates_kind):
    """All band-only inputs for one split/country, keyed by (i1, i23)."""
    i, X = load_pairx(split, c)
    f = i.with_columns(pl.DataFrame(X, schema=PX))
    f = f.join(pl.read_parquet(fpath("em", split, c)), on=["i1", "i23"], how="left")
    f = f.join(pl.read_parquet(fpath("ce43", split, c)), on=["i1", "i23"], how="left")
    m = pl.read_parquet(S3.mates_path(split, c) if mates_kind == "mates" else fpath("mates2", split, c))
    return f.join(m, on=["i1", "i23"], how="left")


def frame(tab, bf, base):
    """tab: all edges (p_l1, p_l2 [, p_43a], blocking cols); context over `base`; band features joined."""
    t = tab.rename({base: "p"}) if base != "p_l2" else tab.rename({"p_l2": "p"})
    if base != "p_l2":
        t = t.with_columns(p_l2=tab["p_l2"])
    ctx = models.context_features(t)          # context needs every edge (list + competition) ...
    ctx = ctx.join(bf, on=["i1", "i23"], how="inner").with_columns(   # ... but only band rows carry the ~220 features
        band=pl.lit(True),
        m_present=pl.col("m_p").is_not_null().cast(pl.Float32),
        ce_present=pl.col("p_ce").is_not_null().cast(pl.Float32))
    return ctx.with_columns(ce_minus_p=pl.when(pl.col("ce_present") > 0).then(pl.col("p_ce") - pl.col("p")).otherwise(0.0),
                            p_ce=pl.col("p_ce").fill_null(-1.0))


def tables(split, c, l2, oof2, with_a=False):
    tab = S3.edges_with_l2(split, c, l2, oof2 if split == "mirror" else None)
    if with_a:
        a = pl.read_parquet(fpath("p43a", split, c))
        tab = tab.join(a, on=["i1", "i23"], how="left").with_columns(p_43a=pl.coalesce("p_43a", "p_l2"))
    return tab


def train_stacker(name, base, feats, mates_kind):
    from .pipeline import MIRROR_C
    from sklearn.metrics import roc_auc_score
    if (CACHE / f"{name}.json").exists():
        log(f"cache hit: {name}"); return
    l2, oof2 = S3._l2(), pl.read_parquet(CACHE / "l2_oof.parquet")
    fr = pl.concat([frame(tables("mirror", c, l2, oof2, with_a=(base == "p_43a")), band_features("mirror", c, mates_kind), base)
                    .with_columns(cty=pl.lit(c)) for c in MIRROR_C], how="diagonal_relaxed")
    trn = fr.filter(pl.col("is_train") & pl.col("band")).join(oof1(), on=["i1", "i23"])
    X = trn.select(feats).to_numpy().astype(np.float32)
    y, folds, cty = trn["y"].to_numpy(), trn["fold"].to_numpy(), trn["cty"].to_numpy()
    prm = models.params(monotone=[1 if f in ("p", "p_ce", "em_total") else 0 for f in feats])
    log(f"{name} on {models.device()}: {X.shape}")
    oof, iters, final = models.train_oof(X, y, folds, prm, name)
    final.save_model(CACHE / f"{name}.json")
    trn.select("i1", "i23").with_columns(p=pl.Series(oof)).write_parquet(CACHE / f"{name}_oof.parquet", compression="zstd")
    msg = [f"{name} band AUC: base {roc_auc_score(y, trn['p'].to_numpy()):.5f} -> OOF {roc_auc_score(y, oof):.5f}"]
    for src, tgt in [("us", "india"), ("india", "us")]:                 # LOCO (France proxy), AUC level
        b = models.fit(X[cty == src], y[cty == src], prm, int(np.mean(iters)))
        pt = models.predict(b, X[cty == tgt])
        msg.append(f"LOCO {src}->{tgt} AUC {roc_auc_score(y[cty == tgt], pt):.5f} (in-country OOF {roc_auc_score(y[cty == tgt], oof[cty == tgt]):.5f})")
        del b; gc.collect()
    log(" | ".join(msg))


def score_stacker(name, base, feats, mates_kind, out_kind):
    """p_<name> for every band edge of every split (TRAIN rows take their OOF value) -> <out_kind>_{split}_{c}.parquet."""
    import xgboost as xgb
    from .pipeline import countries
    l2, oof2 = S3._l2(), pl.read_parquet(CACHE / "l2_oof.parquet")
    bst = xgb.Booster(model_file=str(CACHE / f"{name}.json")); bst.set_param({"device": models.device()})
    oof = pl.read_parquet(CACHE / f"{name}_oof.parquet")
    for split in ["mirror", "test"]:
        for c in countries(split):
            if fpath(out_kind, split, c).exists():
                continue
            fr = frame(tables(split, c, l2, oof2, with_a=(base == "p_43a")), band_features(split, c, mates_kind), base)
            fr = fr.filter(pl.col("band"))
            p = models.predict(bst, fr.select(feats).to_numpy().astype(np.float32))
            out = fr.select("i1", "i23").with_columns(p_new=pl.Series(p))
            if split == "mirror":
                out = (out.join(oof.rename({"p": "p_oof"}), on=["i1", "i23"], how="left")
                          .with_columns(p_new=pl.coalesce("p_oof", "p_new")).drop("p_oof"))
            out.rename({"p_new": f"p_{out_kind[1:]}" if out_kind.startswith("p") else "p"}).write_parquet(
                fpath(out_kind, split, c), compression="zstd")
            log(f"scored {name} {split} {c}: {out.height:,} band edges")
            del fr, out; gc.collect()


def stage_l43a():
    train_stacker("l43a", "p_l2", F43A, "mates")
    score_stacker("l43a", "p_l2", F43A, "mates", "p43a")


# ------------------------------------------------------------------ stage: second evidence-pooling round
def stage_mates2():
    from joblib import Parallel, delayed
    from tqdm.auto import tqdm
    from .features import REC, _feat_batch, PAIR_FEATS
    from .pipeline import countries, load_parsed, offsets
    l2, oof2 = S3._l2(), pl.read_parquet(CACHE / "l2_oof.parquet")
    idx_src = [PAIR_FEATS.index(f) for f in S3.MATE_SRC]
    for split in ["mirror", "test"]:
        o1, o23 = offsets(split)
        for c in countries(split):
            if fpath("mates2", split, c).exists():
                log(f"cache hit: mates2 {split} {c}"); continue
            t = time.time()
            tab = tables(split, c, l2, oof2, with_a=True).select("i1", "i23", "p_l2", "p_43a")
            band = tab.filter(in_band()).select("i1", "i23")
            strong = (tab.filter(pl.col("p_43a") >= S3.STRONG)
                         .with_columns(r=pl.col("p_43a").rank("ordinal", descending=True).over("i1"), n_strong=pl.len().over("i1"))
                         .filter(pl.col("r") <= 2).select("i1", mate="i23", m_p="p_43a", r="r", n_strong="n_strong"))
            m = (band.join(strong, on="i1").filter(pl.col("mate") != pl.col("i23")).sort("r").unique(["i1", "i23"], keep="first")
                     .with_columns(m_n_strong=pl.col("n_strong").cast(pl.Float32)))
            R = load_parsed(split, c, "s23").select(REC)
            x = m["i23"].to_numpy().astype(np.int64) - o23[c]
            yy = m["mate"].to_numpy().astype(np.int64) - o23[c]
            Bn = 40_000
            jobs = (delayed(_feat_batch)(R[x[lo:lo + Bn]].rows(), R[yy[lo:lo + Bn]].rows()) for lo in range(0, len(x), Bn))
            F = np.empty((len(x), len(S3.MATE_SRC)), np.float32)
            pos = 0
            for part in tqdm(Parallel(n_jobs=CFG["n_jobs"], return_as="generator", pre_dispatch="2*n_jobs")(jobs),
                             total=-(-len(x) // Bn), desc=f"mates2 {split} {c}", mininterval=15):
                F[pos:pos + len(part)] = part[:, idx_src]; pos += len(part)
            (m.select("i1", "i23", m_p=pl.col("m_p").cast(pl.Float32), m_n_strong="m_n_strong")
              .with_columns(pl.DataFrame(F, schema=[f"m_{f}" for f in S3.MATE_SRC]))
              .write_parquet(fpath("mates2", split, c), compression="zstd"))
            log(f"mates2 {split} {c}: {m.height:,} band edges with a strong mate | {time.time() - t:.0f}s")
            del tab, band, strong, m, R, F; gc.collect()


def stage_l43b():
    train_stacker("l43b", "p_43a", F43B, "mates2")
    score_stacker("l43b", "p_43a", F43B, "mates2", "p43b")


# ------------------------------------------------------------------ all edges with every score variant
def scored_tab(split, c, l2, oof2):
    tab = tables(split, c, l2, oof2, with_a=True)
    b = pl.read_parquet(fpath("p43b", split, c))
    return tab.join(b, on=["i1", "i23"], how="left").with_columns(p_43b=pl.coalesce("p_43b", "p_43a"))


# ------------------------------------------------------------------ learned policy (V3 L14) on top of the V4 rule
def policy_inputs(st, tab_p, s1i, ci, scope_i1, p_single=None):
    """V4 rule's calibrated + exclusive lists for the S1s in scope -> (ids, P, R, rest, EF, Xpol, sc_ex)."""
    sc = tab_p.select("i1", "i23").with_columns(p=pl.Series(st["iso"].predict(tab_p["p"].to_numpy()).astype(np.float32)))
    sc_ex = decide.exclusivity(sc, st["eps"], st["lam"]).join(
        pl.DataFrame({"i1": scope_i1}).cast({"i1": sc.schema["i1"]}), on="i1", how="semi")
    return lists_from(sc_ex, st, s1i, ci, scope_i1, p_single)


def lists_from(sc_ex, st, s1i, ci, scope_i1, p_single=None):
    ids = np.sort(np.unique(np.asarray(scope_i1)))
    row = pl.DataFrame({"i1": ids, "row": np.arange(len(ids))}).cast({"i1": sc_ex.schema["i1"]})
    s = (sc_ex.sort(["i1", "p"], descending=[False, True]).with_columns(r=pl.int_range(pl.len()).over("i1"))
              .join(row, on="i1"))
    n, K = len(ids), POLICY_K
    P = np.zeros((n, K)); R = np.full((n, K), -1, np.int64)
    top = s.filter(pl.col("r") < K)
    P[top["row"].to_numpy(), top["r"].to_numpy()] = top["p"].to_numpy()
    R[top["row"].to_numpy(), top["r"].to_numpy()] = top["i23"].to_numpy().astype(np.int64)
    rest = np.bincount(s.filter(pl.col("r") >= K)["row"].to_numpy(), weights=s.filter(pl.col("r") >= K)["p"].to_numpy(), minlength=n)
    if p_single is None:
        s1 = s1i.join(pl.DataFrame({"i1": ids}).cast({"i1": s1i.schema["i1"]}), on="i1", how="semi")
        f = decide.singleton_features(sc_ex, s1, ci)
        raw = models.predict(st["sing"], f.select(models.SING_FEATS).to_numpy().astype(np.float32))
        p_single = f.select("i1").with_columns(p_single=pl.Series(1 - st["sing_iso"].predict(raw)))
    pn = row.join(p_single.cast({"i1": row.schema["i1"]}), on="i1", how="left", maintain_order="left")["p_single"].fill_null(0.5).to_numpy()
    EF = DC3.ef_table(None, P, rest, st["m_miss"], K)
    Xpol = DC3.policy_matrix(P, EF, rest, pn, np.zeros((n, 2), np.float32), np.zeros(n), np.ones(n), K)
    return ids, P, R, rest, EF, Xpol


def cost_table(ids, R, truth):
    tset = set(zip(truth["i1"].to_list(), truth["i23"].to_list()))
    n, K = R.shape
    y = np.array([[1.0 if (R[i, k] >= 0 and (int(ids[i]), int(R[i, k])) in tset) else 0.0 for k in range(K)] for i in range(n)])
    ntrue = (pl.DataFrame({"i1": ids}).cast({"i1": truth.schema["i1"]})
               .join(truth.group_by("i1").len("nt"), on="i1", how="left", maintain_order="left")["nt"].fill_null(0).to_numpy().astype(float))
    tp = np.concatenate([np.zeros((n, 1)), np.cumsum(y, 1)], 1)
    npred = np.arange(K + 1)[None, :].repeat(n, 0).astype(float)
    den = 5 * tp + (ntrue[:, None] - tp) + 4 * (npred - tp)
    f = np.where(den == 0, 1.0, 5 * tp / np.maximum(den, 1e-12))
    valid = np.concatenate([np.ones((n, 1), bool), R >= 0], 1)
    return 1 - f, valid, f


def pred_from_k(ids, R, k):
    m = np.arange(R.shape[1])[None, :] < k[:, None]
    rr, cc = np.where(m & (R >= 0))
    return pl.DataFrame({"i1": ids[rr], "i23": R[rr, cc]}).cast({"i1": pl.UInt32, "i23": pl.UInt32})


# ------------------------------------------------------------------ stage: decision + policy + HOLDOUT report
def stage_decide():
    from .pipeline import MIRROR_C, mirror_truth_global, role_global, s1_info, cand_info
    l2, oof2 = S3._l2(), pl.read_parquet(CACHE / "l2_oof.parquet")
    tab = pl.concat([scored_tab("mirror", c, l2, oof2) for c in MIRROR_C], how="diagonal_relaxed")
    truth, T, H = mirror_truth_global(), role_global("train"), role_global("holdout")
    truth_T = truth.join(T, on="i1", how="semi").select("i1", "i23")
    truth_H = truth.join(H, on="i1", how="semi").select("i1", "i23", "cty")
    tab = tab.join(oof1(), on=["i1", "i23"], how="left")
    is_train = tab["is_train"].to_numpy()
    y_edge, f_edge = tab.filter(pl.col("is_train"))["y"].to_numpy(), tab.filter(pl.col("is_train"))["fold"].to_numpy()
    fold_of_i1 = tab.filter(pl.col("is_train")).group_by("i1").agg(pl.col("fold").first())
    hitT = truth_T.join(tab.filter(pl.col("is_train")).select("i1", "i23"), on=["i1", "i23"], how="semi").height
    m_miss = (truth_T.height - hitT) / T.height
    s1i, ci = s1_info("mirror"), cand_info("mirror")
    st = {v: decide.fit_decision(tab.select("i1", "i23", p=v), y_edge, is_train, f_edge, truth_T, T["i1"], fold_of_i1,
                                 s1i, ci, m_miss, tag=v) for v in ["p_l2", "p_43a", "p_43b"]}
    best = max(st, key=lambda k: st[k]["f_train"])
    sb = st[best]
    # ---- learned policy on TRAIN (honest inputs: cross-fitted calibration + OOF singleton probabilities)
    pT = tab[best].to_numpy()
    p_cal = sb["iso"].predict(pT).astype(np.float32)
    p_cal[is_train] = decide.iso_crossfit(pT[is_train], y_edge, f_edge)
    sc_ex = decide.exclusivity(tab.select("i1", "i23").with_columns(p=pl.Series(p_cal)), sb["eps"], sb["lam"]).join(T, on="i1", how="semi")
    ids, P, R, rest, EF, Xpol = lists_from(sc_ex, sb, s1i, ci, T["i1"], sb["p_single_T"])
    C, valid, Fk = cost_table(ids, R, truth_T)
    valid &= EF >= 0
    fo = pl.DataFrame({"i1": ids}).cast({"i1": fold_of_i1.schema["i1"]}).join(fold_of_i1, on="i1", how="left")["fold"].fill_null(0).to_numpy()
    kk = np.zeros(len(ids), np.int64)
    for k in range(CFG["folds"]):
        tr, va = fo != k, fo == k
        kk[va] = DC3.policy_choose(DC3.fit_policy(Xpol[tr], C[tr], valid[tr]), Xpol[va], valid[va])
    f_pol = Fk[np.arange(len(ids)), kk]
    f_ef = Fk[np.arange(len(ids)), EF.argmax(1)]
    from v3.common import bootstrap_delta
    d, lo, hi = bootstrap_delta(f_pol, f_ef)
    policy = DC3.fit_policy(Xpol, C, valid)
    use_policy = bool(lo > 0)
    log(f"learned policy on TRAIN ({best}): {f_pol.mean():.5f} vs expected-F lists {f_ef.mean():.5f} | delta {d:+.5f} "
        f"CI [{lo:+.5f}, {hi:+.5f}] -> {'ADOPTED' if use_policy else 'rejected'}")
    # ---- HOLDOUT report
    names = {"p_l2": "V4.1 (p_l2)", "p_43a": "V4.3 stacker round 1", "p_43b": "V4.3 stacker round 2"}
    rows = []

    def add_row(name, f_train, pred):
        r = [name, f_train, macro_f05(pred, truth_H.select("i1", "i23"), H["i1"])]
        for c in MIRROR_C:
            hc = H.filter(pl.col("cty") == c)
            r.append(macro_f05(pred.join(hc, on="i1", how="semi"), truth_H.filter(pl.col("cty") == c).select("i1", "i23"), hc["i1"]))
        hp = pred.join(H, on="i1", how="semi")
        rows.append(r + [hp.height / H.height, 1 - hp["i1"].n_unique() / H.height])

    for v in st:
        add_row(names[v] + (" (chosen rule)" if v == best else ""), st[v]["f_train"],
                decide.apply(st[v], tab.select("i1", "i23", p=v), s1i, ci, H["i1"]))
    idsH, _, RH, _, EFH, XH = policy_inputs(sb, tab.select("i1", "i23", p=best), s1i, ci, H["i1"])
    validH = np.concatenate([np.ones((len(idsH), 1), bool), RH >= 0], 1) & (EFH >= 0)
    add_row(f"V4.3 {names[best]} + learned policy", float(f_pol.mean()), pred_from_k(idsH, RH, DC3.policy_choose(policy, XH, validH)))
    rep = pl.DataFrame(rows, schema=["model", "train_f05", "holdout_f05", *[f"f05_{c}" for c in MIRROR_C], "pred_per_s1",
                                     "pred_empty"], orient="row")
    log(f"V4.3 HOLDOUT (true matches/S1 {truth_H.height / H.height:.3f}, true empty {1 - truth_H['i1'].n_unique() / H.height:.4f}):\n{rep}")
    pickle.dump(dict(st=sb, variant=best, policy=policy, use_policy=use_policy, report=rep),
                open(CACHE / "decision_v43.pkl", "wb"))


# ------------------------------------------------------------------ stage: export
def stage_export():
    from .pipeline import countries, load_parsed, load_raw, offsets, s1_info, cand_info
    d = pickle.load(open(CACHE / "decision_v43.pkl", "rb"))
    st, var = d["st"], d["variant"]
    l2 = S3._l2()
    s1i, ci = s1_info("test"), cand_info("test")
    o1, o23 = offsets("test")
    preds, cands, stats = [], [], []
    for c in countries("test"):
        tab = scored_tab("test", c, l2, None)
        tp = tab.select("i1", "i23", p=var)
        if d["use_policy"]:
            a_ids = load_parsed("test", c, "s1", ["idx"]).select(i1=(pl.col("idx") + o1[c]).cast(pl.UInt32))["i1"]
            ids, _, R, _, EF, X = policy_inputs(st, tp, s1i, ci, a_ids)
            valid = np.concatenate([np.ones((len(ids), 1), bool), R >= 0], 1) & (EF >= 0)
            pred = pred_from_k(ids, R, DC3.policy_choose(d["policy"], X, valid))
        else:
            pred = decide.apply(st, tp, s1i, ci)
        a = load_parsed("test", c, "s1", ["idx", "entity_id"]).with_columns(i1=(pl.col("idx") + o1[c]).cast(pl.UInt32))
        b = load_parsed("test", c, "s23", ["idx", "entity_id"]).with_columns(i23=(pl.col("idx") + o23[c]).cast(pl.UInt32))
        ids_ = lambda e: (e.join(a.select("i1", s1_id="entity_id"), on="i1").join(b.select("i23", s23_id="entity_id"), on="i23")
                           .select("s1_id", "s23_id"))
        preds.append(ids_(pred)); cands.append(ids_(tab.select("i1", "i23")))
        stats.append((c, a.height, pred.height / a.height, 1 - pred["i1"].n_unique() / a.height))
        del tab, tp; gc.collect()
    log(f"test predictions ({var}{' + policy' if d['use_policy'] else ''}):\n"
        f"{pl.DataFrame(stats, schema=['cty', 'n_s1', 'pred_per_s1', 'pred_empty'], orient='row')}")
    all_pred, all_cand = pl.concat(preds), pl.concat(cands)
    assert all_pred.join(all_cand, on=["s1_id", "s23_id"], how="anti").height == 0, "matches must be a subset of candidates"
    assert all_pred["s23_id"].is_unique().all(), "an S2/S3 record may belong to at most one S1"
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

    m = write(all_pred, "matched_entity_ids", OUTF / "matching_results.tsv")
    write(all_cand, "candidate_entity_ids", OUTF / "candidate_pairs.tsv")
    log(f"wrote {OUTF / 'matching_results.tsv'}: {m.height:,} rows, {(m['matched_entity_ids'] != '').sum():,} non-empty, "
        f"{all_pred.height:,} matches")
    if SMOKE:
        log("smoke run: official validator skipped (the smoke test set is a sample)"); return
    r = subprocess.run([sys.executable, str(ROOT / "student_resource" / "utils" / "validate_submission.py"),
                        "--matching", str(OUTF / "matching_results.tsv"), "--candidate", str(OUTF / "candidate_pairs.tsv"),
                        "--test-dir", str(DATA / "test"), "--check-ids"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-2000:], r.stderr[-1000:])
    log(f"official validator exit code {r.returncode} -> {'PASS' if r.returncode == 0 else 'FIX ISSUES'}")


STAGES = dict(pairx=stage_pairx, em=stage_em, ce43=stage_ce43, l43a=stage_l43a, mates2=stage_mates2, l43b=stage_l43b,
              decide=stage_decide, export=stage_export)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    s = ap.parse_args().stage
    t0 = time.time()
    for name, fn in STAGES.items():
        if s in ("all", name):
            log(f"===== V4.3 stage {name} =====")
            fn()
    log(f"done in {(time.time() - t0) / 60:.1f} min")
