"""V4 pipeline stages. Run all:  python -m er4.pipeline --stage all   (from version 4/src), or use v4_run.ipynb.

Splits: 'mirror' = labelled train data under test conditions (S1 dropout, full-scale per-country pools),
        'test'   = the real test set. Both go through the same code path."""
import argparse, gc, json, os, pickle, subprocess, sys, time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import decide, models
from .blocking import add_dense, blk_matrix, block_country, competition_features
from .config import BASE_CACHE, CACHE, CFG, DATA, DENSE, OUT, SHARED, SMOKE, SRC, STRICT, V4, VALIDATOR
from .features import MONOTONE, V2_FEATS, V4_FEATS, V4_IDX, build_features
from .mining import mine_tables, self_mine_seeds, simple_block, subs_for
from .normalize import LEGAL_SEED, discover_legal, enrich, parse_frame
from .util import cached, check_disk, log, macro_f05

MIRROR_C = ["us", "india"]
CTY = pl.col("country").str.strip_chars().str.to_lowercase()


def read_tsv(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).fill_null("")


def load_raw(split):
    d = DATA / split
    s1 = read_tsv(d / f"{split}_source1.tsv").with_columns(cty=CTY)
    s23 = pl.concat([read_tsv(d / f"{split}_source{s}.tsv") for s in (2, 3)]).with_columns(cty=CTY)
    gt = read_tsv(d / "train_ground_truth.tsv") if split == "train" else None
    if SMOKE:                       # keep ~2.5% of S1, their true matches and ~2.5% of the other S2/S3 records
        keep = lambda df: df.filter(pl.col("entity_id").hash(1) % 40 == 0)
        s1 = keep(s1)
        if gt is not None:
            gt = gt.join(s1.select(source1_entity_id="entity_id"), on="source1_entity_id", how="semi")
            mt = gt_pairs(gt).select(entity_id="s23_id")
            s23 = pl.concat([s23.join(mt, on="entity_id", how="semi"), keep(s23.join(mt, on="entity_id", how="anti"))])
        else:
            s23 = keep(s23)
    return s1, s23, gt


def gt_pairs(gt):
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
              .filter(pl.col("matched_entity_ids") != "")
              .select(s1_id="source1_entity_id", s23_id="matched_entity_ids"))


def write_pq(df, path):
    """write-then-rename: a killed run never leaves a truncated checkpoint that looks complete."""
    tmp = path.with_suffix(".tmp")
    df.write_parquet(tmp, compression="zstd")
    tmp.replace(path)


# ------------------------------------------------------------------ stage: roles (the test mirror)
def stage_roles():
    def build():
        tr1, _, gt = load_raw("train")
        u = np.random.default_rng(CFG["seed"]).random(tr1.height)          # same draw as V2's make_samples
        role = np.full(tr1.height, "context", dtype=object)
        for name, (lo, hi) in CFG["roles"].items():
            role[(u >= lo) & (u < hi)] = name
        r = tr1.select("entity_id", "cty").with_columns(role=pl.Series(role, dtype=pl.String))
        return r.join(gt_pairs(gt).group_by("s1_id").len("n_true").rename({"s1_id": "entity_id"}), on="entity_id", how="left")
    r = cached("roles.parquet", build)
    cached("truth_mirror.parquet", lambda: gt_pairs(load_raw("train")[2]).join(
        r.filter(pl.col("role") != "drop").select(s1_id="entity_id"), on="s1_id", how="semi"))
    tab = r.group_by("cty", "role").len().pivot(on="role", index="cty", values="len")
    log(f"mirror roles:\n{tab}")
    return r


# ------------------------------------------------------------------ stage: mining tables (+ legal forms)
def seeded_tables():
    """The tables the submitted run used: V2's mined tables (train MINE slice = V2's sample M, same RNG; France self-mined
    on test, unsupervised). Written by v2/prepare.py (legal_found / tables_train / tables_france), shipped in anchor_v2/."""
    a = SRC / "anchor_v2"
    lf = pickle.load(open(a / "legal_found.pkl", "rb"))
    return dict(train=pickle.load(open(a / "tables_train.pkl", "rb"))[0], unseen=pickle.load(open(a / "tables_france.pkl", "rb"))[0],
                legal=LEGAL_SEED | {t for v in lf.values() for t in v}, source="seeded from V2 (same MINE slice, same RNG)")


def stage_mine():
    def build():
        if not STRICT and os.environ.get("ER4_REMINE") != "1":     # ER4_REMINE=1: re-mine with the V4 code instead
            return seeded_tables()
        tr1, tr23, gt = load_raw("train")
        te1, te23, _ = load_raw("test")
        roles = stage_roles()
        pr = gt_pairs(gt)
        mine_ids = roles.filter(pl.col("role") == "mine").select(s1_id="entity_id")
        names = tr1.select("business_name", "cty")
        if not STRICT:                           # V4/V4.1/V4.2: legal-form discovery also read unlabeled test France names
            names = pl.concat([names, te1.filter(pl.col("cty") == "france").select("business_name", "cty")])
        legal_found = discover_legal({c: g["business_name"].sample(min(g.height, 150_000), seed=0).to_list()
                                      for (c,), g in names.group_by("cty")})
        legal = LEGAL_SEED | {t for v in legal_found.values() for t in v}
        unseen = {}
        # labelled countries: MINE S1 + their matches + a matching share of unmatched records (V2's M sample)
        rng = np.random.default_rng(CFG["seed"]); rng.random(tr1.height)
        unmatched = tr23.join(pr.select(pl.col("s23_id").alias("entity_id")), on="entity_id", how="anti")
        v = rng.random(unmatched.height)
        lo, hi = CFG["roles"]["mine"]
        s1 = tr1.join(mine_ids.rename({"s1_id": "entity_id"}), on="entity_id", how="semi")
        prm = pr.join(mine_ids, on="s1_id", how="semi")
        s23 = pl.concat([tr23.join(prm.select(pl.col("s23_id").alias("entity_id")), on="entity_id", how="semi"),
                         unmatched.filter(pl.Series((v >= lo) & (v < hi)))])
        a = parse_frame(s1.drop("cty"), legal, desc="parse MINE S1 (pass 1)")
        b = parse_frame(s23.drop("cty"), legal, desc="parse MINE S2S3 (pass 1)")
        cand = pl.concat([simple_block(a.filter(pl.col("cty") == c), b.filter(pl.col("cty") == c)) for c in MIRROR_C])
        truth = (prm.join(a.select(s1_id="entity_id", i1="idx"), on="s1_id")
                    .join(b.select(s23_id="entity_id", i23="idx"), on="s23_id").select("i1", "i23"))
        neg = cand.select("i1", "i23").join(truth, on=["i1", "i23"], how="anti")
        train_tables, _ = mine_tables(a, b, truth, neg, "train MINE")
        for c in ([] if STRICT else sorted(set(te1["cty"].unique().to_list()) - set(MIRROR_C))):  # unseen: self-mining (not in STRICT)
            a = parse_frame(te1.filter(pl.col("cty") == c).drop("cty"), legal, desc=f"parse test {c} S1 (pass 1)")
            b = parse_frame(te23.filter(pl.col("cty") == c).drop("cty"), legal, desc=f"parse test {c} S2S3 (pass 1)")
            seeds, neg = self_mine_seeds(a, b)
            log(f"{c} self-mining: {seeds.height:,} seed pairs | {neg.height:,} negatives")
            t, _ = mine_tables(a, b, seeds, neg, f"test {c}")
            unseen[c] = t[c]
        return dict(train=train_tables, unseen=unseen, legal=legal, source="mined (train only)" if STRICT else "mined")
    t = cached("tables.pkl", build)
    log(f"tables ({t['source']}): " + ", ".join(f"{k}: {sum(len(x) for x in v)}" for k, v in t["train"].items())
        + " | unseen: " + ", ".join(f"{k}: {sum(len(x) for x in v)}" for k, v in t["unseen"].items()) + f" | legal forms: {len(t['legal'])}")
    return t


# ------------------------------------------------------------------ stage: parse + enrich (per split, per country)
def countries(split):
    return MIRROR_C if split == "mirror" else ["us", "france", "india"]


def parsed_path(split, c, side):
    return CACHE / f"parsed_{split}_{c}_{side}.parquet"


def stage_parse():
    t = stage_mine()
    for split in ["mirror", "test"]:
        if all(parsed_path(split, c, s).exists() for c in countries(split) for s in ("s1", "s23")):
            log(f"cache hit: parsed {split}"); continue
        if split == "mirror":
            s1, s23, _ = load_raw("train")
            keep = pl.read_parquet(CACHE / "roles.parquet").filter(pl.col("role") != "drop").select("entity_id")
            s1 = s1.join(keep, on="entity_id", how="semi")
        else:
            s1, s23, _ = load_raw("test")
        for c in countries(split):
            if parsed_path(split, c, "s1").exists() and parsed_path(split, c, "s23").exists():
                continue
            check_disk(f"parse {split} {c}")
            subs = subs_for(c, t["train"], t["unseen"])
            tt = time.time()
            a = parse_frame(s1.filter(pl.col("cty") == c).drop("cty"), t["legal"], subs, desc=f"parse {split} {c} S1")
            b = parse_frame(s23.filter(pl.col("cty") == c).drop("cty"), t["legal"], subs, desc=f"parse {split} {c} S2S3")
            a, b = enrich(a, b)
            write_pq(a, parsed_path(split, c, "s1"))
            write_pq(b, parsed_path(split, c, "s23"))
            cov = b.select(house=(pl.col("house") != "").mean(), street=(pl.col("street").list.len() > 0).mean(),
                           region=(pl.col("region") != "").mean(), postal=(pl.col("postal") != "").mean(),
                           legal=(pl.col("legal") != "").mean())
            log(f"parsed {split} {c}: {a.height:,} S1 | {b.height:,} S2/S3 | {time.time() - tt:.0f}s | S2/S3 coverage {cov.row(0, named=True)}")
            del a, b; gc.collect()
        del s1, s23; gc.collect()


def load_parsed(split, c, side, columns=None):
    return pl.read_parquet(parsed_path(split, c, side), columns=columns)


def offsets(split):
    """Global id offsets per country so that (i1, i23) are unique across countries."""
    o1, o23, n1, n23 = {}, {}, 0, 0
    for c in countries(split):
        o1[c], o23[c] = n1, n23
        n1 += pl.scan_parquet(parsed_path(split, c, "s1")).select(pl.len()).collect().item()
        n23 += pl.scan_parquet(parsed_path(split, c, "s23")).select(pl.len()).collect().item()
    return o1, o23


def role_idx(c, role):
    """idx (per-country row position) of mirror S1s with this role."""
    a = load_parsed("mirror", c, "s1", ["idx", "entity_id"])
    r = pl.read_parquet(CACHE / "roles.parquet").filter(pl.col("role") == role).select("entity_id")
    return a.join(r, on="entity_id", how="semi")["idx"]


def truth_idx(c):
    """mirror truth pairs as per-country (i1, i23)."""
    a = load_parsed("mirror", c, "s1", ["idx", "entity_id"])
    b = load_parsed("mirror", c, "s23", ["idx", "entity_id"])
    return (pl.read_parquet(CACHE / "truth_mirror.parquet")
              .join(a.select(s1_id="entity_id", i1="idx"), on="s1_id")
              .join(b.select(s23_id="entity_id", i23="idx"), on="s23_id").select("i1", "i23"))


# ------------------------------------------------------------------ stage: blocking scorer (MINE, full scale)
def stage_blkfit():
    def build():
        Xs, ys, curves = [], [], []
        for c in MIRROR_C:
            a, b = load_parsed("mirror", c, "s1"), load_parsed("mirror", c, "s23")
            u = block_country(a, b, query_idx=role_idx(c, "mine"), tag=f"MINE {c}")
            tr = truth_idx(c).join(pl.DataFrame({"i1": role_idx(c, "mine")}), on="i1", how="semi")
            y = u.join(tr.with_columns(y=pl.lit(1)), on=["i1", "i23"], how="left", maintain_order="left")["y"].fill_null(0).to_numpy()
            Xs.append(blk_matrix(u)); ys.append(y); curves.append((c, u.select("i1"), tr.height))
            log(f"MINE {c}: union {u.height / tr['i1'].n_unique():.1f} pairs per matched S1 | union recall {y.sum() / tr.height:.4f}")
            del a, b, u; gc.collect()
        X, y = np.vstack(Xs), np.concatenate(ys)
        sc = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=500)).fit(X, y)
        rows, off = [], 0
        for (c, ui, n_true), yc in zip(curves, ys):
            p = sc.predict_proba(X[off:off + len(yc)])[:, 1]; off += len(yc)
            rk = ui.with_columns(p=pl.Series(p), y=pl.Series(yc)).with_columns(r=pl.col("p").rank("ordinal", descending=True).over("i1"))
            rows += [(c, n, rk.filter(pl.col("r") <= n)["y"].sum() / n_true) for n in CFG["n_cand_grid"]]
        curve = pl.DataFrame(rows, schema=["cty", "N", "completeness"], orient="row")
        mean = curve.group_by("N").agg(pl.col("completeness").mean()).sort("N")
        best = mean["completeness"].max()
        n_cand = int(mean.filter(pl.col("completeness") >= best - 0.002)["N"].min())
        return dict(scorer=sc, n_cand=n_cand, curve=curve)
    r = cached("blk_scorer.pkl", build)
    log(f"blocking scorer: n_cand = {r['n_cand']}\n{r['curve'].pivot(on='cty', index='N', values='completeness')}")
    return r


# ------------------------------------------------------------------ stage: blocking all S1s (mirror + test)
def cand_path(split, c):
    return CACHE / f"cand_{split}_{c}.parquet"


def stage_block():
    bs = stage_blkfit()
    for split in ["mirror", "test"]:
        for c in countries(split):
            if cand_path(split, c).exists():
                log(f"cache hit: cand {split} {c}"); continue
            check_disk(f"block {split} {c}")
            a, b = load_parsed(split, c, "s1"), load_parsed(split, c, "s23")
            tt = time.time()
            cand = block_country(a, b, scorer=bs["scorer"], n_cand=bs["n_cand"], tag=f"{split} {c}")
            if DENSE:                                   # V4.1: + dense retrieval edges (er4.dense, GPU)
                from .dense import dense_path
                n0 = cand.height
                cand = add_dense(cand, pl.read_parquet(dense_path(split, c)), bs["scorer"])
                log(f"  dense union {split} {c}: {n0:,} -> {cand.height:,} edges")
            cand = competition_features(cand)
            write_pq(cand, cand_path(split, c))
            log(f"blocked {split} {c}: {a.height:,} S1 x {b.height:,} S2/S3 -> {cand.height:,} pairs "
                f"({cand.height / a.height:.1f}/S1) in {time.time() - tt:.0f}s")
            if split == "mirror":
                tr = truth_idx(c)
                for role in ["train", "holdout"]:
                    ids = pl.DataFrame({"i1": role_idx(c, role)})
                    t_ = tr.join(ids, on="i1", how="semi")
                    hit = t_.join(cand.select("i1", "i23"), on=["i1", "i23"], how="semi")
                    log(f"  {c} {role}: pair completeness {hit.height / t_.height:.4f} | oracle macro F0.5 "
                        f"{macro_f05(hit, t_, ids['i1']):.4f}")
            del a, b, cand; gc.collect()


# ------------------------------------------------------------------ stage: TRAIN features (+ V2 anchor score)
def v2_model():
    return pickle.load(open(SRC / "anchor_v2" / "L1_final.pkl", "rb"))


def fold_of(entity_id):
    return (entity_id.hash(CFG["seed"]) % CFG["folds"]).cast(pl.Int8)


def train_path(c):
    return CACHE / f"train_X_{c}.parquet"


def stage_trainfeat():
    o1, o23 = offsets("mirror")
    for c in MIRROR_C:
        if train_path(c).exists():
            log(f"cache hit: {train_path(c).name}"); continue
        check_disk(f"trainfeat {c}")
        a, b = load_parsed("mirror", c, "s1"), load_parsed("mirror", c, "s23")
        ids = pl.DataFrame({"i1": role_idx(c, "train")})
        cand = pl.read_parquet(cand_path("mirror", c)).join(ids, on="i1", how="semi")
        X = build_features(cand, a, b, desc=f"TRAIN features {c}")
        p_v2 = v2_model().predict(X[:, :len(V2_FEATS)])
        y = cand.select("i1", "i23").join(truth_idx(c).with_columns(y=pl.lit(1, pl.Int8)), on=["i1", "i23"], how="left",
                                          maintain_order="left")["y"].fill_null(0)
        folds = cand.select("i1").join(a.select(i1="idx", fold=fold_of(pl.col("entity_id"))), on="i1", how="left",
                                       maintain_order="left")["fold"]
        df = pl.DataFrame(X[:, V4_IDX], schema=V4_FEATS).with_columns(
            i1=(cand["i1"] + o1[c]).cast(pl.UInt32), i23=(cand["i23"] + o23[c]).cast(pl.UInt32),
            cty=pl.lit(c), y=y, fold=folds, p_v2=pl.Series(p_v2, dtype=pl.Float32))
        write_pq(df, train_path(c))
        log(f"TRAIN features {c}: {X.shape} | positive rate {y.mean():.4f}")
        del a, b, cand, X, df; gc.collect()


def load_train():
    df = pl.concat([pl.read_parquet(train_path(c)) for c in MIRROR_C])
    X = df.select(V4_FEATS).to_numpy().astype(np.float32)
    return df.drop(V4_FEATS), X


# ------------------------------------------------------------------ stage: L1 (GPU) + LOCO
def stage_l1():
    if (CACHE / "l1.json").exists() and (CACHE / "l1_oof.parquet").exists():
        log("cache hit: l1"); return
    meta, X = load_train()
    y, folds = meta["y"].to_numpy(), meta["fold"].to_numpy()
    prm = models.params(monotone=[MONOTONE.get(f, 0) for f in V4_FEATS])
    log(f"L1 on {models.device()}: {X.shape} | params {prm}")
    oof, iters, final = models.train_oof(X, y, folds, prm, "L1")
    final.save_model(CACHE / "l1.json")
    meta.select("i1", "i23", "cty", "y", "fold").with_columns(p=pl.Series(oof)).pipe(write_pq, CACHE / "l1_oof.parquet")
    # LOCO: train on one country's TRAIN, score the other's (the France proxy); same decision protocol for both
    rep = []
    ctys = meta["cty"].to_numpy()
    scored = lambda m, p: meta.filter(m).select("i1", "i23").with_columns(p=pl.Series(p))
    truth = lambda m: meta.filter(m & (meta["y"] == 1)).select("i1", "i23")
    for src, tgt in [("us", "india"), ("india", "us")]:
        ms, mt = pl.Series(ctys == src), pl.Series(ctys == tgt)
        thr_src = max(CFG["thr_grid"], key=lambda t: macro_f05(decide.decide_global(decide.exclusivity(scored(ms, oof[ctys == src])), t),
                                                                truth(ms), meta.filter(ms)["i1"].unique()))
        b = models.fit(X[ctys == src], y[ctys == src], prm, int(np.mean(iters)))
        p_t = models.predict(b, X[ctys == tgt])
        i1_t = meta.filter(mt)["i1"].unique()
        f_loco = macro_f05(decide.decide_global(decide.exclusivity(scored(mt, p_t)), thr_src), truth(mt), i1_t)
        f_in = macro_f05(decide.decide_global(decide.exclusivity(scored(mt, oof[ctys == tgt])), thr_src), truth(mt), i1_t)
        rep.append((f"{src} -> {tgt}", f_loco, f_in, f_loco - f_in))
        del b; gc.collect()
    rep = pl.DataFrame(rep, schema=["LOCO", "macro_f05", "in_country_oof", "gap"], orient="row")
    log(f"L1 iters {iters}\n{rep}")
    json.dump(dict(iters=iters, loco=rep.rows()), open(CACHE / "l1_report.json", "w"))


# ------------------------------------------------------------------ stage: score mirror closure + test (L1, V2 anchor)
BLK_COLS = ["blk", "blk_rank", "blk_rev_rank", "rev_n", "blk_gap", "n_cand", "n_passes"]


def p_path(split, c):
    return CACHE / f"p_{split}_{c}.parquet"


def score_edges(edges, a, b, l1, v2=None, tag=""):
    """Stream features in chunks -> V4 L1 score (+ V2 anchor score on the raw 101 columns)."""
    n, ch = edges.height, CFG["score_chunk"]
    p1 = np.empty(n, np.float32)
    p2 = np.empty(n, np.float32) if v2 is not None else None
    for k, lo in enumerate(range(0, n, ch)):
        X = build_features(edges.slice(lo, ch), a, b, desc=f"score {tag} {k + 1}/{-(-n // ch)}")
        p1[lo:lo + len(X)] = models.predict(l1, X[:, V4_IDX])
        if v2 is not None:
            p2[lo:lo + len(X)] = v2.predict(X[:, :len(V2_FEATS)])
        del X; gc.collect()
    return p1, p2


def stage_score():
    import xgboost as xgb
    l1 = xgb.Booster(model_file=str(CACHE / "l1.json"))
    l1.set_param({"device": models.device()})
    v2 = v2_model()
    oof = pl.read_parquet(CACHE / "l1_oof.parquet")
    for split in ["mirror", "test"]:
        o1, o23 = offsets(split)
        for c in countries(split):
            if p_path(split, c).exists():
                log(f"cache hit: {p_path(split, c).name}"); continue
            check_disk(f"score {split} {c}")
            a, b = load_parsed(split, c, "s1"), load_parsed(split, c, "s23")
            cand = pl.read_parquet(cand_path(split, c))
            gid = lambda d: d.with_columns(i1=(pl.col("i1") + o1[c]).cast(pl.UInt32), i23=(pl.col("i23") + o23[c]).cast(pl.UInt32))
            if split == "mirror":
                T, H = pl.DataFrame({"i1": role_idx(c, "train")}), pl.DataFrame({"i1": role_idx(c, "holdout")})
                eT = cand.join(T, on="i1", how="semi")
                recs = pl.concat([eT, cand.join(H, on="i1", how="semi")]).select("i23").unique()
                comp = cand.join(recs, on="i23", how="semi").filter(pl.col("blk_rev_rank") <= CFG["closure_rev_k"])
                edges = (pl.concat([cand.join(H, on="i1", how="semi"), comp]).unique(["i1", "i23"])
                           .join(T, on="i1", how="anti"))
                log(f"mirror {c}: closure {edges.height:,} edges to score (+ {eT.height:,} TRAIN edges with OOF scores)")
                p1, p2 = score_edges(edges, a, b, l1, v2, tag=f"mirror {c}")
                tv2 = pl.read_parquet(train_path(c), columns=["i1", "i23", "p_v2"])
                tr = (gid(eT.select("i1", "i23", *BLK_COLS)).join(oof.select("i1", "i23", p_l1="p"), on=["i1", "i23"])
                          .join(tv2, on=["i1", "i23"]).with_columns(is_train=pl.lit(True)))
                out = pl.concat([gid(edges.select("i1", "i23", *BLK_COLS)).with_columns(
                                     p_l1=pl.Series(p1), p_v2=pl.Series(p2), is_train=pl.lit(False)), tr], how="diagonal_relaxed")
            else:
                p1, _ = score_edges(cand, a, b, l1, None, tag=f"test {c}")
                out = gid(cand.select("i1", "i23", *BLK_COLS)).with_columns(p_l1=pl.Series(p1))
            write_pq(out, p_path(split, c))
            log(f"scored {split} {c}: {out.height:,} edges")
            del a, b, cand, out; gc.collect()


# ------------------------------------------------------------------ stage: L2 context re-ranker (GPU)
def stage_l2():
    if (CACHE / "l2.json").exists():
        log("cache hit: l2"); return
    tab = pl.concat([pl.read_parquet(p_path("mirror", c)) for c in MIRROR_C], how="diagonal_relaxed")
    ctx = models.context_features(tab.rename({"p_l1": "p"}))
    oof1 = pl.read_parquet(CACHE / "l1_oof.parquet").select("i1", "i23", "y", "fold")
    trn = ctx.filter(pl.col("is_train")).join(oof1, on=["i1", "i23"])
    X, y, folds = trn.select(models.L2_FEATS).to_numpy(), trn["y"].to_numpy(), trn["fold"].to_numpy()
    prm = models.params(monotone=[1 if f == "p" else 0 for f in models.L2_FEATS])
    oof, iters, final = models.train_oof(X, y, folds, prm, "L2")
    final.save_model(CACHE / "l2.json")
    trn.select("i1", "i23").with_columns(p_l2=pl.Series(oof)).pipe(write_pq, CACHE / "l2_oof.parquet")
    log(f"L2 iters {iters}")


def add_l2(tab, l2, oof2=None):
    """tab with p_l1 (+ blocking cols) -> + p_l2 (TRAIN rows take their OOF value)."""
    ctx = models.context_features(tab.rename({"p_l1": "p"}))
    p = models.predict(l2, ctx.select(models.L2_FEATS).to_numpy())
    out = tab.with_columns(p_l2=pl.Series(p))
    if oof2 is not None:
        out = (out.join(oof2.rename({"p_l2": "p_oof"}), on=["i1", "i23"], how="left")
                  .with_columns(p_l2=pl.coalesce("p_oof", "p_l2")).drop("p_oof"))
    return out


# ------------------------------------------------------------------ stage: decision layer (TRAIN) + HOLDOUT report
def s1_info(split):
    o1, _ = offsets(split)
    return pl.concat([load_parsed(split, c, "s1", ["idx", "f_core_s1_r", "f_core_s23_r", "f_addr_s1_r", "core_idf", "a_toks", "house"])
                      .select(i1=(pl.col("idx") + o1[c]).cast(pl.UInt32), f_core_s1_r="f_core_s1_r", f_core_s23_r="f_core_s23_r",
                              f_addr_s1_r="f_addr_s1_r", idf_sum=pl.col("core_idf").list.sum(), a_len=pl.col("a_toks").list.len(),
                              has_house=(pl.col("house") != "").cast(pl.Int8)) for c in countries(split)])


def cand_info(split):
    o1, _ = offsets(split)
    return pl.concat([pl.read_parquet(cand_path(split, c), columns=["i1", "blk", "n_passes"]).group_by("i1")
                      .agg(ncand=pl.len(), blk_max=pl.col("blk").max(), npass_max=pl.col("n_passes").max())
                      .with_columns(i1=(pl.col("i1") + o1[c]).cast(pl.UInt32)) for c in countries(split)])


def mirror_truth_global():
    o1, o23 = offsets("mirror")
    return pl.concat([truth_idx(c).with_columns(i1=(pl.col("i1") + o1[c]).cast(pl.UInt32), i23=(pl.col("i23") + o23[c]).cast(pl.UInt32),
                                                cty=pl.lit(c)) for c in MIRROR_C])


def role_global(role):
    o1, _ = offsets("mirror")
    return pl.concat([pl.DataFrame({"i1": (role_idx(c, role) + o1[c]).cast(pl.UInt32), "cty": c}) for c in MIRROR_C])


def stage_decide():
    import xgboost as xgb
    l2 = xgb.Booster(model_file=str(CACHE / "l2.json")); l2.set_param({"device": models.device()})
    tab = pl.concat([pl.read_parquet(p_path("mirror", c)) for c in MIRROR_C], how="diagonal_relaxed")
    tab = add_l2(tab, l2, pl.read_parquet(CACHE / "l2_oof.parquet"))
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
    log(f"decision layer on {T.height:,} TRAIN S1 | {tab.height:,} mirror edges | m_miss {m_miss:.4f}")
    st = {}
    for var in ["p_l1", "p_l2"]:
        st[var] = decide.fit_decision(tab.select("i1", "i23", p=var), y_edge, is_train, f_edge, truth_T, T["i1"],
                                      fold_of_i1, s1i, ci, m_miss, tag=var)
    best = max(st, key=lambda k: st[k]["f_train"])
    # ---- HOLDOUT (never used above): V4 (both variants) and the V2 anchor
    rows = []
    for name, pred in [(f"V4 {v}" + (" (chosen)" if v == best else ""), decide.apply(st[v], tab.select("i1", "i23", p=v), s1i, ci, H["i1"]))
                       for v in st] + [("V2 anchor (L1 thr 0.7 + exclusivity)",
                                        decide.decide_global(decide.exclusivity(tab.select("i1", "i23", p="p_v2")).join(H, on="i1", how="semi"), 0.7))]:
        r = [name, macro_f05(pred, truth_H.select("i1", "i23"), H["i1"])]
        for c in MIRROR_C:
            hc = H.filter(pl.col("cty") == c)
            r.append(macro_f05(pred.join(hc, on="i1", how="semi"), truth_H.filter(pl.col("cty") == c).select("i1", "i23"), hc["i1"]))
        hp = pred.join(H, on="i1", how="semi")
        r += [hp.height / H.height, 1 - hp["i1"].n_unique() / H.height]
        rows.append(r)
    rep = pl.DataFrame(rows, schema=["model", "holdout_f05", *[f"f05_{c}" for c in MIRROR_C], "pred_per_s1", "pred_empty"], orient="row")
    log(f"HOLDOUT (true matches/S1 {truth_H.height / H.height:.3f}, "
        f"true empty {1 - truth_H['i1'].n_unique() / H.height:.4f}):\n{rep}")
    pickle.dump(dict(st=st[best], variant=best, report=rep, ledgers={k: v["ledger"] for k, v in st.items()}),
                open(CACHE / "decision.pkl", "wb"))


# ------------------------------------------------------------------ stage: test export + validator
def stage_export():
    import xgboost as xgb
    d = pickle.load(open(CACHE / "decision.pkl", "rb"))
    st, var = d["st"], d["variant"]
    l2 = xgb.Booster(model_file=str(CACHE / "l2.json")); l2.set_param({"device": models.device()})
    s1i, ci = s1_info("test"), cand_info("test")
    o1, o23 = offsets("test")
    preds, cands, stats = [], [], []
    for c in countries("test"):
        tab = pl.read_parquet(p_path("test", c))
        if var == "p_l2":
            tab = add_l2(tab, l2)
        pred = decide.apply(st, tab.select("i1", "i23", p=var), s1i, ci)
        a = load_parsed("test", c, "s1", ["idx", "entity_id"]).with_columns(i1=(pl.col("idx") + o1[c]).cast(pl.UInt32))
        b = load_parsed("test", c, "s23", ["idx", "entity_id"]).with_columns(i23=(pl.col("idx") + o23[c]).cast(pl.UInt32))
        ids = lambda e: (e.join(a.select("i1", s1_id="entity_id"), on="i1").join(b.select("i23", s23_id="entity_id"), on="i23")
                          .select("s1_id", "s23_id"))
        preds.append(ids(pred)); cands.append(ids(tab.select("i1", "i23")))
        stats.append((c, a.height, pred.height / a.height, 1 - pred["i1"].n_unique() / a.height))
    log(f"test predictions ({var}):\n{pl.DataFrame(stats, schema=['cty', 'n_s1', 'pred_per_s1', 'pred_empty'], orient='row')}")
    all_pred, all_cand = pl.concat(preds), pl.concat(cands)
    assert all_pred.join(all_cand, on=["s1_id", "s23_id"], how="anti").height == 0, "matches must be a subset of candidates"
    assert all_pred["s23_id"].is_unique().all(), "an S2/S3 record may belong to at most one S1"
    order = load_raw("test")[0]["entity_id"]
    OUT.mkdir(parents=True, exist_ok=True)

    def write(pairs, col, path):
        lists = (pairs.unique().sort("s1_id", "s23_id").group_by("s1_id", maintain_order=True)
                      .agg(pl.col("s23_id").str.join(",").alias(col)))
        out = order.to_frame("source1_entity_id").join(lists.rename({"s1_id": "source1_entity_id"}), on="source1_entity_id",
                                                       how="left").fill_null("")
        assert out.height == order.len() and out["source1_entity_id"].is_unique().all()
        path.parent.mkdir(parents=True, exist_ok=True)
        out.write_csv(path, separator="\t", quote_style="never")
        return out

    m = write(all_pred, "matched_entity_ids", OUT / "matching_results.tsv")
    write(all_cand, "candidate_entity_ids", OUT / "candidate_pairs.tsv")
    fr = load_raw("test")[0].filter(pl.col("cty") == "france")["entity_id"]
    write(all_pred.filter(~pl.col("s1_id").is_in(fr.implode())), "matched_entity_ids", OUT / "probe_france_empty" / "matching_results.tsv")
    log(f"wrote {OUT / 'matching_results.tsv'}: {m.height:,} rows, {(m['matched_entity_ids'] != '').sum():,} non-empty, "
        f"{all_pred.height:,} matches | {all_cand.height:,} candidates")
    if SMOKE:
        log("smoke run: official validator skipped (the smoke test set is a sample)"); return
    r = subprocess.run([sys.executable, str(VALIDATOR),
                        "--matching", str(OUT / "matching_results.tsv"), "--candidate", str(OUT / "candidate_pairs.tsv"),
                        "--test-dir", str(DATA / "test"), "--check-ids"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-2000:], r.stderr[-1000:])
    log(f"official validator exit code {r.returncode} -> {'PASS' if r.returncode == 0 else 'FIX ISSUES'}")


def stage_dense():
    if DENSE:
        from .dense import stage_dense as sd
        sd()


STAGES = dict(roles=stage_roles, mine=stage_mine, parse=stage_parse, blkfit=stage_blkfit, dense=stage_dense, block=stage_block,
              trainfeat=stage_trainfeat, l1=stage_l1, score=stage_score, l2=stage_l2, decide=stage_decide, export=stage_export)


class _Tee:
    """Copy console output (stdout + stderr, where tqdm writes) to a UTF-8 log file."""
    def __init__(self, stream, f):
        self.stream, self.f = stream, f

    def write(self, x):
        self.stream.write(x); self.f.write(x); self.f.flush()

    def flush(self):
        self.stream.flush(); self.f.flush()


def link_shared():
    """Dense mode: hard-link the stage outputs that do not depend on the dense pass from the base cache."""
    if CACHE == BASE_CACHE:
        return
    import os
    CACHE.mkdir(parents=True, exist_ok=True)
    for pat in SHARED:
        for f in BASE_CACHE.glob(pat):
            dst = CACHE / f.name
            if not dst.exists():
                os.link(f, dst)


def main():
    link_shared()
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["all", *STAGES])
    ap.add_argument("--log", help="also append all output to this UTF-8 file")
    args = ap.parse_args()
    s = args.stage
    if args.log:
        f = open(args.log, "a", encoding="utf-8")
        sys.stdout, sys.stderr = _Tee(sys.stdout, f), _Tee(sys.stderr, f)
    t0 = time.time()
    for name, fn in STAGES.items():
        if s in ("all", name):
            log(f"===== stage {name} =====")
            fn()
    log(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
