"""L7a record-side resolution (evidence pooling) and the F-CLUS feature family (design 5.1, 6.1).

1. Record-record candidates: all pairs among each S1's strongest candidates (student p >= pmin, top-n), plus dense
   record-record cosine for every such pair.
2. Companion duplicate model (GPU XGBoost): label = both records are true matches of the same S1 (OOF on the frozen
   split of the S1 whose list produced the pair).
3. High-precision clusters: connected components on p_link >= strict (0.95) with a size cap; soft links (>= 0.8)
   are kept for features only.
4. F-CLUS for every (S1, candidate): cluster size / members present in this list / source mix, the best member-level
   evidence against this S1 (the fused-profile comparison: best postal, house, number and name agreement over the
   cluster), member score spread, and the soft-link inheritance max(p_link * p_mate).
"""
import numpy as np
import polars as pl
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .common import CFG, log
from . import v2_base as B


def record_pairs(sc, pmin=0.05, top=None):
    """sc: (i1, i23, p) -> record pairs (i1, x, y, p_x, p_y) among the top candidates of each S1."""
    top = top or CFG["clus_top"]
    t = (sc.filter(pl.col("p") >= pmin).with_columns(r=pl.col("p").rank("ordinal", descending=True).over("i1"))
           .filter(pl.col("r") <= top).select("i1", "i23", "p"))
    pr = t.join(t, on="i1", suffix="_y").filter(pl.col("i23") < pl.col("i23_y"))
    return pr.rename({"i23": "x", "i23_y": "y", "p": "p_x"})


COMP_FEATS = B.PAIR_FEATS + ["rr_cos", "rr_same_src"]


def companion_matrix(uniq, s23p, e23):
    """features for unique record pairs (x, y): V2 content features S2/S3-vs-S2/S3 + dense cosine + same source."""
    _, Xu = B.companion_features(uniq, s23p)
    x, y = uniq["x"].to_numpy(), uniq["y"].to_numpy()
    cos = np.einsum("ij,ij->i", e23[x].astype(np.float32), e23[y].astype(np.float32))
    src = s23p["entity_id"].str.slice(0, 2).to_numpy()
    return np.column_stack([Xu, cos, (src[x] == src[y]).astype(np.float32)]).astype(np.float32)


def companion_labels(rp, truth):
    """1 if x and y are true matches of the same S1 (any S1 in the truth)."""
    own = truth.select("i23", "i1").rename({"i1": "o"})
    return (rp.join(own.rename({"i23": "x", "o": "ox"}), on="x", how="left")
              .join(own.rename({"i23": "y", "o": "oy"}), on="y", how="left")
              .select(lab=((pl.col("ox") == pl.col("oy")) & pl.col("ox").is_not_null()).fill_null(False).cast(pl.Int8)))["lab"].to_numpy()


def clusters(links, n23, strict=None, cap=None):
    """links (x, y, p_link) -> cluster id per S2/S3 record (singletons get their own id)."""
    strict, cap = strict or CFG["clus_strict"], cap or CFG["clus_cap"]
    active = links.group_by("x", "y").agg(pl.col("p_link").max())
    fixed = active.clear()
    for thr in (strict, 0.98, 0.99, 0.995):
        e = pl.concat([fixed, active.filter(pl.col("p_link") >= thr)])
        g = coo_matrix((np.ones(e.height), (e["x"].to_numpy(), e["y"].to_numpy())), shape=(n23, n23))
        _, lab = connected_components(g, directed=False)
        big = np.bincount(lab)[lab] > cap
        if not big.any():
            break
        # components within the cap are final; only oversized ones are re-split at a stricter threshold
        fixed = e.filter(pl.Series(~big[e["x"].to_numpy()]))
        active = active.filter(pl.Series(big[active["x"].to_numpy()]))
    lab = lab.astype(np.int64)
    size = np.bincount(lab)
    over = size[lab] > cap                                  # still oversized: dissolve into singletons
    lab[over] = n23 + np.where(over)[0]
    return lab


CLUS_FEATS = ["cl_size", "cl_in_list", "cl_src_mix", "cl_p_max_other", "cl_p_mean", "cl_p_std", "cl_best_house",
              "cl_best_postal", "cl_best_num", "cl_best_name", "cl_best_addr", "cl_best_dcos", "cl_inherit",
              "cl_soft_n", "cl_link_max"]


def clus_features(pool, p, X, feats, lab, links, s23p, soft=None):
    """pool (i1, i23) rows aligned with p and X -> F-CLUS matrix [n, len(CLUS_FEATS)]."""
    soft = soft or CFG["clus_soft"]
    src3 = s23p["entity_id"].str.starts_with("S3").to_numpy()
    fx = lambda f: X[:, feats.index(f)]
    d = pl.DataFrame({"row": np.arange(pool.height, dtype=np.uint32), "i1": pool["i1"], "i23": pool["i23"],
                      "p": p, "cl": lab[pool["i23"].to_numpy()], "s3": src3[pool["i23"].to_numpy()],
                      "h": fx("a_house"), "po": fx("a_postal"), "nu": fx("a_num_jacc"), "nm": fx("n_jw_core"),
                      "ad": fx("a_tset"), "dc": fx("dcos")})
    size = np.bincount(lab)
    g = d.group_by("i1", "cl").agg(n=pl.len(), s3n=pl.col("s3").sum(), psum=pl.col("p").sum(), pmax=pl.col("p").max(),
                                   p2=pl.col("p").top_k(2).min(), pmean=pl.col("p").mean(), pstd=pl.col("p").std(),
                                   bh=pl.col("h").max(), bpo=pl.col("po").max(), bnu=pl.col("nu").max(),
                                   bnm=pl.col("nm").max(), bad=pl.col("ad").max(), bdc=pl.col("dc").max())
    d = d.join(g, on=["i1", "cl"], how="left").sort("row")
    other_max = pl.when(pl.col("n") <= 1).then(-1.0).when(pl.col("p") >= pl.col("pmax")).then(pl.col("p2")).otherwise(pl.col("pmax"))
    # soft links: inheritance max over list mates of p_link * p_mate, and count of soft mates
    L = links.filter(pl.col("p_link") >= soft)
    both = pl.concat([L.select("i1", a="x", b="y", pl_="p_link"), L.select("i1", a="y", b="x", pl_="p_link")])
    inh = (both.join(d.select("i1", b="i23", pb="p"), on=["i1", "b"])
               .group_by("i1", "a").agg(inh=(pl.col("pl_") * pl.col("pb")).max(), soft_n=pl.len(), lmax=pl.col("pl_").max())
               .rename({"a": "i23"}))
    d = d.join(inh, on=["i1", "i23"], how="left").sort("row")
    out = d.select(
        cl_size=pl.Series(size[d["cl"].to_numpy()].astype(np.float32)),
        cl_in_list=pl.col("n").cast(pl.Float32),
        cl_src_mix=((pl.col("s3n") > 0) & (pl.col("s3n") < pl.col("n"))).cast(pl.Float32),
        cl_p_max_other=other_max.cast(pl.Float32),
        cl_p_mean=pl.col("pmean"), cl_p_std=pl.col("pstd").fill_null(0),
        cl_best_house=pl.col("bh"), cl_best_postal=pl.col("bpo"), cl_best_num=pl.col("bnu"), cl_best_name=pl.col("bnm"),
        cl_best_addr=pl.col("bad"), cl_best_dcos=pl.col("bdc"),
        cl_inherit=pl.col("inh").fill_null(0), cl_soft_n=pl.col("soft_n").fill_null(0).cast(pl.Float32),
        cl_link_max=pl.col("lmax").fill_null(0))
    return out.to_numpy().astype(np.float32)
