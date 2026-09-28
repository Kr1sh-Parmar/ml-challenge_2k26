"""Inverted-index blocking (ported from V2, Parts 3.1 and 4).

V4 changes: the reverse pass always indexes ALL S1s of the country (even when only a subset is queried), and the
competition features are computed over all edges of the country, so they mean the same thing on mirror and test."""
import math

import numpy as np
import polars as pl
from tqdm.auto import tqdm

from .config import CFG


# ------------------------------------------------------------------ engine
def keys_from(df, exprs):
    """df: parsed frame. exprs: list of (prefix, polars expr giving a String or List[String]) -> (idx, key u64)."""
    parts = []
    for prefix, e in exprs:
        k = df.select("idx", k=e)
        if k.schema["k"] == pl.List(pl.String):
            k = k.explode("k")
        parts.append(k.drop_nulls("k").filter(pl.col("k") != "").select("idx", key=(pl.lit(prefix) + pl.col("k")).hash(7)))
    return pl.concat(parts).unique()


def gram_keys(df, col, n=4, prefix="g"):
    return (df.select("idx", s=pl.col(col)).filter(pl.col("s").str.len_chars() >= n)
              .with_columns(pos=pl.int_ranges(0, pl.col("s").str.len_chars() - n + 1)).explode("pos")
              .select("idx", key=(pl.lit(prefix) + pl.col("s").str.slice(pl.col("pos"), n)).hash(7)))


def postings(keys, n_pool, cap):
    cnt = keys.group_by("key").len()
    return (keys.join(cnt.filter(pl.col("len") <= cap), on="key")
                .with_columns(idf=(n_pool / pl.col("len")).log().cast(pl.Float32)).drop("len"))


def index_join(q, post, top_k, qname="i1", cname="i23"):
    """q: (idx, key) of query records; post: (idx, key, idf) of candidates -> top_k candidates per query."""
    return (q.join(post, on="key", suffix="_c")
             .group_by("idx", "idx_c").agg(score=pl.col("idf").sum())
             .with_columns(rank=pl.col("score").rank("ordinal", descending=True).over("idx").cast(pl.UInt16))
             .filter(pl.col("rank") <= top_k)
             .rename({"idx": qname, "idx_c": cname}))


def chunks_of(df, rows):
    for lo in range(0, df.height, rows):
        yield df.slice(lo, rows)


# ------------------------------------------------------------------ passes
def keys_p1(p):
    return pl.concat([gram_keys(p, "k_joined", 4, "g"), gram_keys(p.filter(pl.col("url") != ""), "url", 4, "g"),
                      keys_from(p, [("j", pl.col("k_joined")), ("j", pl.col("url")), ("k", pl.col("k_skel")),
                                    ("t", pl.col("k_trans"))])]).unique()


def keys_p2(p):
    short = pl.when((pl.col("core").list.len() == 1) & (pl.col("k_joined").str.len_chars() <= 5)).then(pl.col("k_joined"))
    return keys_from(p, [("n", pl.col("core")), ("n", pl.col("trade")), ("s", pl.col("k_sorted")), ("m", pl.col("k_meta")),
                         ("a", pl.col("acr")), ("a", short)])


def keys_p3(p):
    h, c0 = pl.col("house"), pl.col("core").list.first().fill_null("")
    return keys_from(p, [("h", pl.col("anchor")), ("hn", pl.when(h != "").then(h + "|" + c0)), ("p", pl.col("postal")),
                         ("pn", pl.when(pl.col("postal") != "").then(pl.col("postal") + "|" + c0)),
                         ("hu", pl.when((h != "") & (pl.col("unit") != "")).then(h + "|" + pl.col("unit")))])


def keys_p5(p):
    ex = (p.select("idx", t="a_toks").with_columns(pos=pl.int_ranges(pl.col("t").list.len()))
           .explode("t", "pos").drop_nulls("t"))
    nxt = ex.with_columns(pos=pl.col("pos") - 1).rename({"t": "t2"})
    bg = (ex.filter(pl.col("t").str.contains(r"\d")).join(nxt, on=["idx", "pos"])
            .select("idx", key=(pl.lit("b") + pl.col("t") + "_" + pl.col("t2")).hash(7)))
    return pl.concat([keys_from(p, [("x", pl.col("a_toks"))]), bg]).unique()


def keys_rev(p):
    return pl.concat([keys_p2(p), keys_from(p, [("j", pl.col("k_joined")), ("k", pl.col("k_skel")),
                                                ("t", pl.col("k_trans"))])]).unique()


def keys_p7(p):
    return pl.concat([keys_p2(p), keys_p5(p)]).unique()


KEYFN = {"p1": keys_p1, "p2": keys_p2, "p3": keys_p3, "p5": keys_p5, "p7": keys_p7}
PASSES = list(KEYFN)
BLK_FEATS = ([f"score_{p}" for p in PASSES] + [f"lrank_{p}" for p in PASSES] + [f"found_{p}" for p in PASSES]
             + ["rev_score", "lrev_rank", "found_rev", "p6", "n_passes"])


def dup_groups(b, max_size):
    g = (b.filter((pl.col("core").list.len() >= 2) | (pl.col("k_joined").str.len_chars() >= 6))
          .group_by("k_sorted").agg(members=pl.col("idx"), n=pl.len())
          .filter((pl.col("n") >= 2) & (pl.col("n") <= max_size)).with_row_index("gid"))
    return g.select("gid", "members").explode("members").rename({"members": "i23"})


def union_chunk(part, posts, rev, grp):
    """part: a slice of query S1s sorted by idx (its idx range holds no other query S1)."""
    long = []
    for p in PASSES:
        r = index_join(KEYFN[p](part), posts[p], CFG["top_k"][p])
        long.append(r.select("i1", "i23", pl.lit(p).alias("pass_"), "score", "rank"))
    w = pl.concat(long).pivot(on="pass_", index=["i1", "i23"], values=["score", "rank"])
    for p in PASSES:
        for c_ in (f"score_{p}", f"rank_{p}"):
            if c_ not in w.columns:
                w = w.with_columns(pl.lit(None, pl.Float32).alias(c_))
    lo, hi = part["idx"].min(), part["idx"].max()
    w = w.join(rev.filter(pl.col("i1").is_between(lo, hi)), on=["i1", "i23"], how="full", coalesce=True)
    found = [pl.col(f"score_{p}").is_not_null() for p in PASSES]
    w = w.with_columns(n_passes=pl.sum_horizontal(found + [pl.col("rev_rank").is_not_null()]).cast(pl.UInt8),
                       best_rank=pl.min_horizontal([pl.col(f"rank_{p}") for p in PASSES] + [pl.col("rev_rank")]))
    src = w.filter((pl.col("n_passes") >= 2) | (pl.col("rank_p2") <= 3) | (pl.col("rank_p1") <= 3))
    mates = (src.select("i1", "i23", "best_rank").join(grp, on="i23").join(grp.rename({"i23": "mate"}), on="gid")
                .filter(pl.col("mate") != pl.col("i23"))
                .group_by("i1", "mate").agg(pl.col("best_rank").min())
                .rename({"mate": "i23"}).join(w.select("i1", "i23"), on=["i1", "i23"], how="anti")
                .with_columns(r=pl.col("best_rank").rank("ordinal").over("i1")).filter(pl.col("r") <= CFG["p6_max"])
                .select("i1", "i23", p6=pl.lit(1, pl.UInt8)))
    w = pl.concat([w.with_columns(p6=pl.lit(0, pl.UInt8)), mates], how="diagonal_relaxed")
    return w.drop("best_rank").with_columns(pl.col("n_passes").fill_null(0))


def blk_matrix(w):
    ex = []
    for p in PASSES:
        k = CFG["top_k"][p] + 1
        ex += [pl.col(f"score_{p}").fill_null(0).alias(f"score_{p}"),
               pl.col(f"rank_{p}").fill_null(k).cast(pl.Float32).log1p().alias(f"lrank_{p}"),
               pl.col(f"score_{p}").is_not_null().cast(pl.Float32).alias(f"found_{p}")]
    ex += [pl.col("rev_score").fill_null(0), pl.col("rev_rank").fill_null(CFG["rev_k"] + 1).cast(pl.Float32).log1p().alias("lrev_rank"),
           pl.col("rev_rank").is_not_null().cast(pl.Float32).alias("found_rev"), pl.col("p6").cast(pl.Float32),
           pl.col("n_passes").cast(pl.Float32)]
    return w.select(ex).select(BLK_FEATS).to_numpy().astype(np.float32)


def block_country(a_all, b, query_idx=None, scorer=None, n_cand=None, tag=""):
    """Block S1s of one country against its full S2/S3 pool.
    a_all: ALL S1s of the country (the reverse pass indexes all of them); query_idx: the S1 idx to return (None = all).
    With a scorer: keep the top n_cand per S1 by blocking score `blk`."""
    n23, n1 = b.height, a_all.height
    caps = {p: max(20, math.ceil(CFG["cap_frac"][p] * n23)) for p in PASSES}
    posts = {p: postings(KEYFN[p](b), n23, caps[p]) for p in tqdm(PASSES, desc=f"postings {tag}", mininterval=10)}
    rpost = postings(keys_rev(a_all), n1, max(20, math.ceil(CFG["cap_frac"]["rev"] * n1)))
    step_r = max(5_000, int(CFG["chunk_rows"] / max(n1, 1)))
    rev = [index_join(keys_rev(part), rpost, CFG["rev_k"], qname="i23", cname="i1")
           for part in tqdm(list(chunks_of(b, step_r)), desc=f"reverse pass {tag}", mininterval=10)]
    rev = pl.concat(rev).select("i1", "i23", rev_score="score", rev_rank="rank")
    del rpost
    a = a_all if query_idx is None else a_all.join(pl.DataFrame({"idx": query_idx}).cast({"idx": a_all.schema["idx"]}),
                                                   on="idx", how="semi")
    a = a.sort("idx")
    rev = rev.join(a.select(pl.col("idx").alias("i1")), on="i1", how="semi")
    grp = dup_groups(b, CFG["dup_group_max"])
    step = max(5_000, int(CFG["chunk_rows"] / n23))
    out = []
    for part in tqdm(list(chunks_of(a, step)), desc=f"blocking {tag}", mininterval=10):
        w = union_chunk(part, posts, rev, grp)
        if scorer is not None:
            w = (w.with_columns(blk=pl.Series(scorer.predict_proba(blk_matrix(w))[:, 1], dtype=pl.Float32))
                  .with_columns(blk_rank=pl.col("blk").rank("ordinal", descending=True).over("i1").cast(pl.UInt16))
                  .filter(pl.col("blk_rank") <= n_cand))
        out.append(w)
    return pl.concat(out, how="diagonal_relaxed")


def competition_features(cand):
    """Over ALL edges of a country: list context per S1 and competition per S2/S3 record."""
    return cand.with_columns(
        blk_gap=pl.col("blk").max().over("i1") - pl.col("blk"),
        n_cand=pl.len().over("i1").cast(pl.UInt16),
        blk_rev_rank=pl.col("blk").rank("ordinal", descending=True).over("i23").cast(pl.UInt16),
        rev_n=pl.len().over("i23").cast(pl.UInt16),
        blk_rev_gap=pl.col("blk").max().over("i23") - pl.col("blk"))


def add_dense(cand, dense, scorer, k=20):
    """V4.1: union top-N blocking edges with dense top-k (+ reverse) edges. Dense-only rows get the blocking scorer's
    value for 'no lexical pass found it'; blk_rank is recomputed over the union (same on mirror and test)."""
    dd = dense.filter((pl.col("drank") <= k) | pl.col("drev_rank").is_not_null()).cast({"i1": cand.schema["i1"], "i23": cand.schema["i23"]})
    u = cand.with_columns(in_blk=pl.lit(1, pl.UInt8)).join(dd, on=["i1", "i23"], how="full", coalesce=True)
    new = u["in_blk"].is_null()
    u = u.with_columns(in_blk=pl.col("in_blk").fill_null(0), p6=pl.col("p6").fill_null(0), n_passes=pl.col("n_passes").fill_null(0))
    if new.any():
        nb = u.filter(new)
        blk_new = scorer.predict_proba(blk_matrix(nb))[:, 1].astype(np.float32)
        u = pl.concat([u.filter(~new), nb.with_columns(blk=pl.Series(blk_new, dtype=pl.Float32))], how="diagonal_relaxed")
    return u.with_columns(blk_rank=pl.col("blk").rank("ordinal", descending=True).over("i1").cast(pl.UInt16))
