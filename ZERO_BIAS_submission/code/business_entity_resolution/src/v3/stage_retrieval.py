"""Stage L6/L7b: fine-tune the dense bi-encoder, embed a dataset, run the dense pass (GPU)."""
import gc
import numpy as np
import polars as pl

from .common import CFG, CACHE, log, cached
from . import dense as D


def get_encoder():
    from .data import load_train_sample, test_raw
    if (D.MODELS / "dense_ft" / "config.json").exists():
        return D.DenseEncoder(str(D.MODELS / "dense_ft"))
    M = load_train_sample("M")
    te1, te23 = test_raw()
    rng = np.random.default_rng(0)
    pool = pl.concat([te1.select("business_name", "business_address"), te23.select("business_name", "business_address")])
    sel = pool[np.sort(rng.choice(pool.height, 400_000, replace=False))]
    test_texts = D.texts_of(sel)
    enc = D.train_dense(M, test_texts)
    del M
    gc.collect()
    return enc


def embed(d, enc):
    tag = d["name"]
    e1 = cached(f"{tag}_emb1.npy", lambda: enc.encode(D.texts_of(d["raw1"]), desc=f"embed {tag} S1"))
    e23 = cached(f"{tag}_emb23.npy", lambda: enc.encode(D.texts_of(d["raw23"]), desc=f"embed {tag} S2S3"))
    return e1, e23


def dense_candidates(d, enc):
    e1, e23 = embed(d, enc)
    cty1, cty23 = d["s1p"]["cty"].to_numpy(), d["s23p"]["cty"].to_numpy()
    return cached(f"{d['name']}_dense.parquet", lambda: D.dense_pass(e1, e23, cty1, cty23, CFG["dense_topk"]))


if __name__ == "__main__":
    from .data import load_train_sample
    from .common import v2_cached
    enc = get_encoder()
    S = load_train_sample("S")
    dp = dense_candidates(S, enc)
    tr = S["truth"]
    v2 = v2_cached("S_cand.parquet").select("i1", "i23")
    for K in [5, 10, 15, 20]:
        dd = dp.filter((pl.col("drank") <= K) | pl.col("drev_rank").is_not_null())
        hit = tr.join(dd, on=["i1", "i23"], how="semi").height
        u = pl.concat([v2, dd.select("i1", "i23")]).unique()
        hu = tr.join(u, on=["i1", "i23"], how="semi").height
        log(f"K={K}: dense recall {hit / tr.height:.4f} | V2 union dense {hu / tr.height:.4f} | "
            f"added pairs/S1 {(u.height - v2.height) / S['s1p'].height:.1f}")
