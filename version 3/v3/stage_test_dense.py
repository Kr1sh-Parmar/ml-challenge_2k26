"""Precompute test-side L6 embeddings + the L7b dense pass per country (GPU), without loading the parsed tables.
Row order = the country-filtered file order, which is exactly the V2 parse index (parse_frame's with_row_index)."""
import gc
import numpy as np
import polars as pl

from .common import log, cached
from . import dense as D
from .data import test_raw, test_countries
from .stage_retrieval import get_encoder
from .common import CFG

if __name__ == "__main__":
    enc = get_encoder()
    te1, te23 = test_raw()
    for c in test_countries():
        a = te1.filter(pl.col("cty") == c)
        b = te23.filter(pl.col("cty") == c)
        tag = f"test_{c}"
        e1 = cached(f"{tag}_emb1.npy", lambda: enc.encode(D.texts_of(a), desc=f"embed {tag} S1"))
        e23 = cached(f"{tag}_emb23.npy", lambda: enc.encode(D.texts_of(b), desc=f"embed {tag} S2S3"))
        cached(f"{tag}_dense.parquet", lambda: D.dense_pass(e1, e23, np.full(len(a), c), np.full(len(b), c), CFG["dense_topk"]))
        del e1, e23; gc.collect()
    log("test dense done")
