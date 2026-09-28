"""V4.1 dense retrieval pass (GPU): V3's fine-tuned multilingual e5 bi-encoder + exact GPU kNN, per split and country.

The encoder was fine-tuned on V2's mining sample M (train S1 with u in [0.10, 0.20) = V4's MINE role), which is disjoint
from mirror TRAIN / HOLDOUT, so mirror scores stay honest. Rows are aligned to V4's parsed idx via entity_id.

    python -m er4.dense            (embeds mirror + test, writes dense_{split}_{c}.parquet to the shared cache)
"""
import gc, sys, time
from pathlib import Path

import numpy as np
import polars as pl

from .config import V3_WORK, V4, CACHE, CFG, MODELS, STRICT
from .util import cached, log

V3_DIR = V3_WORK
DENSE_K, DENSE_REV = 20, 3


def _v3():
    if str(V3_DIR) not in sys.path:
        sys.path.insert(0, str(V3_DIR))
    from v3 import dense as D3
    return D3


def dense_path(split, c):
    return CACHE / f"dense_{split}_{c}.parquet"


def encoder():
    D3 = _v3()
    if not STRICT:
        return D3.DenseEncoder(str(V3_DIR / "models" / "dense_ft"))
    path = MODELS / "dense_strict"
    if not (path / "config.json").exists():
        train_strict_encoder(path)
    return D3.DenseEncoder(str(path))


def train_strict_encoder(path, steps=3000, bs=384, frac_unsup=0.2):
    """ER4_STRICT: same recipe as V3 (InfoNCE on the MINE sample's true pairs, batches grouped by country|region) but the
    self-supervised noise-variant positives come from TRAIN records only - no test record is seen."""
    D3 = _v3()
    from v3.data import load_train_sample
    M = load_train_sample("M")
    t1, t23 = D3.texts_of(M["raw1"]), D3.texts_of(M["raw23"])
    reg = M["s1p"]["cty"] + "|" + M["s1p"]["region"]
    tr = M["truth"]
    sup = pl.DataFrame({"i1": tr["i1"], "i23": tr["i23"]}).with_columns(
        grp=pl.Series(reg.to_numpy()[tr["i1"].to_numpy()]),
        ta=pl.Series([t1[i] for i in tr["i1"].to_numpy()]), tb=pl.Series([t23[i] for i in tr["i23"].to_numpy()]))
    from .pipeline import load_raw
    s1, s23, _ = load_raw("train")
    pool = pl.concat([s1.select("business_name", "business_address"), s23.select("business_name", "business_address")])
    rng = np.random.default_rng(0)
    unsup = D3.texts_of(pool[np.sort(rng.choice(pool.height, min(400_000, pool.height), replace=False))])
    enc = D3.DenseEncoder()
    log(f"STRICT dense encoder: {sup.height:,} MINE pairs + {len(unsup):,} TRAIN records (no test data), {steps} steps")
    loss = enc.finetune(D3.pair_batches(sup, unsup, bs, frac_unsup), steps, bs)
    path.parent.mkdir(parents=True, exist_ok=True)
    enc.save(path)
    log(f"STRICT dense encoder saved to {path} (loss {loss:.4f})")


def texts_by_idx(parsed_ids, raw):
    """parsed (idx, entity_id) + raw (entity_id, business_name, business_address) -> texts in idx order."""
    D3 = _v3()
    j = parsed_ids.join(raw.select("entity_id", "business_name", "business_address"), on="entity_id", how="left").sort("idx")
    assert j.height == parsed_ids.height and j["business_name"].null_count() == 0
    return D3.texts_of(j)


def stage_dense():
    from .pipeline import countries, load_parsed, load_raw
    D3 = _v3()
    enc = None
    for split in ["mirror", "test"]:
        if all(dense_path(split, c).exists() for c in countries(split)):
            log(f"cache hit: dense {split}"); continue
        s1, s23, _ = load_raw("train" if split == "mirror" else "test")
        for c in countries(split):
            if dense_path(split, c).exists():
                continue
            a = load_parsed(split, c, "s1", ["idx", "entity_id"])
            b = load_parsed(split, c, "s23", ["idx", "entity_id"])
            v3c = V3_DIR / "cache"
            e1 = e23 = None
            if split == "test" and not STRICT and (v3c / f"test_{c}_emb1.npy").exists():   # V3 already embedded the test set
                e1, e23 = np.load(v3c / f"test_{c}_emb1.npy", mmap_mode="r"), np.load(v3c / f"test_{c}_emb23.npy", mmap_mode="r")
                if len(e1) == a.height and len(e23) == b.height:
                    e1, e23 = np.asarray(e1), np.asarray(e23)
                    log(f"dense {split} {c}: reusing V3 test embeddings")
                else:
                    e1 = e23 = None
            if e1 is None:
                enc = enc or encoder()
                t = time.time()
                e1 = enc.encode(texts_by_idx(a, s1), desc=f"embed {split} {c} S1")
                e23 = enc.encode(texts_by_idx(b, s23), desc=f"embed {split} {c} S2S3")
                log(f"dense {split} {c}: embedded {a.height:,} + {b.height:,} records in {time.time() - t:.0f}s")
            d = D3.dense_pass(e1, e23, np.full(a.height, c), np.full(b.height, c), DENSE_K, DENSE_REV)
            d.write_parquet(dense_path(split, c), compression="zstd")
            log(f"dense {split} {c}: {d.height:,} pairs ({d.height / a.height:.1f}/S1)")
            del a, b, e1, e23, d; gc.collect()
        del s1, s23; gc.collect()


if __name__ == "__main__":
    stage_dense()
