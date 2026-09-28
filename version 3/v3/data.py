"""L1 ingestion + the V2 data/representation layers (L4 normalization, V2 blocking) for train samples and test countries.

Train side: the V2 density-preserving samples are reused unchanged
  - S (modelling sample): every learned V3 component, OOF on the frozen split
  - M (mining sample, disjoint): variant tables (V2) and, in V3, the dense bi-encoder + companion pretraining
Test side: one country at a time (country labels are an open set; matched pairs never cross a label).
"""
import gc
import numpy as np
import polars as pl

from .common import CACHE, V2CACHE, DATA, cached, v2_cached, log, load_v2_state
from . import v2_base as B


def read_tsv(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).fill_null("")


# --------------------------------------------------------------------------------------------- train sample S
def load_train_sample(name="S"):
    """-> dict(s1p, s23p, truth (i1,i23), raw1, raw23) for V2 sample S or M (parsed pass-2, enriched)."""
    load_v2_state()
    samples = v2_cached("samples.pkl")
    s1raw, s23raw, pr = samples[name]
    del samples
    if name == "S":
        s1p, s23p = v2_cached("S_parsed.pkl")
    else:
        s1p, s23p = v2_cached("M_parsed.pkl")
    truth = B.truth_idx(s1p, s23p, pr)
    raw1 = s1p.select("idx", "entity_id").join(s1raw.drop("country"), on="entity_id", how="left").sort("idx")
    raw23 = s23p.select("idx", "entity_id").join(s23raw.drop("country"), on="entity_id", how="left").sort("idx")
    assert raw1.height == s1p.height and raw23.height == s23p.height
    return dict(s1p=s1p, s23p=s23p, truth=truth, raw1=raw1, raw23=raw23, name=name)


def blk_scorer():
    sc, curve, _, _ = v2_cached("blk_scorer.pkl")
    return sc


# --------------------------------------------------------------------------------------------- test countries
_TEST = {}


def test_raw():
    if not _TEST:
        te1 = read_tsv(DATA / "test" / "test_source1.tsv")
        te23 = pl.concat([read_tsv(DATA / "test" / f"test_source{s}.tsv") for s in (2, 3)])
        te1 = te1.with_columns(cty=pl.col("country").str.strip_chars().str.to_lowercase())
        te23 = te23.with_columns(cty=pl.col("country").str.strip_chars().str.to_lowercase())
        _TEST.update(te1=te1, te23=te23)
    return _TEST["te1"], _TEST["te23"]


def test_countries():
    te1, _ = test_raw()
    return te1["cty"].unique(maintain_order=True).to_list()


def load_test_country(c):
    """V2 parse (with mined tables) + enrich + V2 blocking (top-50) for one test country; reuses V2 caches."""
    subs = load_v2_state()
    te1, te23 = test_raw()
    a_raw = te1.filter(pl.col("cty") == c).drop("cty")
    b_raw = te23.filter(pl.col("cty") == c).drop("cty")
    if (V2CACHE / f"test_{c}_parsed.pkl").exists():
        a, b = v2_cached(f"test_{c}_parsed.pkl")
    else:
        a, b = cached(f"test_{c}_parsed.pkl", lambda: B.enrich(B.parse_frame(a_raw, subs, desc=f"parse test {c} S1"),
                                                               B.parse_frame(b_raw, subs, desc=f"parse test {c} S2S3")))
    if (V2CACHE / f"test_{c}_cand.parquet").exists():
        cand = v2_cached(f"test_{c}_cand.parquet")
    else:
        sc = blk_scorer()
        cand = cached(f"test_{c}_cand.parquet", lambda: B.block_all(a, b, sc, B.CFG["n_cand"], tag=f"test {c}"))
    raw1 = a.select("idx", "entity_id").join(a_raw.drop("country"), on="entity_id", how="left").sort("idx")
    raw23 = b.select("idx", "entity_id").join(b_raw.drop("country"), on="entity_id", how="left").sort("idx")
    return dict(s1p=a, s23p=b, cand=cand, raw1=raw1, raw23=raw23, name=f"test_{c}", truth=None)


if __name__ == "__main__":
    # pre-compute the V2 base layers for every test country (CPU-heavy; run in the background)
    import sys
    for c in (sys.argv[1:] or test_countries()):
        log(f"==== base layers for test country {c}")
        d = load_test_country(c)
        log(f"{c}: {d['s1p'].height:,} S1 | {d['s23p'].height:,} S2/S3 | {d['cand'].height:,} V2 candidates")
        del d
        gc.collect()
