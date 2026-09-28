"""Smoke test of the full test-inference path on a slice of one country (no cache writes except the reads)."""
import sys, pickle
import numpy as np

from . import infer as I
from .common import CACHE, log

N_S1 = 20_000


def fake_cached(name, fn, refresh=False):
    if name.endswith("_pruned.pkl"):
        with open(CACHE / name, "rb") as f:
            pp, Xp, ps = pickle.load(f)
        m = pp["i1"].to_numpy() < N_S1
        log(f"smoke: {name} sliced to {m.sum():,} pairs of the first {N_S1:,} S1")
        return pp.filter(m), Xp[m], ps[m]
    return fn()


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "us"
    I.cached = fake_cached
    M = I.load_models()
    d, cand, pred, stats = I.run_country(c, M, None)
    k = pred.group_by("i1").len()
    log(f"smoke OK: {cand.height:,} candidates, {pred.height:,} matches, "
        f"{k.height:,} of {N_S1:,} S1 with >=1 match; stats {stats}")
