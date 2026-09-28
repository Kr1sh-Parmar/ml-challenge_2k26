"""Precompute the test pruned sets (pool features + L8 student on GPU + adaptive M) per country.
Needs only models/student.json, so it can run while the later training stages are still going."""
import gc, time
import xgboost as xgb

from .common import CACHE, MODELS, log, cached
from . import pool as PL
from .data import load_test_country, test_countries
from .infer import pruned_pool
from .stage_retrieval import dense_candidates

if __name__ == "__main__":
    st = xgb.Booster(); st.load_model(str(MODELS / "student.json"))
    M = {"student": st}
    for c in test_countries():
        while not (CACHE / f"test_{c}_dense.parquet").exists():   # produced by stage_test_dense (GPU)
            time.sleep(30)
        d = load_test_country(c)
        dense = dense_candidates(d, None)          # cached by stage_test_dense
        pool = PL.build_pool(d["cand"], dense)
        log(f"[{d['name']}] pool {pool.height:,} pairs ({pool.height / d['s1p'].height:.1f}/S1)")
        cached(f"{d['name']}_pruned.pkl", lambda: pruned_pool(d, pool, M))
        del d, dense, pool; gc.collect()
    log("test prune done")
