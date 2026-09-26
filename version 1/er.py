"""Shared helpers for the V1 entity-resolution notebooks (M0 + M1).

Lives in a .py file (not a notebook cell) so joblib workers on Windows can import it.
"""
import math
import re
import time
from pathlib import Path

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from tqdm.auto import tqdm
from unidecode import unidecode

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "student_resource" / "dataset"
CACHE = Path(__file__).resolve().parent / "cache"


# ---------------------------------------------------------------- S0 ingestion
def read_tsv(path):
    """All strings, no NA detection, no quoting (V1 §4)."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).fill_null("")


def load_split(split):
    """-> (s1, s23, gt or None). s23 = S2 and S3 stacked; source comes from the ID prefix."""
    d = DATA / split
    s1 = read_tsv(d / f"{split}_source1.tsv")
    s23 = pl.concat([read_tsv(d / f"{split}_source2.tsv"), read_tsv(d / f"{split}_source3.tsv")])
    gt = read_tsv(d / "train_ground_truth.tsv") if split == "train" else None
    return s1, s23, gt


def gt_pairs(gt):
    """Ground truth -> long table (s1_id, s23_id)."""
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
              .explode("matched_entity_ids")
              .filter(pl.col("matched_entity_ids") != "")
              .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "s23_id"}))


# ------------------------------------------------------------ S2 normalization
# ponytail: small seed map only (V1 §6.2 "contract, never expand"); mined tables come in M2.
ABBR = {
    "street": "st", "road": "rd", "avenue": "ave", "av": "ave", "boulevard": "blvd", "bd": "blvd",
    "drive": "dr", "lane": "ln", "court": "ct", "circle": "cir", "highway": "hwy", "place": "pl",
    "trail": "trl", "parkway": "pkwy", "square": "sq", "terrace": "ter", "suite": "ste",
    "apartment": "apt", "building": "bldg", "floor": "fl", "north": "n", "south": "s",
    "east": "e", "west": "w", "saint": "st", "number": "no", "house": "h",
    "limited": "ltd", "private": "pvt", "corporation": "corp", "incorporated": "inc",
    "company": "co", "and": "&", "et": "&", "enterprises": "enterprise", "traders": "trader",
    "industries": "industry", "services": "service", "associates": "associate",
}
_NON_ALNUM = re.compile(r"[^a-z0-9&]+")
_EMPTY = {"none", "null", "nan", "n/a", "na", "<null>", "-"}


def norm(s):
    if not s.isascii():
        s = unidecode(s)
    s = s.lower().replace("'", "")
    if s.strip() in _EMPTY:
        return ""
    return " ".join(ABBR.get(t, t) for t in _NON_ALNUM.sub(" ", s).split())


def add_norm(df):
    """Adds global row index `idx`, normalized name/address and a trimmed, case-folded country key."""
    tq = dict(total=df.height, mininterval=2, leave=False)
    return df.with_row_index("idx").with_columns(
        pl.Series("name_n", [norm(x) for x in tqdm(df["business_name"], desc="norm names", **tq)]),
        pl.Series("addr_n", [norm(x) for x in tqdm(df["business_address"], desc="norm addrs", **tq)]),
        cty=pl.col("country").str.strip_chars().str.to_lowercase(),
    )


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- S4 blocking
def record_keys(name, addr):
    """Blocking keys: name unigrams, address unigrams, (number, next word) address bigrams,
    sorted-token name key and name char 4-grams. Frequent keys are dropped by the posting cap,
    so no stopword list is needed."""
    nt, at = name.split(), addr.split()
    keys = {"n" + t for t in nt}
    keys.update("a" + t for t in at)
    keys.update(f"b{a}_{b}" for a, b in zip(at, at[1:]) if a[:1].isdigit())
    keys.add("s" + "".join(sorted(nt)))  # word-order swaps
    flat = "".join(nt)
    keys.update("g" + flat[i:i + 4] for i in range(len(flat) - 3))  # typos, joined words
    return list(keys)


def key_table(df, desc="keys", batch=200_000):
    """-> (idx u32, key u64 hash). Batched: Python key strings for millions of records won't fit in RAM."""
    out = []
    for lo in tqdm(range(0, df.height, batch), desc=desc, leave=False):
        d = df.slice(lo, batch)
        keys = [record_keys(n, a) for n, a in zip(d["name_n"], d["addr_n"])]
        out.append(pl.DataFrame({"idx": d["idx"], "k": keys}).explode("k").drop_nulls("k")
                     .select("idx", pl.col("k").hash().alias("key")))
    return pl.concat(out)


def block_all(s1, s23, top_k=30, cap_frac=2e-4, chunk_rows=2e10):
    """Inverted-index blocking, run per country label (matched pairs always share it, A4; labels
    are an open set, so every S1 country is processed, France included).
    Score(s1, s23) = sum of IDF over shared keys; keys with > cap postings are skipped; top_k per S1.
    ponytail: one sparse pass; dense/reverse/transitive passes (V1 P4-P6) come in M2."""
    out = []
    for c in s1["cty"].unique(maintain_order=True):
        a, b = s1.filter(pl.col("cty") == c), s23.filter(pl.col("cty") == c)
        if b.height == 0:
            log(f"{c}: no S2/S3 records, {a.height:,} S1 get no candidates")
            continue
        n23 = b.height
        cap = max(20, math.ceil(cap_frac * n23))
        chunk = max(5_000, int(chunk_rows / n23))  # join size grows with pool size -> smaller chunks
        log(f"{c}: {a.height:,} S1 vs {n23:,} S2/S3 | posting cap {cap} | chunk {chunk:,}")
        k23 = key_table(b, f"keys S2/S3 {c}")
        post = (k23.join(k23.group_by("key").len().filter(pl.col("len") <= cap), on="key")
                   .with_columns(idf=(n23 / pl.col("len")).log().cast(pl.Float32)).drop("len"))
        del k23
        k1 = key_table(a, f"keys S1 {c}")
        ids = a["idx"]  # ascending, so each chunk is a contiguous idx range
        for lo in tqdm(range(0, a.height, chunk), desc=f"block {c}", mininterval=2):
            hi = ids[min(lo + chunk, a.height) - 1]
            part = k1.filter(pl.col("idx").is_between(ids[lo], hi))
            out.append(part.join(post, on="key", suffix="23")
                           .group_by("idx", "idx23")
                           .agg(blk_score=pl.col("idf").sum(), blk_nkeys=pl.len().cast(pl.UInt16))
                           .with_columns(blk_rank=pl.col("blk_score").rank("ordinal", descending=True)
                                         .over("idx").cast(pl.UInt16))
                           .filter(pl.col("blk_rank") <= top_k))
        del post, k1
    cand = pl.concat(out).rename({"idx": "i1", "idx23": "i23"}).sort("i1", "blk_rank")
    return cand.with_columns(  # context features: competition in the S1's list and on the S2/S3 side
        blk_gap=pl.col("blk_score").max().over("i1") - pl.col("blk_score"),
        blk_rev_rank=pl.col("blk_score").rank("ordinal", descending=True).over("i23").cast(pl.UInt16),
        n_cand=pl.len().over("i1").cast(pl.UInt16),
    )


# ---------------------------------------------------------------- S5 features
def _pair_feats(a_names, b_names, a_addrs, b_addrs):
    rows = []
    for an, bn, aa, ba in zip(a_names, b_names, a_addrs, b_addrs):
        at, bt = set(aa.split()), set(ba.split())
        anum = {t for t in at if t[:1].isdigit()}
        bnum = {t for t in bt if t[:1].isdigit()}
        rows.append((
            fuzz.ratio(an, bn), fuzz.token_set_ratio(an, bn), fuzz.token_sort_ratio(an, bn),
            fuzz.partial_ratio(an, bn), JaroWinkler.similarity(an, bn),
            fuzz.token_set_ratio(aa, ba), fuzz.ratio(aa, ba),
            len(at & bt) / max(1, len(at | bt)),
            # numbers: -1 = one side has none (missing is not a mismatch, V1 §9)
            len(anum & bnum) / len(anum | bnum) if anum and bnum else -1.0,
            fuzz.partial_ratio(an, ba),  # name hidden in the other address (landmark / swap)
        ))
    return rows


FEATS = ["name_ratio", "name_tset", "name_tsort", "name_partial", "name_jw",
         "addr_tset", "addr_ratio", "addr_jacc", "num_jacc", "name_in_addr"]
MODEL_FEATS = ["blk_score", "blk_nkeys", "blk_rank", "blk_gap", "blk_rev_rank", "n_cand",
               *FEATS, "len_a", "len_b", "addr_empty_b", "src3"]


def pair_features(pairs, s1, s23, n_jobs=-1, batch=100_000):
    """String-similarity features for every candidate pair (parallel over row batches, with progress)."""
    a = s1.select("name_n", "addr_n")[pairs["i1"].to_numpy()]
    b = s23.select("name_n", "addr_n", "entity_id")[pairs["i23"].to_numpy()]
    cols = [a["name_n"].to_list(), b["name_n"].to_list(), a["addr_n"].to_list(), b["addr_n"].to_list()]
    starts = range(0, pairs.height, batch)
    gen = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_pair_feats)(*(c[i:i + batch] for c in cols)) for i in starts)
    res = [r for part in tqdm(gen, total=len(starts), desc="features", leave=False) for r in part]
    f = pl.DataFrame(res, schema={k: pl.Float32 for k in FEATS}, orient="row")
    extra = pl.DataFrame({
        "len_a": a["name_n"].str.len_chars(), "len_b": b["name_n"].str.len_chars(),
        "addr_empty_b": (b["addr_n"] == "").cast(pl.Int8),
        "src3": b["entity_id"].str.starts_with("S3").cast(pl.Int8),
    })
    return pl.concat([pairs, f, extra], how="horizontal")


# ------------------------------------------------------------ S8/S9 decisions
def decide(scored, thr, max_k=11):
    """Exclusivity (each S2/S3 keeps its best S1, A1) + global threshold + list cap."""
    return (scored.filter(pl.col("p") == pl.col("p").max().over("i23"))
                  .filter(pl.col("p") >= thr)
                  .with_columns(r=pl.col("p").rank("ordinal", descending=True).over("i1"))
                  .filter(pl.col("r") <= max_k)
                  .select("i1", "i23"))


# ---------------------------------------------------------------- metric
def macro_f05(pred, truth, s1_ids):
    """pred/truth: (s1_id, s23_id) long tables. Per-S1 F0.5 = 5TP/(5TP+FN+4FP), macro over s1_ids.
    Both empty -> 1.0 (singleton correctly predicted)."""
    tp = pred.join(truth, on=["s1_id", "s23_id"]).group_by("s1_id").len("tp")
    npred = pred.group_by("s1_id").len("npred")
    ntrue = truth.group_by("s1_id").len("ntrue")
    t = (pl.DataFrame({"s1_id": s1_ids}).join(tp, on="s1_id", how="left")
         .join(npred, on="s1_id", how="left").join(ntrue, on="s1_id", how="left").fill_null(0)
         .with_columns(fp=pl.col("npred") - pl.col("tp"), fn=pl.col("ntrue") - pl.col("tp")))
    f = pl.when((pl.col("npred") == 0) & (pl.col("ntrue") == 0)).then(1.0) \
          .otherwise(5 * pl.col("tp") / (5 * pl.col("tp") + pl.col("fn") + 4 * pl.col("fp")))
    return t.select(f.fill_nan(0.0)).to_series().mean()


# ---------------------------------------------------------------- S10 export
def export(s1_ids, cand, match, s23_ids, out_dir):
    """cand/match: (i1, i23) index pairs. Writes the two submission TSVs, one row per S1 in file order."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = pl.DataFrame({"i1": np.arange(len(s1_ids), dtype=np.uint32), "source1_entity_id": s1_ids})
    for df, col, name in [(cand, "candidate_entity_ids", "candidate_pairs.tsv"),
                          (match, "matched_entity_ids", "matching_results.tsv")]:
        lists = (df.with_columns(s23=s23_ids[df["i23"].to_numpy()]).unique(["i1", "s23"])
                   .sort("i1", "s23").group_by("i1", maintain_order=True)
                   .agg(pl.col("s23").str.join(",").alias(col)))
        out = base.join(lists, on="i1", how="left").fill_null("").select("source1_entity_id", col)
        assert out.height == len(s1_ids) and out["source1_entity_id"].n_unique() == out.height
        out.write_csv(out_dir / name, separator="\t", quote_style="never")


if __name__ == "__main__":
    # self-check: metric matches the README worked example (0.714) and the singleton rules
    truth = pl.DataFrame({"s1_id": ["A", "A"], "s23_id": ["x", "z"]})
    pred = pl.DataFrame({"s1_id": ["A", "A", "A", "C"], "s23_id": ["x", "y", "z", "q"]})
    assert abs(macro_f05(pred.head(3), truth, ["A"]) - 0.7142857) < 1e-6
    assert macro_f05(pred.head(0), truth.head(0), ["B"]) == 1.0          # singleton, empty -> 1
    assert macro_f05(pred.slice(3), truth.head(0), ["C"]) == 0.0         # singleton, any match -> 0
    assert norm("Mólecular  Road & Co.") == "molecular rd & co" and norm("<NULL>") == ""
    print("ok")
