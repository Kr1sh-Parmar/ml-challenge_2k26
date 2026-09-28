"""Rebuild output/candidate_pairs.tsv on a CPU-only machine (how the submitted candidate file was produced).

The submitted model scored   lexical top-N (V4 blocking)  UNION  dense top-20 + reverse top-3 (fine-tuned e5)   per test S1.
The lexical part is CPU code and is re-run here with the same code and inputs (same tables, parser, passes, blocking
scorer refit on MINE at full scale, same n_cand); per-pass top-K ties are broken in thread order, so it matches the original
statistically, not bit for bit. The dense part needs the fine-tuned encoder weights (GPU run, ER_WORK/v3/models/
dense_ft); when they are not available, the dense edges are represented by the ones that were finally matched: every
matched pair was a candidate of the scoring run (asserted by every export stage), so

    lexical  UNION  final matches   is a subset of the true candidate set and a superset of the matches.

The script reports how many final matches came from outside the lexical set (= matched dense-only edges).
With the GPU artefacts present, use the pipeline's own file instead: ER_WORK/output_v5/candidate_pairs.tsv (make_final.py).

    cd src && python rebuild_candidates.py --matching <final matching_results.tsv> --out <output folder>
"""
import argparse, os, subprocess, sys, time

assert os.environ.get("ER4_DENSE") != "1" and os.environ.get("ER4_V5") != "1", "run without ER4_DENSE / ER4_V5 (lexical cache)"
os.environ.setdefault("POLARS_MAX_THREADS", "8")      # fewer concurrent join buffers: fits the India pass in 24 GB RAM
import polars as pl

from er4 import pipeline as P
from er4.blocking import block_country, competition_features
from er4.config import VALIDATOR, DATA
from er4.util import log


# the only parsed columns the blocking passes read (keys_p1/p2/p3/p5/rev, dup_groups); loading just these cuts RAM ~3x
BLOCK_COLS = ["idx", "core", "trade", "k_sorted", "k_meta", "acr", "k_joined", "url", "k_skel", "k_trans", "house", "anchor",
              "postal", "unit", "a_toks"]


def read_pairs(path, col):
    d = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).fill_null("")
    return (d.with_columns(pl.col(col).str.split(",")).explode(col).filter(pl.col(col) != "")
             .select(s1_id="source1_entity_id", s23_id=col))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True, help="final matching_results.tsv (the leaderboard file)")
    ap.add_argument("--out", required=True, help="folder for candidate_pairs.tsv")
    a = ap.parse_args()
    t0 = time.time()
    P.stage_roles(); P.stage_mine(); P.stage_parse()
    bs = P.stage_blkfit()
    match = read_pairs(a.matching, "matched_entity_ids")
    lists, rows = [], []
    for c in P.countries("test"):                 # per country, in integer (i1, i23) space: ~130M edges stay within 24 GB RAM
        if not P.cand_path("test", c).exists():
            s1p, s23p = P.load_parsed("test", c, "s1", BLOCK_COLS), P.load_parsed("test", c, "s23", BLOCK_COLS)
            t = time.time()
            cand = competition_features(block_country(s1p, s23p, scorer=bs["scorer"], n_cand=bs["n_cand"], tag=f"test {c}"))
            P.write_pq(cand, P.cand_path("test", c))
            log(f"blocked test {c}: {s1p.height:,} S1 x {s23p.height:,} S2/S3 -> {cand.height:,} pairs in {time.time() - t:.0f}s")
            del s1p, s23p, cand
        ids1 = P.load_parsed("test", c, "s1", ["idx", "entity_id"]).select(i1=pl.col("idx").cast(pl.UInt32), s1_id="entity_id")
        ids23 = P.load_parsed("test", c, "s23", ["idx", "entity_id"]).select(i23=pl.col("idx").cast(pl.UInt32), s23_id="entity_id")
        lex = pl.read_parquet(P.cand_path("test", c), columns=["i1", "i23"]).cast({"i1": pl.UInt32, "i23": pl.UInt32})
        m = match.join(ids1, on="s1_id").join(ids23, on="s23_id").select("i1", "i23")      # this country's matches
        extra = m.join(lex, on=["i1", "i23"], how="anti")                                   # matched dense-only edges
        cand = pl.concat([lex, extra])                                                      # disjoint by construction
        lists.append(cand.join(ids23, on="i23").group_by("i1").agg(pl.col("s23_id").sort().str.join(",").alias("candidate_entity_ids"))
                         .join(ids1, on="i1").select(source1_entity_id="s1_id", candidate_entity_ids="candidate_entity_ids"))
        rows.append((c, ids1.height, lex.height / ids1.height, m.height, extra.height, cand.height, ids1.height - cand["i1"].n_unique()))
        log(f"test {c}: lexical {lex.height:,} | matches {m.height:,} | outside lexical {extra.height:,} | candidates {cand.height:,}")
        del ids1, ids23, lex, m, extra, cand
    rep = pl.DataFrame(rows, schema=["country", "n_s1", "lexical_per_s1", "matched", "matched_outside_lexical", "candidates",
                                     "s1_without_candidates"], orient="row")
    tot = rep.select(pl.col("matched", "matched_outside_lexical", "candidates").sum()).row(0)
    log(f"n_cand={bs['n_cand']} | final matches {tot[0]:,} | outside the lexical set (dense-only edges) {tot[1]:,} "
        f"({tot[1] / max(1, tot[0]):.2%}) | candidate file {tot[2]:,} pairs\n{rep}")
    te1 = pl.read_csv(DATA / "test" / "test_source1.tsv", separator="\t", quote_char=None, infer_schema=False, columns=["entity_id"])
    out = (te1.select(source1_entity_id="entity_id")
              .join(pl.concat(lists), on="source1_entity_id", how="left", maintain_order="left").fill_null(""))
    assert out.height == te1.height and out["source1_entity_id"].is_unique().all()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "candidate_pairs.tsv")
    out.write_csv(path, separator="\t", quote_style="never")
    log(f"wrote {path} in {(time.time() - t0) / 60:.1f} min")
    r = subprocess.run([sys.executable, str(VALIDATOR), "--matching", a.matching, "--candidate", path, "--test-dir", str(DATA / "test")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-2500:], r.stderr[-1000:])


if __name__ == "__main__":
    main()
