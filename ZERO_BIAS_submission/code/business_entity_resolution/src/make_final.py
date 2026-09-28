"""Assemble the submitted files from the saved stage outputs (training-free; the last step of the pipeline).

  US + India : V5 with its learned decision policy on    ER_WORK/output_v5_policy   (er4_v5_policy_export.py)
  France     : pairs predicted by >= 2 of V4.1 / V4.2 / V4.3 / V5
               ER_WORK/output_dense (V4.1), output_v42 (V4.2), output_final (V4.3), output_v5 (V5)
               a record claimed by two S1s keeps the V5 pair, then the one with more votes (never happened in the submission)
  candidates : V5's scored edge set, ER_WORK/output_v5/candidate_pairs.tsv (V4.1..V5 share cand_test_*, so it holds every
               V4.1-V5 prediction and therefore every submitted pair)

    cd src && python make_final.py [--out <folder>]
"""
import argparse, shutil, subprocess, sys

import polars as pl

from erpaths import DATA, VALIDATOR, WORK

SRC = {"v4.1": "output_dense", "v4.2": "output_v42", "v4.3": "output_final", "v5": "output_v5"}


def pairs(folder):
    d = pl.read_csv(WORK / folder / "matching_results.tsv", separator="\t", quote_char=None, infer_schema=False).fill_null("")
    return (d.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
             .filter(pl.col("matched_entity_ids") != "").select(s1="source1_entity_id", s23="matched_entity_ids"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(WORK / "submission"))
    out_dir = ap.parse_args().out
    te1 = pl.read_csv(DATA / "test" / "test_source1.tsv", separator="\t", quote_char=None, infer_schema=False,
                      columns=["entity_id", "country"])
    isfr = pl.col("s1").is_in(te1.filter(pl.col("country") == "France")["entity_id"].implode())
    usin = pairs("output_v5_policy").filter(~isfr)
    votes = (pl.concat([pairs(f).filter(isfr).with_columns(src=pl.lit(v)) for v, f in SRC.items()])
               .group_by("s1", "s23").agg(votes=pl.len(), in_v5=(pl.col("src") == "v5").any()))
    fr = (votes.filter(pl.col("votes") >= 2).sort(["in_v5", "votes"], descending=True)
               .unique("s23", keep="first").select("s1", "s23"))
    p = pl.concat([usin, fr])
    assert p["s23"].is_unique().all(), "an S2/S3 record may belong to at most one S1"
    lists = p.sort("s1", "s23").group_by("s1", maintain_order=True).agg(pl.col("s23").str.join(",").alias("matched_entity_ids"))
    out = (te1.select(source1_entity_id="entity_id")
              .join(lists.rename({"s1": "source1_entity_id"}), on="source1_entity_id", how="left", maintain_order="left").fill_null(""))
    assert out.height == te1.height and out["source1_entity_id"].is_unique().all()
    import os
    os.makedirs(out_dir, exist_ok=True)
    out.write_csv(f"{out_dir}/matching_results.tsv", separator="\t", quote_style="never")
    shutil.copy(WORK / "output_v5" / "candidate_pairs.tsv", f"{out_dir}/candidate_pairs.tsv")
    print(f"wrote {out_dir}: {p.height:,} matches ({usin.height:,} US/India, {fr.height:,} France)", flush=True)
    r = subprocess.run([sys.executable, str(VALIDATOR), "--matching", f"{out_dir}/matching_results.tsv",
                        "--candidate", f"{out_dir}/candidate_pairs.tsv", "--test-dir", str(DATA / "test"), "--check-ids"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(r.stdout[-2000:], r.stderr[-800:])


if __name__ == "__main__":
    main()
