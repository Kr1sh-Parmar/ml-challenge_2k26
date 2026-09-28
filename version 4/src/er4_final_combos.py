"""Training-free combinations of the saved V4.x / V5 submissions. US/India always from V5 (best on HOLDOUT);
France varied. One S2/S3 record is never given to two S1s (conflicts resolved in favour of V5, then by votes)."""
from pathlib import Path

import polars as pl

S = Path("../submissions")
OUT = S / "final_combos"
te1 = pl.read_csv("../../student_resource/dataset/test/test_source1.tsv", separator="\t", quote_char=None,
                  infer_schema=False, columns=["entity_id", "country"])
order = te1["entity_id"]
FR = pl.Series(te1.filter(pl.col("country") == "France")["entity_id"])


def pairs(path):
    d = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).fill_null("")
    return (d.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
             .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "s23"}))


V = {v: pairs(S / v / "matching_results.tsv") for v in ["v4.1", "v4.2", "v4.3", "v5"]}
isfr = pl.col("s1").is_in(FR.implode())
base = V["v5"].filter(~isfr)                                    # US + India from V5
fr = {v: p.filter(isfr) for v, p in V.items()}
n_fr = FR.len()


def exclusive(p, prefer):
    """keep one S1 per record: preferred source first, then higher vote count."""
    return (p.sort(["pref", "votes"], descending=[True, True]).unique("s23", keep="first").select("s1", "s23"))


def votes_frame():
    allp = pl.concat([p.with_columns(src=pl.lit(v)) for v, p in fr.items()])
    return allp.group_by("s1", "s23").agg(votes=pl.len(), in_v5=(pl.col("src") == "v5").any())


def write(fr_pairs, name, note):
    p = pl.concat([base, fr_pairs.select("s1", "s23")])
    assert p["s23"].is_unique().all(), name
    lists = p.sort("s1", "s23").group_by("s1", maintain_order=True).agg(pl.col("s23").str.join(",").alias("matched_entity_ids"))
    out = order.to_frame("source1_entity_id").join(lists.rename({"s1": "source1_entity_id"}), on="source1_entity_id",
                                                   how="left", maintain_order="left").fill_null("")
    assert out.height == 1_732_544 and out["source1_entity_id"].is_unique().all()
    (OUT / name).mkdir(parents=True, exist_ok=True)
    out.write_csv(OUT / name / "matching_results.tsv", separator="\t", quote_style="never")
    print(f"{name:34s} France {fr_pairs.height:>8,} pairs = {fr_pairs.height / n_fr:.3f}/S1 | total {p.height:,} | {note}")


vf = votes_frame()
print("France pairs per version:", {v: p.height for v, p in fr.items()})
print("France vote histogram (how many of 4 versions predict each pair):",
      dict(zip(*vf.group_by("votes").len().sort("votes").to_dict(as_series=False).values())))
# 1) V5 France (reference = the 0.986 file)
write(fr["v5"], "v5_reference", "exact V5 (0.986)")
# 2) V4.1 France (more recall)
write(fr["v4.1"], "v5_usin__v41_france", "France from V4.1 (3.38/S1, simpler model)")
# 3) France majority: pair kept if >= 2 of 4 versions predict it
maj = exclusive(vf.filter(pl.col("votes") >= 2).with_columns(pref=pl.col("in_v5").cast(pl.Int8)), "v5")
write(maj, "v5_usin__france_vote2of4", "France pairs predicted by >= 2 of V4.1/V4.2/V4.3/V5")
# 4) France union of V5 and V4.1 (recall-leaning), V5 wins record conflicts
u = pl.concat([fr["v5"].with_columns(pref=pl.lit(1, pl.Int8), votes=pl.lit(1)),
               fr["v4.1"].with_columns(pref=pl.lit(0, pl.Int8), votes=pl.lit(1))]).unique(["s1", "s23"], keep="first")
write(exclusive(u, "v5"), "v5_usin__union_v5_v41_france", "France = V5 union V4.1")
