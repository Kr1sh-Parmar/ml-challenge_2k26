"""V5+policy for US/India + France majority vote (>= 2 of V4.1/V4.2/V4.3/V5). Training-free."""
from pathlib import Path

import polars as pl

S = Path("../submissions")
C = S / "final_combos"
te1 = pl.read_csv("../../student_resource/dataset/test/test_source1.tsv", separator="\t", quote_char=None,
                  infer_schema=False, columns=["entity_id", "country"])
FR = pl.Series(te1.filter(pl.col("country") == "France")["entity_id"])


def pairs(path):
    d = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).fill_null("")
    return (d.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
             .filter(pl.col("matched_entity_ids") != "").rename({"source1_entity_id": "s1", "matched_entity_ids": "s23"}))


isfr = pl.col("s1").is_in(FR.implode())
p = pl.concat([pairs(C / "v5_policy" / "matching_results.tsv").filter(~isfr),
               pairs(C / "v5_usin__france_vote2of4" / "matching_results.tsv").filter(isfr)])
assert p["s23"].is_unique().all()
lists = p.sort("s1", "s23").group_by("s1", maintain_order=True).agg(pl.col("s23").str.join(",").alias("matched_entity_ids"))
out = te1.select(source1_entity_id="entity_id").join(lists.rename({"s1": "source1_entity_id"}), on="source1_entity_id",
                                                     how="left", maintain_order="left").fill_null("")
assert out.height == 1_732_544 and out["source1_entity_id"].is_unique().all()
d = C / "v5_policy__france_vote2of4"
d.mkdir(parents=True, exist_ok=True)
out.write_csv(d / "matching_results.tsv", separator="\t", quote_style="never")
print("wrote", d, p.height)
