"""Last shot, training-free: blend V5 and frozen V4.3 stacker scores (p = (p_v5 + p_43a)/2) for US/India.
Decision rule fitted on labelled mirror rows (all OOF / out-of-sample), judged on HOLDOUT against V5's own rule.
If the blend wins, export US/India with it; France = majority vote of V4.1/V4.2/V4.3/V5 (from final_combos)."""
import pickle, sys, time
from pathlib import Path

import numpy as np
import polars as pl

import er4.v5 as V
from er4 import decide
from er4.pipeline import MIRROR_C, mirror_truth_global, role_global, s1_info, cand_info, load_parsed, load_raw, offsets

t0 = time.time()
lab = V.labelled_global()
tab = pl.concat([V.scored5("mirror", c) for c in MIRROR_C], how="diagonal_relaxed")
tab = (tab.join(lab.select("i1", "fold"), on="i1", how="left")
          .join(V.truth_edges(), on=["i1", "i23"], how="left").with_columns(pl.col("y").fill_null(0),
                                                                            p_b=(pl.col("p_v5") + pl.col("p_43a")) / 2))
truth = mirror_truth_global()
H = role_global("holdout")
truth_H = truth.join(H, on="i1", how="semi").select("i1", "i23")
s1i, ci = s1_info("mirror"), cand_info("mirror")
m = tab["fold"].is_not_null().to_numpy()
sub = tab.filter(pl.Series(m))
ids = lab.join(pl.DataFrame({"i1": sub["i1"].unique()}).cast({"i1": lab.schema["i1"]}), on="i1", how="semi")
tT = truth.join(ids.select("i1"), on="i1", how="semi").select("i1", "i23")
mm = (tT.height - tT.join(sub.select("i1", "i23"), on=["i1", "i23"], how="semi").height) / ids.height
st_b = decide.fit_decision(tab.select("i1", "i23", p="p_b"), sub["y"].to_numpy(), m, sub["fold"].to_numpy(), tT, ids["i1"],
                           ids.select("i1", "fold"), s1i, ci, mm, tag="blend p_v5+p_43a")
st_v5 = pickle.load(open(V.CACHE / "decision_v5.pkl", "rb"))["st"]
Hs = np.sort(H["i1"].to_numpy())
F_b = V.f_entities(decide.apply(st_b, tab.select("i1", "i23", p="p_b"), s1i, ci, H["i1"]), truth_H, Hs)
F_v = V.f_entities(decide.apply(st_v5, tab.select("i1", "i23", p="p_v5"), s1i, ci, H["i1"]), truth_H, Hs)
d, lo, hi = V.boot(F_b, F_v)
print(f"HOLDOUT: blend {F_b.mean():.5f} vs V5 {F_v.mean():.5f} | delta {d:+.5f} CI [{lo:+.5f}, {hi:+.5f}] | "
      f"TRAIN-OOF blend {st_b['f_train']:.5f} vs V5 {st_v5['f_train']:.5f} | {time.time() - t0:.0f}s", flush=True)
use_blend = st_b["f_train"] > st_v5["f_train"] and d > 0
print("USE BLEND:", use_blend, flush=True)
del tab, sub
if not use_blend:
    sys.exit(0)

# ---- export: US/India with the blend, France from the majority-vote file
s1t, cit = s1_info("test"), cand_info("test")
o1, o23 = offsets("test")
pairs = []
for c in ["us", "india"]:
    t = V.scored5("test", c).with_columns(p_b=(pl.col("p_v5") + pl.col("p_43a")) / 2)
    ids_c = load_parsed("test", c, "s1", ["idx"]).select(i1=(pl.col("idx") + o1[c]).cast(pl.UInt32))["i1"]
    pr = decide.apply(st_b, t.select("i1", "i23", p="p_b"), s1t, cit, ids_c)
    a = load_parsed("test", c, "s1", ["idx", "entity_id"]).with_columns(i1=(pl.col("idx") + o1[c]).cast(pl.UInt32))
    b = load_parsed("test", c, "s23", ["idx", "entity_id"]).with_columns(i23=(pl.col("idx") + o23[c]).cast(pl.UInt32))
    pairs.append(pr.cast({"i1": pl.UInt32, "i23": pl.UInt32}).join(a.select("i1", s1="entity_id"), on="i1")
                   .join(b.select("i23", s23="entity_id"), on="i23").select("s1", "s23"))
    print(c, "pred/S1", pr.height / len(ids_c), flush=True)
    del t
te1 = load_raw("test")[0]
FR = te1.filter(pl.col("cty") == "france")["entity_id"]
vote = pl.read_csv(V.V4 / "submissions" / "final_combos" / "v5_usin__france_vote2of4" / "matching_results.tsv", separator="\t",
                   quote_char=None, infer_schema=False).fill_null("")
vote = (vote.filter(pl.col("source1_entity_id").is_in(FR.implode()))
            .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
            .filter(pl.col("matched_entity_ids") != "").select(s1="source1_entity_id", s23="matched_entity_ids"))
p = pl.concat([*pairs, vote])
assert p["s23"].is_unique().all()
lists = p.sort("s1", "s23").group_by("s1", maintain_order=True).agg(pl.col("s23").str.join(",").alias("matched_entity_ids"))
out = te1.select(source1_entity_id="entity_id").join(lists.rename({"s1": "source1_entity_id"}), on="source1_entity_id",
                                                     how="left", maintain_order="left").fill_null("")
assert out.height == 1_732_544 and out["source1_entity_id"].is_unique().all()
d_ = V.V4 / "submissions" / "final_combos" / "blend_usin__france_vote2of4"
d_.mkdir(parents=True, exist_ok=True)
out.write_csv(d_ / "matching_results.tsv", separator="\t", quote_style="never")
print("wrote", d_, p.height, f"{time.time() - t0:.0f}s", flush=True)
