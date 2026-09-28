"""Re-export V5 with its learned decision policy switched on (no training). Writes submissions/final_combos/v5_policy/."""
import pickle, shutil
from pathlib import Path

import er4.v5 as V

d = pickle.load(open(V.CACHE / "decision_v5.pkl", "rb"))
d["use_policy"] = True
tmp = V.CACHE / "decision_v5_policy.pkl"
pickle.dump(d, open(tmp, "wb"))
orig_load = pickle.load
V.pickle.load = lambda f: d if getattr(f, "name", "").endswith("decision_v5.pkl") else orig_load(f)
V.match_rate = lambda fn, tp, target, **k: (0.0, fn(tp))          # skip the France prior-matching bisection
V.OUT5 = V.V4 / "output_v5_policy"
V.SMOKE = True                                                     # skip the validator inside export (run separately)
V.stage_export5()
dst = V.V4 / "submissions" / "final_combos" / "v5_policy"
dst.mkdir(parents=True, exist_ok=True)
shutil.copy(V.OUT5 / "matching_results.tsv", dst / "matching_results.tsv")
print("done", dst)
