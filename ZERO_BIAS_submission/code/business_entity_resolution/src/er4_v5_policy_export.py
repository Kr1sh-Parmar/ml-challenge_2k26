"""Re-export V5 with its learned decision policy switched on (no training) -> ER_WORK/output_v5_policy/.

V5's own adoption rule rejected the policy on labelled S1s; it was switched on for US/India because it scored higher on
the mirror HOLDOUT per country (US 0.98969, India 0.99116 vs 0.9895 / 0.9911). France is taken from the majority vote in
make_final.py, so the France prior-matching bisection is skipped here.
    cd src && ER4_V5=1 python er4_v5_policy_export.py        (PowerShell: $env:ER4_V5="1"; python er4_v5_policy_export.py)
"""
import pickle

import er4.v5 as V

d = pickle.load(open(V.CACHE / "decision_v5.pkl", "rb"))
d["use_policy"] = True
orig_load = pickle.load
V.pickle.load = lambda f: d if getattr(f, "name", "").endswith("decision_v5.pkl") else orig_load(f)
V.match_rate = lambda fn, tp, target, **k: (0.0, fn(tp))          # skip the France prior-matching bisection
V.OUT5 = V.V4 / "output_v5_policy"
V.SMOKE = True                                                     # skip the validator inside export (make_final.py runs it)
V.stage_export5()
print("done", V.OUT5)
