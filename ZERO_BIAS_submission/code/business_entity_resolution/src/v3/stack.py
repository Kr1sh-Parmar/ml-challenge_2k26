"""Feature assembly shared by the train (OOF) and test (refit) paths: L10 inputs, F-UNC, the L11 token matrix."""
import numpy as np
import polars as pl

from . import pool as PL
from . import fs_em as EM
from . import clusters as CL

MONO = {"n_tset": 1, "n_jw_core": 1, "n_c3": 1, "n_idf_ov": 1, "a_tset": 1, "a_c3": 1, "dcos": 1}
MONO1 = dict(MONO, p_student=1, em_total=1)
L1F = PL.STUDENT_FEATS + EM.EM_FEATS + CL.CLUS_FEATS + ["p_student", "stu_rank", "stu_gap", "stu_lsum", "list_n"]
UNC_FEATS = ["unc_mean", "unc_std", "unc_min", "unc_max", "unc_n05", "unc_text_vs_feat", "ce_present"]
BASE_FEATS = ["n_tset", "n_jw_core", "a_house", "a_postal", "a_unit", "a_tset", "a_num_jacc", "a_region", "a_loc",
              "a_empty_a", "a_empty_b", "f_core_s1_a", "f_core_s23_b", "dcos", "src3", "q_native_b", "n_legal"]
STK_FEATS = ["lg_xgb", "lg_lgb", "lg_stu", "lg_ce"] + UNC_FEATS + EM.EM_FEATS + CL.CLUS_FEATS + BASE_FEATS


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)


def list_rank_feats(i1, p):
    d = pl.DataFrame({"i1": i1, "p": p})
    d = d.with_columns(r=pl.col("p").rank("ordinal", descending=True).over("i1").cast(pl.Float32),
                       gap=(pl.col("p").max().over("i1") - pl.col("p")).cast(pl.Float32),
                       lsum=pl.col("p").sum().over("i1").cast(pl.Float32),
                       n=pl.len().over("i1").cast(pl.Float32))
    return d.select("r", "gap", "lsum", "n").to_numpy().astype(np.float32)


def level1_matrix(Xp, Fem, Fclus, ps, i1):
    return np.column_stack([Xp, Fem, Fclus, ps, list_rank_feats(i1, ps)]).astype(np.float32)


def unc_feats(p_xgb, p_lgb, p_em, p_stu, p_ce):
    ce_f = np.where(np.isfinite(p_ce), p_ce, p_xgb)
    M = np.column_stack([p_xgb, p_lgb, p_em, p_stu, ce_f])
    return np.column_stack([M.mean(1), M.std(1), M.min(1), M.max(1), (M > 0.5).sum(1),
                            np.abs(ce_f - p_xgb), np.isfinite(p_ce).astype(np.float32)]).astype(np.float32)


def token_matrix(p_xgb, p_lgb, ps, p_ce, Func, Fem, Fclus, X1):
    lvl1 = np.column_stack([logit(p_xgb), logit(p_lgb), logit(ps), logit(np.nan_to_num(p_ce, nan=0.5))])
    base_idx = [L1F.index(f) for f in BASE_FEATS]
    return np.column_stack([lvl1, Func, Fem, Fclus, X1[:, base_idx]]).astype(np.float32)


def link_rows(links, i1, i23):
    """links (i1, x, y, p_link) -> (row_x, row_y, p_link) positions in the pair arrays."""
    lk = links.join(pl.DataFrame({"i1": i1, "x": i23, "rx": np.arange(len(i1))}), on=["i1", "x"]) \
              .join(pl.DataFrame({"i1": i1, "y": i23, "ry": np.arange(len(i1))}), on=["i1", "y"])
    return lk["rx"].to_numpy(), lk["ry"].to_numpy(), lk["p_link"].to_numpy()


def st_pack(T, std, i1, order_score, links_rows, y=None):
    from . import settx as ST
    Ts, miss = std.transform(T)
    pack = ST.pack_lists(i1, order_score, Ts, links_rows, y)
    pack["Xs"] = pack["X"]
    pm = np.zeros(pack["X"].shape, bool)
    pm[pack["mask"]] = miss[pack["prow"][pack["mask"]]]
    pack["miss"] = pm
    return pack
