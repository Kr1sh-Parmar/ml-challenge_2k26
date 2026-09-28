"""Pair features (ported from V2, Part 5) + V4 scale-free frequency rates and competition features.

Column layout of every feature matrix (ALL_COLS):
    V2_FEATS (101, exactly V2's order, raw chain counts)  +  FREQ_R (6 rates)  +  ["rev_n", "blk_rev_gap"]
V2's anchor model reads the first 101 columns; V4 models read V4_FEATS (raw counts replaced by rates, + competition)."""
import math, re

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein, LCSseq
from tqdm.auto import tqdm

from .blocking import PASSES
from .config import CFG, DENSE

REC = ["name_n", "core", "trade", "legal", "acr", "url", "k_sorted", "k_joined", "k_skel", "k_trans", "k_meta", "name_nums",
       "native", "has_dba", "addr_n", "a_toks", "street", "house", "unit", "postal", "region", "loc", "landmark", "numset",
       "core_idf", "a_idf", "f_core_s1", "f_core_s23", "f_addr_s1", "f_addr_s23",
       "f_core_s1_r", "f_core_s23_r", "f_addr_s1_r", "f_addr_s23_r"]

FAM = {
    "name_sim": ["n_ratio", "n_tset", "n_tsort", "n_partial", "n_jw_full", "n_jw_core", "n_lev", "n_lcs", "n_c3", "n_jacc",
                 "n_soft", "n_monge", "n_idf_ov", "n_idf_max", "n_first_eq", "n_last_eq", "n_dba_max", "n_len_ratio", "n_tok_diff"],
    "name_keys": ["k_sorted_eq", "k_joined_eq", "k_skel_eq", "k_trans_eq", "k_meta_eq", "n_acr", "url_match", "url_jw",
                  "n_legal", "n_nums"],
    "addr_comp": ["a_postal", "a_postal3", "a_house", "a_house_num", "a_unit", "a_region", "a_loc", "a_num_jacc",
                  "a_street_jacc", "lm_a", "lm_b", "lm_sim", "lm_name", "miss_a", "miss_b", "miss_diff"],
    "addr_sim": ["a_c3", "a_tset", "a_ratio", "a_partial", "a_idf_ov", "a_jacc", "a_len_ratio", "a_empty_a", "a_empty_b"],
    "cross": ["x_prod", "x_min", "x_name_in_addr_ab", "x_name_in_addr_ba", "x_swap"],
    "quality": ["q_ntok_a", "q_ntok_b", "q_nlen_a", "q_nlen_b", "q_atok_a", "q_atok_b", "q_dba_a", "q_dba_b", "q_legal_a",
                "q_legal_b", "q_native_b", "q_url_b", "q_acr_a", "q_acr_b", "q_idf_sum_a", "q_idf_sum_b"],
    "freq": ["f_core_s1_a", "f_core_s23_a", "f_core_s1_b", "f_core_s23_b", "f_addr_s1_a", "f_addr_s23_b"],
}
PAIR_FEATS = [f for fam in FAM.values() for f in fam]                          # 81, V2 order
PROV = ([f"rank_{p}" for p in PASSES] + [f"score_{p}" for p in PASSES] +
        ["rev_rank", "rev_score", "p6", "n_passes", "blk", "blk_rank", "blk_gap", "blk_rev_rank", "n_cand"])
V2_FEATS = PAIR_FEATS + ["src3"] + PROV                                         # 101
FREQ_R = [f + "_r" for f in FAM["freq"]]
COMP = ["rev_n", "blk_rev_gap"]
DENSE_COLS = ["dcos", "drank", "drev_rank", "in_blk"] if DENSE else []          # V4.1 dense retrieval features
ALL_COLS = V2_FEATS + FREQ_R + COMP + DENSE_COLS                                # 109 (+4 in dense mode)
V4_FEATS = [f for f in ALL_COLS if f not in FAM["freq"]]                        # 103 (+4)
V4_IDX = [ALL_COLS.index(f) for f in V4_FEATS]
MONOTONE = {"n_tset": 1, "n_jw_core": 1, "n_c3": 1, "n_idf_ov": 1, "a_tset": 1, "a_c3": 1, "blk": 1, "dcos": 1}
assert len(PAIR_FEATS) == 81 and len(V2_FEATS) == 101 and len(ALL_COLS) == 109 + len(DENSE_COLS)


def _c3(s):
    return {s[i:i + 3] for i in range(len(s) - 2)} if len(s) >= 3 else ({s} if s else set())


def _cos3(x, y):
    gx, gy = _c3(x), _c3(y)
    return len(gx & gy) / math.sqrt(len(gx) * len(gy)) if gx and gy else 0.0


def _tri(x, y):
    return -1.0 if (not x or not y) else float(x == y)


def _jacc(x, y):
    return len(x & y) / len(x | y) if (x and y) else -1.0


def _soft(A, B, thr=0.9):
    if not A or not B:
        return 0.0
    used, m = set(), 0
    for a in A:
        for j, b in enumerate(B):
            if j not in used and (a == b or JaroWinkler.similarity(a, b) >= thr):
                used.add(j); m += 1; break
    return m / (len(A) + len(B) - m)


def _monge(A, B):
    if not A or not B:
        return 0.0
    f = lambda X, Y: sum(max(JaroWinkler.similarity(x, y) for y in Y) for x in X) / len(X)
    return 0.5 * (f(A, B) + f(B, A))


def _idf_ov(ta, ia, tb, ib):
    da, db = dict(zip(ta, ia)), dict(zip(tb, ib))
    w = lambda t: max(da.get(t, 0.0), db.get(t, 0.0))
    sa, sb = set(ta), set(tb)
    uni = sum(w(t) for t in sa | sb)
    sh = [w(t) for t in sa & sb]
    return (sum(sh) / uni if uni else 0.0), (max(sh) if sh else 0.0)


def pair_row(a, b):
    """-> 81 V2 pair features (raw chain counts last) + 6 frequency rates."""
    (nn_a, core_a, trade_a, legal_a, acr_a, url_a, ks_a, kj_a, kk_a, kt_a, km_a, nums_a, nat_a, dba_a, ad_a, at_a, st_a,
     ho_a, un_a, po_a, re_a, lo_a, lm_a, ns_a, ci_a, ai_a, fc1_a, fc23_a, fa1_a, fa23_a, rc1_a, rc23_a, ra1_a, ra23_a) = a
    (nn_b, core_b, trade_b, legal_b, acr_b, url_b, ks_b, kj_b, kk_b, kt_b, km_b, nums_b, nat_b, dba_b, ad_b, at_b, st_b,
     ho_b, un_b, po_b, re_b, lo_b, lm_b, ns_b, ci_b, ai_b, fc1_b, fc23_b, fa1_b, fa23_b, rc1_b, rc23_b, ra1_b, ra23_b) = b
    cs_a, cs_b = " ".join(core_a), " ".join(core_b)
    sca, scb = set(core_a), set(core_b)
    n_tset = fuzz.token_set_ratio(cs_a, cs_b)
    n_idf_ov, n_idf_max = _idf_ov(core_a, ci_a, core_b, ci_b)
    names_a = [cs_a] + ([" ".join(trade_a)] if trade_a else [])
    names_b = [cs_b] + ([" ".join(trade_b)] if trade_b else [])
    name_sim = [
        fuzz.ratio(nn_a, nn_b), n_tset, fuzz.token_sort_ratio(cs_a, cs_b), fuzz.partial_ratio(cs_a, cs_b),
        JaroWinkler.similarity(nn_a, nn_b), JaroWinkler.similarity(kj_a, kj_b), Levenshtein.normalized_similarity(cs_a, cs_b),
        LCSseq.normalized_similarity(kj_a, kj_b), _cos3(kj_a, kj_b), _jacc(sca, scb), _soft(core_a, core_b),
        _monge(core_a, core_b), n_idf_ov, n_idf_max,
        float(bool(core_a and core_b) and core_a[0] == core_b[0]), float(bool(core_a and core_b) and core_a[-1] == core_b[-1]),
        max(fuzz.token_set_ratio(x, y) for x in names_a for y in names_b),
        min(len(nn_a), len(nn_b)) / max(1, len(nn_a), len(nn_b)), abs(len(core_a) - len(core_b))]
    if url_a or url_b:
        url_match = float(bool(url_a and (url_a == kj_b or url_a == url_b)) or bool(url_b and url_b == kj_a))
        url_jw = max(JaroWinkler.similarity(url_a, kj_b) if url_a else 0.0, JaroWinkler.similarity(url_b, kj_a) if url_b else 0.0)
    else:
        url_match, url_jw = -1.0, -1.0
    if nums_a or nums_b:
        n_nums = 0.0 if not (nums_a and nums_b) else (1.0 if nums_a == nums_b else (0.5 if set(nums_a) & set(nums_b) else 0.0))
    else:
        n_nums = -1.0
    name_keys = [float(ks_a == ks_b), float(kj_a == kj_b), float(kk_a == kk_b), float(kt_a == kt_b), float(km_a == km_b and km_a != ""),
                 float(bool(acr_a) and acr_a == kj_b) + float(bool(acr_b) and acr_b == kj_a), url_match, url_jw,
                 _tri(legal_a, legal_b), n_nums]
    sat, sbt = set(at_a), set(at_b)
    miss = lambda h, s, r, l: float((not h) + (not s) + (not r) + (not l))
    ma, mb = miss(ho_a, st_a, re_a, lo_a), miss(ho_b, st_b, re_b, lo_b)
    hd = lambda h: re.sub(r"\D", "", h)
    if lm_a or lm_b:
        lm_name = float((bool(lm_a) and kj_b != "" and kj_b in lm_a.replace(" ", "")) or
                        (bool(lm_b) and kj_a != "" and kj_a in lm_b.replace(" ", "")))
    else:
        lm_name = -1.0
    addr_comp = [_tri(po_a, po_b), _tri(po_a[:3], po_b[:3]), _tri(ho_a, ho_b), _tri(hd(ho_a), hd(ho_b)), _tri(un_a, un_b),
                 _tri(re_a, re_b), (-1.0 if not lo_a or not lo_b else float(bool(set(lo_a) & set(lo_b)))),
                 _jacc(set(ns_a), set(ns_b)), _jacc(set(st_a), set(st_b)), float(bool(lm_a)), float(bool(lm_b)),
                 (fuzz.token_set_ratio(lm_a, lm_b) if lm_a and lm_b else -1.0), lm_name, ma, mb, ma - mb]
    empty = not ad_a or not ad_b
    a_tset = -1.0 if empty else fuzz.token_set_ratio(ad_a, ad_b)
    a_idf_ov = -1.0 if empty else _idf_ov(at_a, ai_a, at_b, ai_b)[0]
    addr_sim = [(-1.0 if empty else _cos3(ad_a.replace(" ", ""), ad_b.replace(" ", ""))), a_tset,
                (-1.0 if empty else fuzz.ratio(ad_a, ad_b)), (-1.0 if empty else fuzz.partial_ratio(ad_a, ad_b)),
                a_idf_ov, _jacc(sat, sbt), (-1.0 if empty else min(len(ad_a), len(ad_b)) / max(len(ad_a), len(ad_b))),
                float(not ad_a), float(not ad_b)]
    cross = [(-1.0 if empty else n_tset * a_tset / 1e4), (-1.0 if empty else min(n_tset, a_tset)),
             (fuzz.partial_ratio(kj_a, ad_b.replace(" ", "")) if ad_b and len(kj_a) >= 4 else -1.0),
             (fuzz.partial_ratio(kj_b, ad_a.replace(" ", "")) if ad_a and len(kj_b) >= 4 else -1.0),
             max(fuzz.token_set_ratio(cs_a, ad_b) if ad_b else 0.0, fuzz.token_set_ratio(cs_b, ad_a) if ad_a else 0.0)]
    quality = [len(core_a), len(core_b), len(nn_a), len(nn_b), len(at_a), len(at_b), float(dba_a), float(dba_b),
               float(bool(legal_a)), float(bool(legal_b)), float(nat_b), float(bool(url_b)),
               float(len(core_a) == 1 and len(kj_a) <= 5), float(len(core_b) == 1 and len(kj_b) <= 5), sum(ci_a), sum(ci_b)]
    freq = [fc1_a, fc23_a, fc1_b, fc23_b, fa1_a, fa23_b]
    freq_r = [rc1_a, rc23_a, rc1_b, rc23_b, ra1_a, ra23_b]
    return name_sim + name_keys + addr_comp + addr_sim + cross + quality + freq + freq_r


def _feat_batch(A, B):
    return np.array([pair_row(a, b) for a, b in zip(A, B)], dtype=np.float32)


def prov_frame(cand):
    ex = []
    for p in PASSES:
        ex += [pl.col(f"rank_{p}").fill_null(CFG["top_k"][p] + 1).cast(pl.Float32).alias(f"rank_{p}"),
               pl.col(f"score_{p}").fill_null(0).cast(pl.Float32).alias(f"score_{p}")]
    ex += [pl.col("rev_rank").fill_null(CFG["rev_k"] + 1).cast(pl.Float32), pl.col("rev_score").fill_null(0).cast(pl.Float32)]
    ex += [pl.col(c).cast(pl.Float32) for c in ["p6", "n_passes", "blk", "blk_rank", "blk_gap", "blk_rev_rank", "n_cand"]]
    return cand.select(ex)


def build_features(cand, s1p, s23p, batch=40_000, desc="features"):
    """cand: edges (i1, i23 + blocking columns) of one country -> float32 [n, 109] in ALL_COLS order."""
    R1, R23 = s1p.select(REC), s23p.select(REC)
    i1, i23 = cand["i1"].to_numpy(), cand["i23"].to_numpy()

    def jobs():
        for lo in range(0, len(i1), batch):
            yield delayed(_feat_batch)(R1[i1[lo:lo + batch]].rows(), R23[i23[lo:lo + batch]].rows())

    X = np.empty((len(i1), len(ALL_COLS)), dtype=np.float32)
    n_pf, pos = len(PAIR_FEATS), 0
    j_fr = ALL_COLS.index(FREQ_R[0])
    for part in tqdm(Parallel(n_jobs=CFG["n_jobs"], return_as="generator", pre_dispatch="2*n_jobs")(jobs()),
                     total=math.ceil(len(i1) / batch), desc=desc, mininterval=10):
        X[pos:pos + len(part), :n_pf] = part[:, :n_pf]
        X[pos:pos + len(part), j_fr:j_fr + len(FREQ_R)] = part[:, n_pf:]
        pos += len(part)
    X[:, n_pf] = s23p["entity_id"].str.starts_with("S3").cast(pl.Float32).to_numpy()[i23]
    X[:, n_pf + 1:n_pf + 1 + len(PROV)] = prov_frame(cand).to_numpy().astype(np.float32)
    j_c = ALL_COLS.index("rev_n")
    X[:, j_c:j_c + 2] = cand.select(pl.col("rev_n").cast(pl.Float32), pl.col("blk_rev_gap").cast(pl.Float32)).to_numpy()
    if DENSE_COLS:
        j_d = ALL_COLS.index("dcos")
        X[:, j_d:j_d + 4] = cand.select(dcos=pl.col("dcos").fill_null(-1.0), drank=pl.col("drank").fill_null(99),
                                        drev_rank=pl.col("drev_rank").fill_null(9),
                                        in_blk=pl.col("in_blk").fill_null(1)).cast(pl.Float32).to_numpy()
    return X
