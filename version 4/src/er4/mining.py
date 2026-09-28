"""Variant mining (ported from V2, Part 3): token/segment alignment on matched vs non-matched pairs, lift filter,
canonical substitution tables; plus unsupervised self-mining for an unseen country (France)."""
import math
from collections import Counter

import polars as pl
from joblib import Parallel, delayed
from rapidfuzz.distance import JaroWinkler
from tqdm.auto import tqdm

from .blocking import gram_keys, index_join, keys_from, postings
from .config import CFG
from .normalize import CONTRACT, LEGAL_SEED, STOP, UNIT_MARK, translit


def simple_keys(p):
    return pl.concat([
        keys_from(p, [("n", pl.col("core")), ("s", pl.col("k_sorted")), ("j", pl.col("k_joined")), ("t", pl.col("k_trans")),
                      ("a", pl.col("a_toks"))]),
        gram_keys(p, "k_joined")]).unique()


def simple_block(a, b, top_k=10, cap_frac=2e-4):
    """First-pass blocking of one country (for mining negatives and France seed pairs)."""
    post = postings(simple_keys(b), b.height, max(20, int(cap_frac * b.height)))
    qa = simple_keys(a)
    step = max(5_000, int(CFG["chunk_rows"] / b.height))
    ids = a["idx"].sort()
    out = [index_join(qa.filter(pl.col("idx").is_between(ids[lo], ids[min(lo + step, len(ids)) - 1])), post, top_k)
           for lo in tqdm(range(0, len(ids), step), desc="simple block", mininterval=10)]
    return pl.concat(out)


def abbrev(short, long):
    if len(short) >= len(long) or len(short) > 5 or short[0] != long[0]:
        return False
    it = iter(long)
    return all(ch in it for ch in short)


def align(a, b, max_pos=3, jw_min=0.85):
    sa, sb = set(a), set(b)
    A = [t for t in a if t and t not in sb and not any(ch.isdigit() for ch in t)]
    B = [t for t in b if t and t not in sa and not any(ch.isdigit() for ch in t)]
    if not A or not B:
        return []
    if len(A) == len(B) and len(A) <= max_pos:   # positional pairing, only for transliteration look-alikes
        return [tuple(sorted(p)) for p in zip(A, B)
                if p[0] != p[1] and min(len(p[0]), len(p[1])) >= 3 and min(len(translit(p[0])), len(translit(p[1]))) >= 3
                and JaroWinkler.similarity(translit(p[0]), translit(p[1])) >= 0.85]
    out, used = [], set()
    for x in A:
        best, bs = None, 0.0
        for y in B:
            if y in used:
                continue
            s = JaroWinkler.similarity(x, y)
            if abbrev(x, y) or abbrev(y, x):
                s = max(s, 0.95)
            if s > bs:
                best, bs = y, s
        if best is not None and bs >= jw_min:
            out.append(tuple(sorted((x, best)))); used.add(best)
    return out


def seg_similar(x, y):
    xs, ys = x.replace(" ", ""), y.replace(" ", "")
    ix, iy = "".join(w[0] for w in x.split()), "".join(w[0] for w in y.split())
    short, long_ = sorted((xs, ys), key=len)
    return (ix == ys or iy == xs or abbrev(short, long_)
            or (min(len(xs), len(ys)) >= 4 and JaroWinkler.similarity(translit(xs), translit(ys)) >= 0.85))


def _count_batch(rows):
    cn, ca, cs = Counter(), Counter(), Counter()
    for c, n1, n2, a1, a2, s1_, s2_ in rows:
        for p in align(n1, n2):
            cn[(c, p)] += 1
        for p in align(a1, a2):
            ca[(c, p)] += 1
        r1, r2 = [s for s in s1_ if s not in s2_], [s for s in s2_ if s not in s1_]
        if len(r1) == 1 and len(r2) == 1 and seg_similar(r1[0], r2[0]):
            cs[(c, tuple(sorted((r1[0], r2[0]))))] += 1
    return cn, ca, cs


def count_subs(pairs, s1p, s23p, desc):
    a = s1p.select(pl.col("idx").alias("i1"), "cty", n1=pl.col("name_n").str.split(" "), a1="a_toks", s1="segs")
    b = s23p.select(pl.col("idx").alias("i23"), n2=pl.col("name_n").str.split(" "), a2="a_toks", s2="segs")
    rows = pairs.join(a, on="i1").join(b, on="i23").select("cty", "n1", "n2", "a1", "a2", "s1", "s2").rows()
    B = 50_000
    res = Parallel(n_jobs=CFG["n_jobs"], return_as="generator")(delayed(_count_batch)(rows[i:i + B]) for i in range(0, len(rows), B))
    tot = [Counter(), Counter(), Counter()]
    for r in tqdm(res, total=math.ceil(len(rows) / B), desc=desc, mininterval=10):
        for t, x in zip(tot, r):
            t.update(x)
    return tot, len(rows)


def token_freq(s1p, s23p):
    f = Counter()
    for col in ["name_n", "addr_n"]:
        for s in (s1p[col], s23p[col]):
            vc = s.str.split(" ").explode().value_counts()
            f.update(dict(zip(vc[col].to_list(), vc["count"].to_list())))
    for s in (s1p["segs"], s23p["segs"]):
        vc = s.explode().drop_nulls().value_counts()
        f.update(dict(zip(vc["segs"].to_list(), vc["count"].to_list())))
    return f


PROTECT = set(CONTRACT.values()) | UNIT_MARK | LEGAL_SEED | STOP   # canonical seed tokens are never remapped


def build_table(pos, neg, n_pos, n_neg, freq, support, min_lift, country=None):
    agg_p, agg_n = Counter(), Counter()
    for (c, p), k in pos.items():
        if country is None or c == country:
            agg_p[p] += k
    for (c, p), k in neg.items():
        if country is None or c == country:
            agg_n[p] += k
    rows = []
    for p, k in agg_p.items():
        if k < support:
            continue
        lift = (k / n_pos) / ((agg_n.get(p, 0) + 0.5) / n_neg)
        x, y = p
        similar = " " not in x + y and ((min(len(x), len(y)) >= 3 and JaroWinkler.similarity(x, y) >= 0.85)
                                        or abbrev(x, y) or abbrev(y, x))
        if lift >= (CFG["sim_lift"] if similar else min_lift):
            rows.append((x, y, k, agg_n.get(p, 0), lift))
    rows.sort(key=lambda r: -r[2])
    best = {}
    for x, y, k, _, _ in rows:
        if len(x) == 1 or len(y) == 1:
            continue
        lo, hi = (x, y) if (freq.get(x, 0), x) < (freq.get(y, 0), y) else (y, x)
        if lo in PROTECT:
            lo, hi = hi, lo
        if lo not in PROTECT:
            best.setdefault(lo, hi)
    return {k: v for k, v in best.items() if v not in best and v != k}, rows


def mine_tables(s1p, s23p, pos_pairs, neg_pairs, tag):
    (pn, pa, ps), n_pos = count_subs(pos_pairs, s1p, s23p, f"align matched ({tag})")
    (nn, na, ns), n_neg = count_subs(neg_pairs, s1p, s23p, f"align non-matched ({tag})")
    freq = token_freq(s1p, s23p)
    tables, report = {}, {}
    for c in [None] + sorted(s1p["cty"].unique().to_list()):
        key = c or "*"
        tn, rn = build_table(pn, nn, n_pos, n_neg, freq, CFG["mine_support"], CFG["mine_lift"], c)
        ta, ra = build_table(pa, na, n_pos, n_neg, freq, CFG["mine_support"], CFG["mine_lift"], c)
        tsg, rs = build_table(ps, ns, n_pos, n_neg, freq, CFG["mine_support"], CFG["mine_lift"], c)
        tables[key], report[key] = (tn, ta, tsg), (rn, ra, rs)
    return tables, report


def self_mine_seeds(a, b):
    """Unlabelled country: high-precision seed pairs (mutual best + same sorted name, or same house + JW >= 0.9)."""
    cand = simple_block(a, b, top_k=5)
    rev_best = cand.with_columns(rr=pl.col("score").rank("ordinal", descending=True).over("i23")).filter(pl.col("rr") == 1)
    top1 = cand.filter(pl.col("rank") == 1).join(rev_best.select("i1", "i23"), on=["i1", "i23"], how="semi")
    x = (top1.join(a.select(pl.col("idx").alias("i1"), ks1="k_sorted", h1="house", j1="k_joined"), on="i1")
             .join(b.select(pl.col("idx").alias("i23"), ks2="k_sorted", h2="house", j2="k_joined"), on="i23"))
    jw = [JaroWinkler.similarity(p, q) for p, q in zip(x["j1"], x["j2"])]
    seeds = x.with_columns(jw=pl.Series(jw)).filter(
        (pl.col("ks1") == pl.col("ks2")) | ((pl.col("h1") != "") & (pl.col("h1") == pl.col("h2")) & (pl.col("jw") >= 0.9)))
    seeds = seeds.select("i1", "i23")
    neg = cand.join(seeds.select("i1").unique(), on="i1", how="semi").join(seeds, on=["i1", "i23"], how="anti").select("i1", "i23")
    return seeds, neg


def subs_for(country, train_tables, unseen_tables):
    g = train_tables["*"]
    own = train_tables.get(country) or unseen_tables.get(country) or ({}, {}, {})
    return tuple({**g[i], **own[i]} for i in range(3))
