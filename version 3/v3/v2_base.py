"""V2 library, lifted verbatim from `version 2/v2_complete_model.ipynb` (definitions only, no execution).

V3 keeps V2 normalization, parsing, blocking and pair features unchanged (V3 design section 0), so the V2
caches stay valid. Runtime state (LEGAL, SUBS) is injected by `v3.state.load_v2_state()`.
"""
import os, re, sys, gc, math, time, json, pickle, random, unicodedata, warnings
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import polars as pl
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from unidecode import unidecode
import jellyfish
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein, LCSseq

LEGAL = set()   # injected at runtime (seed + discovered legal forms)
SUBS = {}       # injected at runtime (mined substitution tables)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

CFG = dict(
    seed=0,
    frac=0.10,            # share of train S1 in the modelling sample (density-preserving, see Part 1.9)
    mine_frac=0.10,       # disjoint share of train S1 used ONLY for variant mining + blocking-scorer fitting
    n_jobs=16,
    run_eda=True,         # Part 1 EDA tables (cheap, ~1-2 min)
    run_ablation=True,    # Part 7 ablation ledger (~15-25 min)
    refresh=set(),        # cache names to recompute, e.g. {"sample_parsed"}
    # blocking (Part 4): per-pass top-K and posting caps (cap = max(20, cap_frac * pool size))
    top_k=dict(p1=30, p2=20, p3=20, p5=15, p7=20), cap_frac=dict(p1=2e-4, p2=2e-4, p3=5e-4, p5=2e-4, p7=2e-4, rev=1e-3),
    rev_k=5, p6_max=5, dup_group_max=6, n_cand=50,
    chunk_rows=2e10,      # S1 chunk size = chunk_rows / pool size (bounds join memory)
    # mining (Part 3)
    mine_support=5, mine_lift=10.0, sim_lift=2.0,
    # L1 LightGBM (Part 5)
    folds=5,
    lgb=dict(objective="binary", learning_rate=0.08, num_leaves=63, min_data_in_leaf=200,
             feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, max_bin=127,
             lambda_l2=1.0, seed=0, verbose=-1, num_threads=0),
    rounds=3000, early_stop=50,
    # decision layer (Part 6)
    max_list=10, p_floor=0.3, link_thr=0.9,
)

def macro_f05(pred, truth, i1_all):
    """pred / truth: DataFrames (i1, i23). i1_all: every S1 index being scored (singletons included).
    Per S1: F0.5 = 5TP / (5TP + FN + 4FP); an S1 with no truth and no prediction scores 1.0."""
    tp = pred.join(truth, on=["i1", "i23"], how="semi").group_by("i1").len("tp")
    t = (pl.DataFrame({"i1": np.asarray(i1_all, dtype=np.uint32)})
         .join(tp, on="i1", how="left")
         .join(pred.group_by("i1").len("npred"), on="i1", how="left")
         .join(truth.group_by("i1").len("ntrue"), on="i1", how="left").fill_null(0)
         .with_columns(fp=pl.col("npred") - pl.col("tp"), fn=pl.col("ntrue") - pl.col("tp")))
    f = (pl.when((pl.col("npred") == 0) & (pl.col("ntrue") == 0)).then(1.0)
           .otherwise(5 * pl.col("tp") / (5 * pl.col("tp") + pl.col("fn") + 4 * pl.col("fp"))))
    return t.select(f.fill_nan(0.0)).to_series().mean()

def P(pairs):  # helper: list of (i1, i23) -> DataFrame
    return pl.DataFrame(pairs, schema={"i1": pl.UInt32, "i23": pl.UInt32}, orient="row")

def read_tsv(path):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False).fill_null("")

def gt_pairs(gt):
    return (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
              .filter(pl.col("matched_entity_ids") != "")
              .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "s23_id"}))

LIG = str.maketrans({"œ": "oe", "Œ": "oe", "æ": "ae", "Æ": "ae", "ß": "ss", "’": "'", "‘": "'", "`": "'",
                     "´": "'", "ʼ": "'", "–": "-", "—": "-", "‐": "-", "‑": "-", "＆": "&", "№": "no"})

CONTRACT = {
    # street types
    "street": "st", "road": "rd", "avenue": "ave", "av": "ave", "boulevard": "blvd", "bd": "blvd", "drive": "dr",
    "lane": "ln", "court": "ct", "circle": "cir", "highway": "hwy", "place": "pl", "trail": "trl", "parkway": "pkwy",
    "square": "sq", "terrace": "ter", "expressway": "expy", "freeway": "fwy", "crossing": "xing", "mount": "mt",
    "marg": "marg", "saint": "st", "sainte": "ste",
    "rue": "r", "chemin": "ch", "impasse": "imp", "route": "rte", "faubourg": "fbg", "allee": "all",
    # compass
    "north": "n", "south": "s", "east": "e", "west": "w", "northeast": "ne", "northwest": "nw",
    "southeast": "se", "southwest": "sw",
    # units and markers
    "suite": "ste", "apartment": "apt", "building": "bldg", "floor": "fl", "number": "no", "num": "no",
    "house": "h", "door": "h", "plot": "plot", "office": "off", "sector": "sec", "block": "blk",
    # legal forms
    "limited": "ltd", "private": "pvt", "corporation": "corp", "incorporated": "inc", "company": "co",
    "l.l.c": "llc", "and": "&",
    # business plurals -> singular
    "enterprises": "enterprise", "traders": "trader", "industries": "industry", "services": "service",
    "associates": "associate", "solutions": "solution", "technologies": "technology", "systems": "system",
    "products": "product", "holdings": "holding", "partners": "partner", "ventures": "venture",
    "brothers": "brother", "builders": "builder", "consultants": "consultant", "exports": "export",
    "laboratories": "laboratory", "labs": "lab", "communications": "communication",
    # ordinal words
    "first": "1", "second": "2", "third": "3", "fourth": "4", "fifth": "5", "sixth": "6", "seventh": "7",
    "eighth": "8", "ninth": "9", "tenth": "10",
}

LEGAL_SEED = {"llc", "inc", "corp", "co", "ltd", "pvt", "llp", "lp", "pc", "pllc", "plc", "lllp", "pa", "psc",
              "sa", "sas", "sasu", "sarl", "eurl", "sci", "snc", "scop", "scp", "ei", "eirl", "selarl", "gmbh", "ag"}

STOP = {"the", "of", "de", "la", "le", "du", "des", "&", "a", "an"}

HONORIFIC = {"ms", "messrs", "mr", "mrs", "dr", "m s"}

TLD = {"com", "net", "org", "www", "biz", "info"}

UNIT_MARK = {"unit", "apt", "ste", "flat", "room", "rm", "shop", "off", "fl", "suite", "#"}

STREET_WORDS = {"st", "rd", "ave", "blvd", "dr", "ln", "ct", "cir", "hwy", "pl", "trl", "pkwy", "way", "r", "all",
                "ch", "imp", "rte", "fbg", "quai", "cours", "marg", "nagar", "colony"}

JUNK = {"", "none", "null", "<null>", "n/a", "na", "nan", "-", "--"}

_ELISION = re.compile(r"\b(?:l|d|qu|j|m|n|s|t|c)'(?=[a-z])")

_LETNUM = re.compile(r"\b([a-z])\s?-\s?(\d+)\b")

_NUMCOMP = re.compile(r"\b[a-z]?\d+[a-z]?(?:\s?[/-]\s?[a-z]?\d+[a-z]?)+\b")

_SEP = re.compile(r"[^a-z0-9&_]+")

_ORD = re.compile(r"^0*(\d+)(?:st|nd|rd|th)$")

_DBA = re.compile(r"\b(?:d\s?/\s?b\s?/\s?a|dba|t\s?/\s?a|trading as|a\s?k\s?a|aka|formerly)\b")

_URL = re.compile(r"(?:www\s?\.\s?)?([a-z0-9][a-z0-9-]{2,})\s?\.\s?(?:com|net|org|in|co|biz|info|us|fr)\b|#([a-z0-9]{4,})")

_LANDMARK = re.compile(r"\b(?:near|nr|opp|opposite|behind|beside|next to|in front of|adjacent to|adj to|facing)\b\.?\s*([^,]*)")

_POBOX = re.compile(r"\b(?:p\s?\.?\s?o\s?\.?\s?box|post box|bp)\s*#?\s*(\d+)")

_POSTAL = [re.compile(r"\b(?:pin|pincode|pin code|zip|postal code|cp)\s*[:\-]?\s*(\d{3})\s?(\d{3}|\d{2})\b"),
           re.compile(r"[a-z]\s?-\s?(\d{3})\s?(\d{3})\b"),          # "vellore-632 009"
           re.compile(r"\b(\d{5})(?:-\d{4})?\s*$"),                   # trailing US ZIP / ZIP+4
           re.compile(r"\b(\d{5})\s+([a-z][a-z-]+)")]

def fold(s):
    s = unicodedata.normalize("NFKC", s).translate(LIG)
    if not s.isascii():
        s = unidecode(s)
    return s.lower()

def _numcomp(m):
    return "_".join((p.lstrip("0") or "0") if p.isdigit() else p for p in re.split(r"\s?[/-]\s?", m.group()))

def tokenize(s, subs=None):
    """folded text -> canonical tokens (see the list above)."""
    s = _ELISION.sub("", s).replace("'", "")
    s = _LETNUM.sub(r"\1\2", s)
    s = _NUMCOMP.sub(_numcomp, s)
    out, run = [], []
    for t in _SEP.sub(" ", s).split():
        if t.isdigit():
            t = t.lstrip("0") or "0"
        else:
            m = _ORD.match(t)
            if m:
                t = m.group(1)
        if len(t) == 1 and t.isalpha():          # glue runs of single letters: l l c -> llc
            run.append(t); continue
        if run:
            out.append("".join(run) if len(run) > 1 else run[0]); run = []
        out.append(t)
    if run:
        out.append("".join(run) if len(run) > 1 else run[0])
    out = [CONTRACT.get(t, t) for t in out]
    if subs:
        out = [subs.get(t, t) for t in out]
    return [t for t in out if t]

_TRANS = [("ksh", "x"), ("ph", "f"), ("w", "v"), ("aa", "a"), ("ee", "i"), ("oo", "u"), ("th", "t"), ("bh", "b"),
          ("dh", "d"), ("kh", "k"), ("gh", "g"), ("sh", "s"), ("ch", "c"), ("y", "i"), ("z", "j"), ("q", "k")]

_VOW = re.compile(r"[aeiou]")

_DUP = re.compile(r"(.)\1+")

def skeleton(s):
    if not s:
        return ""
    return s[0] + _DUP.sub(r"\1", _VOW.sub("", s[1:]))

def translit(s):
    for a, b in _TRANS:
        s = s.replace(a, b)
    return skeleton(_DUP.sub(r"\1", s))

def discover_legal(names_by_country, min_share=0.008, min_last=0.8, max_len=4):
    found = {}
    for c, names in names_by_country.items():
        toks = [tokenize(fold(n)) for n in names]
        last = Counter(t[-1] for t in toks if len(t) >= 2)
        anyc = Counter(x for t in toks for x in set(t))
        n = max(1, len(toks))
        found[c] = sorted(t for t, k in last.items()
                          if k / n >= min_share and k / anyc[t] >= min_last and len(t) <= max_len and t.isalpha() and t not in TLD)
    return found

REC_FIELDS = ["name_n", "core", "trade", "legal", "acr", "url", "k_sorted", "k_joined", "k_skel", "k_trans", "k_meta",
              "name_nums", "native", "has_dba", "addr_n", "a_toks", "street", "house", "unit", "postal", "segs",
              "landmark", "numset"]

def parse_record(name, addr, subs_n=None, subs_a=None, seg_subs=None):
    # ------------------------------------------------------------ name
    native = any(ord(ch) > 0x08FF for ch in name)
    f = fold(name)
    url = ""
    m = _URL.search(f)
    if m:
        url = (m.group(1) or m.group(2) or "").replace("-", "")
    parts = _DBA.split(f, maxsplit=1)
    toks = tokenize(parts[0], subs_n)
    trade = tokenize(parts[1], subs_n) if len(parts) > 1 else []
    if len(toks) > 1 and toks[0] in HONORIFIC:
        toks = toks[1:]
    legal = " ".join(sorted({t for t in toks if t in LEGAL}))
    core = [t for t in toks if t not in LEGAL and t not in STOP and not (url and t in TLD)]
    if not core:
        core = [t for t in toks if t not in STOP] or toks
    k_joined = "".join(core)
    acr = "".join(t[0] for t in core) if len(core) >= 2 else ""
    name_nums = sorted({t for t in toks if t[:1].isdigit()})
    # ------------------------------------------------------------ address
    fa = fold(addr).strip()
    if fa in JUNK:
        return (" ".join(toks), core, trade, legal, acr, url, " ".join(sorted(core)), k_joined, skeleton(k_joined),
                translit(k_joined), jellyfish.metaphone(" ".join(core))[:12], name_nums, native, len(parts) > 1,
                "", [], [], "", "", "", [], "", [])
    landmark = " ".join(" ".join(tokenize(x.strip(), subs_a)) for x in _LANDMARK.findall(fa))
    fa = _LANDMARK.sub(",", fa)
    fa = _POBOX.sub(",", fa)
    postal = ""
    for i, rx in enumerate(_POSTAL):
        pm = rx.search(fa)
        rest = fa[pm.end(1):].split(",")[0].split() if (i == 3 and pm) else []
        if pm and (i < 3 or not any(CONTRACT.get(t, t) in STREET_WORDS for t in rest)):
            postal = pm.group(1) + (pm.group(2) if i < 2 else "")
            fa = fa[:pm.start(1)] + " " + fa[pm.end(2 if i < 2 else 1):]
            break
    seg_toks = []
    for sgm in fa.split(","):
        st = tokenize(sgm)                        # segment table first (it is keyed on seed-normalized text) ...
        j = " ".join(st)
        if seg_subs and j in seg_subs:
            st = seg_subs[j].split()
        elif subs_a:                              # ... then token-level substitutions
            st = [subs_a.get(t, t) for t in st]
        if st:
            seg_toks.append(st)
    all_toks = [t for s in seg_toks for t in s]
    unit, skip = "", set()
    for i, t in enumerate(all_toks):
        if t in UNIT_MARK:
            j = i + 1
            while j < len(all_toks) and all_toks[j] in UNIT_MARK | {"no", "number"}:
                j += 1
            if j < len(all_toks):
                unit = all_toks[j]; skip.update(range(i, j + 1))
            break
    a_toks = [t for i, t in enumerate(all_toks) if i not in skip]
    house, street = "", []
    for s in seg_toks:
        hs = [t for t in s if any(ch.isdigit() for ch in t) and t != unit]
        if hs:
            house = hs[0].split("_")[0]
            street = [t for t in s if t.isalpha() and t not in UNIT_MARK and t not in {"no", "h"}]
            break
    segs = [" ".join(s) for s in seg_toks if not any(ch.isdigit() for t in s for ch in t)]
    numset = sorted({g.lstrip("0") or "0" for t in a_toks for g in re.findall(r"\d+", t)})
    return (" ".join(toks), core, trade, legal, acr, url, " ".join(sorted(core)), k_joined, skeleton(k_joined),
            translit(k_joined), jellyfish.metaphone(" ".join(core))[:12], name_nums, native, len(parts) > 1,
            " ".join(all_toks), a_toks, street, house, unit, postal, segs, landmark, numset)

def _parse_batch(names, addrs, countries, subs, legal=None):
    global LEGAL
    if legal is not None:          # V3: worker processes import this module fresh, so LEGAL is passed explicitly
        LEGAL = legal
    out = []
    for n, a, c in zip(names, addrs, countries):
        sn, sa, sg = (subs.get(c) or subs.get("*")) if subs else (None, None, None)
        out.append(parse_record(n, a, sn, sa, sg))
    return out

PARSE_SCHEMA = {"name_n": pl.String, "core": pl.List(pl.String), "trade": pl.List(pl.String), "legal": pl.String,
                "acr": pl.String, "url": pl.String, "k_sorted": pl.String, "k_joined": pl.String, "k_skel": pl.String,
                "k_trans": pl.String, "k_meta": pl.String, "name_nums": pl.List(pl.String), "native": pl.Boolean,
                "has_dba": pl.Boolean, "addr_n": pl.String, "a_toks": pl.List(pl.String), "street": pl.List(pl.String),
                "house": pl.String, "unit": pl.String, "postal": pl.String, "segs": pl.List(pl.String),
                "landmark": pl.String, "numset": pl.List(pl.String)}

def parse_frame(df, subs=None, batch=40_000, desc="parse"):
    """df: raw records (entity_id, business_name, business_address, country) -> parsed frame with idx."""
    cty = df["country"].str.strip_chars().str.to_lowercase()
    cols = [df["business_name"].to_list(), df["business_address"].to_list(), cty.to_list()]
    starts = range(0, df.height, batch)
    gen = Parallel(n_jobs=CFG["n_jobs"], return_as="generator")(
        delayed(_parse_batch)(*(c[i:i + batch] for c in cols), subs, LEGAL) for i in starts)
    rows = [r for part in tqdm(gen, total=len(starts), desc=desc) for r in part]
    parsed = pl.DataFrame(rows, schema=PARSE_SCHEMA, orient="row")
    return pl.concat([df.select("entity_id").with_row_index("idx"), cty.to_frame("cty"), parsed], how="horizontal")

def enrich(s1p, s23p, region_min_share=5e-4):
    """Corpus-level fields, computed per country on this corpus only."""
    both = pl.concat([s1p.select("cty", "segs").with_columns(side=pl.lit(1)),
                      s23p.select("cty", "segs").with_columns(side=pl.lit(2))])
    last = (both.filter(pl.col("segs").list.len() > 0).select("cty", seg=pl.col("segs").list.last())
                .group_by("cty", "seg").len())
    tot = both.group_by("cty").len("n")
    vocab = (last.join(tot, on="cty").filter(pl.col("len") >= region_min_share * pl.col("n"))
                 .filter(pl.col("seg").str.count_matches(" ") <= 3).select("cty", "seg", reg=pl.lit(True)))

    def add_region(p):
        ex = (p.select("idx", "cty", "segs").with_columns(pos=pl.int_ranges(pl.col("segs").list.len()))
                .explode("segs", "pos").drop_nulls("segs").rename({"segs": "seg"})
                .join(vocab, on=["cty", "seg"], how="left"))
        reg = (ex.filter(pl.col("reg")).sort("pos").group_by("idx").agg(region=pl.col("seg").last()))
        loc = (ex.join(reg, on="idx", how="left").filter(pl.col("seg") != pl.col("region").fill_null(""))
                 .group_by("idx").agg(loc=pl.col("seg").unique()))
        return (p.join(reg, on="idx", how="left").join(loc, on="idx", how="left")
                 .with_columns(pl.col("region").fill_null(""), pl.col("loc").fill_null([])).sort("idx"))

    s1p, s23p = add_region(s1p), add_region(s23p)

    def idf_table(col):
        ex = pl.concat([s1p.select("cty", col), s23p.select("cty", col)]).with_row_index("r").explode(col).drop_nulls(col)
        df_ = ex.unique(["r", col]).group_by("cty", col).len("df")
        n = pl.concat([s1p.select("cty"), s23p.select("cty")]).group_by("cty").len("n")
        return df_.join(n, on="cty").select("cty", pl.col(col).alias("tok"),
                                            idf=(pl.col("n") / pl.col("df")).log().cast(pl.Float32))

    def add_idf(p, col, table, out):
        ex = p.select("idx", "cty", col).with_columns(pos=pl.int_ranges(pl.col(col).list.len())).explode(col, "pos")
        ex = ex.join(table, left_on=["cty", col], right_on=["cty", "tok"], how="left").sort("idx", "pos")
        agg = ex.group_by("idx", maintain_order=True).agg(pl.col("idf").fill_null(0.0).alias(out))
        return p.join(agg, on="idx", how="left").with_columns(
            pl.when(pl.col(col).list.len() == 0).then(pl.lit([], pl.List(pl.Float32))).otherwise(pl.col(out)).alias(out))

    for col, out in [("core", "core_idf"), ("a_toks", "a_idf")]:
        tb = idf_table(col)
        s1p, s23p = add_idf(s1p, col, tb, out), add_idf(s23p, col, tb, out)

    anchor = pl.when(pl.col("house") != "").then(pl.col("house") + "|" + pl.col("street").list.first().fill_null(""))
    s1p, s23p = s1p.with_columns(anchor=anchor.otherwise(pl.lit(""))), s23p.with_columns(anchor=anchor.otherwise(pl.lit("")))
    for key, nm in [("k_sorted", "core"), ("anchor", "addr")]:
        c1 = s1p.filter(pl.col(key) != "").group_by("cty", key).len(f"f_{nm}_s1")
        c2 = s23p.filter(pl.col(key) != "").group_by("cty", key).len(f"f_{nm}_s23")
        s1p = s1p.join(c1, on=["cty", key], how="left").join(c2, on=["cty", key], how="left")
        s23p = s23p.join(c1, on=["cty", key], how="left").join(c2, on=["cty", key], how="left")
    fcols = ["f_core_s1", "f_core_s23", "f_addr_s1", "f_addr_s23"]
    s1p = s1p.sort("idx").with_columns(pl.col(fcols).fill_null(0).cast(pl.UInt32))
    s23p = s23p.sort("idx").with_columns(pl.col(fcols).fill_null(0).cast(pl.UInt32))
    return s1p, s23p

def keys_from(df, exprs):
    """df: parsed frame. exprs: list of (prefix, polars expr giving a String or List[String]) -> (idx, key u64)."""
    parts = []
    for prefix, e in exprs:
        k = df.select("idx", k=e)
        if k.schema["k"] == pl.List(pl.String):
            k = k.explode("k")
        parts.append(k.drop_nulls("k").filter(pl.col("k") != "").select("idx", key=(pl.lit(prefix) + pl.col("k")).hash(7)))
    return pl.concat(parts).unique()

def gram_keys(df, col, n=4, prefix="g"):
    g = (df.select("idx", s=pl.col(col)).filter(pl.col("s").str.len_chars() >= n)
           .with_columns(pos=pl.int_ranges(0, pl.col("s").str.len_chars() - n + 1)).explode("pos")
           .select("idx", key=(pl.lit(prefix) + pl.col("s").str.slice(pl.col("pos"), n)).hash(7)))
    return g

def postings(keys, n_pool, cap):
    cnt = keys.group_by("key").len()
    return (keys.join(cnt.filter(pl.col("len") <= cap), on="key")
                .with_columns(idf=(n_pool / pl.col("len")).log().cast(pl.Float32)).drop("len"))

def index_join(q, post, top_k, qname="i1", cname="i23"):
    """q: (idx, key) of query records; post: (idx, key, idf) of candidates -> top_k candidates per query."""
    return (q.join(post, on="key", suffix="_c")
             .group_by("idx", "idx_c").agg(score=pl.col("idf").sum(), nk=pl.len().cast(pl.UInt16))
             .with_columns(rank=pl.col("score").rank("ordinal", descending=True).over("idx").cast(pl.UInt16))
             .filter(pl.col("rank") <= top_k)
             .rename({"idx": qname, "idx_c": cname}))

def chunks_of(df, rows):
    for lo in range(0, df.height, rows):
        yield df.slice(lo, rows)

def simple_keys(p):
    return pl.concat([
        keys_from(p, [("n", pl.col("core")), ("s", pl.col("k_sorted")), ("j", pl.col("k_joined")), ("t", pl.col("k_trans")),
                      ("a", pl.col("a_toks"))]),
        gram_keys(p, "k_joined")]).unique()

def simple_block(s1p, s23p, top_k=10, cap_frac=2e-4):
    out = []
    for (c,), a in s1p.group_by("cty"):
        b = s23p.filter(pl.col("cty") == c)
        if b.height == 0:
            continue
        post = postings(simple_keys(b), b.height, max(20, int(cap_frac * b.height)))
        qa = simple_keys(a)
        step = max(5_000, int(CFG["chunk_rows"] / b.height))
        ids = a["idx"].sort()
        for lo in tqdm(range(0, len(ids), step), desc=f"simple block {c}", leave=False):
            sel = ids[lo:lo + step]
            out.append(index_join(qa.filter(pl.col("idx").is_between(sel[0], sel[-1])), post, top_k))
    return pl.concat(out)

def truth_idx(s1p, s23p, pr):
    """ground-truth pairs (s1_id, s23_id) -> (i1, i23) row indices of the parsed frames."""
    return (pr.join(s1p.select(pl.col("entity_id").alias("s1_id"), pl.col("idx").alias("i1")), on="s1_id")
              .join(s23p.select(pl.col("entity_id").alias("s23_id"), pl.col("idx").alias("i23")), on="s23_id")
              .select("i1", "i23"))

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
    """region-like segment pairs: initials ('uttar pradesh' ~ 'up'), abbreviation ('texas' ~ 'tx') or spelling."""
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
    for r in tqdm(res, total=math.ceil(len(rows) / B), desc=desc):
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

PROTECT = set(CONTRACT.values()) | UNIT_MARK | LEGAL_SEED | STOP

def build_table(pos, neg, n_pos, n_neg, freq, support, min_lift, country=None):
    """pos/neg: Counter[(country, (a, b))] -> (canonical map, kept rows).
    Pairs that already look alike (Jaro-Winkler >= 0.85 or abbreviation) need lift >= CFG['sim_lift'];
    pairs aligned only by position (e.g. transliterations) need the strict min_lift."""
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
            lo, hi = hi, lo                      # a protected seed token is always the canonical side
        if lo not in PROTECT:
            best.setdefault(lo, hi)
    # no chain following: a -> b is kept only if b is itself canonical (chains compound alignment errors)
    return {k: v for k, v in best.items() if v not in best and v != k}, rows

def keys_p1(p):
    return pl.concat([gram_keys(p, "k_joined", 4, "g"), gram_keys(p.filter(pl.col("url") != ""), "url", 4, "g"),
                      keys_from(p, [("j", pl.col("k_joined")), ("j", pl.col("url")), ("k", pl.col("k_skel")),
                                    ("t", pl.col("k_trans"))])]).unique()

def keys_p2(p):
    short = pl.when((pl.col("core").list.len() == 1) & (pl.col("k_joined").str.len_chars() <= 5)).then(pl.col("k_joined"))
    return keys_from(p, [("n", pl.col("core")), ("n", pl.col("trade")), ("s", pl.col("k_sorted")), ("m", pl.col("k_meta")),
                         ("a", pl.col("acr")), ("a", short)])

def keys_p3(p):
    h, c0 = pl.col("house"), pl.col("core").list.first().fill_null("")
    ok_h = pl.when(h != "")
    return keys_from(p, [("h", pl.col("anchor")), ("hn", ok_h.then(h + "|" + c0)), ("p", pl.col("postal")),
                         ("pn", pl.when(pl.col("postal") != "").then(pl.col("postal") + "|" + c0)),
                         ("hu", pl.when((h != "") & (pl.col("unit") != "")).then(h + "|" + pl.col("unit")))])

def keys_p5(p):
    ex = (p.select("idx", t="a_toks").with_columns(pos=pl.int_ranges(pl.col("t").list.len()))
           .explode("t", "pos").drop_nulls("t"))
    nxt = ex.with_columns(pos=pl.col("pos") - 1).rename({"t": "t2"})
    bg = (ex.filter(pl.col("t").str.contains(r"\d")).join(nxt, on=["idx", "pos"])
            .select("idx", key=(pl.lit("b") + pl.col("t") + "_" + pl.col("t2")).hash(7)))
    return pl.concat([keys_from(p, [("x", pl.col("a_toks"))]), bg]).unique()

def keys_rev(p):
    return pl.concat([keys_p2(p), keys_from(p, [("j", pl.col("k_joined")), ("k", pl.col("k_skel")),
                                                ("t", pl.col("k_trans"))])]).unique()

def keys_p7(p):
    return pl.concat([keys_p2(p), keys_p5(p)]).unique()

KEYFN = {"p1": keys_p1, "p2": keys_p2, "p3": keys_p3, "p5": keys_p5, "p7": keys_p7}

PASSES = list(KEYFN)

BLK_FEATS = ([f"score_{p}" for p in PASSES] + [f"lrank_{p}" for p in PASSES] + [f"found_{p}" for p in PASSES]
             + ["rev_score", "lrev_rank", "found_rev", "p6", "n_passes"])

def dup_groups(b, max_size):
    g = (b.filter((pl.col("core").list.len() >= 2) | (pl.col("k_joined").str.len_chars() >= 6))
          .group_by("k_sorted").agg(members=pl.col("idx"), n=pl.len())
          .filter((pl.col("n") >= 2) & (pl.col("n") <= max_size)).with_row_index("gid"))
    return g.select("gid", "members").explode("members").rename({"members": "i23"})

def union_chunk(part, posts, rev, grp):
    long = []
    for p in PASSES:
        r = index_join(KEYFN[p](part), posts[p], CFG["top_k"][p])
        long.append(r.select("i1", "i23", pl.lit(p).alias("pass_"), "score", "rank"))
    w = pl.concat(long).pivot(on="pass_", index=["i1", "i23"], values=["score", "rank"])
    for p in PASSES:
        for c_ in (f"score_{p}", f"rank_{p}"):
            if c_ not in w.columns:
                w = w.with_columns(pl.lit(None, pl.Float32).alias(c_))
    lo, hi = part["idx"].min(), part["idx"].max()
    rv = rev.filter(pl.col("i1").is_between(lo, hi))
    w = w.join(rv, on=["i1", "i23"], how="full", coalesce=True)
    found = [pl.col(f"score_{p}").is_not_null() for p in PASSES]
    w = w.with_columns(n_passes=pl.sum_horizontal(found + [pl.col("rev_rank").is_not_null()]).cast(pl.UInt8),
                       best_rank=pl.min_horizontal([pl.col(f"rank_{p}") for p in PASSES] + [pl.col("rev_rank")]))
    # P6 transitive: group mates of strong candidates
    src = w.filter((pl.col("n_passes") >= 2) | (pl.col("rank_p2") <= 3) | (pl.col("rank_p1") <= 3))
    mates = (src.select("i1", "i23", "best_rank").join(grp, on="i23").join(grp.rename({"i23": "mate"}), on="gid")
                .filter(pl.col("mate") != pl.col("i23"))
                .group_by("i1", "mate").agg(pl.col("best_rank").min())
                .rename({"mate": "i23"}).join(w.select("i1", "i23"), on=["i1", "i23"], how="anti")
                .with_columns(r=pl.col("best_rank").rank("ordinal").over("i1")).filter(pl.col("r") <= CFG["p6_max"])
                .select("i1", "i23", p6=pl.lit(1, pl.UInt8)))
    w = pl.concat([w.with_columns(p6=pl.lit(0, pl.UInt8)), mates], how="diagonal_relaxed")
    return w.drop("best_rank").with_columns(pl.col("n_passes").fill_null(0))

def blk_matrix(w):
    ex = []
    for p in PASSES:
        k = CFG["top_k"][p] + 1
        ex += [pl.col(f"score_{p}").fill_null(0).alias(f"score_{p}"),
               (pl.col(f"rank_{p}").fill_null(k).cast(pl.Float32).log1p()).alias(f"lrank_{p}"),
               pl.col(f"score_{p}").is_not_null().cast(pl.Float32).alias(f"found_{p}")]
    ex += [pl.col("rev_score").fill_null(0), (pl.col("rev_rank").fill_null(CFG["rev_k"] + 1).cast(pl.Float32).log1p()).alias("lrev_rank"),
           pl.col("rev_rank").is_not_null().cast(pl.Float32).alias("found_rev"), pl.col("p6").cast(pl.Float32),
           pl.col("n_passes").cast(pl.Float32)]
    return w.select(ex).select(BLK_FEATS).to_numpy().astype(np.float32)

def block_country(a, b, scorer=None, n_cand=None):
    n23, n1 = b.height, a.height
    caps = {p: max(20, math.ceil(CFG["cap_frac"][p] * n23)) for p in PASSES}
    posts = {}
    for p in tqdm(PASSES, desc="postings", leave=False):
        posts[p] = postings(KEYFN[p](b), n23, caps[p])
    rpost = postings(keys_rev(a), n1, max(20, math.ceil(CFG["cap_frac"]["rev"] * n1)))
    rev = []
    step_r = max(5_000, int(CFG["chunk_rows"] / max(n1, 1)))
    for part in tqdm(list(chunks_of(b, step_r)), desc="reverse pass", leave=False):
        rev.append(index_join(keys_rev(part), rpost, CFG["rev_k"], qname="i23", cname="i1"))
    rev = pl.concat(rev).select("i1", "i23", rev_score="score", rev_rank="rank")
    del rpost
    grp = dup_groups(b, CFG["dup_group_max"])
    step = max(5_000, int(CFG["chunk_rows"] / n23))
    out = []
    for part in tqdm(list(chunks_of(a, step)), desc="blocking chunks", leave=False):
        w = union_chunk(part, posts, rev, grp)
        if scorer is not None:
            w = (w.with_columns(blk=pl.Series(scorer.predict_proba(blk_matrix(w))[:, 1], dtype=pl.Float32))
                  .with_columns(blk_rank=pl.col("blk").rank("ordinal", descending=True).over("i1").cast(pl.UInt16))
                  .filter(pl.col("blk_rank") <= n_cand))
        out.append(w)
    return pl.concat(out, how="diagonal_relaxed")

def block_all(s1p, s23p, scorer=None, n_cand=None, tag=""):
    out = []
    for c in s1p["cty"].unique(maintain_order=True).to_list():
        a, b = s1p.filter(pl.col("cty") == c), s23p.filter(pl.col("cty") == c)
        if b.height == 0:
            log(f"[{tag}] {c}: no S2/S3 records -> no candidates for {a.height:,} S1"); continue
        t = time.time()
        r = block_country(a, b, scorer, n_cand)
        log(f"[{tag}] {c}: {a.height:,} S1 x {b.height:,} S2/S3 -> {r.height:,} pairs ({r.height / a.height:.1f}/S1) in {time.time() - t:.0f}s")
        out.append(r)
    cand = pl.concat(out, how="diagonal_relaxed")
    if scorer is not None:
        cand = cand.with_columns(blk_gap=pl.col("blk").max().over("i1") - pl.col("blk"),
                                 n_cand=pl.len().over("i1").cast(pl.UInt16),
                                 blk_rev_rank=pl.col("blk").rank("ordinal", descending=True).over("i23").cast(pl.UInt16))
    return cand

REC = ["name_n", "core", "trade", "legal", "acr", "url", "k_sorted", "k_joined", "k_skel", "k_trans", "k_meta", "name_nums",
       "native", "has_dba", "addr_n", "a_toks", "street", "house", "unit", "postal", "region", "loc", "landmark", "numset",
       "core_idf", "a_idf", "f_core_s1", "f_core_s23", "f_addr_s1", "f_addr_s23"]

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

PAIR_FEATS = [f for fam in FAM.values() for f in fam]

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
    (nn_a, core_a, trade_a, legal_a, acr_a, url_a, ks_a, kj_a, kk_a, kt_a, km_a, nums_a, nat_a, dba_a, ad_a, at_a, st_a,
     ho_a, un_a, po_a, re_a, lo_a, lm_a, ns_a, ci_a, ai_a, fc1_a, fc23_a, fa1_a, fa23_a) = a
    (nn_b, core_b, trade_b, legal_b, acr_b, url_b, ks_b, kj_b, kk_b, kt_b, km_b, nums_b, nat_b, dba_b, ad_b, at_b, st_b,
     ho_b, un_b, po_b, re_b, lo_b, lm_b, ns_b, ci_b, ai_b, fc1_b, fc23_b, fa1_b, fa23_b) = b
    cs_a, cs_b = " ".join(core_a), " ".join(core_b)
    sca, scb = set(core_a), set(core_b)
    # ---------------- name_sim
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
    # ---------------- name_keys
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
    # ---------------- addr_comp
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
    # ---------------- addr_sim
    empty = not ad_a or not ad_b
    a_tset = -1.0 if empty else fuzz.token_set_ratio(ad_a, ad_b)
    a_idf_ov = -1.0 if empty else _idf_ov(at_a, ai_a, at_b, ai_b)[0]
    addr_sim = [(-1.0 if empty else _cos3(ad_a.replace(" ", ""), ad_b.replace(" ", ""))), a_tset,
                (-1.0 if empty else fuzz.ratio(ad_a, ad_b)), (-1.0 if empty else fuzz.partial_ratio(ad_a, ad_b)),
                a_idf_ov, _jacc(sat, sbt), (-1.0 if empty else min(len(ad_a), len(ad_b)) / max(len(ad_a), len(ad_b))),
                float(not ad_a), float(not ad_b)]
    # ---------------- cross
    cross = [(-1.0 if empty else n_tset * a_tset / 1e4), (-1.0 if empty else min(n_tset, a_tset)),
             (fuzz.partial_ratio(kj_a, ad_b.replace(" ", "")) if ad_b and len(kj_a) >= 4 else -1.0),
             (fuzz.partial_ratio(kj_b, ad_a.replace(" ", "")) if ad_a and len(kj_b) >= 4 else -1.0),
             max(fuzz.token_set_ratio(cs_a, ad_b) if ad_b else 0.0, fuzz.token_set_ratio(cs_b, ad_a) if ad_a else 0.0)]
    # ---------------- quality
    quality = [len(core_a), len(core_b), len(nn_a), len(nn_b), len(at_a), len(at_b), float(dba_a), float(dba_b),
               float(bool(legal_a)), float(bool(legal_b)), float(nat_b), float(bool(url_b)),
               float(len(core_a) == 1 and len(kj_a) <= 5), float(len(core_b) == 1 and len(kj_b) <= 5), sum(ci_a), sum(ci_b)]
    freq = [fc1_a, fc23_a, fc1_b, fc23_b, fa1_a, fa23_b]
    return name_sim + name_keys + addr_comp + addr_sim + cross + quality + freq

def _feat_batch(A, B):
    return np.array([pair_row(a, b) for a, b in zip(A, B)], dtype=np.float32)

PROV = ([f"rank_{p}" for p in PASSES] + [f"score_{p}" for p in PASSES] +
        ["rev_rank", "rev_score", "p6", "n_passes", "blk", "blk_rank", "blk_gap", "blk_rev_rank", "n_cand"])

FAM["prov"] = PROV

FEATS = PAIR_FEATS + ["src3"] + PROV

FAM["quality"] = FAM["quality"] + ["src3"]

MONOTONE = {"n_tset": 1, "n_jw_core": 1, "n_c3": 1, "n_idf_ov": 1, "a_tset": 1, "a_c3": 1, "blk": 1}

def prov_frame(cand):
    ex = []
    for p in PASSES:
        ex += [pl.col(f"rank_{p}").fill_null(CFG["top_k"][p] + 1).cast(pl.Float32).alias(f"rank_{p}"),
               pl.col(f"score_{p}").fill_null(0).cast(pl.Float32).alias(f"score_{p}")]
    ex += [pl.col("rev_rank").fill_null(CFG["rev_k"] + 1).cast(pl.Float32), pl.col("rev_score").fill_null(0).cast(pl.Float32)]
    ex += [pl.col(c).cast(pl.Float32) for c in ["p6", "n_passes", "blk", "blk_rank", "blk_gap", "blk_rev_rank", "n_cand"]]
    return cand.select(ex)

def build_features(cand, s1p, s23p, batch=40_000, desc="features"):
    """-> float32 matrix [n_pairs, len(FEATS)] in FEATS order, rows aligned with cand."""
    R1, R23 = s1p.select(REC), s23p.select(REC)
    i1, i23 = cand["i1"].to_numpy(), cand["i23"].to_numpy()

    def jobs():
        for lo in range(0, len(i1), batch):
            yield delayed(_feat_batch)(R1[i1[lo:lo + batch]].rows(), R23[i23[lo:lo + batch]].rows())

    X = np.empty((len(i1), len(FEATS)), dtype=np.float32)          # preallocated: no vstack copy
    pos = 0
    for part in tqdm(Parallel(n_jobs=CFG["n_jobs"], return_as="generator", pre_dispatch="2*n_jobs")(jobs()),
                     total=math.ceil(len(i1) / batch), desc=desc):
        X[pos:pos + len(part), :len(PAIR_FEATS)] = part
        pos += len(part)
    X[:, len(PAIR_FEATS)] = s23p["entity_id"].str.starts_with("S3").cast(pl.Float32).to_numpy()[i23]
    X[:, len(PAIR_FEATS) + 1:] = prov_frame(cand).to_numpy().astype(np.float32)
    return X

def iso_crossfit(p, y, folds):
    out = np.zeros_like(p)
    for k in range(CFG["folds"]):
        tr, va = folds != k, folds == k
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p[tr], y[tr])
        out[va] = iso.predict(p[va])
    return out.astype(np.float32)

def reliability(p, y, groups=None, bins=(0, .1, .3, .5, .7, .8, .9, .95, .99, 1.0001)):
    d = pl.DataFrame({"p": p, "y": y}).with_columns(bin=pl.col("p").cut(list(bins[1:-1])))
    if groups is not None:
        d = d.with_columns(g=pl.Series(groups))
    keys = ["g", "bin"] if groups is not None else ["bin"]
    return d.group_by(keys).agg(n=pl.len(), mean_p=pl.col("p").mean(), rate=pl.col("y").mean()).sort(keys)

def companion_pairs(sc, top=6, pmin=0.2):
    t = (sc.filter(pl.col("p") >= pmin).with_columns(r=pl.col("p").rank("ordinal", descending=True).over("i1"))
           .filter(pl.col("r") <= top).select("i1", "i23", "p"))
    pr = t.join(t, on="i1", suffix="_y").filter(pl.col("i23") < pl.col("i23_y"))
    return pr.rename({"i23": "x", "i23_y": "y", "p": "p_x", "p_y": "p_y"})

def companion_features(pairs, s23p, batch=40_000):
    R = s23p.select(REC)
    uniq = pairs.select("x", "y").unique()
    xa, ya = uniq["x"].to_numpy(), uniq["y"].to_numpy()
    jobs = (delayed(_feat_batch)(R[xa[lo:lo + batch]].rows(), R[ya[lo:lo + batch]].rows()) for lo in range(0, len(xa), batch))
    parts = list(tqdm(Parallel(n_jobs=CFG["n_jobs"], return_as="generator")(jobs), total=math.ceil(len(xa) / batch), desc="companion features"))
    return uniq, (np.vstack(parts) if parts else np.zeros((0, len(PAIR_FEATS)), np.float32))

def propagate(sc, links, thr):
    """one hop: p'(x) = max(p(x), max over strong links (p_link >= thr) of p_link * p(y)), within the same S1 list."""
    L = links.filter(pl.col("p_link") >= thr)
    both = pl.concat([L.select("i1", a="x", b="y", pl_="p_link"), L.select("i1", a="y", b="x", pl_="p_link")])
    inh = (both.join(sc.rename({"i23": "b", "p": "pb"}), on=["i1", "b"])
               .group_by("i1", "a").agg(inh=(pl.col("pl_") * pl.col("pb")).max()).rename({"a": "i23"}))
    return (sc.join(inh, on=["i1", "i23"], how="left")
              .with_columns(p=pl.max_horizontal("p", pl.col("inh").fill_null(0))).drop("inh"))

def exclusivity(sc, eps=0.0, lam=1.0):
    top2 = sc.group_by("i23").agg(p1=pl.col("p").max(), p2=pl.col("p").top_k(2).min(), n=pl.len())
    s = sc.join(top2, on="i23").filter(pl.col("p") == pl.col("p1"))
    s = s.unique(["i23"], keep="first")                               # exact ties: keep one
    amb = (pl.col("n") > 1) & ((pl.col("p1") - pl.col("p2")) < eps)
    return s.with_columns(p=pl.when(amb).then(pl.col("p") * lam).otherwise(pl.col("p"))).select("i1", "i23", "p")

def singleton_features(sc_excl, s1p, cand):
    lst = sc_excl.sort("p", descending=True).group_by("i1").agg(
        top=pl.col("p").head(3), psum=pl.col("p").sum(), n05=(pl.col("p") >= 0.5).sum(), n08=(pl.col("p") >= 0.8).sum())
    cinfo = cand.group_by("i1").agg(ncand=pl.len(), blk_max=pl.col("blk").max(), npass_max=pl.col("n_passes").max())
    base = s1p.select(pl.col("idx").alias("i1"), "f_core_s1", "f_core_s23", "f_addr_s1",
                      idf_sum=pl.col("core_idf").list.sum(), a_len=pl.col("a_toks").list.len(),
                      has_house=(pl.col("house") != "").cast(pl.Int8))
    f = (base.join(lst, on="i1", how="left").join(cinfo, on="i1", how="left")
             .with_columns(t1=pl.col("top").list.get(0, null_on_oob=True), t2=pl.col("top").list.get(1, null_on_oob=True),
                           t3=pl.col("top").list.get(2, null_on_oob=True)).drop("top").fill_null(0))
    return f

SING_FEATS = ["t1", "t2", "t3", "psum", "n05", "n08", "ncand", "blk_max", "npass_max", "f_core_s1", "f_core_s23",
              "f_addr_s1", "idf_sum", "a_len", "has_house"]

def expected_f_select(sc_excl, p_single, m_miss, floor, max_k):
    """Per S1: choose k in 0..max_k maximizing E[F0.5] of predicting its top-k (Poisson-binomial DP, vectorized)."""
    s = (sc_excl.filter(pl.col("p") >= floor).sort(["i1", "p"], descending=[False, True])
                .with_columns(r=pl.int_range(pl.len()).over("i1")))
    rest = sc_excl.group_by("i1").agg(ptot=pl.col("p").sum())
    ids = s["i1"].unique().sort()
    n, K = len(ids), max_k
    row = ids.to_frame().with_row_index("row")
    s = s.join(row, on="i1").filter(pl.col("r") < K)
    Q = np.zeros((n, K), np.float64)
    Q[s["row"].to_numpy(), s["r"].to_numpy()] = s["p"].to_numpy()
    tot = row.join(rest, on="i1", how="left")["ptot"].fill_null(0).to_numpy()
    ps = row.join(p_single, on="i1", how="left")["p_single"].fill_null(0.5).to_numpy()
    dist = np.zeros((n, K + 1)); dist[:, 0] = 1.0
    best_ef, best_k = ps.copy(), np.zeros(n, dtype=np.int64)
    cum = np.zeros(n)
    t = np.arange(K + 1)
    for k in range(1, K + 1):
        q = Q[:, k - 1]
        new = dist * (1 - q)[:, None]
        new[:, 1:] += dist[:, :-1] * q[:, None]
        dist = new
        cum += q
        eu = np.maximum(tot - cum, 0) + m_miss                          # expected true matches left out
        denom = 5 * t[None, :] + eu[:, None] + 4 * (k - t[None, :])
        ef = (dist * np.where(t[None, :] > 0, 5 * t[None, :] / np.maximum(denom, 1e-9), 0)).sum(1)
        ef = np.where(q > 0, ef, -1)                                     # only real candidates
        better = ef > best_ef
        best_ef, best_k = np.where(better, ef, best_ef), np.where(better, k, best_k)
    chosen = row.with_columns(k=pl.Series(best_k))
    return s.join(chosen.select("row", "k"), on="row").filter(pl.col("r") < pl.col("k")).select("i1", "i23")

def write_tsv(pairs, col, path, s1_order):
    lists = (pairs.unique(["s1_id", "s23_id"]).sort("s1_id", "s23_id").group_by("s1_id", maintain_order=True)
                  .agg(pl.col("s23_id").str.join(",").alias(col)))
    out = s1_order.to_frame("source1_entity_id").join(lists.rename({"s1_id": "source1_entity_id"}),
                                                       on="source1_entity_id", how="left").fill_null("")
    assert out.height == s1_order.len() and out["source1_entity_id"].is_unique().all()
    assert not out[col].str.contains("nan").any() and out[col].str.contains(r"(^|,)S1-").sum() == 0
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_csv(path, separator="\t", quote_style="never")
    return out
