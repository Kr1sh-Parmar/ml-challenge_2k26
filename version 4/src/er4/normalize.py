"""Character folding, tokenization, name keys, the record parser and corpus enrichment (ported from V2, Part 2).

V4 changes: French ordinals / bis-ter / cedex / arrondissement; the legal-form set is an explicit argument (worker
processes do not see runtime globals); enrich() adds scale-free frequency rates next to the raw counts."""
import math, re, unicodedata
from collections import Counter

import jellyfish
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from tqdm.auto import tqdm
from unidecode import unidecode

from .config import CFG

LIG = str.maketrans({"œ": "oe", "Œ": "oe", "æ": "ae", "Æ": "ae", "ß": "ss", "’": "'", "‘": "'", "`": "'",
                     "´": "'", "ʼ": "'", "–": "-", "—": "-", "‐": "-", "‑": "-", "＆": "&", "№": "no"})
CONTRACT = {
    # street types
    "street": "st", "road": "rd", "avenue": "ave", "av": "ave", "boulevard": "blvd", "bd": "blvd", "drive": "dr",
    "lane": "ln", "court": "ct", "circle": "cir", "highway": "hwy", "place": "pl", "trail": "trl", "parkway": "pkwy",
    "square": "sq", "terrace": "ter", "expressway": "expy", "freeway": "fwy", "crossing": "xing", "mount": "mt",
    "marg": "marg", "saint": "st", "sainte": "ste",
    "rue": "r", "chemin": "ch", "impasse": "imp", "route": "rte", "faubourg": "fbg", "allee": "all", "allees": "all",
    "arrondissement": "arr",
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
DROP_TOK = {"cedex"}                                   # French mail-routing noise
HOUSE_SUFFIX = {"bis", "ter", "quater"}                # "12 bis rue ..." -> house 12

_ELISION = re.compile(r"\b(?:l|d|qu|j|m|n|s|t|c)'(?=[a-z])")
_LETNUM = re.compile(r"\b([a-z])\s?-\s?(\d+)\b")
_NUMCOMP = re.compile(r"\b[a-z]?\d+[a-z]?(?:\s?[/-]\s?[a-z]?\d+[a-z]?)+\b")
_SEP = re.compile(r"[^a-z0-9&_]+")
_ORD = re.compile(r"^0*(\d+)(?:st|nd|rd|th|er|e|eme)$")     # + French 1er / 2e / 3eme
_DBA = re.compile(r"\b(?:d\s?/\s?b\s?/\s?a|dba|t\s?/\s?a|trading as|a\s?k\s?a|aka|formerly)\b")
_URL = re.compile(r"(?:www\s?\.\s?)?([a-z0-9][a-z0-9-]{2,})\s?\.\s?(?:com|net|org|in|co|biz|info|us|fr)\b|#([a-z0-9]{4,})")
_LANDMARK = re.compile(r"\b(?:near|nr|opp|opposite|behind|beside|next to|in front of|adjacent to|adj to|facing)\b\.?\s*([^,]*)")
_POBOX = re.compile(r"\b(?:p\s?\.?\s?o\s?\.?\s?box|post box|bp)\s*#?\s*(\d+)")
_POSTAL = [re.compile(r"\b(?:pin|pincode|pin code|zip|postal code|cp)\s*[:\-]?\s*(\d{3})\s?(\d{3}|\d{2})\b"),
           re.compile(r"[a-z]\s?-\s?(\d{3})\s?(\d{3})\b"),
           re.compile(r"\b(\d{5})(?:-\d{4})?\s*$"),
           re.compile(r"\b(\d{5})\s+([a-z][a-z-]+)")]


def fold(s):
    s = unicodedata.normalize("NFKC", s).translate(LIG)
    if not s.isascii():
        s = unidecode(s)
    return s.lower()


def _numcomp(m):
    return "_".join((p.lstrip("0") or "0") if p.isdigit() else p for p in re.split(r"\s?[/-]\s?", m.group()))


def tokenize(s, subs=None):
    """folded text -> canonical tokens."""
    s = _ELISION.sub("", s).replace("'", "")
    s = _LETNUM.sub(r"\1\2", s)
    s = _NUMCOMP.sub(_numcomp, s)
    out, run = [], []
    for t in _SEP.sub(" ", s).split():
        if t in DROP_TOK or (t in HOUSE_SUFFIX and out and out[-1].isdigit() and not run):
            continue
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


assert tokenize(fold("19320- 1ND PL")) == ["19320", "1", "pl"]
assert tokenize(fold("0015 HAGER LN")) == ["15", "hager", "ln"]
assert tokenize(fold("Zeclarsoft, L.L.C.")) == ["zeclarsoft", "llc"]
assert tokenize(fold("École de l'Hôpital")) == ["ecole", "de", "hopital"]
assert tokenize(fold("12 bis Rue du 1er Mai, 75011 Paris CEDEX")) == ["12", "r", "du", "1", "mai", "75011", "paris"]

_TRANS = [("ksh", "x"), ("ph", "f"), ("w", "v"), ("aa", "a"), ("ee", "i"), ("oo", "u"), ("th", "t"), ("bh", "b"),
          ("dh", "d"), ("kh", "k"), ("gh", "g"), ("sh", "s"), ("ch", "c"), ("y", "i"), ("z", "j"), ("q", "k")]
_VOW = re.compile(r"[aeiou]")
_DUP = re.compile(r"(.)\1+")


def skeleton(s):
    return s[0] + _DUP.sub(r"\1", _VOW.sub("", s[1:])) if s else ""


def translit(s):
    for a, b in _TRANS:
        s = s.replace(a, b)
    return skeleton(_DUP.sub(r"\1", s))


assert translit("saaphttveer") == translit("software") == "sftvr"


def discover_legal(names_by_country, min_share=0.008, min_last=0.8, max_len=4):
    """Legal-form candidates: frequent last tokens that are almost always last, short, not a TLD (per country)."""
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


def parse_record(name, addr, legal_set, subs_n=None, subs_a=None, seg_subs=None):
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
    legal = " ".join(sorted({t for t in toks if t in legal_set}))
    core = [t for t in toks if t not in legal_set and t not in STOP and not (url and t in TLD)]
    if not core:
        core = [t for t in toks if t not in STOP] or toks
    k_joined = "".join(core)
    acr = "".join(t[0] for t in core) if len(core) >= 2 else ""
    name_nums = sorted({t for t in toks if t[:1].isdigit()})
    name_part = (" ".join(toks), core, trade, legal, acr, url, " ".join(sorted(core)), k_joined, skeleton(k_joined),
                 translit(k_joined), jellyfish.metaphone(" ".join(core))[:12], name_nums, native, len(parts) > 1)
    # ------------------------------------------------------------ address
    fa = fold(addr).strip()
    if fa in JUNK:
        return name_part + ("", [], [], "", "", "", [], "", [])
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
        st = tokenize(sgm)
        j = " ".join(st)
        if seg_subs and j in seg_subs:
            st = seg_subs[j].split()
        elif subs_a:
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
    return name_part + (" ".join(all_toks), a_toks, street, house, unit, postal, segs, landmark, numset)


_r = dict(zip(REC_FIELDS, parse_record("Dr coinsintered.com", "DOOR NO 35 , MANKAPUR, OPP.RANU PRIMARY SCHOOL, NAGPUR, Maharashtra", LEGAL_SEED)))
assert _r["url"] == "coinsintered" and _r["landmark"] == "ranu primary school" and _r["house"] == "35"
_r = dict(zip(REC_FIELDS, parse_record("Umbraavi d/b/a Centre Médical du Marie", "ALLÉE CLÉMENCEAU, 33260 LA TESTE DE BUCH", LEGAL_SEED)))
assert _r["postal"] == "33260" and _r["trade"] == ["centre", "medical", "du", "marie"] and _r["has_dba"]

PARSE_SCHEMA = {"name_n": pl.String, "core": pl.List(pl.String), "trade": pl.List(pl.String), "legal": pl.String,
                "acr": pl.String, "url": pl.String, "k_sorted": pl.String, "k_joined": pl.String, "k_skel": pl.String,
                "k_trans": pl.String, "k_meta": pl.String, "name_nums": pl.List(pl.String), "native": pl.Boolean,
                "has_dba": pl.Boolean, "addr_n": pl.String, "a_toks": pl.List(pl.String), "street": pl.List(pl.String),
                "house": pl.String, "unit": pl.String, "postal": pl.String, "segs": pl.List(pl.String),
                "landmark": pl.String, "numset": pl.List(pl.String)}


def _parse_batch(names, addrs, legal_set, subs):
    sn, sa, sg = subs if subs else (None, None, None)
    return pl.DataFrame([parse_record(n, a, legal_set, sn, sa, sg) for n, a in zip(names, addrs)],
                        schema=PARSE_SCHEMA, orient="row")


def parse_frame(df, legal_set, subs=None, batch=40_000, desc="parse"):
    """df: raw records of ONE country (entity_id, business_name, business_address, country)
    -> parsed frame with idx (row position) and cty. Each worker returns an Arrow-backed frame (no big Python lists)."""
    cty = df["country"].str.strip_chars().str.to_lowercase()
    names, addrs = df["business_name"].to_list(), df["business_address"].to_list()
    starts = range(0, df.height, batch)
    gen = Parallel(n_jobs=CFG["n_jobs"], return_as="generator")(
        delayed(_parse_batch)(names[i:i + batch], addrs[i:i + batch], legal_set, subs) for i in starts)
    parts = list(tqdm(gen, total=len(starts), desc=desc, mininterval=10))
    parsed = pl.concat(parts) if parts else pl.DataFrame(schema=PARSE_SCHEMA)
    return pl.concat([df.select("entity_id").with_row_index("idx"), cty.to_frame("cty"), parsed], how="horizontal")


FREQ_COLS = ["f_core_s1", "f_core_s23", "f_addr_s1", "f_addr_s23"]


def enrich(s1p, s23p, region_min_share=5e-4):
    """Corpus-level fields for ONE country's corpus: region/locality, IDF lists, chain/mall counts.
    V4: also `*_r` = log1p(count per 100k records of that side), which does not change with corpus size."""
    both = pl.concat([s1p.select("cty", "segs"), s23p.select("cty", "segs")])
    last = (both.filter(pl.col("segs").list.len() > 0).select("cty", seg=pl.col("segs").list.last())
                .group_by("cty", "seg").len())
    tot = both.group_by("cty").len("n")
    vocab = (last.join(tot, on="cty").filter(pl.col("len") >= region_min_share * pl.col("n"))
                 .filter(pl.col("seg").str.count_matches(" ") <= 3).select("cty", "seg", reg=pl.lit(True)))
    del both

    def add_region(p):
        ex = (p.select("idx", "cty", "segs").with_columns(pos=pl.int_ranges(pl.col("segs").list.len()))
                .explode("segs", "pos").drop_nulls("segs").rename({"segs": "seg"})
                .join(vocab, on=["cty", "seg"], how="left"))
        reg = ex.filter(pl.col("reg")).sort("pos").group_by("idx").agg(region=pl.col("seg").last())
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
    n1, n23 = max(1, s1p.height), max(1, s23p.height)
    rate = lambda c: (pl.col(c).cast(pl.Float32) * (1e5 / (n1 if c.endswith("_s1") else n23))).log1p().alias(c + "_r")
    fix = lambda p: (p.sort("idx").with_columns(pl.col(FREQ_COLS).fill_null(0).cast(pl.UInt32))
                      .with_columns([rate(c) for c in FREQ_COLS]))
    return fix(s1p), fix(s23p)
