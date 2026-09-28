"""L6 representation v3 + the L7b dense retrieval pass (GPU).

- Bi-encoder: multilingual e5 (MIT), mean pooling, L2-normalised; bge-m3 / Qwen3-Embedding are drop-in via CFG.
- Fine-tuning (design 4.5): symmetric InfoNCE with in-batch negatives, batches grouped by (country, region) so
  negatives are hard; positives = true pairs of the *mining sample M* (disjoint from S, so S retrieval is honest)
  plus self-supervised noise-model variants of unlabeled *test* records (adapts the space to France, no labels).
- Retrieval: exact chunked GPU kNN per country label, S1 -> S2/S3 top-K plus the reverse S2/S3 -> S1 top-r.
"""
import math, random, re, time

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F

from .common import CFG, CACHE, MODELS, log, cached

_JUNK = {"", "none", "null", "<null>", "n/a", "na", "nan", "-", "--"}


def record_text(name, addr):
    a = (addr or "").strip()
    if a.lower() in _JUNK:
        a = ""
    n = (name or "").strip()
    return f"query: {n} | {a}" if a else f"query: {n}"


def texts_of(raw):
    return [record_text(n, a) for n, a in zip(raw["business_name"].to_list(), raw["business_address"].to_list())]


# --------------------------------------------------------------------------------------------- noise model (light)
_LEGAL_WORDS = ["pvt", "ltd", "private", "limited", "llc", "inc", "corp", "co", "sarl", "sas", "sa", "eurl"]


def noise_variant(text, rng):
    """A synthetic duplicate of a record (operators sampled per record, V3 section 4.1 light version)."""
    body = text[len("query: "):]
    name, _, addr = body.partition(" | ")
    ops = rng.random(8)
    toks = name.split()
    if ops[0] < 0.3 and len(toks) > 1:                                  # word-order transposition
        i = rng.integers(0, len(toks) - 1); toks[i], toks[i + 1] = toks[i + 1], toks[i]
    if ops[1] < 0.3:                                                    # drop / add a legal form
        low = [t.lower().strip(".,") for t in toks]
        keep = [t for t, l in zip(toks, low) if l not in _LEGAL_WORDS]
        toks = keep if (keep and len(keep) < len(toks)) else toks + [_LEGAL_WORDS[rng.integers(0, len(_LEGAL_WORDS))].upper()]
    name = " ".join(toks)
    if ops[2] < 0.4 and len(name) > 4:                                  # character typo
        i = int(rng.integers(1, len(name) - 1))
        k = rng.integers(0, 3)
        name = name[:i] + name[i + 1:] if k == 0 else (name[:i] + name[i] + name[i:] if k == 1 else
                                                       name[:i] + name[i + 1] + name[i] + name[i + 2:])
    if ops[3] < 0.3:
        name = name.upper() if rng.random() < 0.5 else name.lower()
    segs = [s.strip() for s in addr.split(",") if s.strip()]
    if segs:
        if ops[4] < 0.35:                                               # drop address components
            segs = [s for s in segs if rng.random() > 0.35] or segs[:1]
        if ops[5] < 0.35:                                               # reorder components
            rng.shuffle(segs)
        if ops[6] < 0.15:                                               # missing address entirely
            segs = []
    if ops[7] < 0.2:
        segs = [re.sub(r"\b0*(\d+)", lambda m: m.group(1)[1:] or m.group(1), s, count=1) for s in segs]
    addr = ", ".join(segs)
    return f"query: {name} | {addr}" if addr else f"query: {name}"


# --------------------------------------------------------------------------------------------- encoder
class DenseEncoder:
    def __init__(self, path_or_name=None, max_len=None):
        from transformers import AutoModel, AutoTokenizer
        self.name = path_or_name or CFG["dense_model"]
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModel.from_pretrained(self.name).to("cuda")
        self.max_len = max_len or CFG["dense_dim_max_len"]

    def _embed(self, batch_texts):
        enc = self.tok(batch_texts, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to("cuda")
        out = self.model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).to(out.dtype)
        e = (out * m).sum(1) / m.sum(1).clamp(min=1)
        return F.normalize(e.float(), dim=-1)

    @torch.no_grad()
    def encode(self, texts, bs=1024, desc="encode"):
        from tqdm.auto import tqdm
        self.model.eval()
        order = np.argsort([len(t) for t in texts])[::-1]
        out = None
        for lo in tqdm(range(0, len(texts), bs), desc=desc, mininterval=10):
            sel = order[lo:lo + bs]
            with torch.autocast("cuda", dtype=torch.float16):
                e = self._embed([texts[i] for i in sel])
            if out is None:
                out = np.empty((len(texts), e.shape[1]), dtype=np.float16)
            out[sel] = e.half().cpu().numpy()
        return out

    def finetune(self, pairs, steps, bs, lr=3e-5, temp=0.05, desc="dense fine-tune"):
        """pairs: list of batches' source -> iterator yielding (texts_a, texts_b) lists of length bs."""
        from tqdm.auto import tqdm
        self.model.train()
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * max(0.0, 1 - s / steps))
        scaler = torch.amp.GradScaler("cuda")
        it = iter(pairs)
        bar = tqdm(range(steps), desc=desc, mininterval=10)
        run = None
        for step in bar:
            a, b = next(it)
            with torch.autocast("cuda", dtype=torch.float16):
                ea, eb = self._embed(a), self._embed(b)
                logits = ea @ eb.T / temp
                lab = torch.arange(len(a), device="cuda")
                loss = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run = loss.item() if run is None else 0.98 * run + 0.02 * loss.item()
            if step % 200 == 0:
                bar.set_postfix(loss=f"{run:.4f}")
        self.model.eval()
        return run

    def save(self, path):
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)


def pair_batches(sup, unsup, bs, frac_unsup, seed=0):
    """Infinite generator of (a_texts, b_texts) batches.
    sup: DataFrame (grp, ta, tb, i1) of labelled pairs; batches are drawn inside one grp (country|region) to make
    in-batch negatives hard, with at most one pair per S1 per batch (avoids false negatives).
    unsup: list of record texts; a positive is a noise-model variant of the same record."""
    rng = np.random.default_rng(seed)
    groups = {}
    for g, ta, tb, i1 in sup.select("grp", "ta", "tb", "i1").iter_rows():
        groups.setdefault(g, []).append((ta, tb, i1))
    keys = list(groups)
    w = np.array([len(groups[k]) for k in keys], dtype=np.float64)
    w /= w.sum()
    n_sup = bs - int(round(bs * frac_unsup))
    while True:
        a, b, seen = [], [], set()
        while len(a) < n_sup:
            g = keys[rng.choice(len(keys), p=w)]
            lst = groups[g]
            for j in rng.integers(0, len(lst), min(len(lst), n_sup - len(a))):
                ta, tb, i1 = lst[j]
                if i1 in seen:
                    continue
                seen.add(i1); a.append(ta); b.append(tb)
        if unsup is not None and len(a) < bs:
            for j in rng.integers(0, len(unsup), bs - len(a)):
                t = unsup[j]
                a.append(noise_variant(t, rng)); b.append(noise_variant(t, rng) if rng.random() < 0.5 else t)
        yield a, b


# --------------------------------------------------------------------------------------------- GPU kNN
@torch.no_grad()
def knn(q, p, k, q_chunk=None, p_chunk=1_000_000, exclude_self=False, score_bytes=768 << 20):
    """Exact inner-product top-k: q [nq, d], p [np, d] (fp16 numpy, L2-normalised) -> (idx int32, sim float16)."""
    k = min(k, p.shape[0] - int(exclude_self))
    if k <= 0:
        return np.zeros((q.shape[0], 0), np.int32), np.zeros((q.shape[0], 0), np.float16)
    P = [torch.from_numpy(p[lo:lo + p_chunk]).to("cuda") for lo in range(0, p.shape[0], p_chunk)]
    if q_chunk is None:                       # bound the fp16 score block q_chunk x p_chunk to ~score_bytes
        q_chunk = int(max(128, min(8192, score_bytes // (2 * max(x.shape[0] for x in P)))))
    offs = np.cumsum([0] + [x.shape[0] for x in P])
    I = np.empty((q.shape[0], k), np.int32)
    S = np.empty((q.shape[0], k), np.float16)
    kk = k + int(exclude_self)
    for lo in range(0, q.shape[0], q_chunk):
        Q = torch.from_numpy(q[lo:lo + q_chunk]).to("cuda")
        best_s, best_i = None, None
        for j, Pc in enumerate(P):
            s = Q @ Pc.T
            ts, ti = s.topk(min(kk, Pc.shape[0]), dim=1)
            ti = ti + int(offs[j])
            if best_s is None:
                best_s, best_i = ts, ti
            else:
                cs, ci = torch.cat([best_s, ts], 1), torch.cat([best_i, ti], 1)
                best_s, o = cs.topk(kk, dim=1)
                best_i = ci.gather(1, o)
        if exclude_self:
            rows = torch.arange(lo, lo + Q.shape[0], device="cuda").unsqueeze(1)
            mask = best_i == rows
            best_s = best_s.masked_fill(mask, -2)
            best_s, o = best_s.topk(k, dim=1)
            best_i = best_i.gather(1, o)
        I[lo:lo + Q.shape[0]] = best_i.cpu().numpy()
        S[lo:lo + Q.shape[0]] = best_s.cpu().numpy()
    del P
    torch.cuda.empty_cache()
    return I, S


def dense_pass(e1, e23, cty1, cty23, k, rev_k=3):
    """Per country label: S1 -> S2/S3 top-k and S2/S3 -> S1 top-rev_k. -> DataFrame(i1, i23, dcos, drank, drev_rank)."""
    out = []
    for c in np.unique(cty1):
        a = np.where(cty1 == c)[0]
        b = np.where(cty23 == c)[0]
        if len(b) == 0:
            continue
        t = time.time()
        I, S = knn(e1[a], e23[b], k)
        fw = pl.DataFrame({"i1": np.repeat(a, I.shape[1]).astype(np.uint32), "i23": b[I.ravel()].astype(np.uint32),
                           "dcos": S.ravel().astype(np.float32),
                           "drank": np.tile(np.arange(1, I.shape[1] + 1, dtype=np.uint16), len(a))})
        Ir, Sr = knn(e23[b], e1[a], rev_k)
        rv = pl.DataFrame({"i23": np.repeat(b, Ir.shape[1]).astype(np.uint32), "i1": a[Ir.ravel()].astype(np.uint32),
                           "dcos_r": Sr.ravel().astype(np.float32),
                           "drev_rank": np.tile(np.arange(1, Ir.shape[1] + 1, dtype=np.uint16), len(b))})
        m = fw.join(rv, on=["i1", "i23"], how="full", coalesce=True).with_columns(
            dcos=pl.coalesce("dcos", "dcos_r")).drop("dcos_r")
        out.append(m)
        log(f"dense pass [{c}]: {len(a):,} S1 x {len(b):,} S2/S3 -> {m.height:,} pairs in {time.time() - t:.0f}s")
    return pl.concat(out)


def s23_neighbours(e23, cty23, queries, k=5):
    """Record-record dense neighbours (for L7a clusters / cluster-aware retrieval): queries = S2/S3 indices."""
    out = []
    for c in np.unique(cty23):
        b = np.where(cty23 == c)[0]
        q = np.intersect1d(queries, b)
        if len(q) == 0:
            continue
        pos = np.searchsorted(b, q)
        I, S = knn(e23[q], e23[b], k + 1)
        nb = b[I]
        ok = nb != q[:, None]
        out.append(pl.DataFrame({"x": np.repeat(q, I.shape[1])[ok.ravel()].astype(np.uint32),
                                 "y": nb.ravel()[ok.ravel()].astype(np.uint32),
                                 "cos": S.ravel()[ok.ravel()].astype(np.float32)}))
    return pl.concat(out) if out else pl.DataFrame(schema={"x": pl.UInt32, "y": pl.UInt32, "cos": pl.Float32})


# --------------------------------------------------------------------------------------------- training entry point
def train_dense(M, test_texts=None, steps=None, bs=None, frac_unsup=0.2):
    """Fine-tune the bi-encoder on the mining sample M (+ self-supervised test records). Saved under models/."""
    path = MODELS / "dense_ft"
    if (path / "config.json").exists():
        log(f"dense encoder: using fine-tuned checkpoint {path}")
        return DenseEncoder(str(path))
    steps, bs = steps or CFG["dense_ft_steps"], bs or CFG["dense_ft_bs"]
    t1 = texts_of(M["raw1"])
    t23 = texts_of(M["raw23"])
    reg = M["s1p"]["cty"] + "|" + M["s1p"]["region"]
    tr = M["truth"]
    sup = pl.DataFrame({"i1": tr["i1"], "i23": tr["i23"]}).with_columns(
        grp=pl.Series(reg.to_numpy()[tr["i1"].to_numpy()]),
        ta=pl.Series([t1[i] for i in tr["i1"].to_numpy()]),
        tb=pl.Series([t23[i] for i in tr["i23"].to_numpy()]))
    enc = DenseEncoder()
    gen = pair_batches(sup, test_texts, bs, frac_unsup if test_texts else 0.0)
    log(f"dense fine-tune: {sup.height:,} labelled pairs (sample M) | "
        f"{0 if test_texts is None else len(test_texts):,} unlabeled test records | {steps} steps x {bs}")
    loss = enc.finetune(gen, steps, bs)
    log(f"dense fine-tune done: final loss {loss:.4f}")
    enc.save(path)
    return enc
