"""L10 text-view scorer: cross-encoder (default microsoft/mdeberta-v3-base, MIT) fine-tuned on pruned candidate pairs (GPU).

Serialization (design 6.2): raw name + raw address of both records (native script kept), plus a compact pooled-evidence
summary of the candidate's L7a cluster. OOF by S1 group; test scores are the average of the fold models (bagging).
bf16 autocast (Ampere); length-sorted dynamic padding for throughput.
"""
import math, time

import numpy as np
import torch
import torch.nn.functional as F

from .common import CFG, MODELS, log

_JUNK = {"", "none", "null", "<null>", "n/a", "na", "nan", "-", "--"}


def _clean(s):
    s = (s or "").strip()
    return "" if s.lower() in _JUNK else s


def serialize(raw1, raw23, i1, i23, cl_size=None):
    n1, a1 = raw1["business_name"].to_list(), raw1["business_address"].to_list()
    n2, a2 = raw23["business_name"].to_list(), raw23["business_address"].to_list()
    A = [f"{n1[i]} ; {_clean(a1[i])}" for i in i1]
    if cl_size is None:
        Bt = [f"{n2[j]} ; {_clean(a2[j])}" for j in i23]
    else:
        Bt = [f"{n2[j]} ; {_clean(a2[j])} ; dup {int(c)}" for j, c in zip(i23, cl_size)]
    return A, Bt


class CrossEncoder:
    def __init__(self, name=None):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.name = name or CFG["ce_model"]
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.name, num_labels=1, dtype=torch.float32).to("cuda")
        self.max_len = CFG["ce_max_len"]

    def _batch(self, A, Bt):
        return self.tok(A, Bt, padding=True, truncation="longest_first", max_length=self.max_len,
                        return_tensors="pt").to("cuda")

    def fit(self, A, Bt, y, epochs=None, bs=None, lr=None, desc="CE train"):
        from tqdm.auto import tqdm
        epochs, bs, lr = epochs or CFG["ce_epochs"], bs or CFG["ce_bs"], lr or CFG["ce_lr"]
        n = len(y)
        steps = epochs * math.ceil(n / bs)
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        warm = max(100, steps // 20)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.02, 1 - s / steps))
        # bucket by length so batches are homogeneous (fast), then shuffle the buckets
        lens = np.array([len(a) + len(b) for a, b in zip(A, Bt)])
        rng = np.random.default_rng(0)
        self.model.train()
        step, run = 0, None
        bar = tqdm(total=steps, desc=desc, mininterval=15)
        yt = torch.tensor(y, dtype=torch.float32)
        for ep in range(epochs):
            perm = rng.permutation(n)
            chunks = [perm[i:i + bs * 100] for i in range(0, n, bs * 100)]
            batches = []
            for ch in chunks:
                ch = ch[np.argsort(lens[ch])]
                batches += [ch[i:i + bs] for i in range(0, len(ch), bs)]
            rng.shuffle(batches)
            for b in batches:
                enc = self._batch([A[i] for i in b], [Bt[i] for i in b])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit = self.model(**enc).logits.squeeze(-1).float()
                loss = F.binary_cross_entropy_with_logits(logit, yt[b].to("cuda"))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step(); sched.step()
                run = loss.item() if run is None else 0.99 * run + 0.01 * loss.item()
                step += 1
                bar.update(1)
                if step % 500 == 0:
                    bar.set_postfix(loss=f"{run:.4f}")
        bar.close()
        self.model.eval()
        return run

    @torch.no_grad()
    def predict(self, A, Bt, bs=512, desc="CE predict"):
        from tqdm.auto import tqdm
        self.model.eval()
        order = np.argsort([len(a) + len(b) for a, b in zip(A, Bt)])[::-1]
        out = np.empty(len(A), np.float32)
        for lo in tqdm(range(0, len(A), bs), desc=desc, mininterval=30):
            sel = order[lo:lo + bs]
            enc = self._batch([A[i] for i in sel], [Bt[i] for i in sel])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out[sel] = torch.sigmoid(self.model(**enc).logits.squeeze(-1).float()).cpu().numpy()
        return out

    def save(self, path):
        self.model.save_pretrained(path)
        self.tok.save_pretrained(path)


def ce_band_mask(p_student):
    lo, hi = CFG["ce_band"]
    return (p_student >= lo) & (p_student <= hi)


def ce_oof(A, Bt, y, groups, band, p_hard, tag="S"):
    """OOF cross-encoder on the pairs inside the uncertainty band. groups: CE fold id per pair (by S1).
    Returns oof score (NaN outside band) and the list of saved fold-model paths."""
    oof = np.full(len(y), np.nan, np.float32)
    paths = []
    K = int(groups.max()) + 1
    for k in range(K):
        path = MODELS / f"ce_fold{k}"
        tr = np.where((groups != k) & band)[0]
        va = np.where((groups == k) & band)[0]
        if (path / "config.json").exists():
            ce = CrossEncoder(str(path))
            log(f"CE fold {k}: loaded {path}")
        else:
            cap = CFG["ce_train_pairs"]
            if len(tr) > cap:                                # keep every positive, fill with the hardest negatives first
                pos = tr[y[tr] == 1]
                neg = tr[y[tr] == 0]
                hard = neg[np.argsort(-p_hard[neg])]
                n_hard = max(0, min(len(hard), cap - len(pos)))
                tr = np.sort(np.concatenate([pos[:cap], hard[:n_hard]]))
            ce = CrossEncoder()
            t = time.time()
            loss = ce.fit([A[i] for i in tr], [Bt[i] for i in tr], y[tr].astype(np.float32), desc=f"CE fold {k}")
            ce.save(path)
            log(f"CE fold {k}: trained on {len(tr):,} pairs (pos rate {y[tr].mean():.3f}) loss {loss:.4f} in {time.time() - t:.0f}s")
        oof[va] = ce.predict([A[i] for i in va], [Bt[i] for i in va], desc=f"CE OOF fold {k}")
        paths.append(path)
        del ce
        torch.cuda.empty_cache()
    return oof, paths


def ce_predict_bag(A, Bt, paths):
    s = np.zeros(len(A), np.float32)
    for p in paths:
        ce = CrossEncoder(str(p))
        ce.model.to(torch.bfloat16)                     # inference only: bf16 weights, ~2x faster
        s += ce.predict(A, Bt, desc=f"CE test {p.name}")
        del ce
        torch.cuda.empty_cache()
    return s / len(paths)
