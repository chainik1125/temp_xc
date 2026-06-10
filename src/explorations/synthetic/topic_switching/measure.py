"""Autoresearch #2 — topic-switching measurement (stages 2-3).

Per synthetic/topic_switching/prereg.md (frozen). Operationalises
topic as a per-sentence cluster id on fineweb-edu TEXT, then measures whether
the topic stream is temporal — and crucially whether the dwell is HEAVY-TAILED
beyond a first-order Markov chain (semi-Markov) or merely geometric (memoryless
slow state). That verdict sets the change-point generator's persistence knob.

Labeler (realized; deviations from the prereg's exact tools noted in the record):
  - sentences: regex splitter (spaCy unavailable in this env)
  - embeddings: sentence-transformers/all-MiniLM-L6-v2 via `transformers`
    (mean-pool + L2-normalise — identical to the sentence-transformers wrapper)
  - clustering: sklearn MiniBatchKMeans, K=20

Nulls: N1 within-doc permutation; N2 first-order Markov at the empirical
transition matrix (geometric dwell); N3 iid marginal.

    .venv/bin/python -m explorations.synthetic.topic_switching.measure

Deterministic (SEED=0). Writes synthetic/topic_switching/results/topic_switching_stats.json +
figs. Uses the GPU for embedding — run when the backtracking grid is idle.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SEED = 0
N_DOC = 2000
MIN_SENTS = 8
K_CLUST = 20
K_ALT = [12, 32]
MAXLAG = 12
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "results" / "topic_switching_stats.json"
FIG_DIR = HERE / "figs"

_SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')


def split_sents(text: str) -> list[str]:
    text = text.replace("\n", " ").strip()
    return [s.strip() for s in _SENT.split(text) if len(s.strip()) >= 3]


def load_docs(n_doc=N_DOC, min_sents=MIN_SENTS, seed=SEED):
    """Stream fineweb-edu; keep the first n_doc docs with >= min_sents sentences."""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    docs, n_seen = [], 0
    for r in ds:
        n_seen += 1
        sents = split_sents(r["text"])
        if len(sents) >= min_sents:
            docs.append(sents[:64])         # cap doc length for tractability
        if len(docs) >= n_doc:
            break
        if n_seen > 200_000:
            break
    return docs


def embed(sentences, device, batch=256):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(EMB_MODEL)
    mdl = AutoModel.from_pretrained(EMB_MODEL).to(device).eval()
    out = []
    for i in range(0, len(sentences), batch):
        b = sentences[i:i + batch]
        enc = tok(b, padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        with torch.no_grad():
            o = mdl(**enc)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        emb = (o.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        out.append(emb.cpu().numpy())
    return np.concatenate(out)


# ── statistics on a list of per-doc topic-id sequences ────────────────

def same_topic_acf(seqs, K, maxlag=MAXLAG):
    """C(k) = P(z_{i+k}=z_i) at each lag (raw, not chance-subtracted)."""
    out = []
    for k in range(1, maxlag + 1):
        num = den = 0
        for z in seqs:
            if len(z) > k:
                num += int(np.sum(z[:-k] == z[k:])); den += len(z) - k
        out.append(num / max(den, 1))
    return out


def chance_same(seqs):
    c = Counter();
    for z in seqs: c.update(z.tolist())
    tot = sum(c.values()); p = np.array([v / tot for v in c.values()])
    return float((p ** 2).sum())


def switch_rate(seqs):
    sw = tot = 0
    for z in seqs:
        sw += int(np.sum(z[1:] != z[:-1])); tot += len(z) - 1
    return sw / max(tot, 1)


def run_lengths(seqs):
    runs = []
    for z in seqs:
        if len(z) == 0: continue
        r = 1
        for i in range(1, len(z)):
            if z[i] == z[i - 1]: r += 1
            else: runs.append(r); r = 1
        runs.append(r)
    return np.array(runs)


def run_length_survival(runs, maxr=20):
    return [float(np.mean(runs > r)) for r in range(maxr + 1)]


def transition_matrix(seqs, K):
    P = np.ones((K, K))                      # Laplace smoothing
    for z in seqs:
        for a, b in zip(z[:-1], z[1:]):
            P[a, b] += 1
    return P / P.sum(1, keepdims=True)


def mi_vs_lag(seqs, K, maxlag=MAXLAG):
    out = []
    for k in range(1, maxlag + 1):
        joint = np.zeros((K, K));
        for z in seqs:
            if len(z) > k:
                for a, b in zip(z[:-k], z[k:]):
                    joint[a, b] += 1
        if joint.sum() == 0: out.append(0.0); continue
        joint /= joint.sum()
        pa = joint.sum(1, keepdims=True); pb = joint.sum(0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            mi = np.nansum(joint * np.log(joint / (pa @ pb) + 1e-12))
        out.append(float(max(mi, 0.0)))
    return out


def fano_switches(seqs, w=10):
    counts = []
    for z in seqs:
        s = (z[1:] != z[:-1]).astype(float)
        for i in range(0, len(s) - w + 1, w):
            counts.append(s[i:i + w].sum())
    counts = np.array(counts)
    return float(counts.var() / max(counts.mean(), 1e-9)) if len(counts) else float("nan")


def markov_order_test(seqs, K):
    """Conditional entropy H(z_i | order ctx) for order 0/1/2 (nats)."""
    def cond_entropy(order):
        ctx = defaultdict(Counter)
        for z in seqs:
            for i in range(order, len(z)):
                ctx[tuple(z[i - order:i])][z[i]] += 1
        H = tot = 0.0
        for c in ctx.values():
            n = sum(c.values()); tot += n
            p = np.array([v / n for v in c.values()])
            H += n * (-(p * np.log(p + 1e-12)).sum())
        return H / max(tot, 1)
    return {f"H_order{o}": float(cond_entropy(o)) for o in (0, 1, 2)}


# ── null models ───────────────────────────────────────────────────────

def null_permute(seqs, rng):                 # N1: destroy order, keep marginal+count
    return [z[rng.permutation(len(z))] for z in seqs]


def null_markov1(seqs, K, rng):              # N2: first-order Markov (geometric dwell)
    P = transition_matrix(seqs, K)
    pi = np.zeros(K)
    for z in seqs: pi[z[0]] += 1
    pi /= pi.sum()
    out = []
    for z in seqs:
        L = len(z); s = np.empty(L, dtype=np.int64)
        s[0] = rng.choice(K, p=pi)
        for i in range(1, L):
            s[i] = rng.choice(K, p=P[s[i - 1]])
        out.append(s)
    return out


def null_iid(seqs, K, rng):                  # N3: iid marginal
    c = Counter()
    for z in seqs: c.update(z.tolist())
    tot = sum(c.values()); p = np.array([c.get(i, 0) / tot for i in range(K)])
    return [rng.choice(K, size=len(z), p=p) for z in seqs]


def signature(seqs, K):
    runs = run_lengths(seqs)
    return {
        "switch_rate": switch_rate(seqs),
        "acf": same_topic_acf(seqs, K),
        "chance_same": chance_same(seqs),
        "mean_run": float(runs.mean()), "run_cv": float(runs.std() / max(runs.mean(), 1e-9)),
        "run_survival": run_length_survival(runs),
        "mi": mi_vs_lag(seqs, K),
        "fano_switches": fano_switches(seqs),
    }


def main():
    import torch
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[topic] loading docs (device={device})...", flush=True)
    docs = load_docs()
    flat, doc_bounds = [], []
    for d in docs:
        doc_bounds.append((len(flat), len(flat) + len(d))); flat.extend(d)
    print(f"[topic] {len(docs)} docs, {len(flat)} sentences. embedding...", flush=True)
    emb = embed(flat, device)

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    km = MiniBatchKMeans(n_clusters=K_CLUST, random_state=SEED, n_init=3, batch_size=2048)
    labels = km.fit_predict(emb)
    seqs = [np.asarray(labels[a:b]) for a, b in doc_bounds]

    # noise floor
    sil = float(silhouette_score(emb[:20000], labels[:20000])) if len(emb) > 50 else float("nan")
    km2 = MiniBatchKMeans(n_clusters=K_CLUST, random_state=SEED + 1, n_init=3, batch_size=2048)
    ari_reseed = float(adjusted_rand_score(labels, km2.fit_predict(emb)))
    ari_K = {}
    for Ka in K_ALT:
        kma = MiniBatchKMeans(n_clusters=Ka, random_state=SEED, n_init=3, batch_size=2048)
        ari_K[str(Ka)] = float(adjusted_rand_score(labels, kma.fit_predict(emb)))

    real = signature(seqs, K_CLUST)
    n1 = signature(null_permute(seqs, rng), K_CLUST)
    n2 = signature(null_markov1(seqs, K_CLUST, rng), K_CLUST)
    n3 = signature(null_iid(seqs, K_CLUST, rng), K_CLUST)
    markov = markov_order_test(seqs, K_CLUST)

    # position trend: switch rate by normalized position bin
    pos_bins = 10; ptrend = np.zeros(pos_bins); pcnt = np.zeros(pos_bins)
    for z in seqs:
        for i in range(1, len(z)):
            bdx = min(pos_bins - 1, int((i / len(z)) * pos_bins))
            ptrend[bdx] += int(z[i] != z[i - 1]); pcnt[bdx] += 1
    pos_switch = (ptrend / np.maximum(pcnt, 1)).tolist()

    # Verdict — the GENUINE temporal/order signal is real vs N1 (the within-doc
    # permutation that preserves per-doc topic COMPOSITION but destroys order).
    # Comparing to N2 (first-order Markov) is confounded: N2 draws from the
    # global stationary dist, so it does NOT preserve per-doc composition, and a
    # doc concentrated on a few topics shows high long-lag same-topic ACF that
    # is composition, not temporal structure. So: order = real - N1.
    ra = np.array(real["acf"]); n1a = np.array(n1["acf"]); n3a = np.array(n3["acf"])
    order_lag1 = float(ra[0] - n1a[0])                  # adjacency order beyond composition
    order_tail = float(np.mean((ra - n1a)[3:7]))        # long-range order beyond composition
    composition_frac = float(n1a[0] / max(ra[0], 1e-9))  # share of ACF(1) that is composition
    sticky_vs_n3 = float(ra[0] - n3a[0])                # any stickiness at all (vs iid)
    surv_real, surv_n2 = np.array(real["run_survival"]), np.array(n2["run_survival"])
    run_tail_excess = float(np.mean((surv_real - surv_n2)[3:8]))  # heavy dwell beyond Markov-1
    labeler_ok = sil >= 0.05
    if order_lag1 < 0.10 and order_tail < 0.03:
        verdict = "ABORT_composition_dominated" + ("" if labeler_ok else "_labeler_inadequate")
    elif order_tail > 0.05:
        verdict = "TEMPORAL_long_memory_order"
    else:
        verdict = "TEMPORAL_weak_short_range_order" + ("" if labeler_ok else "_labeler_caveat")

    stats = {
        "seed": SEED, "n_docs": len(docs), "n_sentences": len(flat), "K_clust": K_CLUST,
        "noise_floor": {"silhouette": sil, "ari_reseed": ari_reseed, "ari_K": ari_K},
        "real": real, "null_permute": n1, "null_markov1": n2, "null_iid": n3,
        "markov_order": markov, "pos_switch_rate": pos_switch,
        "verdict": verdict,
        "discriminators": {"order_lag1_vs_N1": order_lag1, "order_tail_k4_7_vs_N1": order_tail,
                           "composition_frac_of_acf1": composition_frac,
                           "run_tail_excess_vs_N2": run_tail_excess, "sticky_vs_N3": sticky_vs_n3},
    }
    OUT_JSON.write_text(json.dumps(stats, indent=2))
    _plot(stats)
    _print(stats)
    return stats


def _print(s):
    print("\n========= TOPIC-SWITCHING MEASUREMENT =========")
    print(f"docs={s['n_docs']} sentences={s['n_sentences']} K={s['K_clust']}")
    nf = s["noise_floor"]
    print(f"noise floor: silhouette={nf['silhouette']:.3f} ari_reseed={nf['ari_reseed']:.3f} ari_K={nf['ari_K']}")
    print(f"switch_rate real={s['real']['switch_rate']:.3f} | mean_run real={s['real']['mean_run']:.2f} "
          f"(N2 {s['null_markov1']['mean_run']:.2f}, N3 {s['null_iid']['mean_run']:.2f})")
    print(f"ACF(1): real={s['real']['acf'][0]:.3f}  N1={s['null_permute']['acf'][0]:.3f}  "
          f"N2={s['null_markov1']['acf'][0]:.3f}  N3={s['null_iid']['acf'][0]:.3f} (chance {s['real']['chance_same']:.3f})")
    d = s["discriminators"]
    print(f"GENUINE ORDER (real-N1): lag1={d['order_lag1_vs_N1']:+.3f}  tail(4-7)={d['order_tail_k4_7_vs_N1']:+.3f}"
          f"  | composition share of ACF(1)={d['composition_frac_of_acf1']:.2f}")
    print(f"dwell heavy-tail vs N2 (run-tail excess)={d['run_tail_excess_vs_N2']:+.3f}; stickiness vs N3={d['sticky_vs_N3']:+.3f}")
    print(f"VERDICT: {s['verdict']}")
    print(f"-> {OUT_JSON}")


def _plot(s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    lags = np.arange(1, len(s["real"]["acf"]) + 1)
    for key, lab, c, ls in [("real", "real", "#1f77b4", "-"), ("null_markov1", "N2 Markov-1", "#d62728", "--"),
                            ("null_permute", "N1 permute", "#7f7f7f", ":"), ("null_iid", "N3 iid", "#2ca02c", "-.")]:
        ax[0].plot(lags, s[key]["acf"], ls, color=c, label=lab, lw=1.8)
    ax[0].axhline(s["real"]["chance_same"], color="k", lw=0.6, alpha=0.4)
    ax[0].set_xlabel("lag (sentences)"); ax[0].set_ylabel("P(same topic)"); ax[0].set_title("Same-topic autocorrelation")
    ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.25)
    r = np.arange(len(s["real"]["run_survival"]))
    for key, lab, c, ls in [("real", "real", "#1f77b4", "-"), ("null_markov1", "N2 Markov-1", "#d62728", "--")]:
        ax[1].semilogy(r, np.array(s[key]["run_survival"]) + 1e-4, ls, color=c, label=lab, lw=1.8)
    ax[1].set_xlabel("run length r"); ax[1].set_ylabel("P(run > r)"); ax[1].set_title("Dwell-time survival (heavy tail?)")
    ax[1].legend(fontsize=8); ax[1].grid(True, alpha=0.25)
    ax[2].plot(np.linspace(0, 1, len(s["pos_switch_rate"])), s["pos_switch_rate"], "o-", color="#9467bd")
    ax[2].set_xlabel("normalized position in doc"); ax[2].set_ylabel("switch rate"); ax[2].set_title("Switch rate vs position")
    ax[2].grid(True, alpha=0.25)
    fig.suptitle(f"Topic-switching measurement — verdict: {s['verdict']}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext, dpi in [("pdf", None), ("png", 120), ("thumb.png", 55)]:
        fig.savefig(FIG_DIR / f"topic_switching_signature.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
