"""Stage 3 — calibrate one selected candidate (the `measure.py` template, run
by the factory harness).

Pipeline per candidate (all gates preregistered in its frozen card):

1. **Bulk-label** every document of its domain with the Haiku judge using the
   card's frozen `judge_instruction` (chunked, order-preserving; a doc with a
   dead chunk is dropped whole, coverage reported).
2. **Validate the labeler**: Sonnet relabels a held-out doc sample →
   agreement, Cohen's κ, noise floor ε̂; PLUS the card's independent heuristic
   cross-check. κ < 0.30 ⇒ ABORT (labeler inadequate — the topic_switching
   precedent).
3. **Measure the signature** (kind per card) + the N1/N2/N3 null battery +
   bootstrap CIs + a split-half stability check.
4. **Gate**: the card's PRIMARY ordered statistic must exceed BOTH the N1 and
   N2 97.5% null bands, and still do so after ε̂-noise perturbation of the
   labels. Otherwise ABORT.
5. **PROCEED only**: fit the card's Appendix-B mirror on a 70% doc split,
   validate on the held-out 30%; then the adversarial SKEPTIC pass (Opus,
   fixed 5-item kill-rubric) — any kill ⇒ verdict demoted to ABORT.
6. Write `records/<name>/` : labels, stats JSON, figure, and the
   `calibration.md` record (written for ABORTs too — an abort is a success).

    .venv/bin/python -m experiments.explorations.synthetic.expansion.calibrate <name>
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

from explorations.synthetic.expansion import labeler as lab
from explorations.synthetic.expansion import mirrors
from explorations.synthetic.expansion import signature as sig
from explorations.synthetic.expansion.client import Judge, Meter
from explorations.synthetic.expansion.corpus import load_reasoning_traces

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SEED = 0
KAPPA_FLOOR = 0.30          # labeler adequacy (below ⇒ ABORT, topic_switching-style)
N_NULL, N_BOOT = 200, 300
INTERJUDGE_DOCS = 12

# ── the card's independent heuristics (implemented verbatim from prereg/) ──

_HEDGE = ("maybe", "perhaps", "might", "i think", "not sure", "let me try",
          "possibly", "seems")
_COMMIT = ("therefore", "thus", "clearly", "definitely", "must be",
           "the answer is", "hence", "obviously")


def _heur_hedging(sents):
    out = []
    for s in sents:
        t = s.lower()
        score = sum(w in t for w in _COMMIT) - sum(w in t for w in _HEDGE)
        out.append(2 if score > 0 else (0 if score < 0 else 1))
    return np.array(out, dtype=np.int8)


_ASSUME = re.compile(r"\b(suppose|assume|let|if|consider the case|in case)\b", re.I)
_CONSEQ = re.compile(r"\b(then|therefore|so|thus|hence|it follows|this implies|which means)\b", re.I)


def _heur_assumption(sents):
    return np.array([1 if _ASSUME.search(s) else (2 if _CONSEQ.search(s) else 0)
                     for s in sents], dtype=np.int8)


_QSTART = ("what", "why", "how", "when", "who", "is", "are", "do", "does", "can")
_ASTART = ("yes", "no", "the answer", "this is", "because", "it is")


def _heur_qa(sents):
    out = []
    prev_q = False
    for s in sents:
        t = s.strip().lower()
        if s.strip().endswith("?") or any(t.startswith(w + " ") for w in _QSTART):
            out.append(1)
            prev_q = True
        elif prev_q and any(t.startswith(w) for w in _ASTART):
            out.append(2)
            prev_q = False
        else:
            out.append(0)
            prev_q = False
    return np.array(out, dtype=np.int8)


_QUOTE_RE = re.compile(r"(said|says|stated|according to|told|remarked|noted that|claimed)", re.I)


def _heur_quote(sents):
    def has_pair(s):
        return s.count('"') >= 2 or ("“" in s and "”" in s)
    return np.array([1 if (has_pair(s) or _QUOTE_RE.search(s)) else 0
                     for s in sents], dtype=np.int8)


# ── per-candidate calibration config (statistic kinds + primary gate stat) ──

CFG = {
    "uncertainty-hedging-drift": dict(
        domain="reasoning-trace", kind="scalar", pair=None,
        primary=("acf", 0), heuristic=_heur_hedging,
        mirror="ar1", mirror_kw={"position": True}, mirror_kind="scalar"),
    "assumption-then-consequence": dict(
        domain="reasoning-trace", kind="categorical", pair=(1, 2),
        primary=("asym", None), heuristic=_heur_assumption,
        mirror="markov", mirror_kw={}, mirror_kind="categorical"),
    "question-answer-adjacency": dict(
        domain="text-corpus", kind="categorical", pair=(1, 2),
        primary=("asym", None), heuristic=_heur_qa,
        mirror="markov", mirror_kw={}, mirror_kind="categorical"),
    "quotation-burst": dict(
        domain="text-corpus", kind="binary", pair=None,
        primary=("acf", 0), heuristic=_heur_quote,
        mirror="logistic_ar", mirror_kw={"K": 8}, mirror_kind="binary"),
}


def load_domain(domain: str):
    if domain == "reasoning-trace":
        return load_reasoning_traces(REPO / "results/c7_backtracking/stage_a")
    return json.loads((HERE / "data/fineweb_sample.json").read_text())


def primary_value(h: dict, primary) -> float:
    key, idx = primary
    v = np.asarray(h[key], dtype=float)
    return float(v.ravel()[idx]) if idx is not None else float(v)


def band_value(band: dict, primary, which: str) -> float:
    key, idx = primary
    v = np.asarray(band[key][which], dtype=float)
    return float(v.ravel()[idx]) if idx is not None else float(v)


# ── skeptic pass (Opus, fixed kill-rubric — briefing guardrail #5) ──────────

SKEPTIC_SYSTEM = """\
You are the adversarial SKEPTIC in a measure->mirror benchmark loop. A candidate
temporal property has provisionally PASSED its null-battery gate and its mirror was
fit. Your job is to KILL it if it does not deserve to be frozen as a benchmark spec.
You will see the frozen prereg card, the labeler-validation numbers, the measured
statistics vs the nulls, and the mirror validation.

Fill this fixed kill-rubric. For each item answer kill=true only with concrete
evidence from the numbers given (kill=false needs a one-line justification too):
 a. noise_floor: is the ordered-vs-shuffled gap within the labeler noise floor
    (i.e. could label noise alone produce it)?
 b. leakage: could the labeler be leaking the target (the label definition itself
    builds in the temporal statistic, e.g. an 'answer' label that requires a prior
    question makes question->answer ordering circular)?
 c. composition: is the effect per-document composition / marginal, not within-
    document order (the topic_switching trap — check the N1 comparison)?
 d. circularity: does the mirror match the statistic by construction in a way that
    makes the validation vacuous (validating on the same quantity that was fit is
    expected — kill only if the SPEC would test nothing beyond what was inserted)?
 e. segmentation: is the effect plausibly an artifact of sentence segmentation or
    windowing choices (e.g. the splitter creating alternation by construction)?

Respond with ONLY a JSON object, no prose, no fence:
{"a_noise_floor": {"kill": bool, "evidence": "..."},
 "b_leakage": {"kill": bool, "evidence": "..."},
 "c_composition": {"kill": bool, "evidence": "..."},
 "d_circularity": {"kill": bool, "evidence": "..."},
 "e_segmentation": {"kill": bool, "evidence": "..."},
 "overall_note": "..."}"""


def skeptic_pass(judge: Judge, name: str, card_md: str, summary: dict) -> dict:
    user = (f"## Frozen prereg card\n\n{card_md}\n\n## Calibration numbers\n\n"
            + json.dumps(summary, indent=1, default=float)
            + "\n\nFill the kill-rubric. JSON only.")
    text = judge.call("think", SKEPTIC_SYSTEM, user, max_tokens=2500,
                      tag=f"skeptic:{name}")
    return json.loads(re.search(r"\{.*\}", text, re.S).group(0))


# ── main per-candidate pipeline ────────────────────────────────────────────

def run(name: str):
    cfg = CFG[name]
    card = json.loads((HERE / "results/candidates.json").read_text())
    cand = next(c for c in card["candidates"] if c["name"] == name)
    spec = {"name": name, "kind": cand["label_kind"], "n_values": cand["n_values"],
            "judge_instruction": cand["judge_instruction"]}
    out_dir = HERE / "records" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    meter = Meter()
    judge = Judge(meter)
    rng = np.random.default_rng(SEED)

    data = load_domain(cfg["domain"])
    docs = [d["sentences"] for d in data["docs"]]
    doc_ids = [d["id"] for d in data["docs"]]
    print(f"[{name}] domain={cfg['domain']}  docs={len(docs)}  "
          f"sents={sum(map(len, docs))}  spend=${meter.spent:.2f}")

    # 1 ── bulk label (cache to disk so a crashed run never re-spends)
    labels_path = out_dir / "labels.json"
    if labels_path.exists():
        blob = json.loads(labels_path.read_text())
        seqs = [np.array(x, dtype=np.int8) if x is not None else None
                for x in blob["labels"]]
        coverage = blob["coverage"]
        print(f"[{name}] labels loaded from cache ({coverage['doc_coverage']:.3f} coverage)")
    else:
        seqs, coverage = lab.label_stream(judge, docs, spec, role="bulk",
                                          chunk=50, ctx=3, workers=8, tag=f"bulk:{name}")
        labels_path.write_text(json.dumps(
            {"doc_ids": doc_ids, "coverage": coverage,
             "labels": [s.tolist() if s is not None else None for s in seqs]}))
        print(f"[{name}] labeled: {coverage}  spend=${meter.spent:.2f}")

    # 2 ── labeler validation: inter-judge + heuristic cross-check
    val_path = out_dir / "labeler_validation.json"
    if val_path.exists():
        val = json.loads(val_path.read_text())
    else:
        inter = lab.validate_interjudge(judge, docs, seqs, spec,
                                        sample_docs=INTERJUDGE_DOCS, seed=SEED,
                                        tag=f"interjudge:{name}")
        heur = [cfg["heuristic"](d) for d in docs]
        xc = (lab.crosscheck_binary(seqs, heur) if cand["label_kind"] == "binary"
              else lab.crosscheck_categorical(seqs, heur, cand["n_values"]))
        d = inter["disagreement"]
        eps = (lab.noise_floor_from_disagreement(d) if cand["label_kind"] == "binary"
               else lab.noise_floor_categorical(d))
        val = {"interjudge": inter, "heuristic_crosscheck": xc, "noise_floor_eps": eps}
        val_path.write_text(json.dumps(val, indent=2, default=float))
    eps = val["noise_floor_eps"]
    kappa = val["interjudge"]["kappa"]
    print(f"[{name}] interjudge κ={kappa:.3f} agree={val['interjudge']['agreement']:.3f} "
          f"ε̂={eps:.3f}  spend=${meter.spent:.2f}")

    ok = [s for s in seqs if s is not None]
    if cfg["kind"] == "scalar":
        ok = [s.astype(float) for s in ok]

    # 3 ── signature + nulls + stability
    stats = sig.measure(ok, cfg["kind"], seed=SEED, n_null=N_NULL, n_boot=N_BOOT,
                        pair=cfg["pair"],
                        noise_eps=(eps,) if cfg["kind"] == "binary" and eps > 0 else ())
    half = rng.permutation(len(ok))
    h1 = [ok[i] for i in half[: len(ok) // 2]]
    h2 = [ok[i] for i in half[len(ok) // 2:]]
    hkw = dict(pair=cfg["pair"])
    stability = {"half1": primary_value(sig.headline(h1, cfg["kind"], **hkw), cfg["primary"]),
                 "half2": primary_value(sig.headline(h2, cfg["kind"], **hkw), cfg["primary"])}

    # 4 ── noise-perturbed primary (effect must survive the noise floor)
    if cfg["kind"] == "binary":
        pert = sig.flip_labels(ok, eps, rng)
    else:
        pert = sig.perturb_categorical([s.astype(np.int8) for s in ok] if cfg["kind"] == "scalar" else ok,
                                       eps, rng)
        if cfg["kind"] == "scalar":
            pert = [p.astype(float) for p in pert]
    pert_h = sig.headline(pert, cfg["kind"], **hkw)

    real_p = primary_value(stats["real"], cfg["primary"])
    n1_hi = band_value(stats["nulls"]["N1_permute"], cfg["primary"], "hi")
    n2_hi = band_value(stats["nulls"]["N2_trend"], cfg["primary"], "hi")
    pert_p = primary_value(pert_h, cfg["primary"])

    gate = {
        "primary_stat": f"{cfg['primary'][0]}"
                        + (f"[lag{cfg['primary'][1] + 1}]" if cfg["primary"][1] is not None else ""),
        "real": real_p, "noise_perturbed": pert_p,
        "N1_hi": n1_hi, "N2_hi": n2_hi,
        "labeler_kappa": kappa, "kappa_floor": KAPPA_FLOOR,
        "noise_floor_eps": eps, "stability": stability,
        "clears_sampling": bool(real_p > n1_hi and real_p > n2_hi),
        "clears_noise": bool(pert_p > n1_hi and pert_p > n2_hi),
        "labeler_ok": bool(kappa >= KAPPA_FLOOR),
    }
    verdict = "PROCEED" if all(
        (gate["clears_sampling"], gate["clears_noise"], gate["labeler_ok"])) else "ABORT"
    gate["verdict_pre_skeptic"] = verdict
    print(f"[{name}] GATE: real={real_p:.4f} pert={pert_p:.4f} "
          f"N1hi={n1_hi:.4f} N2hi={n2_hi:.4f} κ={kappa:.2f} -> {verdict}")

    # 5 ── PROCEED only: mirror fit + held-out validation + skeptic
    # The skeptic verdict is CACHED across reruns (like the labels): a rerun —
    # e.g. after an infrastructure fix to the validation sampling — must never
    # re-roll the adversarial pass hoping for a different answer.
    prev_path = out_dir / "calibration_stats.json"
    prev_skeptic = None
    if prev_path.exists():
        prev_skeptic = json.loads(prev_path.read_text()).get("skeptic")
    mirror_blob = skeptic = None
    if verdict == "PROCEED":
        idx = rng.permutation(len(ok))
        cut = int(0.7 * len(ok))
        train = [ok[i] for i in idx[:cut]]
        ev = [ok[i] for i in idx[cut:]]
        fit_fn, gen_fn = mirrors.MENU[cfg["mirror"]]
        m_train = train if cfg["mirror_kind"] != "categorical" else [s.astype(np.int8) for s in train]
        params = fit_fn(m_train, **cfg["mirror_kw"])
        syn = gen_fn(params, [s.size for s in ev], rng)
        if cfg["kind"] == "scalar":
            syn = [np.asarray(s, dtype=float) for s in syn]
        mv = mirrors.validate_mirror(ev, syn, cfg["kind"], maxlag=12)
        if cfg["pair"] is not None:
            mv["real_directed"] = sig.directed_transition(ev, *cfg["pair"])
            mv["syn_directed"] = sig.directed_transition(
                [np.asarray(s, dtype=np.int8) for s in syn], *cfg["pair"])
        mirror_blob = {"params": params, "n_train": len(train), "n_eval": len(ev),
                       "validation": mv}
        print(f"[{name}] mirror fit+validated (n_train={len(train)})")

        if prev_skeptic is not None:
            skeptic = dict(prev_skeptic, reused_from_prior_run=True)
            print(f"[{name}] skeptic verdict reused from prior run (never re-rolled)")
        else:
            card_md = (HERE / "prereg" / f"{name}.md").read_text()
            summary = {"gate": gate, "labeler_validation": val,
                       "null_bands_primary": {
                           k: {w: band_value(stats["nulls"][k], cfg["primary"], w)
                               for w in ("mean", "lo", "hi")} for k in stats["nulls"]},
                       "mirror": mirror_blob, "coverage": coverage}
            skeptic = skeptic_pass(judge, name, card_md, summary)
        kills = [k for k, v in skeptic.items()
                 if isinstance(v, dict) and v.get("kill")]
        if kills:
            verdict = "ABORT"
            gate["killed_by_skeptic"] = kills
            print(f"[{name}] SKEPTIC KILLED: {kills}")
        else:
            print(f"[{name}] skeptic pass: survived all 5 items")

    gate["verdict"] = verdict

    blob = {"name": name, "config": {k: str(v) if callable(v) else v
                                     for k, v in cfg.items()},
            "coverage": coverage, "labeler_validation": val, "signature": stats,
            "noise_perturbed_headline": {k: (np.asarray(v).tolist() if np.ndim(v) else float(v))
                                         for k, v in pert_h.items()},
            "gate": gate, "mirror": mirror_blob, "skeptic": skeptic,
            "spend_usd_after": meter.spent}
    (out_dir / "calibration_stats.json").write_text(
        json.dumps(blob, indent=2, default=float))
    _figure(name, cfg, stats, gate, out_dir)
    print(f"[{name}] VERDICT: {verdict}   spend=${meter.spent:.2f} of ${meter.cap:.0f}")
    return blob


def _figure(name, cfg, stats, gate, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lags = np.arange(1, len(stats["real"]["acf"]) + 1)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(lags, stats["real"]["acf"], "o-", color="#1f77b4", label="real", lw=2)
    for nm, col in [("N1_permute", "#999"), ("N2_trend", "#d62728"), ("N3_iid", "#2ca02c")]:
        m = np.array(stats["nulls"][nm]["acf"]["mean"])
        lo = np.array(stats["nulls"][nm]["acf"]["lo"])
        hi = np.array(stats["nulls"][nm]["acf"]["hi"])
        ax[0].plot(lags, m, "--", color=col, label=nm, lw=1)
        ax[0].fill_between(lags, lo, hi, color=col, alpha=0.15)
    ax[0].axhline(0, color="k", lw=0.6, alpha=0.4)
    ax[0].set_xlabel("lag (sentences)")
    ax[0].set_ylabel("autocorrelation")
    ax[0].set_title("ACF: real vs nulls")
    ax[0].legend(fontsize=8)

    pr = stats["position_profile"]
    ax[1].plot(np.linspace(0, 1, len(pr)), pr, "o-", color="#1f77b4")
    ax[1].set_xlabel("normalized position")
    ax[1].set_ylabel("mean label")
    ax[1].set_title("Position profile")

    g = gate
    bars = ["real", "noise_perturbed", "N1_hi", "N2_hi"]
    ax[2].bar(bars, [g[b] for b in bars],
              color=["#1f77b4", "#9467bd", "#999", "#d62728"])
    ax[2].set_title(f"Gate on {g['primary_stat']} → {g['verdict']}")
    ax[2].tick_params(axis="x", rotation=20)
    for a in ax:
        a.grid(True, alpha=0.25)
    fig.suptitle(f"{name} — temporal signature (n={stats['n_seqs']} docs, "
                 f"{stats['n_spans']} sentences)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext, dpi in [("pdf", None), ("png", 120), ("thumb.png", 55)]:
        fig.savefig(out_dir / f"signature.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run(sys.argv[1])
