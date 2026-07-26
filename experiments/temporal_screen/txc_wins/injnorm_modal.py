"""Was "matched injected norm" actually matched? Measure it rather than argue it. CPU only.

Every write is normalised to unit Frobenius norm over the (T, d) slab and then applied as
`h[:, a:b+1, :] += alpha * scale * W[t]` -- the same vector added to EVERY token in segment
t's span. So the norm actually injected into the residual stream is

    alpha * scale * sqrt( sum_t  len_t * ||W[t]||^2 )

with `len_t` the segment's token count. Matching ||W||_F matches the SLAB norm, and the two
coincide only when all segments have the same token length. They do not: sentences vary from
about four to nine words. An arm whose profile happens to weight the long segments therefore
injects more at identical Frobenius norm.

Whether that biases anything depends on whether segment LENGTH correlates with segment INDEX.
In these corpora sentences are drawn into slots at random, so it should not, and the effect
should be variance rather than bias -- but that is an argument, and the whole comparison
rests on the norms being matched, so it should be a number. This reproduces each run's test
documents from its stored seed, tokenises them, and reports the realised injected norm per
arm from the `write_profile` already saved in the results JSON. No GPU and no model weights:
the tokenizer and the stored per-position norms are sufficient.

It matters prospectively more than retrospectively. Any design that SELECTS its conditions on
a score -- best-versus-worst permutation search, say -- can select on where the long segments
sit, at which point length becomes correlated with condition and the mismatch is systematic
rather than random.

    modal run experiments/temporal_screen/txc_wins/injnorm_modal.py --runs recency_v2,evidence_v2
"""
import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("txcwins-injnorm")
image = (
    modal.Image.debian_slim()
    .pip_install("transformers", "numpy")
    .add_local_dir(str(_here.parent), "/work/txc_wins")
)


@app.function(image=image, timeout=3600)
def injnorm(specs: list, model_id: str):
    import sys
    sys.path.insert(0, "/work")
    import random
    import numpy as np
    from transformers import AutoTokenizer
    from txc_wins.tasks import TASKS

    tok = AutoTokenizer.from_pretrained(model_id)
    out = {}

    for spec in specs:
        name, task, k_seg, n_train, n_test, seed, profiles = spec
        make_pair = TASKS[task](k_seg)
        rng = random.Random(seed)

        def draw():
            p = make_pair(rng)
            return p if len(p) == 5 else (p[0], p[1], p[2], None, None)

        # Reproduce the run's rng stream exactly: n_train training draws (each of which
        # also consumed one randint for the class label), then the test draws.
        for _ in range(n_train):
            draw()
            rng.randint(0, 1)

        lens = []
        for _ in range(n_test):
            sa, sb, car, _, _ = draw()
            for sents in (sa, sb):
                text, spans = car, []
                for j, s in enumerate(sents):
                    if j:
                        text += " "
                    spans.append((len(text), len(text) + len(s)))
                    text += s
                e = tok(text, return_offsets_mapping=True)
                offs = e["offset_mapping"]
                row = []
                for (a, b) in spans:
                    idx = [i for i, (s0, s1) in enumerate(offs)
                           if s0 >= a and s1 <= b and s1 > s0]
                    row.append(len(idx))
                lens.append(row)
        Lm = np.array(lens, dtype=float)                       # (2*n_test, T)

        rows = {}
        for arm, prof in profiles.items():
            w2 = np.array(prof, dtype=float) ** 2              # ||W[t]||^2, ||W||_F = 1
            inj = np.sqrt(Lm @ w2)                             # per document
            rows[arm] = {"mean": float(inj.mean()), "sd": float(inj.std()),
                         "min": float(inj.min()), "max": float(inj.max())}
        base = rows.get("random_broadcast") or next(iter(rows.values()))
        out[name] = {"task": task, "n_docs": int(Lm.shape[0]),
                     "seg_len_mean": float(Lm.mean()), "seg_len_sd": float(Lm.std()),
                     # Does segment length depend on segment INDEX? If it does, the
                     # mismatch is systematic; if not, it is variance.
                     "len_by_position": [float(v) for v in Lm.mean(0)],
                     "len_position_sd": float(Lm.mean(0).std()),
                     "arms": rows}
        print(f"\n=== {name} ({task}) — {Lm.shape[0]} documents, "
              f"segment length {Lm.mean():.2f} +- {Lm.std():.2f} tokens", flush=True)
        print(f"  mean length by position: "
              f"{' '.join(f'{v:.1f}' for v in Lm.mean(0))}  "
              f"(sd across positions {Lm.mean(0).std():.3f})", flush=True)
        for arm, r in sorted(rows.items(), key=lambda kv: -kv[1]["mean"]):
            print(f"  {arm:<20} injected {r['mean']:.3f} +- {r['sd']:.3f}   "
                  f"ratio to broadcast {r['mean'] / base['mean']:.4f}", flush=True)
    return out


@app.local_entrypoint()
def main(runs: str = "recency_v2,evidence_v2",
         model: str = "Qwen/Qwen2.5-1.5B-Instruct", tag: str = ""):
    import json
    specs = []
    for name in runs.split(","):
        name = name.strip()
        p = ROOT / "results" / "txc_wins" / f"{name}.json"
        if not p.exists():
            print(f"[skip] {p} missing")
            continue
        r = json.loads(p.read_text())
        if not r.get("write_profile"):
            print(f"[skip] {name} has no write_profile")
            continue
        specs.append([name, r["task"], r["k_seg"], r["n_train"], r["n_test"],
                      r.get("seed", 31415), r["write_profile"]])
    if not specs:
        print("[none] nothing to measure")
        return
    res = injnorm.remote(specs, model)
    outdir = ROOT / "results" / "txc_wins"
    (outdir / f"injected_norm{tag}.json").write_text(json.dumps(res, indent=2))
    print("[saved]", outdir / f"injected_norm{tag}.json")
