"""Modal: graded amplitude control — steering a WAVEFORM, not a sign pattern.

Every result so far schedules a sign (±1 per segment). This asks the sharper
question: can the handle track a GRADED profile — five urgency levels — so that the
thing being controlled is an amplitude envelope rather than a binary alternation?

Design (realmodel agent's deesc_profile): the assistant's own urgency register, five
levels, twelve sentences each, banks split into disjoint A (fits the direction) and
B (builds eval pairs). No distressing content — only the speaker's register.

  target = descending ramp [5,4,3,2,1]   foil = ascending [1,2,3,4,5]
  target = [1,3,5,3,1] (peak)            foil = [5,3,1,3,5] (trough)

Both contrasts are exact multiset matches — the same five sentences in a different
order — so a constant (DC) write is inert by construction. The second contrast also
removes the global slope, leaving only the shape.

Coefficient s_t = (level_t − 3)/2 ∈ {−1, −0.5, 0, +0.5, +1}: a real waveform.

Gates and controls, all registered before the run:
  - MONOTONICITY GATE: project each level's mean segment activation onto u and
    require the five projections to be ordered. Without that, a graded claim is not
    supported and the honest fallback is the binary construction.
  - BIDIRECTIONALITY: run the sign-flipped schedule; a usable handle should drive the
    margin negative by a similar magnitude, not merely fail to help.
  - Arms: graded template, binary template (sign only, amplitude discarded — does
    amplitude buy anything over sign?), broadcast, random-direction graded.

    modal run experiments/temporal_screen/trajectory_steering/graded_modal.py
"""

import pathlib
import modal

_here = pathlib.Path(__file__).resolve()
ROOT = _here.parents[3] if len(_here.parents) > 3 else pathlib.Path("/work")

app = modal.App("trajectory-graded")
image = modal.Image.debian_slim().pip_install("torch", "transformers", "accelerate", "numpy")

LEVELS = {
    5: ["This needs to be dealt with right now.",
        "We have to move on this immediately.",
        "There is no time left to wait on it.",
        "This has to be settled before anything else.",
        "It cannot sit any longer than today.",
        "Everything else has to stop for this.",
        "This is the thing that has to happen first.",
        "We are out of room to delay it.",
        "It has to be handled at once.",
        "This one cannot wait another hour.",
        "We need to act on it straight away.",
        "There is no slack left here at all."],
    4: ["This should be sorted out fairly soon.",
        "It would be better not to leave it long.",
        "We ought to get to this shortly.",
        "This is worth moving up the list.",
        "It should not sit for too many days.",
        "We should keep this one near the front.",
        "This deserves attention before long.",
        "It is better handled sooner than later.",
        "We should put some time against it soon.",
        "This ought to move fairly quickly.",
        "It would help to close this out early.",
        "This should not drift much further."],
    3: ["This can be handled in the usual way.",
        "It sits alongside the other items.",
        "We can take this in the normal order.",
        "This fits into the regular schedule.",
        "It can go through the standard process.",
        "This is roughly like the rest of the list.",
        "We can treat it as ordinary work.",
        "It belongs with the usual batch.",
        "This can follow the normal timeline.",
        "It is a routine item among the others.",
        "We can slot it in where it fits.",
        "This proceeds at the usual pace."],
    2: ["This can wait until things quieten down.",
        "There is room to leave it a while.",
        "It can sit until the busier work clears.",
        "We can come back to this later on.",
        "It is fine to park this for now.",
        "This can wait for a calmer stretch.",
        "There is no pressure on this one.",
        "It can hold until later in the month.",
        "We can leave this resting for a bit.",
        "This one can drift without harm.",
        "It is comfortable to postpone.",
        "There is plenty of room on this."],
    1: ["There is no rush at all; we can take this slowly.",
        "This can sit for as long as it needs to.",
        "Nothing here needs any hurry whatsoever.",
        "We can leave this entirely to one side.",
        "It can rest indefinitely without issue.",
        "There is no clock on this one at all.",
        "This can take all the time it wants.",
        "We can be completely relaxed about it.",
        "It matters very little when this happens.",
        "This can wait as long as you like.",
        "There is no urgency attached to it.",
        "We can let this sit quite peacefully."],
}
CARRIERS = ["Planning note.\n", "From the handover:\n", "Status update.\n",
            "Weekly note.\n", "Team memo.\n", "End of day summary.\n"]


@app.function(gpu="A10G", image=image, timeout=3600)
def graded(model_id: str, layer: int, n_train: int, n_eval: int, fracs: list):
    import random
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[load] {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    dev = model.device
    layers_ = model.model.layers
    L = layer if layer >= 0 else len(layers_) // 2
    print(f"[cfg] L={L}")

    A = {lv: s[:6] for lv, s in LEVELS.items()}
    B = {lv: s[6:] for lv, s in LEVELS.items()}
    cap, steer = {}, {"v": []}

    def cap_hook(_m, _i, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    def steer_hook(_m, _i, out):
        if not steer["v"]:
            return out
        hs = out[0] if isinstance(out, tuple) else out
        for a, b, vec in steer["v"]:
            hs[:, a:b + 1, :] = hs[:, a:b + 1, :] + vec.to(hs.dtype)
        return (hs, *out[1:]) if isinstance(out, tuple) else hs

    layers_[L].register_forward_hook(steer_hook)
    rng = random.Random(555)
    unit = lambda x: x / (np.linalg.norm(x) + 1e-8)

    def build(carrier, sents):
        text, spans = carrier, []
        for j, s in enumerate(sents):
            if j:
                text += " "
            spans.append((len(text), len(text) + len(s)))
            text += s
        return text, spans

    def encode(text, cs):
        enc = tok(text, return_tensors="pt", return_offsets_mapping=True)
        offs = enc["offset_mapping"][0].tolist()
        ts = []
        for (a, b) in cs:
            ix = [i for i, (x, y) in enumerate(offs) if y > x and y > a and x < b]
            ts.append((min(ix), max(ix)))
        return enc["input_ids"].to(dev), ts

    def seg_logprob(ids, ts):
        with torch.no_grad():
            lp = model(ids).logits[0].log_softmax(-1).float()
        return float(sum(lp[p - 1, ids[0, p]]
                         for a, b in ts for p in range(a, b + 1) if p >= 1))

    def capture(text, cs):
        ids, ts = encode(text, cs)
        h = layers_[L].register_forward_hook(cap_hook)
        with torch.no_grad():
            model(ids)
        h.remove()
        hh = cap["h"][0].float().cpu().numpy()
        return ([hh[a:b + 1].mean(0) for a, b in ts],
                [float(np.linalg.norm(hh[p])) for a, b in ts
                 for p in range(a, b + 1)])

    def margin(pair, vecs, m):
        ids, ts = pair
        steer["v"] = []
        if m > 0:
            for i, v in enumerate(vecs):
                if v is None:
                    continue
                a, b = ts[i]
                steer["v"].append((max(a - 1, 0), max(b - 1, 0), m * v))
        val = seg_logprob(ids, ts)
        steer["v"] = []
        return val

    # ---- direction from bank A: high (4,5) vs low (1,2); level 3 excluded ----
    by_lv, norms_all = {lv: [] for lv in LEVELS}, []
    for _ in range(n_train):
        lvs = [rng.choice([1, 2, 3, 4, 5]) for _ in range(5)]
        sents = [A[lv][rng.randrange(6)] for lv in lvs]
        text, cs = build(rng.choice(CARRIERS), sents)
        segs, norms = capture(text, cs)
        for lv, sv in zip(lvs, segs):
            by_lv[lv].append(sv)
        norms_all += norms
    hi = np.mean(by_lv[4] + by_lv[5], 0)
    lo = np.mean(by_lv[1] + by_lv[2], 0)
    u_np = unit(hi - lo)
    u = torch.tensor(u_np, device=dev, dtype=torch.float32)
    bn = float(np.mean(norms_all))
    proj = {lv: float(np.mean(by_lv[lv], 0) @ u_np) for lv in sorted(LEVELS)}
    monotone = all(proj[i] < proj[i + 1] for i in range(1, 5))
    print(f"[dir] base_norm={bn:.1f}")
    print(f"[gate] level projections onto u: "
          + "  ".join(f"L{lv}={proj[lv]:+.2f}" for lv in sorted(proj))
          + f"   MONOTONE={monotone}")

    CONTRASTS = {"ramp_down_vs_up": ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
                 "peak_vs_trough": ([1, 3, 5, 3, 1], [5, 3, 1, 3, 5])}
    results = {"model": model_id, "layer": int(L), "base_norm": bn,
               "level_projections": proj, "monotone_gate": bool(monotone),
               "contrasts": {}}

    for cname, (ptar, pfoil) in CONTRASTS.items():
        pairs = []
        for _ in range(n_eval):
            idx = {lv: rng.randrange(6) for lv in LEVELS}
            car = rng.choice(CARRIERS)
            tT, cT = build(car, [B[lv][idx[lv]] for lv in ptar])
            tF, cF = build(car, [B[lv][idx[lv]] for lv in pfoil])
            pairs.append((encode(tT, cT), encode(tF, cF)))
        base = [margin(t, [], 0) - margin(f, [], 0) for t, f in pairs]
        s_graded = [(lv - 3) / 2 for lv in ptar]
        s_binary = [1.0 if lv > 3 else (-1.0 if lv < 3 else 0.0) for lv in ptar]
        g = torch.Generator(device="cpu").manual_seed(11)
        u_rand = torch.tensor(unit(torch.randn(u.shape[0], generator=g).numpy()),
                              device=dev, dtype=torch.float32)
        row = {"base_margin": float(np.mean(base))}
        ARMS = {
            "graded": [c * u for c in s_graded],
            "binary": [c * u for c in s_binary],
            "graded_flipped": [-c * u for c in s_graded],
            "broadcast": [u] * 5,
            "random_graded": [c * u_rand for c in s_graded],
        }
        for arm, vecs in ARMS.items():
            per = {}
            for fr in fracs:
                m = fr * bn
                ds = np.array([(margin(t, vecs, m) - margin(f, vecs, m)) - b
                               for (t, f), b in zip(pairs, base)])
                per[round(fr, 3)] = {"mean": float(ds.mean()),
                                     "sem": float(ds.std(ddof=1) / np.sqrt(len(ds)))}
            best = max(per, key=lambda x: abs(per[x]["mean"]))
            row[arm] = {"peak_frac": best, **per[best], "curve": per}
        results["contrasts"][cname] = row
        print(f"[{cname}] base={row['base_margin']:+.2f}  "
              + "  ".join(f"{a}={row[a]['mean']:+.2f}±{row[a]['sem']:.2f}"
                          for a in ARMS))
    return results


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen2.5-1.5B-Instruct", layer: int = -1,
         n_train: int = 40, n_eval: int = 32, fracs: str = "0.2,0.35,0.5"):
    import json
    res = graded.remote(model, layer, n_train, n_eval,
                        [float(x) for x in fracs.split(",")])
    outdir = ROOT / "results" / "temporal_screen"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "graded.json").write_text(json.dumps(res, indent=2))
    print("[saved]", outdir / "graded.json")
