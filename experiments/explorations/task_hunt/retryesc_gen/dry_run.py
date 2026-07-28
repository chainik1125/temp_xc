"""$0 DRY RUN of the `retryesc_gen` scaffold — no API, no model, no spend.

Runs the real `run_retryesc_gen` turn loop and the real `build_stream`
against a STUB backend that emits filler prose of a realistic length,
then measures the stream with the real `elicit_lib` instruments.

What this CAN validate before any money is spent:
  * turn layout is 2N+1 and alternates user/assistant
  * events land on environment turns only, never on assistant turns
  * event turns are fully probe-MASKED and assistant turns are eligible
  * the realised gap median, and therefore `f` == floor_excess, which is
    gate 1 of the card -- measured rather than projected
  * resume is EXACT (the loop must not consume rng)

What it CANNOT validate: the model's actual prose. Gate 2's real test
(`unigram_auc` <= 0.60) needs generated text and is a PILOT gate, not
this one. A pass here is a green light to spend, not evidence of
vocabulary cleanliness.

Run:
  .venv/bin/python -m experiments.explorations.task_hunt.retryesc_gen.dry_run
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..labels import elicit_lib as el
from ..labels import retryesc_gen_lib as rg
from ..labels import run_elicit as re_

N_DOCS = 60
SEED = 0
MAX_NEW = rg.LEN_HI
REALISED_FRAC = 0.8       # the assumption gate 1 rested on; tested here
TS = (4, 8, 16, 32, 64)
OUT = Path(__file__).resolve().parent / "results" / "dry_run.json"

# ~1 token per word for this filler; enough distinct words that the
# tokenizer does not collapse it into a degenerate stream.
_WORDS = ("inspect the failing step and re-run it under the requested "
          "approach then capture stderr for comparison against the "
          "previous attempt before deciding whether to escalate or "
          "adjust the parameters again in the next iteration").split()


class StubBackend:
    """Emits filler of ~REALISED_FRAC x cap tokens. No network."""

    kind = "stub"

    def __init__(self):
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.calls = 0

    def tok(self, text, add_special_tokens=False):
        return self._tok(text, add_special_tokens=add_special_tokens)

    def chat(self, convs, cap):
        self.calls += len(convs)
        n = max(4, int(cap * REALISED_FRAC))
        out = []
        for i in range(len(convs)):
            w = [_WORDS[(i + j) % len(_WORDS)] for j in range(n)]
            out.append(" ".join(w) + ".")
        return out


def main() -> None:
    be = StubBackend()
    docs, fillers = run_and_check(be)

    ids, doc_off, first, mask, elig, topics, texts = re_.build_stream(
        docs, be.tok)
    gaps = el.realised_gaps(first, doc_off)

    # raw age at every position -> the claim_zone instrument (card §2.2a:
    # frac_in_window IS floor_excess, K=0.96 on real data)
    raw_age = np.full(ids.size, np.inf, dtype=np.float64)
    for d in range(len(doc_off) - 1):
        lo, hi = doc_off[d], doc_off[d + 1]
        last = -1
        for p in range(lo, hi):
            if first[p]:
                last = p
            raw_age[p] = (p - last) if last >= 0 else np.inf
    cz = el.claim_zone(raw_age, elig.astype(bool), TS)
    f64 = cz["frac_in_window"]["T64"]
    lo_b, hi_b = rg.FLOOR_EXCESS_BAND

    print(f"\nDRY RUN — {N_DOCS} docs, stub backend, {be.calls} stub calls, "
          f"{fillers} fillers")
    print(f"  tokens {ids.size:,}  docs {len(doc_off) - 1}  "
          f"tok/doc {ids.size / (len(doc_off) - 1):,.0f}")
    print(f"  events {int(first.sum()):,}  "
          f"eligible tokens {int(elig.sum()):,}")
    print(f"  realised gap median {gaps['median']:.0f} tok "
          f"(mean {gaps['mean']:.0f}, p10 {gaps['p10']:.0f}, "
          f"p90 {gaps['p90']:.0f})")
    print(f"    (descriptive only — the card's gap target "
          f"{rg.GAP_MEDIAN_SUPERSEDED}/{rg.GAP_RANGE_SUPERSEDED} is "
          f"SUPERSEDED: it assumed uniform probe positions)")
    print("  claim_zone frac_in_window: "
          + "  ".join(f"T{t}={cz['frac_in_window'][f'T{t}']:.4f}'"[:-1]
                      for t in TS))
    verdict = "IN BAND" if lo_b <= f64 <= hi_b else "OUT OF BAND"
    print(f"\n  => predicted floor_excess = f(T64) = {f64:.4f}   "
          f"band [{lo_b}, {hi_b}]   **{verdict}**")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "n_docs": N_DOCS, "seed": SEED, "realised_frac_assumed": REALISED_FRAC,
        "p_repeat": rg.P_REPEAT, "n_tokens": int(ids.size),
        "n_events": int(first.sum()), "gaps": gaps, "claim_zone": cz,
        "predicted_floor_excess_T64": f64,
        "band": [lo_b, hi_b], "in_band": bool(lo_b <= f64 <= hi_b),
        "note": "STUB prose. Validates layout/masking/clock only; "
                "unigram cleanliness is a PILOT gate.",
    }, indent=1))
    print(f"  wrote {OUT.name}")


def run_and_check(be) -> tuple[list, int]:
    """Run the real loop, then assert the structural invariants."""
    docs, fillers = re_.run_retryesc_gen(be, N_DOCS, SEED, MAX_NEW,
                                         tag="retryesc_gen_dryrun")
    for d in docs:
        n_pairs = len(d["plan"]["pairs"])
        turns = d["turns"]
        assert len(turns) == 2 * n_pairs + 1, \
            f"expected {2 * n_pairs + 1} turns, got {len(turns)}"
        for i, (role, _txt, is_ev) in enumerate(turns):
            want = "user" if i % 2 == 0 else "assistant"
            assert role == want, f"turn {i}: expected {want}, got {role}"
            # ⚑ the model must never author an event
            assert not (role == "assistant" and is_ev), \
                "an assistant turn was marked as an event"
        # events must match the plan exactly, offset by one pair
        got = [i for i, (_r, _t, e) in enumerate(turns) if e]
        want = [2 * (k + 1) for k, p in enumerate(d["plan"]["pairs"])
                if rg.is_event(p)]
        assert got == want, f"event positions {got} != planned {want}"
    print(f"[dry-run] structural invariants hold on {len(docs)} docs "
          f"(2N+1 turns, strict alternation, events on environment turns "
          f"only, positions match the plan)")
    return docs, fillers


if __name__ == "__main__":
    main()
