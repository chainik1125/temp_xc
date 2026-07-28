# `facecmp` — CUMULATIVE face: **arm at chance. Direction not viable as specified.**

Feasibility probe, frozen at `cb8ba2d38` before any activations were read.
**Not a hunt4 § 4 verdict** — a re-labelling of a borrowed corpus, and it
must not be quoted as a screened candidate.

**Total cost: $0.** No pod, no generation. gpt2 (124M) cached on local
MPS in 50 s; every bar in this document is either ground-truth-derived or
a 2-feature probe.

## Verdict

**The activations do not encode a long-horizon event count.** Same
corpus, same activations, same code path, same bars — only the label
changes:

| face | tok | best window | **gain** | foreign null | floor @ bestT |
|---|---|---|---|---|---|
| **RECENCY age** (positive control) | 0.3781 | 0.4377 (T64) | **+0.0596** | 0.3643 | 0.5859 |
| **CUMULATIVE `rate_H512`** | 0.3505 | 0.3529 (T32) | **+0.0024** | **0.3580** | 0.4172 |

**The foreign-context null (0.3580) BEATS the real arm (0.3529).** That
is not a weak signal, it is the absence of one: a window of activations
drawn from a *different* document does as well as the true window.
`label_null` 0.3402 and `position_floor` 0.3318 sit in the same band.
Every cumulative-face number is within ~0.02 of chance.

## The positive control is what makes this worth reporting

Run on the **identical local MPS cache and the identical code path**, the
recency label reproduces the pod: **local tok 0.3781 → window 0.4377
(+0.0596)** against the pod's **0.3669 → 0.4314 (+0.0645)**, with the
foreign null correctly collapsing to 0.3643. So the pipeline, the cache,
the manifest and the transplant all work. **The cumulative result is
about the LABEL, not about my instrumentation** — which is exactly the
thing a bare negative could not have distinguished.

## Pre-registration audit (card § "Pre-registration", written before any GPU ran)

| I predicted | outcome |
|---|---|
| 1. position floor ≈ chance | ✅ **0.3318 (−0.0016)** |
| 2. visible floor T64 ≈ +0.13 | ✅ **0.4679 (+0.1345)** vs +0.1331 label-side |
| 3. arm clears: **~35–40 %**, "genuinely uncertain" | ❌ **failed, and not narrowly** — +0.0024 against a +0.05 bar |
| 4. if it clears, expect it at large T | — n/a, nothing cleared |

**2/2 on the bars I could compute in advance, 0/1 on the one I could
not.** The 35–40 % was honestly uncertain and it was wrong; the arm did
not merely miss the bar, it never left chance.

## What this kills, and what it does NOT

**Killed:** `rate_H512` as a face, and with it the specific plan to build
a corpus around a long-horizon *count*. **It also refutes the reasoning
that motivated it.** I argued the cumulative face escapes the scissors
because both its bars are lower — and both bars *are* lower (visible
floor +0.13 vs +0.26, position floor 0.000 vs +0.039). **Lowering the bar
did not help, because the arm fell further than the bar did.** Bar-first
reasoning is only as good as the assumption that the arm survives the
change, and here it did not.

**NOT established — and I am not going to overclaim this into a general
result:**

- **"Cumulative safety states are unreadable."** Not shown. The tercile
  edges here are **1.0 / 2.0**, so the task is literally *"distinguish
  ≤1 vs exactly 2 vs ≥3 events in the last 512 tokens"* — a **precise
  small-integer counting** discrimination, which is a known weak spot and
  is not the same thing as a graded accumulator ("how much pressure am I
  under"). A smooth cumulative face on a corpus designed for it is
  untested.
- **A corpus built for the face.** `retryesc_gen` was generated with a
  roughly constant per-document event rate, so `rate_H512` here is coarse
  and heavily tied. This was disclosed in the frozen card **before** the
  result, precisely so a failure could not be quietly upgraded into a
  kill.

Per the asymmetry I pre-registered: **this is a feasibility negative, not
a KILL.** It is reported as the weaker thing it is.

## Why it was still worth doing

It cost **$0** and it stopped a **~$21** generation for a face whose arm
sits at chance. The $0 bar-side work (`floor_by_face_shape.py`) made the
direction look good on both bars; only the arm test showed the direction
was empty. **Cheap arm tests on borrowed corpora should come before
generation money** — that is the transferable lesson, and it is now a
one-command pattern (`facecmp/arm_test.py` + a local MPS cache).

_Recorded-by: claude-opus-5 (mac-c, owner + executor)_
