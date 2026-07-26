---
author: Dmitry Manning-Coe
date: 2026-07-26
tags:
  - results
  - in-progress
---

## 22:33 — kickoff

Sprint window 2026-07-25 22:33 PDT → 2026-07-26 08:33 PDT, branch `dmitry-txcwins-10h`.

Goal: find further tasks where the temporal crosscoder beats a TopK SAE and a tSAE, ideally
ones corresponding to documented model behaviours rather than constructs.

The procedure being repeated, from the previous sprint:

1. identify a property that structurally favours a window code;
2. build a task with that property and matched foils, so generic effects cancel;
3. test reading and steering **separately** — they came apart last time and that was the
   result;
4. run the controls that can kill it — time-averaged profile, random profile, random
   direction, supervised ceiling.

Three agents launched:

| agent | job |
| --- | --- |
| `theory` | enumerate structural properties favouring a window code; concrete task designs with registered predictions and the control that would refute each; rank by (probability real) × (relevance to a behaviour anyone wants to steer) |
| `implement` | calibrate the tSAE baseline first (carried-over debt), then build and run task designs on Modal from the `steer_order_modal.py` template |
| `review` | continuously scan the literature; catalogue candidate behaviours and model organisms for TXC vs SAE vs tSAE, prioritising ones where an organism already exists and can be obtained tonight |

`implement` starts unblocked on the tSAE calibration, which does not depend on any task
design: at the repo's documented `l1_coef=1e-3` the tSAE code is dense (2989/4096 latents
active, alive 0.999) and a 100× sweep moved realised L0 by 0.3%. The `lam = 1/(4·d_in)`
scaling in `han_tsae/saeTemporal.py` means `l1` needs to be ~1–10 at this activation scale.
Until that is fixed there is no third baseline.

Standing methodology, carried forward and non-negotiable: realised coefficients per segment
as the axis (never nominal k), stride-1 windowing, `batchtopk` without ReLU, `m.eval()`
before scoring, and every steering claim accompanied by the time-averaged / random-profile /
random-direction / supervised-ceiling controls.
