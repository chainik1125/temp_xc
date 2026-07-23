# Working state — agent `runpod-b`

**Last rewrite:** 2026-07-23 ~07:10 UTC — **FB-C1 COMPLETE, STOPPED AT THE
ACCEPTANCE GATE.** Awaiting mac-local review; briefing
`briefings/freqbench-c1.md` deliberately NOT deleted.

## Who / where
Second RunPod box, repo `/workspace/temp_xc`, 32 CPU. `/workspace/.agent_id`
= runpod-b. Git identity set (Han); push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
`export ANTHROPIC_API_KEY=$(cat /workspace/.tokens/anthropic_key)` for API.

## FB-C1 final state (everything committed + pushed)

- **Phase 1** — widened FreqFrac pass, 132 cells: PORT § G.1 (seed-stable
  coordinates; T=8 high-pass PASS).
- **FB-2 multilane POSITIVE** — grid 708/708; sprint band headline FAILED
  its frozen T=8 bar; spectral k_pos-collapse found. `multilane/`.
- **FB-3 colored_sources POSITIVE (weak realization)** — grid 582/582;
  CS-1 floor wholesale; ≤21 % of the provable ceiling realized, by
  txc-pre (ordering inversion). `colored_sources/`.
- **FB-1 phasepair POSITIVE** — grid 636/636; post sign 1.000; spectral
  singleton-band phase-blindness at T≤4; exact bag null. `phasepair/`.
- **The triple dissociation** (cycle headline): spectral/power,
  pre/lag-covariance, post/phase — see PORT.md § H (the FB-C1 cycle log).
- REPORT 78/78 · BENCHMARKS +3 · registry +3 · FreqFrac coords ×3 benches
  · spend $1.04/$25 · ~2,100 grid cells, 0 failures · tests 159+ green.
- Phase 4 (T=16 frontier addendum + `--T 16` FreqFrac) — **NOT RUN**
  (acceptance gate prioritized); queued as the natural follow-up briefing.

## Next actions
**None — STOPPED for review** per the briefing's acceptance gate. On
approval, candidates: the T=16 addendum; porting the remaining
verify_theory checks as permanent tests; the acid-test rows (predict a
held-out bench's ranking from coordinates alone — the triple dissociation
gives three fresh rows).

## Gotchas for the next session on this box
- Grid pace ≈ 8-9 cells/min at 28 free workers (OMP1, GIT_OPTIONAL_LOCKS=0,
  TQDM_DISABLE=1); eval probes ≤ 2 s/cell; checkpoint store now populated
  (~2,300 keys) so overlapping reruns fast-forward.
- `git pull --rebase` only in quiet windows (no grid appending); STATUS.md
  must be committed before rebasing; keep BOTH sides on data.yaml /
  synthetic.py / STATUS § 0 conflicts (runpod is active on stage-6 #3).
- One-sided floor checks for eigen/probe artifacts; orthonormal null for
  eigen-estimators; document every gate-check fix and disclose to the
  skeptic — the pattern that kept this cycle honest.
- freqfrac_report needs the bench registered in registry.py first.
- BatchTopKSAE.train_step wants (B, d_in); tsae class = TSAEPaper.
