# Working state — agent `runpod-c`

**Last rewrite:** 2026-07-23 (session 1 COMPLETE — conversion-depth
briefing executed end-to-end; awaiting mac-local review).

## Who / where
GPU RunPod box: **H100 80 GB** (briefing said A40; actual pod is H100),
16 vCPU / 251 GB RAM, 652 GB network volume, repo at
`/workspace/temp_xc`, `/workspace/.agent_id` = `runpod-c`. Role:
conversion-depth / substrate-audit empirical arms. NO dictionary
training, NO leaderboard writes on this line.

## Session 1: DONE — all five phases

Everything is in `experiments/explorations/conversion_depth/RECORD.md`
(§§ 0–6) + `results/` + `figs/`. Verdict quantities:
`results/depth_verdicts.json`. Freeze order provable from git log
(scripts + prereg committed before first runs). Falsifier not
triggered. Summary bullet appended to research STATUS § 0.

- § 0 provenance PINNED (traces = R1-Distill; math500 misnomer).
- § 1 GPT-2 machinery ALL PASS; probe stack frozen (problib.py).
- § 2 prereg frozen pre-probe (commit b25201ed).
- § 3 both 8B arms: flat open g plateau +0.03..+0.06 all layers;
  reader-predictability verdict (P3 falsified); L10 fine-not-special;
  post-hoc: mostly order-free aggregation.
- § 4 EM: P5 FALSIFIED — inverted-U g, peak +0.134 at L13, +0.097 at
  L15; g_order +0.108 at L13. EM negative depth+readout-confounded.
- § 5 gemma base/IT probing-equivalent (mean |Δ| 0.005; winogrande
  flag = degenerate probe both models).

## Volume assets (KEEP — TXC-tracking follow-up inputs)
- `/workspace/conv_depth_caches/ward_stream/` — canonical 4044×128
  stream + label sidecars (+ `results/probe_rows.npz` = frozen rows).
- `/workspace/conv_depth_caches/{base,distill}/hs*.npy` — 17-point
  multi-layer caches, 72 GB each. KEEP (briefing directive).
- `/workspace/conv_depth_caches/em_medical/` — cohort sidecars only
  (judge_outputs + labels/lens/qids/meta); the 29-point activation
  shards and gemma_probing pooled caches were DELETED per the briefing
  after their probe stats were written. probe_tasks texts (390 MB)
  kept (cheap, re-derivable).
- `/workspace/hf` — HF_HOME (Llama base, R1-distill, Qwen-7B, gemma×2,
  gpt2; ~55 GB).

## Next session (design AFTER mac-local review)
TXC-tracking: does trained-TXC advantage track g(ℓ)? Concrete
predictions in RECORD § 6: flat +0.04 margin for backtracking at any
layer; EM mid-depth curve with the g_order slice (L13) as the best
candidate for a position-aware architecture win.

## Gotchas for future me
- Distill tokenizer: AutoTokenizer resolves to a whitespace-mangling
  SLOW LlamaTokenizer under transformers 5.7 — always force
  PreTrainedTokenizerFast. Id→token maps are identical to base;
  only `</think>` tokenizes differently.
- Gated repos (gemma) need HF_TOKEN env even with weights cached.
- Don't `git add` a results dir while jobs write into it; logs are
  gitignored; use pull --rebase --autostash only when no live writer
  owns an untracked results file.
- peft 0.19.1 installed in .venv (not in pyproject).
- Window-MLP presence probes occasionally collapse on 65k-dim inputs
  (frozen hyperparams overfit); linear pair unaffected.

## Git
On `arxiv`; all session commits local through `eca24f94` + STATUS/
research-STATUS updates; push after final pull --rebase (remote had
runpod/runpod-b evening pushes). Briefing `briefings/conversion-depth.md`
STAYS until mac-local review (its own acceptance-gate rule).
