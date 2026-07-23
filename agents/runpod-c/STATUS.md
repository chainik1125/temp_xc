# Working state — agent `runpod-c`

**Last rewrite:** 2026-07-23 (seeded by mac-local at pod creation — no
session run yet).

## Who / where
The GPU RunPod box: A40 48 GB, ≥ 300 GB volume, Linux at
`/workspace/temp_xc`, identity file `/workspace/.agent_id` = `runpod-c`.
Role: **conversion-depth ablation + the substrate audit's empirical
arms** — subject-model forward passes, multi-layer activation caches,
raw-activation g(ℓ) probes. NOT a grid pod: no dictionary training, no
leaderboard writes this line of work.

## First session
Execute `briefings/conversion-depth.md` end-to-end (governing docs:
`docs/ideas/conversion_depth.md` + `docs/substrate_audit_2026-07.md`).
Setup checklist before phase 0: `.venv` per repo conventions; verify
CUDA torch (`python -c "import torch; print(torch.cuda.is_available())"`);
HF cache on the big volume (`HF_HOME=/workspace/hf`); git creds at
`/workspace/.tokens/` — never hardcode; `git pull --rebase` before every
push (two CPU agents run in parallel tonight).

## Git
Fresh clone of `arxiv`. No local state yet.
