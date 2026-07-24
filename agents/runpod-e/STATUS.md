# Working state — agent `runpod-e`

**Last rewrite:** 2026-07-24 (first session, mid-task).

## Who / where
GPU RunPod pod (H100 80GB), Linux at `/workspace/temp_xc`, identity
`/workspace/.agent_id` = `runpod-e` (seeded this session). Role:
**task-hunt arm B** (`briefings/task-hunt-b.md`, governed by
`briefings/task-hunt.md`) — repetition-lag Δ across model scale
(gpt2-small / gemma-2-2b base / Llama-3.1-8B base) + confidence-trend /
emotional-instability backups. Own independent 700 GB volume at
`/workspace` (693 GB free); all caches local under
`/workspace/replag_caches/`.

## Session state (2026-07-24)
- Env DONE: `.venv` has CUDA torch 2.8.0 / transformers 5.7.0 /
  datasets 4.8.5 / accelerate; `HF_HOME=/workspace/hf_cache`; git
  identity `runpod-e-agent`, creds via
  `store --file=/workspace/.git-credentials` (token from
  `/workspace/.tokens/gh_token`, never hardcode). Model weights
  (gpt2, gemma-2-2b, Llama-3.1-8B base) downloading to HF cache in
  background.
- runpod-b labels had NOT landed when caches were ready → building the
  Δ-label builder myself per briefing (exact computation from tokens +
  shuffled-window null + sanity tests), committed before outputs.
- Corpus: the committed pinned fineweb sample
  (`experiments/explorations/synthetic/expansion/data/fineweb_sample.json`,
  400 docs) — the prep briefing's sanctioned default.
- Next: freeze `experiments/explorations/task_hunt/replag/CARD.md`
  (commit BEFORE screen) → build labels → cache one mid-depth layer ×3
  models → Stage-1 screen → verdict to `../LOG.md` → then candidate 2
  vs 3 decision (check runpod-b clock-bridge stats first).

## Standing rules I'm operating under
Pull-rebase before every push (5 agents on `arxiv`). No reviewer/
meeting quotes in tracked files. Stage-2 only through the canonical
runner. Screen budget ~2-4 h/candidate; fail fast; deadline
2026-07-26 morning PT.
