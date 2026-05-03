# CLAUDE.md — purified/ subtree

You are an AI agent working on the paper-ready `final` branch. **All work
happens inside `purified/`.** The root-level `temp_xc/{src,experiments,docs}`
is the "wasteland" — read it for context, never import or modify it.

## First actions on every session

1. Read `purified/PROTOCOL.md` (operational rules).
2. Read your agent dir's `briefing.md` (your specific mandate).
3. Read `docs/components/c{N}.md` for any component you'll touch.
4. Skim the last 5 entries of `results/leaderboard.jsonl` to see what
   other agents have just produced.

## Hard rules

1. **Never import from `temp_xc/{src,experiments,docs}`.** If you need
   reference code, copy it into `purified/src/temp_bench/` (duplication
   is fine; coupling is not).
2. **Never edit `purified/agents/<other_agent>/`** — those are owned by
   other agents.
3. **Use `purified/.venv`** built from `purified/pyproject.toml`. Run
   `uv sync` from inside `purified/`. Do not use the root `temp_xc/.venv`.
4. **Always set `TQDM_DISABLE=1`** before any Python invocation.
5. **The two TXC architectures are locked**: `TXC-base` =
   `txc_bare_antidead_t5`; `TXC-pro` = `phase5b_subseq_h8`. Do not
   introduce a third TXC variant. Sparsity (k) and dictionary size
   (d_sae) are the only free parameters per component.
6. **Writeups go in `docs/components/c{N}.md`**, not in agent dirs.
   Agent dirs hold your *briefing* and *log* only — ephemeral state.

## Hardware quotas

| Pod | Agents | Note |
|---|---|---|
| Local 5090 (32GB) | Agent PAPER | Orchestrator + C1 + C2 + paper drafting |
| 2× H100 RunPod | Agent NLP, Agent EM | NLP=C3+C4 (shared cache); EM=C6 (Qwen-14B) |
| 3× A40 RunPod | Agent STEER, Agent BACK, optional 3rd | STEER=C5; BACK=C7 |
| H200 | reserve | Only for EM if R32 organism blows H100 mem |

## How to record results

- Pick a run id: `<component>_<arch>_<seed>_<short-hash>` where short-hash
  is `python -c "import secrets; print(secrets.token_hex(4))"`.
- Write outputs to `results/runs/<run_id>/`:
  - `manifest.json` — config, command, host, start time
  - `metrics.json` — final numbers
  - `plots/*.png` + `plots/*.thumb.png` (use `temp_bench.plotting.save_figure`)
  - `log.txt` — stdout/stderr
- When done, append one line to `results/leaderboard.jsonl`:
  ```json
  {"run_id": "...", "component": "c3", "arch": "txc_bare_antidead_t5",
   "seed": 42, "k": 20, "metric": "probing_auc_S32", "value": 0.9127,
   "ckpt_hf": "chainik1125/temp-bench/<run_id>",
   "agent": "agent_nlp", "ts": "2026-05-03T12:00:00Z"}
  ```
  Use `flock` if scripting concurrent appends.

## How to record checkpoints

- Upload to HF: `chainik1125/temp-bench/<run_id>` (uses HF_TOKEN).
- Append one line to `checkpoints/manifest.jsonl`:
  ```json
  {"run_id": "...", "hf_url": "https://huggingface.co/chainik1125/temp-bench/<run_id>",
   "local_path": "/workspace/.../<run_id>.pt", "size_mb": 412}
  ```
- Local-only checkpoints are allowed during development but must be HF-backed
  before the agent finishes its session.

## Markdown style

Same as the root CLAUDE.md: ATX headings, no H1, dash bullets, fenced code
blocks with language, YAML frontmatter (author/date/tags) on `docs/`.

## Quick reference

```bash
# bootstrap (RunPod, idempotent)
cd /workspace/temp_xc/purified && bash scripts/bootstrap_runpod.sh

# run any python in purified env
cd purified && TQDM_DISABLE=1 .venv/bin/python -m experiments.c1_synthetic_topk.run

# append a leaderboard row
python -c "import json,fcntl,sys; row={...}; \
  f=open('results/leaderboard.jsonl','a'); fcntl.flock(f,fcntl.LOCK_EX); \
  f.write(json.dumps(row)+'\n'); f.flush(); fcntl.flock(f,fcntl.LOCK_UN)"
```
