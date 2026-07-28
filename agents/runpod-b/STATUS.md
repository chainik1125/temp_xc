# runpod-b STATUS — POD A CLOSING (2026-07-28 13:08 BST). Fully drained, nothing at risk.

**I am `runpod-b`** — was pod A GPU 1 (container `ac9c40aafb66`,
shared with runpod-a). **Han closed the pod at ~13:07.** Alarm posted
to LOG as `b10125c65`.

## Drain state — verified 13:07, all clean

- **Working tree clean, 0 unpushed commits, 0 running processes,
  both GPUs 0 MiB / 0%.** Nothing died mid-flight.
- **Durability 20/20**: every ckpt mirrored to HF
  (`han1823123123/temp-bench-data`, `ckpts/<train_key>/`), sha256
  receipts in `agents/runpod-b/hf_ckpt_receipts.json`, pushed.
  **Nothing on that pod's disk was unique.**
- rmx_b lane CLOSED 6/6 this morning (actuals $28); folded into
  RM_CERTIFICATE v1.0 and ratified.

## If I resume on a new venue — do these first

1. `git pull --rebase origin arxiv`, read the LOG tail, re-arm a
   listener (150s poll over `experiments/explorations/task_hunt
   briefings agents/runpod-a agents/runpod-2 agents/runpod-1
   agents/runpod-c`).
2. **Re-measure the substrate before trusting any of my numbers on
   the new box** — `cat /sys/fs/cgroup/cpu.max` and `memory.max`
   (NOT `nproc`/`free`, which report the host with no lxcfs),
   `torch.get_num_threads()`, and the co-tenancy curve. See "still
   unexplained" below.
3. `set_agent_env.sh runpod-b` now sets `CVD=1 OMP=16 MKL=16` for
   **pod A specifically** — on a new venue the GPU index and thread
   count both need revisiting.

## Delivered today (all pushed, all PTR)

- **Substrate**: cgroup quota trap — torch autosizes threads from the
  *host* count, cgroup throttles rather than masks, so no API sees the
  cap (`0bed01849`). Pod A = 47.6 cores / 468 GiB, ONE container shared
  with runpod-a, with `cpu.stat` throttle receipts (`e4038e0f2`).
  Reproduced on pod B (runpod-c) and the old pod (runpod-2) ⇒
  image-level, not pod-specific.
- **⚑ Feed does not saturate** (`ade801886`): 12 concurrent lanes,
  ~1.0× per-lane, ~10.5 GiB/s aggregate ⇒ lane packing is
  GPU-memory + cores bound only. runpod-a withdrew their
  ~3-lanes/pod cap on this (`85d8658d7`). Also: fp16 prize is ~7% of
  wall, not ~47% — 86% of the feed is not byte movement.
- **Two launch traps** in `actmix_rlhf/run_cells.py` (`0be31500b`),
  both confirmed at source by runpod-a: `AGENT_NAME` setdefault
  `runpod-2` at line 24 (would misattribute ~8 lanes of rows), and no
  thread guard.
- **Seed-column lanes** `pf_s42{,_a,_b,_c}` / `pf_s1` / `pf_s2`
  (`0d4f471fb`) — Han's directive 3 was **unrunnable** through the
  T-blocked `pf_{lo,mid,hi}` lanes. Additive-only: 19 pre-existing
  lanes byte-identical, zero new cell_ids, `validate` OK.
- **pf table replicated 14/14** to 4 dp; "131 frozen btk rows"
  confirmed (135 `btk-only` − 4 `smoke=True`); two caption nits
  (l0 column is k_feat-independent; E1's exact-cap claim inverted at
  T1 vs T6).
- Own env arm fixed (`d56026a6b`): `CUDA_VISIBLE_DEVICES ""` → `1`.

## Open / handed off

- **Deferred twin relay — ANY agent can run it.** If btk T10 s1/s2
  twins appear, `torch.equal` them against my anchors
  `cd2f6e8ab14fa3e0` (s1) / `d3e331643b765baf` (s2). Both on HF with
  receipts. Protocol `83dc80d37`: expect 7/7 on shared tensors, with
  `threshold_set` present only on the btk side.
- **⚑ Still unexplained**: naive 2-lane co-tenancy measured 0.75–0.88×
  on pod A and by runpod-2 on the old pod, but **1.82× on pod B**
  (runpod-c) — at an identical 2.35× oversubscription ratio. I tested
  and **rejected NUMA** (taskset per socket: no effect). Do not assume
  my penalty transfers to a new venue.
- Wave-1 packing plan (`ade801886` §4) needs re-mapping off pod A onto
  pod B + old pod; the arithmetic stands, the venue assignment does not.

## Standing rules

- Explicit-path commits (never `git add -A` with live jsonls);
  keep-BOTH LOG conflicts only after verifying both parents have zero
  legitimate `=======` lines; stamps only from already-printed `date`
  output; 15-min ack discipline; PTR everything; gold → HANDOFF
  same-beat.
- `AGENT_NAME=runpod-b` inline on every launch (the rlhf driver
  defaults it to `runpod-2`).
- Stuck rebase escape: `commit --no-edit` + `rm -rf .git/rebase-merge`
  + `checkout -B arxiv HEAD`.

*Rewrite before any compact. — runpod-b*
