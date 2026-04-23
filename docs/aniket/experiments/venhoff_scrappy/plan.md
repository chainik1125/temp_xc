---
author: Aniket
date: 2026-04-23
tags:
  - proposal
  - in-progress
---

## venhoff_scrappy autoresearch — full plan

Scrappy, iteration-friendly version of the Venhoff MATH500 Gap Recovery experiment, wired into an autoresearch loop. Pattern adopted from Han's `experiments/phase5_downstream_utility/` (Phase 5.7 setup): thin orchestrator → append-only JSONL ledger → per-candidate commit rhythm → agentic research log as state machine.

**Status**: scaffold committed 2026-04-23 (`7487d64`), run_cycle.py is a **stub** — real dispatch to the vendor pipeline is what this plan describes. Looking for a web-claude sanity check before implementation.

## 1. Motivation

The full-budget run currently executing on 4× H100 delivers the headline: SAE / TempXC / MLC vs Venhoff's paper baseline (3.5% Gap Recovery on Llama-3.1-8B ↔ DeepSeek-R1-Distill-Llama-8B cell). That answers the **first-order** question: "can temporal crosscoders do better than per-sentence SAE steering?" The running number at task 152/500 on this arch shows ~18% Gap Recovery, which if it holds is ~5× Venhoff's baseline. Good.

Autoresearch answers the **second-order** question: **within an arch family, is there a variant that meaningfully outperforms the one we're running?** Specifically:

- Steering layer choice (we use L12 because Venhoff did; we haven't tested other layers)
- Temporal reduction operator for TempXC (sum vs mean vs max over the time window)
- Loss function tweaks on Phase 2 (orthogonality between clusters, contrastive term)
- Coefficient scheme (fixed scalar vs per-cluster learned)
- Cluster count (15 is Venhoff's default; unclear if optimal for our cell)
- Token window strategy (paper sweeps {0, -1, -15, -50, -100}; may not be Pareto-optimal)

These are the dials we'd like an LLM agent (Claude Code in a loop) to turn overnight, evaluating each on a scrappy 20-task slice of MATH500 (~10 min per cycle on a single GPU). Winning candidates promote to paper-budget runs.

## 2. Success criteria

**Infrastructure**:
- `bash experiments/venhoff_scrappy/run_autoresearch.sh baseline_sae` completes one full Phase 0→grade cycle in **<15 min wall-clock on a single H100** (or <30 min on A5000).
- Ledger row appended to `experiments/venhoff_scrappy/results/autoresearch_index.jsonl` with real Gap Recovery (not the current placeholder).
- Commit + push rhythm per-candidate so partial progress survives pod reboot.

**Scientific**:
- At least one candidate lands a **FINALIST** verdict (Δ > +10 pp vs `baseline_sae`) on the scrappy slice.
- Mechanism insight: the verdict pattern across ~10-20 cycles should produce a narrative sentence like "TempXC+L16 is the strongest axis" or "cluster count dominates reduction operator."

## 3. Relevant files

### 3.1 Scrappy scaffold (already committed, needs wiring)

| Path | Role | State |
|------|------|-------|
| `experiments/venhoff_scrappy/README.md` | quickstart | final |
| `experiments/venhoff_scrappy/config.yaml` | scrappy defaults (n_tasks=20, coef=0.5, max_iters=5, n_clusters=4) | final |
| `experiments/venhoff_scrappy/run_autoresearch.sh` | orchestrator; loops over candidates, calls run_cycle.py + summariser, commits per result | final |
| `experiments/venhoff_scrappy/run_cycle.py` | per-candidate Phase 0→grade driver | **scaffold — this is what needs wiring** |
| `experiments/venhoff_scrappy/autoresearch_summarise.py` | reads grade_results.json, computes Δ vs baseline, appends ledger row | final |
| `experiments/venhoff_scrappy/candidates/baseline_sae.yaml` | SAE-with-shipped-vectors reference point | final |
| `docs/aniket/experiments/venhoff_scrappy/agentic-log.md` | append-only research log (cycle hypotheses + takeaways) | template |
| `docs/aniket/experiments/venhoff_scrappy/plan.md` | this doc | draft |

### 3.2 Existing vendor pipeline entry points (to be called from run_cycle.py)

| Path | Purpose | Key args run_cycle.py will set |
|------|---------|------|
| `src/bench/venhoff/run_steering.py` | CLI: train all steering vectors (bias + N clusters) for one arch | `--arch`, `--max-iters`, `--n-training-examples`, `--n-clusters`, `--top-k-clusters`, `--steering-layer`, `--sae-layer`, `--optim-minibatch-size` |
| `src/bench/venhoff/run_hybrid.py` | CLI: run hybrid inference on MATH500 via Venhoff's `hybrid_token.py` | `--arch`, `--coefficients`, `--token-windows`, `--n-tasks`, `--max-new-tokens`, `--max-thinking-tokens` |
| `src/bench/venhoff/run_grade.py` | CLI: compute Gap Recovery from hybrid results; emits summary JSON | `--arch`, `--dataset`, `--out` |
| `src/bench/venhoff/steering.py` | `SteeringConfig` dataclass + `train_all_vectors` (dispatches per-cluster + bias) | — |
| `src/bench/venhoff/hybrid.py` | `HybridConfig` dataclass + `run_hybrid` (applies vendor_patches, subprocess-calls vendor `hybrid_token.py`) | — |
| `src/bench/venhoff/grade.py` | `compute_gap_recovery(results_dir, thinking_jsonl, base_jsonl, coefficients, token_windows)` — returns `best_cell` + `best_gap_recovery` + `per_cell` | — |
| `src/bench/venhoff/paths.py` | `ArtifactPaths`, `RunIdentity` — canonicalizes result dirs by (model, dataset, split, n_traces, layer, seed) | — |
| `src/bench/venhoff/vendor_patches.py` | Idempotent vendor-tree monkey-patches (judge model, `load_in_8bit` drop, BatchEncoding clone unwrap) | — |
| `src/bench/venhoff/generate_traces.py` | Phase 0: collect DeepSeek thinking traces. Expensive; run once, reuse across candidates. | — |
| `src/bench/venhoff/activation_collection.py` | Phase 1: extract activations at steering layer from traces | — |

### 3.3 Reference — Han's autoresearch setup (read-only; pattern source)

| Path | Role |
|------|------|
| `experiments/phase5_downstream_utility/run_autoresearch.sh` | orchestrator template |
| `experiments/phase5_downstream_utility/autoresearch_summarise.py` | Δ + verdict + ledger row template |
| `experiments/phase5_downstream_utility/train_primary_archs.py` | training dispatcher; our analog is run_cycle.py |
| `docs/han/research_logs/phase5_downstream_utility/2026-04-21-agentic-log.md` | research log format |
| `experiments/phase5_downstream_utility/results/autoresearch_index.jsonl` | ledger schema reference |

### 3.4 Operational scripts

| Path | Purpose |
|------|---------|
| `scripts/runpod_venhoff_bootstrap.sh` | fresh-pod setup — clone repo, install uv, create 3.11 + 3.12 venvs, clone vendor, write .env template |
| `scripts/runpod_venhoff_paper_run.sh` | one-shot paper-budget launcher (not used in scrappy — left for full-budget promotion) |

## 4. Candidate model

A **candidate** is a YAML file at `experiments/venhoff_scrappy/candidates/<name>.yaml`. It declares:

```yaml
name: <string>              # must match filename
arch: sae | tempxc | mlc    # which steering-vector architecture
baseline: <candidate_name>  # the candidate to compute Δ against (often baseline_sae)
hypothesis: |
  <free-text prose explaining the mechanism under test>

# Any subset of config.yaml keys may be overridden:
phase2_steering:
  max_iters: 10
  n_clusters: 8
  reduction: mean
  steering_layer: 16
phase3_hybrid:
  coefficients: [0.3, 0.5, 0.8]
  token_windows: [0]
```

`run_cycle.py` deep-merges the candidate YAML over `config.yaml` (scrappy defaults) and dispatches the three vendor CLIs with the merged config.

**Immutability rule**: once a candidate has landed a result row in `autoresearch_index.jsonl`, its YAML is frozen. To re-run with different settings, create a new candidate with a different name. This matches Han's pattern and avoids silent config drift.

## 5. Pipeline: what run_cycle.py must do

This is the implementation gap. Current state: stub that writes a placeholder JSON. Target state: full Phase 0→grade dispatch in ~100 LOC. Pseudocode:

```python
def run_cycle(cfg, candidate, result_dir):
    # Resolve paths under a candidate-specific results subtree so
    # cycles don't clobber each other.
    paths = ArtifactPaths(
        root=result_dir / "venhoff_eval",
        identity=RunIdentity(
            model="deepseek-r1-distill-llama-8b",
            dataset="math500",
            dataset_split=cfg["phase3_hybrid"]["dataset_split"],
            n_traces=cfg["phase0_activations"]["n_traces"],
            layer=cfg["model"]["steering_layer"],
            seed=cfg["autoresearch"]["seed"],
        ),
    )

    # Phase 0: reuse the repo-level cached traces/activations.
    # Symlink or copy the cached artifacts into paths.* so Phase 2/3
    # find them where they expect.
    _reuse_phase0_cache(paths, cfg)

    # Phase 2: train steering vectors. Skip entirely for baseline_sae
    # (venhoff ships 16 vectors under vendor/.../optimized_vectors/).
    if not cfg.get("reuse_venhoff_vectors", False):
        subprocess.run([
            "/workspace/spar-temporal-crosscoders/.venv/bin/python",
            "-m", "src.bench.venhoff.run_steering",
            "--root", str(paths.root),
            "--model", paths.identity.model,
            "--dataset", paths.identity.dataset,
            "--n-traces", str(paths.identity.n_traces),
            "--layer", str(paths.identity.layer),
            "--arch", cfg["arch"],
            "--steering-layer", str(cfg["model"]["steering_layer"]),
            "--sae-layer", str(cfg["model"]["sae_layer"]),
            "--n-clusters", str(cfg["phase2_steering"]["n_clusters"]),
            "--max-iters", str(cfg["phase2_steering"]["max_iters"]),
            "--n-training-examples", str(cfg["phase2_steering"]["n_training_examples"]),
            "--optim-minibatch-size", str(cfg["phase2_steering"]["optim_minibatch_size"]),
            "--num-gpus", "1",
        ], check=True)

    # Phase 3: hybrid inference.
    subprocess.run([
        "/workspace/spar-temporal-crosscoders/.venv/bin/python",
        "-m", "src.bench.venhoff.run_hybrid",
        "--root", str(paths.root),
        "--arch", cfg["arch"],
        "--n-tasks", str(cfg["phase3_hybrid"]["n_tasks"]),
        "--max-new-tokens", str(cfg["phase3_hybrid"]["max_new_tokens"]),
        "--max-thinking-tokens", str(cfg["phase3_hybrid"]["max_thinking_tokens"]),
        "--coefficients", *[str(c) for c in cfg["phase3_hybrid"]["coefficients"]],
        "--token-windows", *[str(w) for w in cfg["phase3_hybrid"]["token_windows"]],
        # plus --model, --dataset, --n-traces, --layer (identity fields)
    ], check=True)

    # Grade: compute Gap Recovery.
    grade_out = result_dir / "grade_raw.json"
    subprocess.run([
        "/workspace/spar-temporal-crosscoders/.venv/bin/python",
        "-m", "src.bench.venhoff.run_grade",
        "--arch", cfg["arch"],
        "--dataset", "math500",
        "--out", str(grade_out),
    ], check=True)

    # Normalize grade output into the ledger-friendly shape.
    raw = json.loads(grade_out.read_text())
    return {
        "candidate": candidate,
        "arch": cfg["arch"],
        "n_tasks": cfg["phase3_hybrid"]["n_tasks"],
        "thinking_acc": raw["thinking_accuracy"],
        "base_acc": raw["base_accuracy"],
        "hybrid_acc": raw["best_cell"]["hybrid_accuracy"],
        "gap_recovery": raw["best_gap_recovery"],
        "best_cell": raw["best_cell"],
        "wall_time_s": time.time() - t0,
    }
```

### 5.1 Key implementation decisions

1. **Phase 0 reuse**. Traces + activations are ~30 GB and take ~1 h to collect. They don't vary across candidates (all candidates use the same thinking-model + base-model + dataset + steering-layer). Run Phase 0 **once** at repo level when the pod is bootstrapped, then `run_cycle.py` symlinks or copies the cached artifacts into the candidate's `paths.*` tree. Exception: if a candidate changes `steering_layer`, Phase 0 for the new layer is needed — cache per layer.

2. **Per-candidate result isolation**. Results land under `experiments/venhoff_scrappy/results/cycles/<candidate>/` with its own `venhoff_eval/` subtree. This prevents vendor hybrid_token.py's internal resume-cache from mixing cycle outputs. The cost: each cycle re-pays the hybrid inference cost. Worth it for clean provenance.

3. **SAE baseline shortcut**. For `baseline_sae`, set `reuse_venhoff_vectors: true` in the YAML. Phase 2 is skipped entirely; `steering.py` already has the sidecar logic (`source=venhoff_shipped`) that detects and uses the pre-shipped vectors.

4. **Single-GPU constraint**. Scrappy pod is 1 GPU (A5000/A100/PRO 6000 depending on availability). `run_steering.py --num-gpus 1` is fine. `run_hybrid.py` is single-GPU by default.

5. **Grade output location**. `run_grade.py` writes to `{root}/grades/{arch}_{dataset}.json` by default. `run_cycle.py` passes `--out` to force it into the cycle's result dir.

### 5.2 Estimated per-cycle wall-clock

On 1× H100 at scrappy budget:

| Phase | Wall-clock |
|-------|------------|
| 0 (activations, cached) | ~0s (symlink reuse) |
| 2 (steering vector train) | ~5 min (4 clusters × max_iters=5 × 128 examples, serial) |
| 3 (hybrid, n=20, 1 coef × 2 windows = 2 cells) | ~4 min (vLLM warmup + 20 × ~10s/task) |
| Grade | ~30 s |
| **Total** | **~10 min** |

On 1× A5000 or similar: roughly 2× slower → ~20 min/cycle. Still fine for overnight.

## 6. The autoresearch loop

This is the **agent cadence** — how does an LLM propose the next candidate?

### 6.1 Option A: manual-curated (recommended for first N cycles)

Human hand-writes the first 5-10 candidate YAMLs based on the plan below. Each cycle commits its YAML + result row. Gap Recovery numbers accumulate in the ledger. Human (or Claude Code session) reads the ledger + agentic-log and proposes the next candidate by hand.

Pros: tight control, can correct misdirection early.
Cons: slow, requires human-in-loop each cycle.

### 6.2 Option B: fully agentic (Han's pattern)

A Claude Code session runs in a tmux tab on the pod. Its system prompt is "read agentic-log.md, read the latest ledger rows, propose the next candidate, write its YAML, commit, trigger `run_autoresearch.sh <candidate>`, then repeat after the ledger row lands." Stop criterion: "5 consecutive cycles with Δ < +10 pp and no new mechanism insight."

Pros: zero human intervention overnight.
Cons: agent may converge on a pathological local optimum (e.g., all cycles differ only in `coefficient`, never touch the architecture axis). Needs a curated candidate-exploration plan in the system prompt.

**Recommendation**: start with A (5 curated cycles to validate the infra). If infra is stable, switch to B for overnight runs with an explicit candidate-axis plan in the prompt.

### 6.3 Exploration strategy — screening then exploit (web-claude revision)

Pure coordinate descent on a O(10³) space traps the agent in **cheap-axis gravity** (it tweaks `coefficient` and `token_windows` because they're one-line YAML edits, skipping the real-effect axes). Two-phase plan:

**Screening phase (scope of current wiring)** — 3 × 3 full factorial on `(arch ∈ {sae, tempxc, mlc}) × (n_clusters ∈ {4, 8, 15})`. 9 cycles. ~90 min on a single GPU at ~10 min/cycle.

Why only these two axes in the first wired batch:
- `arch` is a hardcoded case in the vendor `optimize_steering_vectors.py` — no YAML override needed.
- `n_clusters` is already a `SteeringConfig` field and flows through to Phase 2 + Phase 3.
- `steering_layer` variation requires a second Phase 0 run per layer (activations are layer-specific) — deferred to a follow-up batch.
- `reduction` (sum/mean/max for TempXC) is NOT a `SteeringConfig` field yet. Adding it requires a vendor-level patch or a new loss override. Deferred.
- `coefficient` and `token_windows` are the cheap-axis gravity trap; excluded from screening by design. The agent can explore them in the exploit phase.

**Exploit phase** — agent-driven coordinate descent on the screening winner, adding `steering_layer` + `coefficient` + `token_windows` as axes. System prompt constraint: *every axis in the ledger must have ≥3 cycles before any verdict claim*. ~50 cycles, ~10 h.

### 6.4 Initial curated batch (cycles 00-02, before the factorial)

1. **`baseline_sae`** — reference. Δ = 0 by definition. Establishes the absolute GR number on the scrappy slice.
2. **`baseline_tempxc`** — TempXC at scrappy budget. Confirms scrappy slice preserves TempXC > SAE ordering (smoke test for the full-budget headline).
3. **`baseline_mlc`** — MLC at scrappy budget. Same smoke test for MLC.

Then the 9-cycle factorial (§6.3). ~12 cycles × 10 min = ~2 h total.

**Shuffle-control note**: `baseline_sae_shuffled` is scaffolded in `candidates/` but excluded from `run_all.sh`. Running it right with the Venhoff pipeline requires either retraining SAE from scratch on permuted activations (~60 min at n_clusters=15; ~15 min at n_clusters=4) or patching `hybrid_token.py` to shuffle steering-vector → cluster assignment at inference time. Treated as a post-hoc confound check invoked manually once the main batch is in.

## 7. Metric, noise, verdict thresholds

### 7.1 Metric: Gap Recovery (absolute and paired Δ)

Per `grade.py`:
```
gap_recovery = (hybrid_acc - base_acc) / (thinking_acc - base_acc)
```

**Absolute noise (n=20)**:
- thinking_acc ≈ 0.80 (SE ≈ 0.09), base_acc ≈ 0.20 (SE ≈ 0.09), hybrid_acc SE ≈ 0.09
- Denominator ≈ 0.60, unpaired numerator SE ≈ 0.13 → **absolute Gap Recovery SE ≈ ±22 pp**.

**Paired Δ noise (what we actually care about)** — web-claude review (2026-04-23):
The metric we verdict on is Δ = GR_cand − GR_baseline evaluated on **the same 20 tasks**. Denominator cancels in the Δ (thinking/base grades are fixed across candidates); numerator is `hybrid_cand − hybrid_base` on paired tasks with SE = √(2·p(1−p)(1−ρ)/n). With moderate task-level correlation ρ ≈ 0.4–0.6 (plausible — task difficulty is mostly intrinsic), **paired Δ SE ≈ ±10–15 pp**, not ±22.

**Decision**: keep the +10 pp FINALIST threshold, with two mandatory additions:

1. **Deterministic 20-task slice** fixed across all candidates so pairing actually holds. Use `MATH500.test[:20]` (or a pre-computed stratified-by-level slice of 20; see §11). No per-cycle reseed.
2. **Per-task outcomes vector** in every ledger row (20-element bool array). ~40 bytes/row, unlocks paired McNemar and paired bootstrap post-hoc for *any* pair of candidates without re-running.

Any FINALIST re-runs at **n=100** (~50 min scrappy) for confirmation before paper-budget promotion.

### 7.2 Verdict thresholds

| Δ Gap Recovery | Verdict | Action |
|----|----|----|
| > +10 pp | FINALIST | promote to n=100 re-run |
| +3 pp to +10 pp | PROMISING | accumulate; if 3 consecutive on same axis, re-run at n=100 |
| −10 pp to +3 pp | AMBIGUOUS | continue search, don't discard |
| < −10 pp | DISCARD | retire candidate family |

### 7.3 Ledger schema (v1)

Each row in `autoresearch_index.jsonl`:
```json
{
  "schema_version": 1,
  "ts": "...",
  "candidate": "...", "baseline": "...",
  "arch_cand": "...", "arch_base": "...",
  "n_tasks": 20,
  "thinking_acc": 0.80, "base_acc": 0.20, "hybrid_acc": 0.42,
  "gap_recovery_cand": 0.367, "gap_recovery_base": 0.183,
  "delta_pp": 18.4,
  "absolute_gap_cand": 0.22, "absolute_gap_base": 0.11,
  "best_cell": {"coefficient": 0.5, "token_window": 0, "hybrid_accuracy": 0.42},
  "per_task_outcomes": [true, false, true, ...],
  "verdict": "FINALIST",
  "wall_time_s": 612,
  "gap_recovery_per_gpu_minute": 0.0036,
  "phase0_cache_hash": "sha256:abcd..."
}
```

### 7.4 Stop criterion

Adapted from Han's "5 consecutive cycles with Δ < +0.01 and no informative insight":

> If 5 consecutive cycles produce Δ < +3 pp vs `baseline_sae` (PROMISING threshold) without a new mechanism insight, escalate to human for re-hypothesis.

"Insight" = a qualitative pattern in the ledger (e.g., "all TempXC layers underperform SAE; reduction operator doesn't matter"). Agent decides this based on the ledger rows, not just the threshold number.

### 7.2 Verdict thresholds

| Δ Gap Recovery | Verdict | Action |
|----|----|----|
| > +10 pp | FINALIST | promote to n=100 re-run |
| +3 pp to +10 pp | PROMISING | accumulate; if 3 consecutive on same axis, re-run at n=100 |
| −10 pp to +3 pp | AMBIGUOUS | continue search, don't discard |
| < −10 pp | DISCARD | retire candidate family |

### 7.3 Stop criterion

Adapted from Han's "5 consecutive cycles with Δ < +0.01 and no informative insight":

> If 5 consecutive cycles produce Δ < +3 pp vs `baseline_sae` (PROMISING threshold) without a new mechanism insight, escalate to human for re-hypothesis.

"Insight" = a qualitative pattern in the ledger (e.g., "all TempXC layers underperform SAE; reduction operator doesn't matter"). Agent decides this based on the ledger rows, not just the threshold number.

## 8. Review decisions (web-claude, 2026-04-23)

Questions from the previous revision, with resolutions now baked into the plan + code:

1. **Noise floor vs threshold** → keep +10 pp FINALIST + n=100 re-run. Paired Δ SE is ~±10–15 pp (not the ±22 pp of absolute GR), because the same 20 tasks are scored for every candidate and task-level correlation cancels. Enforced by deterministic slice + per-task outcomes vector (§7.3).

2. **Phase 0 cache scope** → ship symlinks; nothing writes to Phase 0 post-collection, so race risk is nil. Stamp a `phase0_cache_hash` in every ledger row so scrappy vs paper-budget provenance is auditable.

3. **Candidate explosion** → 20-cycle fractional factorial screening phase (arch × layer × reduction) before the agent runs free (§6.3). Agent system prompt mandates ≥3 cycles per axis before verdict claims — anti cheap-axis gravity.

4. **Metric alternatives** → track Gap Recovery, hybrid_acc, absolute gap, `best_cell`, **and per-task outcomes** in every row (§7.3). Adds ~80 bytes; unlocks paired post-hoc analysis.

5. **Agent loop fragility** → stateless agent sessions (each turn reads ledger + log from disk, no in-session memory dependence). Bash orchestrator per-cycle timeout kills stuck vLLM and emits a `FAILED` row. Optional tmux respawn loop. No heartbeat file.

6. **Loss tweaks via monkey-patch** → extend `vendor_patches.py` with `register_loss_override(name, fn)`; candidate YAML references `loss_override: <name>`. Override functions live under `src/bench/venhoff/loss_overrides/` in version control. Patch-file fallback only for structural changes that can't be expressed as function substitution. No vendor fork.

7. **Judge cost** → cache thinking/base grades once per slice (they're constants across candidates). Per-cycle judge calls drop from 60 → 20. ~$9/day → ~$3/day. Trivial optimization.

## 9. Extra web-claude flags (incorporated)

- **Shuffle control cycle** added as cycle 01 (`baseline_sae_shuffled`). Per our own "shuffle control is mandatory" learning — if TFA-style dense-channel confound is present on Venhoff pipeline, we catch it before spending 10 h on the agent loop.
- **Slice stratification**: when MATH500 has level tags, take 4 tasks from each of the 5 difficulty levels (20 total) instead of first-20. Confirm `thinking_acc ∈ [0.75, 0.85]` and `base_acc ∈ [0.15, 0.25]` pre-flight; abort if collapsed (GR denominator unstable).
- **Schema versioning**: `schema_version: 1` on every ledger row. Future schema additions stay queryable.
- **Cost-adjusted Δ**: `gap_recovery_per_gpu_minute` in ledger. A +15 pp candidate that costs 3× more isn't obviously better than a +12 pp one at baseline cost. Helpful for paper-budget promotion decisions.
- **Schema validation in summariser**: reject grade JSON with missing keys; don't silently score a partial run.
- **Pre-write the story**: before launching the agent, draft the 4–5 sentences the exploration is trying to land (e.g., "TempXC at L16 dominates L12 across all reductions; reduction operator is a second-order axis"). Focuses the search; makes a non-story obvious.

## 9. Execution plan

Phased. Each phase has a concrete deliverable + go/no-go check.

### Phase 0 — plan review (now)

- [x] Scaffold committed (`7487d64`)
- [ ] This plan shared with web-claude for review
- [ ] Incorporate feedback; finalize candidate exploration strategy

### Phase 1 — wire run_cycle.py (~2 h)

- [ ] Replace stub in `run_cycle.py` with real subprocess dispatch (pseudocode §5)
- [ ] Unit test: `baseline_sae` cycle completes on laptop with `n_tasks=2` (smoke)
- [ ] Pod test: `baseline_sae` full scrappy cycle completes in <15 min

### Phase 2 — first curated batch (~3 h)

- [ ] Hand-write YAMLs for candidates 1-8 (§6.3)
- [ ] Run each sequentially via `bash run_autoresearch.sh cand1 cand2 ... cand8`
- [ ] Populate agentic-log.md cycles 01-08 with results

### Phase 3 — agentic loop (overnight)

- [ ] Draft system prompt for Claude Code autoresearch session (exploration strategy baked in)
- [ ] Launch in tmux on pod
- [ ] Monitor for 1 hour to confirm progress rate + absence of pathological local optima
- [ ] Leave overnight; review ledger + agentic-log in the morning

### Phase 4 — FINALIST re-run + paper-budget promotion

- [ ] Re-run FINALIST candidates at n=100 on same pod
- [ ] For any that survive: run at paper budget (n=500, paper Phase 2 config) on a multi-GPU pod
- [ ] Compare against Venhoff's 3.5% + our initial TempXC result

## 10. Risks

- **Phase 0 re-run trap**: if the scrappy pod's Phase 0 output has different hash than the paper-budget pod's, candidates that shine scrappy may not replicate at paper budget. Mitigation: pin the Phase 0 artifacts to the paper-budget pod's cache (copy over once), don't regenerate.
- **Judge drift**: Haiku 4.5 is deterministic at temperature=0 but model updates happen. If the dated id rotates, scrappy ledger rows become non-comparable to paper-budget. Mitigation: pin judge to dated id in `vendor_patches.py` (already done).
- **Candidate YAML schema drift**: agent may invent keys not honored by run_cycle.py. Mitigation: validate merged config against a schema before dispatch; error loud.

## 11. Bugs-as-tests (main run lessons baked in)

The full-budget MATH500 run took ~5 iterations to get green on 2026-04-22/23. Every one of those bugs is a latent trap for the scrappy pod too — the vendor tree and transformers version are identical. Lessons + preventive measures:

### 11.1 Vendor patches (already in `src/bench/venhoff/vendor_patches.py`, auto-applied)

Fresh scrappy pods pull `aniket` branch → get all of these for free. Listed here so future-you knows *why* each exists:

| Fix | Commit | Symptom on fresh pod |
|-----|--------|----------------------|
| Byte-level BPE normalize (`Ġ`→space, `Ċ`→newline) in `responses.py` + `activation_collection.py` | `6df2ff9` | Phase 1: "no sentence activations collected" — traces saved in BPE-encoded form, sentence splitter returns nothing. |
| Force `use_fast=True`; swap `encode_plus(return_offsets_mapping=True)` → `tokenizer(...).input_ids` in `tokenization.py` | `27774e6` | `LlamaTokenizer has no attribute encode_plus` — `encode_plus` removed in modern transformers; only fast tokenizers expose offset mapping. |
| Drop `load_in_8bit=` kwarg at `optimize_steering_vectors.py` call site (Phase 2) | `12b6241` | `TypeError: LlamaForCausalLM.__init__() got an unexpected keyword argument 'load_in_8bit'` during Phase 2. |
| Drop `load_in_8bit=` kwarg at `utils/utils.py:509` `LanguageModel(...)` call (Phase 3) | `8f73b7f` | Same TypeError but during Phase 3 hybrid inference. Two different call sites, both need patching. |
| Swap `encode_plus` → `tokenizer(...)` in `utils/utils.py:243` (`get_char_to_token_map`) | `e657900` | Same `encode_plus` removal, different site. |
| Venhoff SAE shipped-vector reuse (writes `source=venhoff_shipped` sidecar; SAE Phase 2 no-ops) | `551bcb7` | SAE arch re-trains all 16 vectors from scratch instead of reading `vendor/.../optimized_vectors/llama-3.1-8b_{bias,idx0..14}.pt`. Wastes ~40 GPU-min per run. |
| Judge model id: `claude-haiku-4-5-20251001` (dated, not bare) | `8f73b7f` | `404 Not Found for url 'api.anthropic.com/v1/messages' \| model: claude-haiku-4.5` — bare name is OpenRouter-only; direct Anthropic API needs the dated id. |
| BatchEncoding `.clone()` unwrap via `hasattr(x, 'clone')` probe at `hybrid_token.py:381` | `0cbe084` | `KeyError: 'clone'` from `BatchEncoding.__getattr__` — modern transformers `apply_chat_template(return_tensors="pt")` returns BatchEncoding, not Tensor, in some configs. |
| `Patch.optional=True` for self-healing migration patches | `a87caf2` | `patch precondition failed` when fresh pod has a mix of previously-applied and new patches. Optional skip preserves idempotency. |

**Smoke test** after bootstrap — run this to verify the vendor tree is in the expected state:
```bash
cd /workspace/spar-temporal-crosscoders
.venv/bin/python -c "
from pathlib import Path
from src.bench.venhoff.vendor_patches import (
    ensure_hybrid_judge_patched, ensure_steering_patched
)
root = Path('vendor/thinking-llms-interp')
ensure_hybrid_judge_patched(root)
ensure_steering_patched(root)
print('[ok] all vendor patches applied/confirmed')
"
```

### 11.2 Pod-level traps

| Trap | Mitigation |
|------|------------|
| `/root/.cache/uv` eats container disk (17 GB after first `uv sync`) | Bootstrap deletes `~/.cache/uv` after sync. 30 GB container disk recommended (not 20). |
| HF cache on container disk (32 GB for 2× 8B models) | Export `HF_HOME=/workspace/.cache/huggingface` **before** `hf auth login` or any model download. |
| Torch inductor cache on container disk | Export `TORCHINDUCTOR_CACHE_DIR=/workspace/.cache/torchinductor`. |
| `huggingface-cli` deprecated on modern pods | Use `hf auth login`. |
| GPU orphan after subprocess crash (pkill pattern misses the actual python subprocess) | `run_autoresearch.sh` per-cycle timeout (30 min, 3× expected 10 min). On timeout: `ps -ef \| grep vendor/thinking-llms-interp/.venv \| awk '{print $2}' \| xargs -r kill -9`; wait 60 s for driver to reclaim VRAM; emit FAILED ledger row; continue to next candidate. |
| vLLM "Free memory X/80 GiB" transient after orphan kill | Wait ~60 s before relaunching; driver reclaims memory asynchronously. |
| SSH disconnect kills long-running process | Always run orchestrator inside `tmux new -s scrappy`. |
| `uv sync` fails on py3.11 with `separation-scaling` extra | `separation-scaling` block removed from `pyproject.toml` on `aniket` branch; pull before sync. |
| Model load race: vLLM + nnsight model load contend on memory | Serialize — don't run multiple cycles in parallel on a single GPU. Orchestrator is already serial per $@. |

### 11.3 What stays risky on the scrappy pod

- **New HuggingFace revision of `transformers`** between now and the scrappy run. If pip/uv pulls a newer version, new deprecations may break patches. Pin `transformers` version in `pyproject.toml`-or-vendor-venv if you want zero risk. Currently we don't pin.
- **Venhoff upstream edits to `hybrid_token.py` or `utils.py`**. Our patches use literal-string `find`. If upstream reshuffles whitespace or renames variables, patches fail at precondition. Provenance-pinned commit in `docs/aniket/experiments/venhoff_eval/VENHOFF_PROVENANCE.md`; verify on bootstrap.
- **Judge id rotation**. If Anthropic retires `claude-haiku-4-5-20251001`, patches silently start calling a 404. Mitigation: bootstrap smoke-test does a 1-prompt judge call before spending cycles.

## 12. Not in scope

- Non-MATH500 datasets (ARC-C, GSM8K, etc.). Venhoff tests a suite; we're scoped to MATH500 for the scrappy loop.
- Other model cells (Llama-70B etc.). Scoped to the one cell where Venhoff's baseline is 3.5%.
- Architectures outside {SAE, TempXC, MLC}. New archs would require first-class Python classes, not just YAML tweaks.

---

*Looking for: (a) sanity check on the noise-floor / threshold tradeoff in §7, (b) opinion on the agentic-loop fragility question in §8.5, (c) any patterns from your own autoresearch work that I'm missing.*
