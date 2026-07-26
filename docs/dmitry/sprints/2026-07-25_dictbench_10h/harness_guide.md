---
author: Claude harness agent
date: 2026-07-25
tags:
  - reference
  - in-progress
---

## Bench harness integration guide — TXC + TopK SAE on one cache, steering from decoder rows

Everything below was read out of the code and, where a shape or a norm is asserted, checked
by running it on CPU. File:line references are against the tree as of this sprint's start
([[start|sprint kickoff]]).

Reading order if you are short on time: § Five things, then § 3 (decoder rows), then § 6
(traps). §§ 1–2 are the mechanical contract.

### Five things most likely to cost you an hour

1. **`TemporalCrosscoder(d_in, d_sae, T, k)` multiplies your `k` by `T` internally**
   (`src/bench/architectures/crosscoder.py:36`). `k` is **per-position**; the realized L0 is
   `k*T` **per window**. `window_l0` in the training log is per window, not per token.
2. **The TXC's decoder is normalized as a `(T, d_in)` slab, not per position**
   (`crosscoder.py:47-50`, `norm(dim=(1,2))`). The SAE's is normalized per column
   (`topk_sae.py:43-46`, `norm(dim=0)`). So an SAE decoder column has norm 1 and a TXC
   per-position row has norm ≈ `1/sqrt(T)` (measured mean 0.442 at T=5, spread 0.25–0.62).
   Adding `α · row` to both arms at the same `α` hands the SAE a `sqrt(T)`× bigger write.
3. **Protocols A and B are numerically identical at T=5** (`saebench/configs.py:67-80`), and
   the prose in `saebench/matching_protocols.py:9-18` contradicts the code. Verified values in
   § 5. Pick T ≠ 5 or you will train two identical models and report them as a matched pair.
4. **At equal `batch_size` and equal steps, the TXC sees T× more token-activations than the
   SAE.** `gen_flat` yields `(B, d)` = B tokens; `gen_windows[T]` yields `(B, T, d)` = B·T
   token-slots (`data.py:397-421`). Nothing in the harness corrects for this. This is the
   fairness objection your review agent will raise, and it is the same class of bug as Han's
   2026-05-05 purified-sampling fix (the purified SAE was getting ~25× more tokens/step than
   the TXC it was being compared against).
5. **`CrosscoderSpec.decoder_directions()` returns a live autograd view; `TopKSAESpec`'s
   returns `.data`.** Verified: `requires_grad=True` vs `False`. Detach the TXC one or your
   steering loop silently builds a graph and `.numpy()` throws.

### 1. Data contract

#### What each `train()` pulls from `gen_fn`

Both training loops do exactly one thing with `gen_fn`, on the first line of the step:

```python
x = gen_fn(batch_size).to(device)   # crosscoder.py:120 and topk_sae.py:89
```

`gen_fn` takes **one positional argument, the batch size, and nothing else**. It is not
passed a scaling factor, a step index, or a device. The `.to(device)` is applied by the
train loop, so `gen_fn` may return CPU tensors (and the cached pipeline does — see below).

| arch | `data_format` | `gen_fn(B)` must return | dtype |
| --------------------- | --------------------------------- | ------------------------- | --------- |
| `TopKSAESpec` | `"flat"` (`topk_sae.py:74`) | `(B, d_in)` | float32 |
| `CrosscoderSpec(T)` | `"window"` (`crosscoder.py:87`) | `(B, T, d_in)` | float32 |

dtype: nothing casts for you. `TemporalCrosscoder`'s parameters are float32 by default, and
`torch.einsum` on a bf16 input against fp32 weights raises. Cast the cache to float32 —
`build_cached_activations_pipeline` already does (`data.py:376`, `.float()`).

#### What `data_format` means concretely

`data_format` is a routing string, nothing more. It selects the generator in
`sweep.py:_get_generator` (`sweep.py:84-94`) and the eval branch in
`eval.py:evaluate_model` (`eval.py:119-128`):

- `"flat"` → `pipeline.gen_flat`, batches of `(B, d)`, one independent token per row.
- `"seq"` → `pipeline.gen_seq`, batches of `(B, seq_len, d)`, whole sequences.
- `"window"` → `pipeline.gen_windows[T]`, batches of `(B, T, d)`. A **window** is `T`
  *consecutive* tokens from one sequence, sliced at stride 1 — `seqs[:, t:t+T, :]` for every
  valid `t` (`data.py:416`). Order along axis 1 is real token order; I verified nothing in
  the window path permutes it.
- `"multi_layer"` → `(B, n_layers, d)`, the MLC's layer axis. Irrelevant to this sprint.

There is **no automatic conversion between formats.** `ModelEntry.gen_key` is set by hand in
`architectures/__init__.py:42-73` ("flat" for the SAE, `f"window_{T}"` for the crosscoder) and
the pipeline builds both generators independently over the *same* underlying `train_pool`.
That is the sense in which the two archs "share one cache": the same array, two samplers.

#### Minimal `gen_fn` for a cache of shape (N, T_seq, d_model)

If you have `cache: torch.Tensor` of shape `(N, S, d)`, float32, on CPU:

```python

# SAE — one independent token per row.

def gen_flat(B: int) -> torch.Tensor:
    idx = torch.randint(0, N * S, (B,))
    return cache[idx // S, idx % S]                       # (B, d)

# TXC — B windows of T consecutive tokens, each from a uniformly random

# (sequence, start) pair. Simpler and better-mixed than data.py's version.

def gen_window(B: int, T: int = T) -> torch.Tensor:
    seq = torch.randint(0, N, (B,))
    start = torch.randint(0, S - T + 1, (B,))
    off = torch.arange(T)
    return cache[seq[:, None], start[:, None] + off[None, :]]   # (B, T, d)
```

Sixty lines is an overestimate; this is the whole thing. See § 4 for why I recommend writing
it rather than reusing.

### 2. Construction and training

#### Exact call sequence

```python
import torch
from src.bench.architectures.topk_sae import TopKSAESpec
from src.bench.architectures.crosscoder import CrosscoderSpec

device = torch.device("cuda")

sae_spec = TopKSAESpec()                 # no constructor args
txc_spec = CrosscoderSpec(T=T)           # T is the ONLY constructor arg

# create(d_in, d_sae, k, device) -> nn.Module, already .to(device)

sae = sae_spec.create(d_in=d_model, d_sae=d_sae, k=sae_k, device=device)
txc = txc_spec.create(d_in=d_model, d_sae=d_sae, k=txc_k_per_position, device=device)

# train(model, gen_fn, total_steps, batch_size, lr, device,

#       log_every=500, grad_clip=1.0,

#       plateau_pct=None, plateau_min_steps=5000) -> dict[str, list[float]]

sae_log = sae_spec.train(sae, gen_flat, total_steps=10_000, batch_size=2048,
                         lr=3e-4, device=device, log_every=500, grad_clip=1.0)
txc_log = txc_spec.train(txc, gen_window, total_steps=10_000, batch_size=2048,
                         lr=3e-4, device=device, log_every=500, grad_clip=1.0)
```

`create` is `crosscoder.py:97-98` / `topk_sae.py:76-77`; `train` is `crosscoder.py:100-162` /
`topk_sae.py:79-131`. All `train` args after `device` are keyword-with-defaults. The return
value is `{"loss": [...], "l0": [...]}` sampled at `log_every` — not per step.

Both loops hardcode `torch.optim.Adam` with no weight decay and no LR schedule
(`crosscoder.py:115`, `topk_sae.py:84`). There is no dead-latent resampling, no auxiliary
loss, and no L1 term: the loss is pure `(x_hat - x).pow(2).sum(-1).mean()`
(`crosscoder.py:71`, `topk_sae.py:66`). Note that is **sum over d, mean over batch** — for the
TXC, `.sum(dim=-1)` sums over `d_in` only and `.mean()` then averages over both `B` and `T`,
so the two archs' loss numbers are on comparable per-token scales. Their `l0` numbers are not
(see below).

`train()` calls `model.train()` at entry and `model.eval()` at exit, so the returned model is
already in eval mode.

#### What `k` means, per arch

- **`TopKSAE`**: `self.k = k` (`topk_sae.py:31`), TopK taken over the last axis of a `(B, d_sae)`
  pre-activation (`topk_sae.py:52`). **k active latents per token.** Verified: exactly `k`
  non-zeros per row.
- **`TemporalCrosscoder`**: `self.k = k * T` (`crosscoder.py:36`, comment: "Match stacked SAE's
  total L0"). The encoder sums the per-position projections into a single shared `(B, d_sae)`
  vector and takes TopK once (`crosscoder.py:54-58`). So **`k` is per-position, and `k*T`
  latents are active per window.** Verified: `TemporalCrosscoder(16, 64, T=5, k=3)` gives
  `self.k == 15` and exactly 15 non-zeros per window.

`window_l0` in the log is `(z > 0).float().sum(dim=-1).mean()` over `z` of shape `(B, d_sae)`
(`crosscoder.py:131`) — **per window**. It will read `k*T`, i.e. 500 at k=100, T=5, next to the
SAE's `l0` of 100. Those two numbers are not comparable; divide the TXC's by `T` for a
per-token figure.

The same asymmetry leaks into eval: `eval_forward` sets `n_tokens=x.shape[0]` for both
(`crosscoder.py:170`, `topk_sae.py:138`), which is `B` — the number of *windows* for the TXC.
`evaluate_model` then reports `l0 = sum_l0 / n_tokens` (`eval.py:131`), so the `l0` field of
any sweep JSON is per-window for the crosscoder and per-token for the SAE. NMSE is a ratio of
sums and is unaffected.

#### Normalization you must not fight

Both loops call `model._normalize_decoder()` **after every optimizer step**
(`crosscoder.py:127`, `topk_sae.py:96`), under `@torch.no_grad()`:

- `TopKSAE._normalize_decoder` (`topk_sae.py:43-46`): `W_dec.norm(dim=0)` on a `(d_in, d_sae)`
  matrix → every **column** is a unit vector in R^d_in.
- `TemporalCrosscoder._normalize_decoder` (`crosscoder.py:47-50`): `W_dec.norm(dim=(1,2))` on a
  `(d_sae, T, d_in)` tensor → every **`(T, d_in)` slab** has unit Frobenius norm. Individual
  per-position rows are *not* unit; their squared norms sum to 1 across the `T` positions.

Consequences you have to live with rather than undo:

- Decoder magnitude carries no information. All of a latent's scale lives in the encoder and
  therefore in `z`. A steering coefficient must come from observed `z` values or from an
  explicit energy match — never from the decoder's own norm.
- Within a TXC latent, the *relative* norms across the `T` positions **are** meaningful: they
  are the latent's temporal profile, and they are exactly what the sprint is trying to
  exploit. Do not renormalize per position; that destroys the signal.
- Re-normalizing after loading a checkpoint is a no-op (the loop already did it) but harmless.

The `_normalize_decoder` call sites are inside `train()`, so if you write your own training
loop you must call it yourself or the TopK objective is degenerate.

### 3. Decoder rows — the exact indexing

The layout, from `crosscoder.py:38-45`:

```text
TemporalCrosscoder.W_dec : (d_sae, T, d_in)     b_dec : (T, d_in)
TopKSAE.W_dec            : (d_in, d_sae)        b_dec : (d_in,)
```

#### TemporalCrosscoder

```python
CrosscoderSpec(T).decoder_directions(model, pos=p)  # -> model.decoder_directions_at(p)
model.decoder_directions_at(p)   # crosscoder.py:74-76  ==  model.W_dec[:, p, :].T
model.decoder_dirs_averaged      # crosscoder.py:78-81  ==  model.W_dec.mean(dim=1).T
```

- **Shape**: both `(d_in, d_sae)`. Orientation is `d_in × d_sae` — **columns are latents**,
  matching `ArchSpec.decoder_directions`'s documented contract (`base.py:151-164`) and
  `feature_recovery_auc`, which normalizes with `dim=0` (`eval.py:72`).
- **Unit-normalized?** **No, neither.** `decoder_directions_at(p)` columns have norm ≈
  `1/sqrt(T)` with real spread (measured at T=5: mean 0.442, min 0.251, max 0.617 on a fresh
  init; after training the spread is the temporal profile and will be much wider).
  `decoder_dirs_averaged` is worse — measured mean column norm 0.201, because averaging `T`
  rows that partially cancel shrinks them. `decoder_dirs_averaged` is a UMAP/AUC convenience
  (`eval.py:245-255` stacks the per-position ones and means them, giving the same thing) and is
  **the wrong object for steering**: it throws away the schedule, which is the entire claim.
- **`requires_grad=True`** — it is a view on the `nn.Parameter`, not `.data`. Verified.
  Always `.detach()`.

**The write pattern for latent `j` of a TXC with window `T` is a single contiguous slice:**

```python
W = model.W_dec.detach()                # (d_sae, T, d_in)
pattern_j = W[j]                        # (T, d_in)  <- the T vectors in R^d_model, in
                                        #                token order: row t is position t
```

Verified `pattern_j[p] == decoder_directions_at(p)[:, j]` exactly. `W_dec[j]` is the direct
analogue of `win_atoms[j]` in `experiments/temporal_screen/trajectory_steering/dict_modal.py:183`
(`(n_atoms, k, d)`), so the existing least-squares/energy-match code drops in with
`win_atoms → model.W_dec.detach().cpu().numpy()`.

For a set of `m` latents at once: `W[torch.as_tensor(js)]` → `(m, T, d_in)`.

Do **not** include `b_dec` in a steering write. It is the per-position reconstruction offset
(`crosscoder.py:65`), constant across all inputs, and adding it steers nothing while injecting
a large fixed vector.

#### TopKSAE

```python
TopKSAESpec().decoder_directions(model, pos=None)   # topk_sae.py:140-141

# -> model.W_dec.data, shape (d_in, d_sae)

```

- `pos` is accepted and **ignored** — the SAE has no position axis. `n_decoder_positions` is
  `None` (`base.py:191-194`), so `_get_decoder_averaged` takes the single-decoder branch.
- **Shape** `(d_in, d_sae)`, columns are latents, same orientation as the TXC's.
- **Unit-normalized? Yes** — columns have norm exactly 1.0, both at init (`topk_sae.py:40-41`)
  and after every training step. Verified min = max = 1.0000.
- Returns `.data`, so `requires_grad=False` already. Asymmetric with the TXC; don't rely on it
  in shared code.

Direction for latent `j`: `model.W_dec.data[:, j]` → `(d_in,)`, the analogue of
`tok_atoms[j]` in `dict_modal.py:178`.

Note the SAE's encoder subtracts `b_dec` before projecting (`topk_sae.py:49`) while the TXC's
does not (`crosscoder.py:54`). Only matters if you read coefficients back by encoding; it does
not affect the decoder rows.

### 4. Existing activation-caching code

Two candidates. **My recommendation: write your own, using the ~15 lines in § 1.** Reasons
below, then what to reuse anyway.

#### `venhoff/activation_collection.py` `collect_path3` — do not use

It does preserve the temporal axis (`activation_collection.py:276-371`, output
`(N_sentences, T, d_model)`), but it is not a general activation cacher — it is a
*sentence-window* cacher for reasoning traces, and every one of its assumptions is wrong for
this sprint:

- **Input is a traces JSON**, read from `paths.traces_json` (`line 304-305`), whose entries must
  have a `full_response` containing a `<think>` block; `extract_thinking_process` returns empty
  and the sentence yields nothing otherwise (`line 91-92`). Your semisynthetic carrier
  documents have no `<think>` block, so you would get zero windows.
- **One window per *sentence*, centered on the sentence midpoint** (`line 326-331`):
  `mid = (token_start + token_end) // 2`, `win_start = max(0, mid - T//2)`. That is a
  sentence-anchored sample, not the stride-1 window stream the TXC trains on, and it
  right-aligns near the end and zero-pads short traces (`line 333-337`) — zero rows in a
  training batch are a silent NMSE distortion.
- **Output is a pickle, not `.npy`**: `pickle.dump((windows_np, sentence_texts), ...)` at
  `line 352-354`, written to `paths.activations_pkl("path3")`.
  `build_cached_activations_pipeline` expects `<root>/<layer_key>.npy` plus a
  `layer_specs.json` sidecar (`data.py:355-372`). **You cannot point the pipeline at Path 3
  output** without a conversion step.
- It needs a `ModelConfig` from the registry (`line 307`) — and there is no Qwen entry (§ 6).

#### `temporal_crosscoders/NLP/cache_activations.py` — the thing `build_cached_activations_pipeline` actually expects

This is the producer for the `.npy` contract: `<key>.npy` of shape `(N, seq_len, d_model)`
float32, plus `layer_specs.json` and `token_ids.npy`, under
`cache_dir_for(model, dataset)` = `data/cached_activations/<model>/<dataset>/`
(`temporal_crosscoders/NLP/config.py:120-121`). Layer keys are `f"{component}_L{idx}"`, e.g.
`resid_L14` (`NLP/config.py:66`). If you want `build_cached_activations_pipeline` and
`run_cached_sweep` for free, produce that layout — writing an `(N, S, d)` float32 array to
`resid_L14.npy` and a `{"d_model": 1536}` JSON is enough for the loader.

But even then the pipeline's own generators have properties you probably don't want (§ 6:
whole-cache RAM load, only ~74 distinct sequences per batch, per-window correlation). For a
10-hour sprint where you control the corpus and want the mix ratio to be a reported parameter,
the two closures in § 1 are less code than the config plumbing to reach the same place.

#### What you *should* reuse

`experiments/temporal_screen/trajectory_steering/dict_modal.py` is the same experiment with
PCA dictionaries in place of trained ones, and the whole eval loop transfers:

- `steer_hook` / `cap_hook` and the `hs[:, a:b+1, :] += vec` residual write (`lines 90-101`),
- span→token mapping via `return_offsets_mapping` (`lines 114-121`),
- teacher-forced segment log-prob and the multiset-matched foil margin (`lines 123-151`,
  `246-259`),
- the energy match `sc = ||T|| / ||R||` (`line 286`) — which is exactly what neutralizes the
  decoder-normalization asymmetry in trap #2,
- the `fracs × base_norm` magnitude sweep with peak selection (`lines 261-297`).

The swap is `win_atoms[:m]` → `model.W_dec.detach().cpu().numpy()[js]` and `tok_atoms[:p]` →
`sae.W_dec.data.T.cpu().numpy()[js]`. Note `dict_modal.py` picks atoms by PCA rank (`[:m]`);
you need a *stated, matched* selection rule instead (the kickoff commits to this), and the
SAE arm there spends `m // k` atoms per segment (`line 226`) — that budget convention is the
load-bearing choice in the comparison and should be re-derived, not inherited by accident.

### 5. Sparsity matching, as actually configured

`PROTOCOL_A = MatchingProtocol(name="A", sae_k=100, mlc_k=100, tempxc_k_base=100)` and
`PROTOCOL_B = MatchingProtocol(name="B", sae_k=100, mlc_k=100, tempxc_k_base=500)`
(`saebench/configs.py:83-84`). `protocol_k(arch, protocol, t)` (`matching_protocols.py:48-65`)
returns `tempxc_k_at(t)` for `tempxc`, which is (`configs.py:67-80`):

- **A**: `return tempxc_k_base` → per-position `k` is **always 100**, so window budget `= 100·T`
  grows with T.
- **B**: `return max(1, tempxc_k_base // t)` → per-position `k = 500 // T`, so window budget is
  **pinned near 500** for all T.

**The returned value is the per-position `k` you pass to `create()`, and the crosscoder
multiplies it by T.** Verified values:

| T | A: per-pos k | A: window k | B: per-pos k | B: window k | SAE k |
| ---- | -------------- | ------------- | -------------- | ------------- | ------- |
| 2 | 100 | 200 | 250 | 500 | 100 |
| 5 | 100 | **500** | 100 | **500** | 100 |
| 8 | 100 | 800 | 62 | 496 | 100 |
| 10 | 100 | 1000 | 50 | 500 | 100 |
| 12 | 100 | 1200 | 41 | 492 | 100 |

Things to get right:

- **A and B are the same model at T=5.** `configs.py:77` says so ("At T=5, B coincides with A
  by design"). The kickoff's "both will be run" only produces two distinct arms at T ≠ 5.
- **The prose in `matching_protocols.py:9-18` is stale and wrong.** It claims A gives "TempXC
  k=100 … effectively 5× sparser in total-activation terms at T=5", which would be true only if
  `k=100` meant a window budget of 100. It does not — `crosscoder.py:36` multiplies by T. Under
  the code, A is per-*token* matched (100 per token for both archs) and B is per-*window*
  matched (500 per window for the TXC vs 100 per token → 500 per 5-token window for the SAE),
  which at T=5 is the same constraint. Trust `configs.py`; the docstring at `configs.py:67-80`
  agrees with the implementation.
- `max(1, ...)` floors B at k=1, so at T > 500 protocol B silently stops being budget-matched.
  Not a risk at your T.
- `topk(self.k)` requires `k*T <= d_sae`. At Qwen-1.5B `d_model=1536` and ×8 expansion,
  `d_sae=12288`; protocol A at T=12 needs 1200. Fine, but if you shrink `d_sae` for the smoke
  test, shrink `k` too or `topk` raises.
- `regressions.py:77-92` (check B4) asserts protocol invariants against Gemma's
  `D_SAE = 18432`. It will still pass, but it is not checking *your* `d_sae`.

### 6. Traps

Ordered roughly by how much time each would cost.

#### Sparsity and budget

- `k → k*T` inside `TemporalCrosscoder.__init__` (`crosscoder.py:36`). Passing `k=500`
  "because protocol B says 500" gives you `window_k = 2500` at T=5. Pass what `protocol_k`
  returns, unmodified.
- `window_l0` vs `l0` in the logs, and `l0` in every sweep JSON, are per-window for the TXC and
  per-token for the SAE (`crosscoder.py:131`, `eval.py:131` with `n_tokens = x.shape[0]` at
  `crosscoder.py:170`). Any table that puts them in the same column is wrong by a factor of T.

#### Training-data budget

- **T× token asymmetry.** `gen_flat(B)` returns B tokens; `gen_windows[T](B)` returns B·T
  token-slots. Same `batch_size`, same `total_steps` ⇒ the TXC has consumed T× more activation
  vectors. If your headline is "matched budget", either match token-slots (`sae_batch = B*T`)
  or report the discrepancy explicitly. `run_cached_sweep` does not correct for it
  (`sweep.py:314-320` reads one `batch_size` for all entries).
- Windows within a batch **overlap heavily**. `data.py:411-421` draws only
  `n = B // (S - T + 1) + 1` distinct sequences per step and emits every stride-1 window of
  each. At S=32, T=5, B=2048 that is 74 sequences producing 2072 windows — consecutive windows
  share T−1 of T tokens. Effective batch diversity is ~74, not 2048. My § 1 `gen_window` draws
  independent `(sequence, start)` pairs and avoids this.
- `_shuffle_within_sequence_` (`data.py:319-329`) fires on the **whole cache including the eval
  slice** when `DataConfig.shuffle_within_sequence=True`, at load time, unseeded. That is the
  temporal-control knob; leave it False unless you are deliberately running the shuffled arm.
  It is a different function from `base.py:shuffle_within_sequence` (seeded, probing-time) —
  `base.py:34-38` explicitly warns about the pair. Neither is invoked by the training path
  unless you ask for it, and nothing in the window generators reorders within a window (I
  checked `unfold` and the `cat` bank; both preserve token order).

#### Decoder and steering magnitude

- The `sqrt(T)` norm asymmetry (trap #2 at the top). Concretely: at T=5 a fresh TXC's
  per-position rows average 0.442 while the SAE's columns are exactly 1.0. Energy-match the
  assembled write (`dict_modal.py:286`) or normalize per arm before applying `α`; never compare
  raw `α`.
- `decoder_dirs_averaged` shrinks columns further (measured 0.201 vs 0.442) and discards the
  schedule. Use `decoder_directions_at(p)` or `W_dec[j]`, never the average, for steering.
- `CrosscoderSpec.decoder_directions` hands back an autograd view (`requires_grad=True`);
  `TopKSAESpec.decoder_directions` hands back `.data`. Detach both defensively.
- `b_dec` is not part of any latent's write: TXC `(T, d_in)` at `crosscoder.py:45`, SAE
  `(d_in,)` at `topk_sae.py:33`. Excluding it is correct; including it injects a constant.
- Decoder norms are pinned every step, so decoder magnitude cannot tell you how strongly a
  latent normally writes. If you want a natural scale, take it from observed `z` on real data.

#### Off-by-ones and alignment

- `activation_collection.py:224` (`acts[:, token_start - 1 : token_end, :]`) preserves
  Venhoff's `-1` deliberately; the module docstring flags it as a hazard (`lines 5-9`) and
  `_iter_sentence_spans` returns raw offsets with no `-1` applied (`lines 84-86`). If you touch
  Path 1 or Path MLC, the `-1` is intentional. Path 3 does *not* apply it.
- The steering hook in `dict_modal.py:148` writes at `(a-1, b-1)` — one token left of the
  segment span — because a residual write at position `p` affects the prediction of token
  `p+1`. Keep that shift; dropping it moves your whole intervention one token late and quietly
  halves the effect.
- Path 3 zero-pads windows that run past the end of a trace
  (`activation_collection.py:334-336`). Zero rows are not activations.

#### Devices, dtypes, memory

- `gen_fn` may return CPU tensors; `train()` calls `.to(device)` (`crosscoder.py:120`). But
  `eval_forward` does **not** move anything — `evaluate_model`'s helpers do
  (`eval.py:263`, `283`). If you call `spec.eval_forward` directly, move the batch yourself.
- `build_cached_activations_pipeline` does `np.asarray(arr)` on the mmap
  (`data.py:376`), i.e. **loads the entire cache into RAM as float32**, then `.float()` again.
  The `mmap_mode="r"` on line 364 buys you nothing. For (24000, 32, 1536) that is ~4.7 GB;
  size accordingly.
- The eval slice is the **last** `n_eval` sequences, clamped to 20% of the cache
  (`data.py:385-388`). If your cache is written task-first-then-general-text, the held-out
  slice is 100% general text. Interleave when you build it.
- `CrosscoderSpec.encode` materializes a `(chunk, T, d_sae)` tensor and runs an assertion over
  a `B×T×d_sae` boolean mask (`crosscoder.py:237-248`). Set `CROSSCODER_SKIP_MASK_CHECK=1` if
  you call `encode` in a loop. Also note `encode` is *not* the native `(B, d_sae)` forward —
  it is the per-position masked contribution, deliberately (`crosscoder.py:186-196`). For
  reading coefficients to steer with, you want `model.encode(x)` (the module method,
  `crosscoder.py:52-61`), not `spec.encode(model, x)`.

#### Registry and CLI

- **There is no Qwen entry in `MODEL_REGISTRY`** (`model_registry.py:31-72` — only DeepSeek-8B,
  Llama-3.1-8B, Gemma-2-2b, Gemma-2-2b-it). `get_model_config("qwen2.5-1.5b-instruct")` raises,
  and `sweep.py:556-558` uses `choices=list_models()` so argparse rejects the name outright.
  Adding an entry is ~10 lines and `resid_hook_target` already handles `family="qwen"` via the
  `model.model.layers[i]` path (`model_registry.py:91-94`). Qwen-2.5-1.5B: `d_model=1536`,
  `n_layers=28`. Do this first if you intend to use any registry-backed code path.
- `run_cached_sweep` writes checkpoints as bare `model.state_dict()` (`sweep.py:409-419`) with
  **no config in the file**. Nothing records `T`, `k`, or `d_sae` except the filename. Save a
  JSON sidecar; reconstructing `TemporalCrosscoder(d_in, d_sae, T, k)` to load a state dict
  requires all four, and getting `k` wrong loads cleanly while changing the sparsity.
- `--stop-on-plateau` compares means of 4 *single-batch* logged losses against the previous 4
  (`plateau.py:49-56`), so it cannot fire before `8 × log_every` steps *and* `min_steps=5000`.
  Harmless, but don't expect it to save time on a short run.
- `run_eval.py` is the *probing* entry point (`--architecture {sae,tempxc,mlc}`), routed to
  SAEBench sparse probing. It is not a steering harness and has nothing you need this sprint
  beyond the regression gate.

### Related

- [[start]] — sprint kickoff and locked design decisions
- `src/bench/architectures/README.md` — how to register a new arch (only needed if you subclass)
- `docs/dmitry/c6_em/2026-05-07_em_repl/lessons_learned.md` — the "delegate to the reference's
  function" rule, which is why § 4 recommends reusing `dict_modal.py`'s eval loop rather than
  reimplementing it
