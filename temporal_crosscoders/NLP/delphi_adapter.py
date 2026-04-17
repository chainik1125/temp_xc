"""delphi_adapter.py — Bridge our custom SAE / crosscoder architectures into
EleutherAI's `delphi` auto-interp pipeline (Paulo et al., ICLR 2025).

The key value here is `precomputed_to_safetensors`: delphi's default
caching stage re-runs the subject transformer through its hook API, but
our activation caches are already on disk as (n_seq, seq_len, d_model)
float32 .npy files. This function walks those caches, applies the
architecture's `encode()` directly, and writes sharded safetensors in
the exact format `delphi.latents.LatentDataset` expects:

    {
        activations: Float[N]      # nonzero latent activation values
        locations:   Int[N, 3]     # (batch, seq, latent_id) per nonzero
        tokens:      Int[B, S]     # the token ids for each (batch, seq)
    }

Downstream we hand the directory to `LatentDataset(raw_dir=...)` and
run the standard Pipeline(DefaultExplainer, DetectionScorer, ...).

Architecture support:
  - TopKSAE: applies model.encode per-token, flattens B×T to batch axis.
  - StackedSAE T=5: per-position encode, concatenates positions along
    the seq axis of the flattened (B, T) layout.
  - TemporalCrosscoder: uses spec.encode which returns (B, T, d_sae)
    per our Option B (per-position contributions masked by shared-z TopK).

See `docs/aniket/sprint_coding_dataset_plan.md` § Encode contract for
the rationale behind the uniform (B, T, d_sae) shape.

Usage:
    from temporal_crosscoders.NLP.delphi_adapter import precomputed_to_safetensors

    precomputed_to_safetensors(
        checkpoint="results/nlp/step1-unshuffled/ckpts/crosscoder__...pt",
        arch="crosscoder",
        subject_model="gemma-2-2b",
        cached_dataset="fineweb",
        layer_key="resid_L13",
        k=100, T=5, expansion_factor=8,
        out_dir="reports/step1-gemma-replication/delphi/step1-unshuffled__crosscoder",
        n_splits=5,
    )
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from src.bench.model_registry import get_model_config
from src.bench.architectures.topk_sae import TopKSAESpec
from src.bench.architectures.stacked_sae import StackedSAESpec
from src.bench.architectures.crosscoder import CrosscoderSpec
from temporal_crosscoders.NLP.config import cache_dir_for


ARCH_SPECS = {
    "topk_sae": lambda T: TopKSAESpec(),
    "stacked_sae": lambda T: StackedSAESpec(T=T),
    "crosscoder": lambda T: CrosscoderSpec(T=T),
}


def _load_arch(
    checkpoint: str,
    arch: str,
    subject_model: str,
    k: int,
    T: int,
    expansion_factor: int,
    device: torch.device,
):
    """Load a trained checkpoint via the spec-based architecture classes
    that already implement `encode()`. Same pattern as
    scripts/compute_temporal_metrics.py.
    """
    cfg = get_model_config(subject_model)
    d_in = cfg.d_model
    d_sae = d_in * expansion_factor
    spec = ARCH_SPECS[arch](T)
    model = spec.create(d_in=d_in, d_sae=d_sae, k=k, device=device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return spec, model


def _collect_nonzeros_from_window(
    z_window: torch.Tensor,       # (B, T, d_sae)
    window_start: int,            # offset of this window inside the sequence
    batch_offsets: torch.Tensor,  # (B,) — the global batch index for each row
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract nonzero (activation, (batch, seq, latent)) records from a
    window of encoded activations.

    Returns (values, locations) where values is (N,) float32 and
    locations is (N, 3) int64 with columns (batch_idx, seq_pos, latent_idx).
    """
    B, T, d_sae = z_window.shape
    nonzero = z_window.nonzero(as_tuple=False)  # (N, 3): (b_local, t_local, latent)
    if nonzero.shape[0] == 0:
        return (
            torch.empty(0, dtype=torch.float32),
            torch.empty(0, 3, dtype=torch.int64),
        )

    values = z_window[nonzero[:, 0], nonzero[:, 1], nonzero[:, 2]]

    global_batch = batch_offsets[nonzero[:, 0].cpu()]
    global_seq = nonzero[:, 1] + window_start
    latent_idx = nonzero[:, 2]
    locations = torch.stack(
        [global_batch.to(torch.int64),
         global_seq.to(torch.int64).cpu(),
         latent_idx.to(torch.int64).cpu()],
        dim=1,
    )
    return values.to(torch.float32).cpu(), locations


def precomputed_to_safetensors(
    checkpoint: str,
    arch: str,
    subject_model: str,
    cached_dataset: str,
    layer_key: str,
    k: int,
    T: int,
    expansion_factor: int,
    out_dir: str,
    n_splits: int = 5,
    max_sequences: int | None = None,
    device: str = "auto",
    window_stride: int = 1,
) -> Path:
    """Run our custom encode() over the cached activations and emit delphi-
    compatible sharded safetensors.

    Each shard is a dict of three tensors:
        activations: (N,) float32        nonzero latent values
        locations:   (N, 3) int64        (batch, seq, latent)
        tokens:      (B, seq_len) int64  per-sequence token ids

    `n_splits` controls how many feature-axis shards to produce — delphi's
    LatentDataset sweeps them in order. Splitting by feature range keeps
    per-shard memory manageable for large d_sae.

    Returns the output directory path.
    """
    from safetensors.torch import save_file

    torch_device = torch.device(
        "cuda" if (device == "auto" and torch.cuda.is_available())
        else ("cpu" if device == "auto" else device)
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cache_dir = cache_dir_for(subject_model, cached_dataset)
    act_path = os.path.join(cache_dir, f"{layer_key}.npy")
    token_path = os.path.join(cache_dir, "token_ids.npy")
    if not os.path.exists(act_path):
        raise FileNotFoundError(f"activation cache not found: {act_path}")
    if not os.path.exists(token_path):
        raise FileNotFoundError(
            f"token_ids.npy not found (required by delphi): {token_path}"
        )

    acts_mm = np.load(act_path, mmap_mode="r")   # (n_seq, seq_len, d_model)
    token_ids = np.load(token_path)               # (n_seq, seq_len) int
    n_seq, seq_len, d_model = acts_mm.shape
    if max_sequences is not None:
        n_seq = min(n_seq, max_sequences)

    print(f"delphi_adapter: acts={acts_mm.shape[:2]} seq_len={seq_len} "
          f"d_model={d_model}")

    spec, model = _load_arch(
        checkpoint, arch, subject_model, k, T, expansion_factor, torch_device,
    )
    d_sae = model.d_sae
    print(f"  arch={arch}  d_sae={d_sae}  T={T}")

    # Accumulate per-split buffers. We partition the feature axis into
    # n_splits equal chunks; each shard covers its own latent range.
    chunk = max(1, (d_sae + n_splits - 1) // n_splits)
    per_split_acts: list[list[torch.Tensor]] = [[] for _ in range(n_splits)]
    per_split_locs: list[list[torch.Tensor]] = [[] for _ in range(n_splits)]

    batch_size = 32
    n_windows = seq_len - T + 1

    for batch_start in tqdm(range(0, n_seq, batch_size), desc="encode"):
        batch_end = min(batch_start + batch_size, n_seq)
        B = batch_end - batch_start

        chains_np = np.stack([acts_mm[i].copy() for i in range(batch_start, batch_end)])
        chains = torch.from_numpy(chains_np).float().to(torch_device)

        # Build all length-T sliding windows: (B, n_windows, T, d_model)
        windows = chains.unfold(1, T, window_stride).permute(0, 1, 3, 2).contiguous()
        flat = windows.reshape(B * n_windows, T, d_model)

        batch_offsets = torch.arange(batch_start, batch_end, dtype=torch.int64)

        with torch.no_grad():
            z = spec.encode(model, flat)  # (B*n_windows, T, d_sae)

        # Reassemble to (B, n_windows, T, d_sae) for per-window accounting
        z = z.view(B, n_windows, T, d_sae)

        # For each window, emit nonzero records referring to the CENTRE
        # position of that window. This matches the common "a feature at
        # position t" convention — delphi expects one (batch, seq) per
        # activation event, not T of them per window.
        centre = T // 2
        for w in range(n_windows):
            z_centre = z[:, w, centre, :]                   # (B, d_sae)
            nonzero = z_centre.nonzero(as_tuple=False)       # (N, 2)
            if nonzero.shape[0] == 0:
                continue
            values = z_centre[nonzero[:, 0], nonzero[:, 1]]
            global_batch = batch_offsets[nonzero[:, 0].cpu()].to(torch.int64)
            global_seq = torch.full_like(global_batch, w + centre)
            latent_idx = nonzero[:, 1].to(torch.int64).cpu()

            for split_id in range(n_splits):
                lo, hi = split_id * chunk, min((split_id + 1) * chunk, d_sae)
                mask = (latent_idx >= lo) & (latent_idx < hi)
                if not mask.any():
                    continue
                m_latent = latent_idx[mask] - lo   # remap to shard-local latent id
                m_vals = values.cpu()[mask].to(torch.float32)
                m_batch = global_batch[mask]
                m_seq = global_seq[mask]
                m_loc = torch.stack([m_batch, m_seq, m_latent], dim=1)
                per_split_acts[split_id].append(m_vals)
                per_split_locs[split_id].append(m_loc)

    # Write one safetensors file per shard. Include the shared tokens
    # tensor in each file — delphi's loader reads it from every shard
    # independently.
    tokens_t = torch.from_numpy(token_ids[:n_seq]).to(torch.int64)

    for split_id in range(n_splits):
        acts = (torch.cat(per_split_acts[split_id])
                if per_split_acts[split_id]
                else torch.empty(0, dtype=torch.float32))
        locs = (torch.cat(per_split_locs[split_id])
                if per_split_locs[split_id]
                else torch.empty(0, 3, dtype=torch.int64))
        out_file = out_path / f"latents_{split_id:05d}.safetensors"
        save_file(
            {
                "activations": acts,
                "locations": locs,
                "tokens": tokens_t,
            },
            str(out_file),
            metadata={
                "arch": arch,
                "layer_key": layer_key,
                "shard_id": str(split_id),
                "n_shards": str(n_splits),
                "latent_range_lo": str(split_id * chunk),
                "latent_range_hi": str(min((split_id + 1) * chunk, d_sae)),
                "d_sae": str(d_sae),
                "T": str(T),
            },
        )
        print(f"  wrote {out_file}  ({acts.numel()} nonzeros)")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out_path


# ═══════════════════════════════════════════════════════════════════════════
# Custom Anthropic client for delphi's Explainer/Scorer pipes
# ═══════════════════════════════════════════════════════════════════════════

class AnthropicClient:
    """Minimal `delphi.clients.Client`-compatible wrapper around the
    Anthropic SDK. Mirrors `temporal_crosscoders.NLP.autointerp.ClaudeAPIBackend`
    so we reuse the same retry / semaphore pattern.

    Delphi's Client ABC (as of the current main branch) exposes:
        async def generate(prompt: str, **kwargs) -> str

    with chat-style kwargs (max_tokens, temperature, ...). This subclass
    routes those to `client.messages.create`.
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        max_concurrent: int = 5,
        max_retries: int = 3,
        default_max_tokens: int = 512,
    ):
        import anthropic
        import asyncio
        self._client = anthropic.AsyncAnthropic()
        self.model = model
        self.max_retries = max_retries
        self.default_max_tokens = default_max_tokens
        self._sem = asyncio.Semaphore(max_concurrent)
        self.n_calls = 0
        self.n_errors = 0

    async def generate(self, prompt: str, **kwargs) -> str:
        """Async single-turn generation. Accepts `system`, `max_tokens`,
        `temperature` in kwargs, matching delphi's conventions. Uses
        the module-level asyncio semaphore for concurrency control and
        retries transient errors with exponential backoff.
        """
        import asyncio
        import random

        system = kwargs.get("system", "")
        max_tokens = kwargs.get("max_tokens", self.default_max_tokens)
        temperature = kwargs.get("temperature", 0.0)

        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                async with self._sem:
                    resp = await self._client.messages.create(
                        model=self.model,
                        max_tokens=max_tokens,
                        system=system,
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    self.n_calls += 1
                    return resp.content[0].text
            except Exception as e:
                self.n_errors += 1
                if attempt == self.max_retries - 1:
                    return ""
                await asyncio.sleep(delay + random.uniform(0, 0.5))
                delay = min(delay * 2, 60.0)
        return ""

    # Alias so it works if delphi expects `.call` like our autointerp backend
    async def call(self, system: str, user: str) -> str:
        return await self.generate(user, system=system)
