"""Rolling activation buffer fed by on-the-fly Qwen-7B forward passes.

Stores activations as (N_seqs, chunk_len, d_model) so we can pull T-token
windows with guaranteed in-sequence contiguity for TXC training.

Refills when the fraction of already-sampled slots crosses
``refill_threshold``. Each refill runs one Qwen forward over
``refill_chunks`` newly-tokenized sequences and overwrites a random subset of
the buffer (keeping the buffer mixed rather than FIFO-biased).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class StreamingBufferConfig:
    layer: int
    d_model: int
    buffer_seqs: int = 4000
    chunk_len: int = 256
    refill_chunks: int = 64
    refill_threshold: float = 0.5
    device: str = "cuda"
    dtype: torch.dtype = torch.float16


class StreamingActivationBuffer:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        text_iter: Iterator[str],
        config: StreamingBufferConfig,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.text_iter = text_iter
        self.cfg = config

        self._buf = torch.empty(
            config.buffer_seqs, config.chunk_len, config.d_model,
            device=config.device, dtype=config.dtype,
        )
        # How many times each sequence slot has been sampled since last refill.
        # Used only to drive the refill_threshold heuristic.
        self._samples_drawn = 0
        self._filled = False

    @torch.no_grad()
    def _tokenize_batch(self, n: int) -> torch.Tensor:
        texts: list[str] = []
        for _ in range(n):
            texts.append(next(self.text_iter))
        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.cfg.chunk_len,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return enc.input_ids.to(self.cfg.device)

    @torch.no_grad()
    def _forward_and_extract(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the model and extract layer-`layer` resid_post activations.

        Uses ``output_hidden_states`` — hidden_states[L+1] is the residual
        stream after transformer block L (the `resid_post` of layer L).
        """
        out = self.model(
            input_ids=input_ids,
            output_hidden_states=True,
            use_cache=False,
        )
        hs = out.hidden_states[self.cfg.layer + 1]  # (B, chunk_len, d_model)
        return hs.to(self.cfg.dtype)

    @torch.no_grad()
    def _refill(self, n_seqs: int) -> None:
        """Overwrite ``n_seqs`` random slots with fresh activations."""
        remaining = n_seqs
        while remaining > 0:
            b = min(self.cfg.refill_chunks, remaining)
            ids = self._tokenize_batch(b)
            acts = self._forward_and_extract(ids)
            if not self._filled:
                start = self.cfg.buffer_seqs - remaining
                self._buf[start:start + b] = acts
            else:
                idx = torch.randint(
                    self.cfg.buffer_seqs, (b,), device=self.cfg.device,
                )
                self._buf[idx] = acts
            remaining -= b
        if not self._filled:
            self._filled = True
        self._samples_drawn = 0

    @torch.no_grad()
    def warmup(self) -> None:
        """Initial full fill. Call once before training starts."""
        self._refill(self.cfg.buffer_seqs)

    def _maybe_refill(self) -> None:
        frac = self._samples_drawn / max(1, self.cfg.buffer_seqs)
        if frac >= self.cfg.refill_threshold:
            self._refill(max(1, self.cfg.refill_chunks))

    @torch.no_grad()
    def sample_txc_windows(self, batch_size: int, T: int) -> torch.Tensor:
        """Return ``(batch_size, T, d_model)`` with each window taken from a
        random sequence at a random in-sequence start position."""
        if not self._filled:
            raise RuntimeError("call warmup() before sampling")
        self._maybe_refill()
        if T > self.cfg.chunk_len:
            raise ValueError(f"T={T} > chunk_len={self.cfg.chunk_len}")
        device = self.cfg.device
        seq_idx = torch.randint(self.cfg.buffer_seqs, (batch_size,), device=device)
        start_idx = torch.randint(self.cfg.chunk_len - T + 1, (batch_size,), device=device)
        # Gather: result[b, t, :] = self._buf[seq_idx[b], start_idx[b] + t, :]
        pos = start_idx.unsqueeze(-1) + torch.arange(T, device=device).unsqueeze(0)  # (B, T)
        seqs = self._buf[seq_idx]  # (B, chunk_len, d_model)
        gather_idx = pos.unsqueeze(-1).expand(-1, -1, self.cfg.d_model)  # (B, T, d_model)
        windows = seqs.gather(1, gather_idx)  # (B, T, d_model)
        self._samples_drawn += batch_size
        return windows.contiguous()

    @torch.no_grad()
    def sample_flat(self, batch_size: int) -> torch.Tensor:
        """Return ``(batch_size, d_model)`` samples (for baseline SAE training)."""
        return self.sample_txc_windows(batch_size, T=1).squeeze(1)


# ---------------------------------------------------------------------------
# Multi-layer buffer (for MultiLayerCrosscoder training). Same refill logic,
# but stores per-layer activations and samples one-token-across-layers batches
# instead of T-token single-layer windows.
# ---------------------------------------------------------------------------


@dataclass
class MultiLayerBufferConfig:
    layers: list[int]
    d_model: int
    buffer_seqs: int = 800
    chunk_len: int = 256
    refill_chunks: int = 64
    refill_threshold: float = 0.5
    device: str = "cuda"
    dtype: torch.dtype = torch.float16


class MultiLayerStreamingBuffer:
    """Rolling buffer of residual activations at multiple layers.

    Shape: ``(buffer_seqs, chunk_len, n_layers, d_model)``.
    For a d_model=3584 Qwen-7B with 5 layers and 800 seqs × 256 toks × fp16
    → ~7 GB VRAM (same order as the single-layer buffer at 4 k seqs).
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        text_iter: Iterator[str],
        config: MultiLayerBufferConfig,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.text_iter = text_iter
        self.cfg = config
        self.n_layers = len(config.layers)
        self._buf = torch.empty(
            config.buffer_seqs, config.chunk_len, self.n_layers, config.d_model,
            device=config.device, dtype=config.dtype,
        )
        self._samples_drawn = 0
        self._filled = False

    @torch.no_grad()
    def _tokenize_batch(self, n: int) -> torch.Tensor:
        texts: list[str] = []
        for _ in range(n):
            texts.append(next(self.text_iter))
        enc = self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.cfg.chunk_len, return_tensors="pt", add_special_tokens=True,
        )
        return enc.input_ids.to(self.cfg.device)

    @torch.no_grad()
    def _forward_and_extract(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run the model and stack residuals at each configured layer.

        hidden_states[L+1] is `resid_post` after block L.
        """
        out = self.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        per_layer = [out.hidden_states[L + 1] for L in self.cfg.layers]  # each (B, T, d)
        stacked = torch.stack(per_layer, dim=2)  # (B, T, n_layers, d)
        return stacked.to(self.cfg.dtype)

    @torch.no_grad()
    def _refill(self, n_seqs: int) -> None:
        remaining = n_seqs
        while remaining > 0:
            b = min(self.cfg.refill_chunks, remaining)
            ids = self._tokenize_batch(b)
            acts = self._forward_and_extract(ids)
            if not self._filled:
                start = self.cfg.buffer_seqs - remaining
                self._buf[start:start + b] = acts
            else:
                idx = torch.randint(self.cfg.buffer_seqs, (b,), device=self.cfg.device)
                self._buf[idx] = acts
            remaining -= b
        if not self._filled:
            self._filled = True
        self._samples_drawn = 0

    @torch.no_grad()
    def warmup(self) -> None:
        self._refill(self.cfg.buffer_seqs)

    def _maybe_refill(self) -> None:
        frac = self._samples_drawn / max(1, self.cfg.buffer_seqs)
        if frac >= self.cfg.refill_threshold:
            self._refill(max(1, self.cfg.refill_chunks))

    @torch.no_grad()
    def sample_mlc_batch(self, batch_size: int) -> torch.Tensor:
        """Return ``(batch_size, n_layers, d_model)`` — one token per sample,
        its activations across the configured layers.
        """
        if not self._filled:
            raise RuntimeError("call warmup() before sampling")
        self._maybe_refill()
        seq_idx = torch.randint(self.cfg.buffer_seqs, (batch_size,), device=self.cfg.device)
        pos_idx = torch.randint(self.cfg.chunk_len, (batch_size,), device=self.cfg.device)
        batch = self._buf[seq_idx, pos_idx]  # (B, n_layers, d_model)
        self._samples_drawn += batch_size
        return batch.contiguous()


# ---------------------------------------------------------------------------
# Hookpoint-aware extraction (resid_pre / resid_post / resid_mid /
# ln1_normalized) for training SAEs at non-resid_post hookpoints.
# ---------------------------------------------------------------------------


VALID_HOOKPOINTS = ("resid_pre", "resid_post", "resid_mid", "ln1_normalized")


class HookpointExtractor:
    """Captures activations at a named transformer-block hookpoint during a
    forward pass. Intended for use with Qwen2 / LLaMA-style HF architectures.

    Supported ``kind`` values:
      - ``resid_post`` (default): output of the full block. Equal to
        ``hidden_states[layer+1]`` in ``output_hidden_states``.
      - ``resid_pre``: input to the block. Equal to ``hidden_states[layer]``.
      - ``resid_mid``: residual after attention, before MLP — captured as the
        input to ``post_attention_layernorm`` via a forward-pre-hook.
      - ``ln1_normalized``: output of ``input_layernorm`` — what attention
        sees after LN (TransformerLens's ``ln_1_hook_normalized``).

    The extractor must be entered as a context manager. After ``model(...)``
    returns, ``self.captured`` holds a fresh tensor of shape ``(B, T, d)``.
    """

    def __init__(self, model: PreTrainedModel, kind: str, layer: int):
        if kind not in VALID_HOOKPOINTS:
            raise ValueError(f"kind must be one of {VALID_HOOKPOINTS}, got {kind}")
        self.model = model
        self.kind = kind
        self.layer = layer
        self.captured: torch.Tensor | None = None
        self._handles: list = []

    def _get_block(self):
        return self.model.model.layers[self.layer]

    def _capture_out(self, module, inputs, output):
        if isinstance(output, tuple):
            out = output[0]
        else:
            out = output
        self.captured = out.detach()

    def _capture_in(self, module, inputs):
        self.captured = inputs[0].detach()

    def __enter__(self):
        self.captured = None
        if self.kind == "ln1_normalized":
            block = self._get_block()
            self._handles.append(block.input_layernorm.register_forward_hook(self._capture_out))
        elif self.kind == "resid_mid":
            block = self._get_block()
            self._handles.append(block.post_attention_layernorm.register_forward_pre_hook(self._capture_in))
        # resid_pre / resid_post rely on output_hidden_states — no hook.
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    @classmethod
    def from_output(cls, kind: str, layer: int, model_outputs) -> torch.Tensor:
        """Extract for hookpoints that can be read straight off
        ``output_hidden_states``. Raises if called with module-hook kinds."""
        if kind == "resid_post":
            return model_outputs.hidden_states[layer + 1]
        if kind == "resid_pre":
            return model_outputs.hidden_states[layer]
        raise ValueError(f"{kind} is module-hook only; use as context manager")


@dataclass
class HookpointBufferConfig:
    hookpoint: str
    layer: int
    d_model: int
    buffer_seqs: int = 4000
    chunk_len: int = 256
    refill_chunks: int = 64
    refill_threshold: float = 0.5
    device: str = "cuda"
    dtype: torch.dtype = torch.float16


class HookpointStreamingBuffer:
    """Single-hookpoint streaming buffer. Stores ``(N_seqs, chunk_len, d_model)``.

    Used by run_training_sae_custom.py to train a TopKSAE at one of
    {resid_pre, resid_mid, resid_post, ln1_normalized}. Sample with
    ``sample_flat(batch_size)`` → ``(B, d_model)``.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        text_iter: Iterator[str],
        config: HookpointBufferConfig,
    ):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.text_iter = text_iter
        self.cfg = config
        self._buf = torch.empty(
            config.buffer_seqs, config.chunk_len, config.d_model,
            device=config.device, dtype=config.dtype,
        )
        self._samples_drawn = 0
        self._filled = False

    @torch.no_grad()
    def _tokenize_batch(self, n: int) -> torch.Tensor:
        texts: list[str] = []
        for _ in range(n):
            texts.append(next(self.text_iter))
        enc = self.tokenizer(
            texts, padding="max_length", truncation=True,
            max_length=self.cfg.chunk_len, return_tensors="pt", add_special_tokens=True,
        )
        return enc.input_ids.to(self.cfg.device)

    @torch.no_grad()
    def _forward_and_extract(self, input_ids: torch.Tensor) -> torch.Tensor:
        if self.cfg.hookpoint in ("resid_post", "resid_pre"):
            out = self.model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
            acts = HookpointExtractor.from_output(self.cfg.hookpoint, self.cfg.layer, out)
        else:
            with HookpointExtractor(self.model, self.cfg.hookpoint, self.cfg.layer) as ext:
                self.model(input_ids=input_ids, use_cache=False)
            acts = ext.captured
            if acts is None:
                raise RuntimeError("hook did not capture — check layer/hookpoint config")
        return acts.to(self.cfg.dtype)

    @torch.no_grad()
    def _refill(self, n_seqs: int) -> None:
        remaining = n_seqs
        while remaining > 0:
            b = min(self.cfg.refill_chunks, remaining)
            ids = self._tokenize_batch(b)
            acts = self._forward_and_extract(ids)
            if not self._filled:
                start = self.cfg.buffer_seqs - remaining
                self._buf[start:start + b] = acts
            else:
                idx = torch.randint(self.cfg.buffer_seqs, (b,), device=self.cfg.device)
                self._buf[idx] = acts
            remaining -= b
        if not self._filled:
            self._filled = True
        self._samples_drawn = 0

    @torch.no_grad()
    def warmup(self) -> None:
        self._refill(self.cfg.buffer_seqs)

    def _maybe_refill(self) -> None:
        frac = self._samples_drawn / max(1, self.cfg.buffer_seqs)
        if frac >= self.cfg.refill_threshold:
            self._refill(max(1, self.cfg.refill_chunks))

    @torch.no_grad()
    def sample_flat(self, batch_size: int) -> torch.Tensor:
        if not self._filled:
            raise RuntimeError("call warmup() before sampling")
        self._maybe_refill()
        seq_idx = torch.randint(self.cfg.buffer_seqs, (batch_size,), device=self.cfg.device)
        pos_idx = torch.randint(self.cfg.chunk_len, (batch_size,), device=self.cfg.device)
        batch = self._buf[seq_idx, pos_idx]  # (B, d_model)
        self._samples_drawn += batch_size
        return batch.contiguous()


# ---------------------------------------------------------------------------
# Corpus iterator (mixed pile + lmsys-chat) — lazy to avoid HF datasets
# import at module load.
# ---------------------------------------------------------------------------


def mixed_text_iter(
    tokenizer: PreTrainedTokenizerBase,
    mix: list[dict],
    *,
    seed: int = 0,
) -> Iterator[str]:
    """Yields raw text chunks sampled from the mix according to weights.

    Each entry of ``mix`` is ``{name, weight, split}``. The Qwen chat template
    is applied to lmsys rows to match what the model sees at deployment.
    """
    import random

    from datasets import load_dataset

    rng = random.Random(seed)
    streams: list[tuple[float, str, Iterator]] = []
    total_w = sum(float(m["weight"]) for m in mix)
    for entry in mix:
        w = float(entry["weight"]) / total_w
        ds = load_dataset(entry["name"], split=entry.get("split", "train"), streaming=True)
        streams.append((w, entry["name"], iter(ds)))

    while True:
        r = rng.random()
        cum = 0.0
        picked = streams[-1]
        for tup in streams:
            cum += tup[0]
            if r <= cum:
                picked = tup
                break
        _, name, it = picked
        try:
            row = next(it)
        except StopIteration:
            # Restart the stream on exhaustion.
            ds = load_dataset(name, split="train", streaming=True)
            it_new = iter(ds)
            idx = [i for i, (_, n, _) in enumerate(streams) if n == name][0]
            streams[idx] = (streams[idx][0], name, it_new)
            row = next(it_new)

        if "messages" in row:
            text = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=False,
            )
        elif "text" in row:
            text = row["text"]
        else:
            text = next(iter(row.values()))
        if text:
            yield text
