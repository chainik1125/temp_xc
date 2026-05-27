"""Single SAE trainer for every architecture in the framework.

One ``train_arch(arch_spec, data_spec, …)`` function that:

1. Instantiates the arch via ``arch_spec.class_path``.
2. Builds the right :class:`BatchIter` based on ``arch.consumes``:
   - ``"token"`` → :class:`ActivationBuffer` (token-level shuffle)
   - ``"window"`` → :class:`WindowBuffer` (window-level shuffle)
3. Runs the optimizer loop. Each step:
       arch.pre_step()
       optim.zero_grad()
       metrics = arch.train_step(batch)
       metrics["loss"].backward()
       optim.step()
       arch.post_step()
4. Saves the checkpoint to ``checkpoints/<train_key>/model.safetensors``.
5. Appends a manifest entry.

No arch-specific branching. Per-arch loss + optional plug-ins live inside
``arch.train_step()`` and ``arch.post_step()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch

from temp_bench.core.cache import append_manifest, now_iso
from temp_bench.core.config import checkpoint_dir, import_by_path
from temp_bench.core.schemas import (
    ArchSpec,
    CheckpointManifest,
    CodeVersion,
    DataSourceSpec,
    TrainingConfig,
)
from temp_bench.interfaces.architecture import TempBenchArch


def train_arch(
    *,
    arch_spec: ArchSpec,
    data_spec: DataSourceSpec,
    seed: int,
    training_cfg: TrainingConfig,
    train_key: str,
    code_version: CodeVersion,
    agent: str | None = None,
) -> TempBenchArch:
    """Train one arch on one datasource. Saves checkpoint + manifest row.

    Returns the trained model in eval mode.
    """
    torch.manual_seed(int(seed))

    # 1) Instantiate the arch.
    cls = import_by_path(arch_spec.class_path)
    d_in = _infer_d_in(data_spec)
    model = cls(d_in=d_in, **arch_spec.hparams)
    if not isinstance(model, TempBenchArch):
        raise TypeError(
            f"{arch_spec.class_path} did not return a TempBenchArch instance. "
            "Subclass temp_bench.interfaces.architecture.TempBenchArch."
        )

    device = _select_device()
    model.to(device).train()

    # 2) Build the right BatchIter.
    refill = _build_refill_source(data_spec, seed=seed)
    if model.consumes == "token":
        from temp_bench.data.activation_buffer import ActivationBuffer
        batch_iter = ActivationBuffer(
            refill,
            capacity=training_cfg.buffer_tokens,
            refill_threshold=training_cfg.refill_threshold,
            seq_len=getattr(data_spec, "seq_len", None) or 128,
            device=device,
            seed=seed,
        )
    elif model.consumes == "window":
        from temp_bench.data.window_buffer import WindowBuffer
        T = arch_spec.hparams.get("T") or arch_spec.hparams.get("T_max") or 5
        # buffer ~16k sequences ≈ 16k × 128 × d_in bytes
        capacity_seqs = max(1024, int(training_cfg.buffer_tokens // 128))
        batch_iter = WindowBuffer(
            refill,
            T=int(T),
            capacity_seqs=capacity_seqs,
            refill_threshold=training_cfg.refill_threshold,
            device=device,
            seed=seed,
        )
    elif model.consumes == "sequence":
        # Legacy mode for v1 archs that do internal window sampling.
        from temp_bench.data.sequence_buffer import SequenceBuffer
        seq_len = getattr(data_spec, "seq_len", None) or 128
        batch_iter = SequenceBuffer(
            refill, seq_len=seq_len, device=device, seed=seed,
        )
    else:
        raise ValueError(
            f"Unknown arch.consumes={model.consumes!r} for "
            f"{arch_spec.class_path}; expected 'token' / 'window' / 'sequence'."
        )

    # 3) Optimizer + warmup schedule.
    optim = torch.optim.Adam(model.parameters(), lr=training_cfg.learning_rate)
    if training_cfg.warmup_steps > 0:
        warmup_lambda = lambda step: min(1.0, (step + 1) / training_cfg.warmup_steps)
        sched = torch.optim.lr_scheduler.LambdaLR(optim, warmup_lambda)
    else:
        sched = None

    # 4) Train loop.
    n_steps = int(training_cfg.n_steps)
    log_every = max(1, n_steps // 20)
    last_loss = float("inf")
    for step in range(n_steps):
        model.pre_step()
        optim.zero_grad(set_to_none=True)
        batch = batch_iter(training_cfg.batch_size)
        result = model.train_step(batch)
        # Tolerate both v1 tuple contract (loss, info_dict) and v2 dict.
        if isinstance(result, tuple):
            loss, info = result
            metrics = {"loss": loss, **info}
        else:
            metrics = result
            loss = metrics["loss"]
        loss.backward()
        optim.step()
        if sched is not None:
            sched.step()
        model.post_step()
        last_loss = float(loss.detach().item())
        if step % log_every == 0 or step == n_steps - 1:
            print(f"  [train] step={step:>6} loss={last_loss:.4f}")

    # 5) Save checkpoint + manifest.
    model.eval()
    ckpt_dir = checkpoint_dir(train_key)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "model.safetensors"
    try:
        from safetensors.torch import save_file
        # safetensors requires contiguous tensors on cpu
        state = {
            k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()
        }
        save_file(state, str(ckpt_path))
    except Exception:
        torch.save(model.state_dict(), str(ckpt_path.with_suffix(".pt")))
        ckpt_path = ckpt_path.with_suffix(".pt")

    size_mb = ckpt_path.stat().st_size / (1024 * 1024)

    append_manifest(
        CheckpointManifest(
            train_key=train_key,
            data_key=_data_key_from_spec(data_spec),
            arch=arch_spec.name,
            arch_version=arch_spec.arch_version,
            seed=int(seed),
            datasource=data_spec.name,
            training_cfg=training_cfg,
            local_path=str(ckpt_path),
            size_mb=float(size_mb),
            hf_url=None,
            code_version=code_version,
            agent=agent,
            ts=now_iso(),
        )
    )

    return model


# ── Helpers ───────────────────────────────────────────────────────────


def _data_key_from_spec(data_spec: DataSourceSpec) -> str:
    from temp_bench.core.config import compute_data_key
    return compute_data_key(data_spec)


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _infer_d_in(data_spec: DataSourceSpec) -> int:
    """Get the activation dim from the datasource spec.

    Real-LM: depends on subject_model (Gemma-2-2B = 2304, Llama-3.1-8B = 4096, …).
    Synthetic: explicitly in params["d_in"] or params["d"].
    """
    if data_spec.category == "real_lm":
        return _resid_dim_for_model(data_spec.subject_model)
    # Synthetic
    if data_spec.params is None:
        raise ValueError(
            f"Synthetic datasource {data_spec.name!r} has no params; "
            "include d_in (or 'd') in configs/data.yaml entry."
        )
    if "d_in" in data_spec.params:
        return int(data_spec.params["d_in"])
    if "d" in data_spec.params:
        return int(data_spec.params["d"])
    raise ValueError(
        f"Synthetic datasource {data_spec.name!r} params missing 'd_in' / 'd'."
    )


def _resid_dim_for_model(name: str | None) -> int:
    """Hard-coded resid_post dim for the subject models in this paper."""
    if name is None:
        raise ValueError("Real-LM datasource missing 'subject_model'.")
    table = {
        "google/gemma-2-2b": 2304,
        "google/gemma-2-2b-it": 2304,
        "NousResearch/Meta-Llama-3.1-8B": 4096,
        "meta-llama/Llama-3.1-8B": 4096,
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": 4096,
        "Qwen/Qwen2.5-7B-Instruct": 3584,
        "Qwen/Qwen2.5-14B-Instruct": 5120,
    }
    if name in table:
        return table[name]
    raise ValueError(
        f"Unknown subject_model {name!r}. Add to _resid_dim_for_model "
        "in temp_bench/core/trainer.py."
    )


def _build_refill_source(data_spec: DataSourceSpec, *, seed: int) -> Callable[[int], torch.Tensor]:
    """Return a callable that yields ``(n_seqs, seq_len, d_in)`` tensors.

    Synthetic: invokes the generator (in-memory).
    Real-LM: slices the cached ``acts.npy``.
    """
    if data_spec.category == "synthetic":
        from temp_bench.data.synthetic import build_refill as build_synth_refill
        return build_synth_refill(data_spec, seed=seed)
    elif data_spec.category == "real_lm":
        from temp_bench.data.real_lm import build_refill as build_real_refill
        return build_real_refill(data_spec, seed=seed)
    else:
        raise ValueError(f"Unknown data category: {data_spec.category}")
