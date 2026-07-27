"""tscale L0/L1 harness — scratch trainer + dev-8 probe eval (CARD_SPLIT § 3).

Scratch-loop policy (CARD § 5): mirrors ``temp_bench.core.trainer.train_arch``
semantics exactly (registry instantiation, override merge, buffer choice per
``consumes``, Adam + warmup, pre/post_step hooks) but writes checkpoints to
this exploration's ``results/ckpts/`` and NEVER touches the canonical
manifest/leaderboard. The dev eval imports the canonical probe primitives
(``_fit_probe`` / ``_score_probe`` / ``_encode_pool``) so the probe math is
the protocol's, restricted to the 8 frozen dev tasks — the holdout is
structurally out of reach here.

Batch conventions (pre-registered, CARD § 4): window archs train at
``max(64, base//T)`` windows (P1 Amendment-1 token-slot rule, base 4096);
token archs at 4096; sequence archs at ``seq_batch`` (txc_pro_r1: 1024 =
the v1 c3_b1024 convention).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from temp_bench.core.config import load_arch, load_datasource
from temp_bench.core.schemas import TrainingConfig
from temp_bench.core.trainer import _build_refill_source, _infer_d_in
from temp_bench.data.probe_cache import load_probe_cache
from temp_bench.evals.probing import _encode_pool, _fit_probe, _score_probe
from temp_bench.interfaces.architecture import TempBenchArch

from experiments.explorations.tscale.make_split import DEV_FROZEN

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CKPTS = RESULTS / "ckpts"
DATASOURCE = "gemma_2_2b_it_l13_fineweb_24k128"
BASE_TOKEN_SLOTS = 4096


class _FastSequenceServer:
    """GPU-resident twin of refill+SequenceBuffer for `consumes='sequence'`.

    Bitwise-identical batches to the canonical path: same
    ``np.random.default_rng(seed)`` stream, same ``integers(0, N, size=B)``
    draws, fp16 → fp32 widening is exact on either device.
    """

    def __init__(self, data_spec, *, device: str, seed: int):
        from temp_bench.core.config import compute_data_key, data_cache_dir
        acts_path = data_cache_dir(compute_data_key(data_spec)) / "acts.npy"
        arr = np.load(acts_path, mmap_mode="r")
        if arr.dtype != np.float16:
            raise ValueError(f"expected fp16 cache, got {arr.dtype}")
        self.data = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
        self.n_total = self.data.shape[0]
        self.rng = np.random.default_rng(seed)
        self.device = device

    def __call__(self, batch_size: int) -> torch.Tensor:
        idx = self.rng.integers(0, self.n_total, size=batch_size)
        sel = torch.as_tensor(idx, device=self.device)
        return self.data[sel].float()


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True,
        check=True,
    ).stdout.strip()


def batch_size_for(consumes: str, T: int | None, seq_batch: int) -> int:
    if consumes == "window":
        return max(64, BASE_TOKEN_SLOTS // max(1, int(T or 1)))
    if consumes == "sequence":
        return int(seq_batch)
    return BASE_TOKEN_SLOTS


def config_hash(cfg: dict[str, Any]) -> str:
    js = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(js.encode()).hexdigest()[:16]


def build_cell_cfg(
    *, arch_name: str, override: dict[str, Any] | None, seed: int,
    n_steps: int, seq_batch: int, lr: float = 3e-4, warmup: int = 1000,
) -> dict[str, Any]:
    """Resolve the full cell config (merged hparams + schedule) up front so
    the config hash covers everything that determines the checkpoint."""
    spec = load_arch(arch_name, section="probing")
    hparams = dict(spec.hparams)
    if override:
        hparams.update(override)
    cls_probe = spec.class_path
    T_eff = hparams.get("T") or hparams.get("T_max")
    return {
        "arch": arch_name,
        "class_path": cls_probe,
        "arch_version": spec.arch_version,
        "hparams": hparams,
        "seed": int(seed),
        "n_steps": int(n_steps),
        "lr": float(lr),
        "warmup_steps": int(warmup),
        "seq_batch": int(seq_batch),
        "base_token_slots": BASE_TOKEN_SLOTS,
        "datasource": DATASOURCE,
        "T_eff": T_eff,
    }


def scratch_train(cfg: dict[str, Any], *, device: str = "cuda") -> tuple[TempBenchArch, dict[str, Any]]:
    """Train per the canonical loop; return (model, l0_metrics). Caches by
    config hash under results/ckpts/ (safetensors)."""
    from temp_bench.core.config import import_by_path

    chash = config_hash(cfg)
    CKPTS.mkdir(parents=True, exist_ok=True)
    ckpt = CKPTS / f"{chash}.safetensors"
    meta_p = CKPTS / f"{chash}.json"

    spec = load_arch(cfg["arch"], section="probing")
    data_spec = load_datasource(DATASOURCE)
    d_in = _infer_d_in(data_spec)
    torch.manual_seed(int(cfg["seed"]))
    cls = import_by_path(spec.class_path)
    model = cls(d_in=d_in, **cfg["hparams"])
    model.to(device)

    if ckpt.exists():
        from safetensors.torch import load_file
        model.load_state_dict(load_file(str(ckpt), device=device))
        model.eval()
        train_info = json.loads(meta_p.read_text()) if meta_p.exists() else {}
        return model, train_info

    model.train()
    training_cfg = TrainingConfig(
        n_steps=cfg["n_steps"], batch_size=0,  # batch decided below
        learning_rate=cfg["lr"], warmup_steps=cfg["warmup_steps"],
    )
    refill = _build_refill_source(data_spec, seed=int(cfg["seed"]))
    consumes = model.consumes
    T_eff = cfg.get("T_eff")
    bsz = batch_size_for(consumes, T_eff, cfg["seq_batch"])

    if consumes == "token":
        from temp_bench.data.activation_buffer import ActivationBuffer
        batch_iter = ActivationBuffer(
            refill, capacity=training_cfg.buffer_tokens,
            refill_threshold=training_cfg.refill_threshold,
            seq_len=getattr(data_spec, "seq_len", None) or 128,
            device=device, seed=int(cfg["seed"]),
        )
    elif consumes == "window":
        from temp_bench.data.window_buffer import WindowBuffer
        T_buf = cfg["hparams"].get("T") or cfg["hparams"].get("T_max") or 5
        batch_iter = WindowBuffer(
            refill, T=int(T_buf),
            capacity_seqs=max(1024, int(training_cfg.buffer_tokens // 128)),
            refill_threshold=training_cfg.refill_threshold,
            device=device, seed=int(cfg["seed"]),
        )
    elif consumes == "sequence":
        # Scratch-level serving acceleration (disclosed in RESULTS.md):
        # the canonical SequenceBuffer refills from the mmap'd acts.npy
        # every step (~1 s/step at b1024 — data-bound, GPU idle). This
        # server preloads the cache to GPU fp16 and replays the SAME
        # rng stream (default_rng(seed).integers) with the same
        # fp16→fp32 widening, so served batches are bitwise identical
        # to the canonical path — only faster.
        batch_iter = _FastSequenceServer(
            data_spec, device=device, seed=int(cfg["seed"]),
        )
    else:
        raise ValueError(f"unknown consumes={consumes!r}")

    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda step: min(1.0, (step + 1) / cfg["warmup_steps"]),
    )
    n_steps = int(cfg["n_steps"])
    log_every = max(1, n_steps // 10)
    t0 = time.time()
    loss_trace: list[float] = []
    for step in range(n_steps):
        model.pre_step()
        optim.zero_grad(set_to_none=True)
        batch = batch_iter(bsz)
        result = model.train_step(batch)
        if isinstance(result, tuple):
            loss, info = result
        else:
            info = result
            loss = info["loss"]
        loss.backward()
        optim.step()
        sched.step()
        model.post_step()
        if step % log_every == 0 or step == n_steps - 1:
            lv = float(loss.detach().item())
            loss_trace.append(lv)
            print(f"    [train {cfg['arch']}/T={T_eff}/s{cfg['seed']}] "
                  f"step {step:>5} loss {lv:.4f}", flush=True)
    wall = time.time() - t0

    model.eval()
    train_info = {
        "config_hash": chash, "batch_size": bsz, "consumes": consumes,
        "train_wall_s": round(wall, 1), "loss_trace": loss_trace,
        **l0_health(model, batch_iter, bsz),
    }
    from safetensors.torch import save_file
    save_file({k: v.detach().contiguous().cpu()
               for k, v in model.state_dict().items()}, str(ckpt))
    meta_p.write_text(json.dumps({**cfg, **train_info}, indent=1, default=str))
    return model, train_info


@torch.no_grad()
def l0_health(model: TempBenchArch, batch_iter, bsz: int) -> dict[str, Any]:
    """L0 training-health census on one held batch (CARD § 3 L0)."""
    out: dict[str, Any] = {}
    try:
        batch = batch_iter(bsz)
        was_training = model.training
        model.eval()
        d_sae = getattr(model, "_d_sae", None) or model.config.d_sae
        # Activation census through the arch's own train_step path where
        # possible (uses training-time sparsity), else encode.
        res = model.train_step(batch) if hasattr(model, "train_step") else None
        info = res[1] if isinstance(res, tuple) else (res or {})
        z = info.get("z")
        if z is not None:
            zf = z.reshape(-1, z.shape[-1])
            out["l0_train_path"] = float((zf != 0).float().sum(-1).mean())
            out["frac_latents_active_batch"] = float(((zf != 0).any(0)).float().mean())
        if hasattr(model, "num_tokens_since_fired"):
            nts = model.num_tokens_since_fired
            thr = getattr(model, "dead_threshold_tokens", 10_000_000)
            out["frac_dead_threshold"] = float((nts >= thr).float().mean())
            out["frac_never_fired_recent"] = float((nts > 0).float().mean())
        if was_training:
            model.train()
        _ = d_sae
    except Exception as e:  # health metrics must never kill a run
        out["l0_health_error"] = f"{type(e).__name__}: {e}"
    return out


def eval_dispatch_is_window(model: TempBenchArch) -> bool:
    """The dispatch rule proposed for evals/probing.py (CARD § 4 seam):
    eval_consumes overrides consumes; existing archs are unaffected."""
    return getattr(model, "eval_consumes",
                   getattr(model, "consumes", "token")) == "window"


@torch.no_grad()
def dev_eval(
    model: TempBenchArch, *, k_feats: tuple[int, ...] = (20,),
    S: int = 32, encode_batch: int = 64, shuffle_seed: int = 0,
    device: str | None = None,
) -> dict[str, Any]:
    """Dev-8 probe eval, canonical primitives, canonical semantics.

    Window-path dispatch uses ``eval_dispatch_is_window`` (the harness-side
    twin of the proposed probing.py generalization). Returns per-k dicts:
    mean/per-task auc, shuffled twin, realized l0.
    """
    if device is None:
        device = str(next(model.parameters()).device)
    dev_t = torch.device(device)
    model.eval()

    # Present the model to _encode_pool under its EVAL contract: the
    # canonical helper reads .consumes; give it a proxy view when the two
    # contracts differ (scratch-harness-local; probing.py gets the real fix).
    class _EvalView:
        def __init__(self, m):
            self._m = m
            self.consumes = "window" if eval_dispatch_is_window(m) else getattr(m, "consumes", "token")
        def __getattr__(self, name):
            return getattr(self._m, name)
        def encode(self, x):
            return self._m.encode(x)
        def eval(self):
            self._m.eval(); return self
        def parameters(self):
            return self._m.parameters()

    view = _EvalView(model)
    identity = int(getattr(model, "T", 1) or 1) == 1

    out: dict[str, Any] = {"tasks": list(DEV_FROZEN)}
    for k_feat in k_feats:
        aucs, aucs_sh, l0s = {}, {}, []
        for tname in DEV_FROZEN:
            task = load_probe_cache(DATASOURCE, tname)
            train_feats, _ = _encode_pool(
                view, task["X_train"], S=S, batch_size=encode_batch,
                device=dev_t, first_real=task.get("first_real_train"),
            )
            clf, top_idx = _fit_probe(train_feats, task["y_train"], k_feat)
            test_feats, l0 = _encode_pool(
                view, task["X_test"], S=S, batch_size=encode_batch,
                device=dev_t, first_real=task.get("first_real_test"),
            )
            r = _score_probe(clf, top_idx, test_feats, task["y_test"])
            if identity:
                r_sh = dict(r)
            else:
                sh_feats, _ = _encode_pool(
                    view, task["X_test"], S=S, batch_size=encode_batch,
                    device=dev_t, first_real=task.get("first_real_test"),
                    shuffle_seed=shuffle_seed,
                )
                r_sh = _score_probe(clf, top_idx, sh_feats, task["y_test"])
            aucs[tname] = r["auc"]
            aucs_sh[tname] = r_sh["auc"]
            l0s.append(l0)
        out[f"k{k_feat}"] = {
            "dev_mean_auc": float(np.mean(list(aucs.values()))),
            "dev_mean_auc_shuf": float(np.mean(list(aucs_sh.values()))),
            "realized_l0": float(np.mean(l0s)),
            "per_task": aucs,
            "per_task_shuf": aucs_sh,
        }
    return out


def append_row(row: dict[str, Any], fname: str = "l1_rows.jsonl") -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with (RESULTS / fname).open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
