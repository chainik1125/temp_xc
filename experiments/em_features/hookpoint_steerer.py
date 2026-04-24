"""Activation-addition steering at {resid_pre, resid_post, resid_mid,
ln1_normalized} of a given transformer-block layer.

ActivationSteerer from open-source-em-features only hooks layer module
outputs (== resid_post). For the other three hookpoints we need module-
specific hooks. This class mirrors ActivationSteerer's __enter__/__exit__
contract so frontier_sweep.py uses it interchangeably.
"""

from __future__ import annotations

from typing import Sequence

import torch
from transformers import PreTrainedModel


class HookpointSteerer:
    def __init__(
        self,
        model: PreTrainedModel,
        hookpoint: str,
        layer: int,
        steering_vectors: Sequence[torch.Tensor],
        coefficient: float,
    ):
        self.model = model
        self.hookpoint = hookpoint
        self.layer = layer
        self.coefficient = float(coefficient)
        param = next(model.parameters())
        stacked = torch.stack([v.to(param.dtype).to(param.device) for v in steering_vectors])
        # All directions applied with the same coefficient → sum once.
        self._total = (self.coefficient * stacked.sum(dim=0))
        self._handles: list = []

    def _block(self):
        return self.model.model.layers[self.layer]

    def _add_output(self, module, inputs, output):
        if isinstance(output, tuple):
            head, *rest = output
            return (head + self._total.to(head.dtype),) + tuple(rest)
        return output + self._total.to(output.dtype)

    def _add_input(self, module, inputs):
        head = inputs[0]
        patched = head + self._total.to(head.dtype)
        return (patched,) + tuple(inputs[1:])

    def __enter__(self):
        if self.coefficient == 0.0:
            return self
        block = self._block()
        if self.hookpoint == "resid_post":
            self._handles.append(block.register_forward_hook(self._add_output))
        elif self.hookpoint == "resid_pre":
            if self.layer == 0:
                target = self.model.model.embed_tokens
            else:
                target = self.model.model.layers[self.layer - 1]
            self._handles.append(target.register_forward_hook(self._add_output))
        elif self.hookpoint == "ln1_normalized":
            self._handles.append(block.input_layernorm.register_forward_hook(self._add_output))
        elif self.hookpoint == "resid_mid":
            self._handles.append(block.post_attention_layernorm.register_forward_pre_hook(self._add_input))
        else:
            raise ValueError(f"unknown hookpoint {self.hookpoint}")
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    # Backwards-compat alias; ActivationSteerer exposes .remove().
    def remove(self):
        self.__exit__(None, None, None)


class TXCWindowSteerer:
    """TXC-specific steerer that adds per-position decoder directions to the
    last min(seq_len, T) positions of the block-L residual output at every
    forward pass.

    With kv-cache enabled:
      - prefill (seq_len=prompt_len, often >= T): steering writes positions
        prompt_len-T..prompt_len-1 using decoder slots 0..T-1. Those steered
        residuals are cached, so subsequent decoded tokens attend to the
        steered kv-cache entries.
      - decode step (seq_len=1): applies only slot T-1 at the current token.

    With kv-cache disabled: every decode step sees full prompt + generated-so-far;
    steering applies to last T positions each step.

    Feature directions are unit-normed per (position, feature) slice so that α
    has a consistent magnitude meaning. ``coefficient`` is a scalar applied to
    each selected feature's contribution.
    """

    def __init__(
        self,
        model,
        txc,
        layer: int,
        feature_ids,
        coefficient: float,
    ):
        import torch
        self.model = model
        self.layer = layer
        self.coefficient = float(coefficient)
        self.T = int(txc.T)
        param = next(model.parameters())
        rows = txc.W_dec.data[:, feature_ids, :]  # (T, k, d_in)
        rows = rows / (rows.norm(dim=-1, keepdim=True) + 1e-8)
        # Summed over selected features, per position: (T, d_in)
        self._per_pos = (self.coefficient * rows.sum(dim=1)).to(param.dtype).to(param.device)
        self._handle = None

    def _hook(self, module, inputs, output):
        import torch
        if isinstance(output, tuple):
            head = output[0]
            rest = output[1:]
        else:
            head = output
            rest = None
        if self.coefficient == 0.0 or head.shape[1] == 0:
            return output
        seq_len = head.shape[1]
        n = min(seq_len, self.T)
        add = self._per_pos[self.T - n:self.T].to(head.dtype)  # (n, d_in)
        patched = head.clone()
        patched[:, -n:, :] = patched[:, -n:, :] + add.unsqueeze(0)
        if rest is None:
            return patched
        return (patched,) + rest

    def __enter__(self):
        if self.coefficient == 0.0:
            return self
        block = self.model.model.layers[self.layer]
        self._handle = block.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def remove(self):
        self.__exit__(None, None, None)
