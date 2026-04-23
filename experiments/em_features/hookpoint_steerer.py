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
