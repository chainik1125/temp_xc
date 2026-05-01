"""Batched per-element steering for the Wang procedure.

Replaces the serial cell-by-cell pattern of generate_longform_completions
(one steering direction broadcast over the whole batch) with a per-element
hook so K cells × N questions × M rollouts can all run in a single
model.generate() call.

Key idea: the standard steering hook adds α·direction uniformly across all
batch elements. Here we accept a stack of per-element (direction, magnitude)
tuples and add a different α·direction to each batch row. Since the hook
operates on the residual stream at the target layer, this produces the same
logits per element as if each row were generated in its own forward pass —
making this a *correct* parallelization of the Wang generation step.

Determinism: with do_sample=False (greedy), batched and unbatched outputs
are token-identical for matched (direction, magnitude). With do_sample=True
the RNG state evolves differently between batch sizes, so byte-identity is
not guaranteed; aggregate statistics (mean align/coh) should still match
within typical seed noise.

    from experiments.em_features.batched_steering import generate_batched_steered_completions
    completions = generate_batched_steered_completions(
        model=model, tokenizer=tok, prompts=prompts,
        steering_directions=[d1, d2, ...], magnitudes=[a1, a2, ...],
        layer_idx=15, max_new_tokens=200, temperature=1.0, batch_size=32,
    )
"""

from __future__ import annotations

from typing import List, Optional

import torch


def _format_prompt(tokenizer, question: str) -> str:
    """Apply the model's chat template to a single user question."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )


def _make_per_element_hook(steering: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
    """Return a forward hook that adds steering[i, None, :] to batch row i.

    `steering` has shape (B, d_model). When `attention_mask` is provided
    (B, T_input), the steering is zeroed at padded positions during the
    prompt-encoding pass (where h.shape[1] == T_input). During decode steps
    (h.shape[1] == 1, KV cache) there is no padding so the mask is skipped.

    Without the mask, left-padded prompts in a batched generate() get
    steering added at padding positions, contaminating the residual stream
    and flipping argmax even at greedy. This breaks bit-identity vs the
    serial-batch-of-1 invocation.
    """
    def hook(module, args, output):
        if isinstance(output, tuple):
            h = output[0]
        else:
            h = output

        seq_len = h.shape[1]
        steer = steering[:, None, :].to(h.dtype)
        if attention_mask is not None and seq_len == attention_mask.shape[1]:
            mask = attention_mask[:, :, None].to(h.dtype)
            steer = steer * mask
        new_h = h + steer

        if isinstance(output, tuple):
            return (new_h,) + output[1:]
        return new_h
    return hook


def generate_batched_steered_completions(
    *,
    model,
    tokenizer,
    prompts: List[str],
    steering_directions: List[torch.Tensor],   # one per prompt, shape (d_model,)
    magnitudes: List[float],                    # one per prompt
    layer_idx: int,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    batch_size: int = 32,
    do_sample: Optional[bool] = None,
) -> List[str]:
    """Generate one completion per prompt, with each prompt getting its own
    steering vector applied to the residual stream at layer `layer_idx`.

    Returns: List[str] of length len(prompts), aligned with the input order.
    """
    assert len(prompts) == len(steering_directions) == len(magnitudes), \
        f"prompts/directions/mags must align: {len(prompts)}/{len(steering_directions)}/{len(magnitudes)}"
    if len(prompts) == 0:
        return []

    device = next(model.parameters()).device
    if do_sample is None:
        do_sample = temperature > 0

    # Format with chat template + left-pad for batched generation
    prev_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    formatted = [_format_prompt(tokenizer, q) for q in prompts]

    # Build per-element steering vectors as a single (N, d) tensor
    direction_dtype = torch.float32  # safer; cast inside hook
    steering_all = torch.stack([
        (float(magnitudes[i]) * steering_directions[i].to(device).to(direction_dtype))
        for i in range(len(prompts))
    ], dim=0)  # (N, d)

    all_completions: List[str] = []
    try:
        for batch_start in range(0, len(prompts), batch_size):
            batch_prompts = formatted[batch_start: batch_start + batch_size]
            batch_steering = steering_all[batch_start: batch_start + batch_size]

            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True,
                add_special_tokens=False,
            ).to(device)
            input_lens = inputs.attention_mask.sum(dim=1).tolist()

            handle = model.model.layers[layer_idx].register_forward_hook(
                _make_per_element_hook(batch_steering, inputs.attention_mask)
            )
            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        temperature=temperature if do_sample else 1.0,
                        pad_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
                    )
            finally:
                handle.remove()

            # Each row of `outputs` is left-padded prompt + generated suffix.
            # Strip the input portion (length is unpadded prompt length).
            for i, row in enumerate(outputs):
                # The full sequence length minus the unpadded input length
                # gives us the generated suffix length. With left-padding the
                # generated tokens start after position (full_len - new_len).
                full_len = row.shape[0]
                input_len_unpadded = input_lens[i]
                # Account for left-padding: the input occupies the LAST
                # input_len_unpadded positions of the padded prefix.
                # The generated tokens are everything past the padded-input region.
                pad_len = inputs.input_ids[i].shape[0] - input_len_unpadded
                # New tokens start at index (pad_len + input_len_unpadded) in the OUTPUT
                # because the model's output sequence is [padding, prompt, generation].
                new_tokens_start = pad_len + input_len_unpadded
                new_tokens = row[new_tokens_start:]
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                all_completions.append(text)
    finally:
        tokenizer.padding_side = prev_padding_side

    return all_completions
