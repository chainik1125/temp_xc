from __future__ import annotations

import numpy as np
import torch

from experiments.power_spectrum.backtracking_sae_pooling.evaluate import (
    encode_topk,
    pool_dense_codes,
    truncate_topk,
)


def test_pooling_is_aligned_and_mean_max_are_permutation_invariant() -> None:
    z = torch.arange(2 * 5 * 7, dtype=torch.float32).reshape(2, 5, 7)
    pools = pool_dense_codes(z)
    reversed_pools = pool_dense_codes(z.flip(1))
    assert torch.equal(pools["last"], z[:, -1])
    assert torch.equal(pools["first"], z[:, 0])
    assert torch.equal(pools["mean"], reversed_pools["mean"])
    assert torch.equal(pools["max"], reversed_pools["max"])
    assert torch.allclose(pools["recency"], reversed_pools["reverse_recency"])


def test_topk_encoder_selects_before_relu_like_paper_v1() -> None:
    w_enc = torch.eye(4)
    b_enc = torch.zeros(4)
    b_dec = torch.zeros(4)
    x = torch.tensor([[[-4.0, -3.0, 2.0, 1.0]]])
    z = encode_topk(x, w_enc, b_enc, b_dec, k=3)
    assert z.tolist() == [[[0.0, 0.0, 2.0, 1.0]]]


def test_truncate_topk_keeps_largest_positive_entries_per_row() -> None:
    x = np.asarray([[1.0, 4.0, 0.0, 3.0], [2.0, 0.0, 5.0, 1.0]], dtype=np.float32)
    truncated = truncate_topk(x, 2, chunk_rows=1)
    np.testing.assert_array_equal(
        truncated,
        np.asarray([[0.0, 4.0, 0.0, 3.0], [2.0, 0.0, 5.0, 0.0]], dtype=np.float32),
    )

