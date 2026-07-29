from __future__ import annotations

import numpy as np

from experiments.power_spectrum.code.run_task_screen import (
    TARGETS,
    _labels,
    _leading_tiles,
)


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _Data:
    extra = {
        "velocity_labels": _Tensor([[0, 0], [1, 1]]),
        "omega": [3, 98],
        "M": 101,
    }


def test_phasepair_transforms_split_power_pair_and_sign() -> None:
    pair = _labels(_Data(), TARGETS["phasepair_pair"])
    sign = _labels(_Data(), TARGETS["phasepair_sign"])
    assert np.array_equal(pair, [3, 3])
    assert np.array_equal(sign, [1, 0])


def test_leading_tiles_keep_one_independent_example_per_sequence() -> None:
    x = np.zeros((7, 12, 3))
    labels = np.arange(84).reshape(7, 12)
    tiles, target = _leading_tiles(x, labels, TARGETS["changepoint_mode"], 4)
    assert tiles.shape == (7, 4, 3)
    assert np.array_equal(target, labels[:, 3])
