import torch

from experiments.c6_em.window_sweep import _paper_shuffle, training_cfg


def test_window_sweep_changes_only_t_from_paper_recipe():
    cfg = training_cfg(4)
    assert cfg.n_steps == 25_000
    assert cfg.batch_size == 1_024
    assert cfg.bricken_enabled is True
    assert cfg.ema_auxk_alpha == 1.0 / 8.0
    assert cfg.dead_threshold_tokens == 128_000
    assert cfg.arch_hparams_override == {"T": 4}


def test_paper_shuffle_uses_per_row_randperm_sequence():
    windows = torch.arange(3 * 4).reshape(3, 4, 1)
    generator = torch.Generator().manual_seed(42)
    expected = torch.stack(
        [windows[row, torch.randperm(4, generator=generator)] for row in range(3)]
    )
    assert torch.equal(_paper_shuffle(windows, 4), expected)
