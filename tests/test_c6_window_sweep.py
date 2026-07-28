from experiments.c6_em.window_sweep import training_cfg


def test_window_sweep_changes_only_t_from_paper_recipe():
    cfg = training_cfg(4)
    assert cfg.n_steps == 25_000
    assert cfg.batch_size == 1_024
    assert cfg.bricken_enabled is True
    assert cfg.ema_auxk_alpha == 1.0 / 8.0
    assert cfg.dead_threshold_tokens == 128_000
    assert cfg.arch_hparams_override == {"T": 4}
