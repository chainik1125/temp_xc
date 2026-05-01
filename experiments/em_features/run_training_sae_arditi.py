"""Train a TopK SAE on streaming activations — arditi recipe.

Plain TopK SAE (Bricken-style top-k) trained on per-token activations from a
single hookpoint. This is functionally identical to ``run_training_sae_custom``
with ``training_recipe = "topk_sae_arditi-style"``; it exists as a separate
launcher entry point so brief/launcher scripts can reference the recipe by
name.

    uv run python -m experiments.em_features.run_training_sae_arditi \\
        --config experiments/em_features/config_qwen14b.yaml \\
        --out_prefix /root/em_features/checkpoints/qwen14b_l24_sae_arditi_k128_em_nanda \\
        --total_steps 10000 --snapshot_at 10000 \\
        --d_sae 32768 --k 128 \\
        --batch_size 256 --lr 3e-4 \\
        --layer 24 --hookpoint resid_post
"""

from experiments.em_features.run_training_sae_custom import main


if __name__ == "__main__":
    main()
