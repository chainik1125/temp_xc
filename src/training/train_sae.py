"""SAE-lens BatchTopK training infrastructure."""

from collections.abc import Callable, Iterator

import torch
from sae_lens import BatchTopKTrainingSAE, BatchTopKTrainingSAEConfig, SAETrainer
from sae_lens.config import LoggingConfig, SAETrainerConfig

from src.training.toy_configs import TrainingConfig
from src.utils.device import DEFAULT_DEVICE


class DataIterator(Iterator[torch.Tensor]):
    """Wraps a callable that yields hidden activations as an infinite iterator for SAETrainer.

    Each call to __next__ invokes the callable to produce a batch of hidden
    activations ready for SAE training.
    """

    def __init__(
        self,
        generate_hidden_acts_fn: Callable[[int], torch.Tensor],
        batch_size: int,
    ):
        self.generate_hidden_acts_fn = generate_hidden_acts_fn
        self.batch_size = batch_size

    def __next__(self) -> torch.Tensor:
        with torch.no_grad():
            return self.generate_hidden_acts_fn(self.batch_size)

    def __iter__(self) -> "DataIterator":
        return self


def create_sae(
    d_in: int,
    d_sae: int,
    k: float,
    device: torch.device = DEFAULT_DEVICE,
) -> BatchTopKTrainingSAE:
    """Factory for creating a BatchTopK training SAE.

    Args:
        d_in: Input dimension (hidden_dim of toy model).
        d_sae: Number of SAE latents.
        k: Average number of active latents per sample (L0).
        device: Torch device.

    Returns:
        Initialized BatchTopKTrainingSAE.
    """
    cfg = BatchTopKTrainingSAEConfig(
        d_in=d_in,
        d_sae=d_sae,
        k=k,
        device=str(device),
        dtype="float32",
        apply_b_dec_to_input=True,
    )
    return BatchTopKTrainingSAE(cfg)


def train_sae(
    sae: BatchTopKTrainingSAE,
    generate_hidden_acts_fn: Callable[[int], torch.Tensor],
    training_config: TrainingConfig,
    device: torch.device = DEFAULT_DEVICE,
) -> BatchTopKTrainingSAE:
    """Train an SAE on hidden activations using SAE-lens.

    Args:
        sae: The SAE to train.
        generate_hidden_acts_fn: Callable that takes batch_size and returns
            hidden activations of shape (batch_size, d_in).
        training_config: Training hyperparameters.
        device: Torch device.

    Returns:
        The trained SAE.
    """
    data_iter = DataIterator(generate_hidden_acts_fn, training_config.batch_size)

    trainer_cfg = SAETrainerConfig(
        n_checkpoints=0,
        checkpoint_path=None,
        save_final_checkpoint=False,
        total_training_samples=training_config.total_training_samples,
        device=str(device),
        autocast=False,
        lr=training_config.lr,
        lr_end=training_config.lr,
        lr_scheduler_name="constant",
        lr_warm_up_steps=0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_decay_steps=0,
        n_restart_cycles=1,
        train_batch_size_samples=training_config.batch_size,
        dead_feature_window=training_config.dead_feature_window,
        feature_sampling_window=training_config.feature_sampling_window,
        logger=LoggingConfig(log_to_wandb=False),
    )

    trainer = SAETrainer(
        cfg=trainer_cfg,
        sae=sae,
        data_provider=data_iter,
    )

    trained_sae = trainer.fit()
    return trained_sae
