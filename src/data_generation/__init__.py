"""Synthetic data generation pipeline for temporal crosscoder experiments."""

from src.data_generation.configs import DataGenerationConfig
from src.data_generation.dataset import generate_dataset

__all__ = ["DataGenerationConfig", "generate_dataset"]
