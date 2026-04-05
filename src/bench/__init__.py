"""Standardized architecture comparison benchmarking module.

Provides a plug-and-play framework for comparing temporal SAE architectures
(TopKSAE, StackedSAE, TemporalCrosscoder, TFA, etc.) on synthetic data.

To add a new architecture:
    1. Create src/bench/architectures/my_arch.py with nn.Module + ArchSpec
    2. Register it in src/bench/architectures/__init__.py
    3. It's automatically available in sweeps
"""
