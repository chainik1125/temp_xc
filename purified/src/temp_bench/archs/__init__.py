"""Architecture registry.

Drop a .py file here + register in ``configs/archs.yaml`` → the runner
picks it up. All classes subclass
:class:`temp_bench.interfaces.architecture.TempBenchArch`.

This package contains both ours (TXC-base, TXC-pro, TopK-SAE,
SAE-arditi, MLC, Stacked-SAE) and adapter-wrapped upstream baselines
(T-SAE wraps AI4LIFE-GROUP/temporal-saes; TFA wraps the TFA paper's
reference impl). From the framework's perspective they're uniform.
"""
