"""Small actual historical models: trainer parity, RNG resume, batch semantics."""
from __future__ import annotations

import copy
import importlib.util
import os
from pathlib import Path
import random
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get("C7_HISTORICAL_ROOT", str(ROOT / "purified/artifacts/camera_ready_2026/sources/extended-300k/purified")))
RUNNER = Path(os.environ.get("C7_TRAIN_RUNNER", str(ROOT / "purified/experiments/backtracking_camera_ready_2026/train.py")))
os.environ["TEMP_BENCH_ROOT"] = str(SOURCE)
sys.path.insert(0, str(SOURCE / "src"))
spec = importlib.util.spec_from_file_location("c7_camera_ready_train_under_test", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

from temp_bench.config import instantiate_arch, load_arch
from temp_bench.schemas import TrainingConfig
from temp_bench.training.sae_trainer import _make_optimizer, train_sae
from temp_bench.utils.seed import set_seed


def make_model(arch, seed=42):
    set_seed(seed)
    spec = load_arch(arch, component="c7")
    hparams = {**spec.hparams, "d_sae": 32, "k_pos": 2}
    spec = spec.model_copy(update={"hparams": hparams})
    model = instantiate_arch(spec, d_in=4)
    if arch == "tsae_paper":
        model.threshold_start_step = 0
    return model.train()


class TrainingInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        # The actual historical modules must be under test, not current v2.
        import temp_bench.config
        if Path(temp_bench.config.__file__).resolve() != SOURCE / "src/temp_bench/config.py":
            raise RuntimeError("run this training test in a fresh Python process")

    def setUp(self):
        self.acts = torch.tensor(np.random.default_rng(9).normal(size=(7, 9, 4)), dtype=torch.float32)
        self.cfg = TrainingConfig(n_steps=4, batch_size=6, warmup_steps=2, precision="fp32")

    def assert_states_equal(self, left, right):
        self.assertEqual(set(left), set(right))
        for name in left:
            self.assertTrue(torch.equal(left[name], right[name]), name)

    def test_full_batch_matches_pinned_historical_trainer_all_families(self):
        for arch in runner.ARCHS:
            with self.subTest(arch=arch):
                reference = make_model(arch)
                samples = runner.WindowSampler(self.acts, 42)
                reference_result = train_sae(reference, samples, self.cfg, device="cpu")
                candidate = make_model(arch)
                optimizer = _make_optimizer(candidate, self.cfg)
                samples = runner.WindowSampler(self.acts, 42)
                losses = []
                for step in range(self.cfg.n_steps):
                    metrics = runner.one_update(candidate, optimizer, samples(6), self.cfg, step,
                                                arch=arch, microbatch_size=6)
                    losses.append(metrics["loss"])
                self.assertEqual(losses, reference_result["log"]["loss"])
                self.assert_states_equal(candidate.state_dict(), reference.state_dict())

    def test_atomic_resume_reproduces_uninterrupted_model_optimizer_and_rng(self):
        for arch in runner.ARCHS:
            with self.subTest(arch=arch), tempfile.TemporaryDirectory() as temporary:
                model = make_model(arch)
                optimizer = _make_optimizer(model, self.cfg)
                sampler = runner.WindowSampler(self.acts, 42)
                identity = {"arch": arch, "training_cfg": self.cfg.model_dump()}
                for step in range(2):
                    metrics = runner.one_update(model, optimizer, sampler(6), self.cfg, step,
                                                arch=arch, microbatch_size=6)
                path = Path(temporary) / "latest-resume.pt"
                runner.atomic_torch_save(path, runner.resume_payload(
                    model, optimizer, sampler, identity=identity, completed=2, metrics=metrics, elapsed=3.5))
                self.assertFalse(path.with_suffix(".pt.tmp").exists())
                for step in range(2, 4):
                    runner.one_update(model, optimizer, sampler(6), self.cfg, step, arch=arch, microbatch_size=6)
                expected_model = copy.deepcopy(model.state_dict())
                expected_opt = copy.deepcopy(optimizer.state_dict())
                expected_draws = (random.random(), np.random.rand(), torch.rand(3), sampler(6))
                resumed = make_model(arch, seed=999)
                resumed_optimizer = _make_optimizer(resumed, self.cfg)
                resumed_sampler = runner.WindowSampler(self.acts, 999)
                saved = torch.load(path, map_location="cpu", weights_only=False)
                completed = runner.restore_payload(saved, resumed, resumed_optimizer, resumed_sampler, identity)
                self.assertEqual(completed, 2)
                for step in range(completed, 4):
                    runner.one_update(resumed, resumed_optimizer, resumed_sampler(6), self.cfg, step,
                                      arch=arch, microbatch_size=6)
                self.assert_states_equal(expected_model, resumed.state_dict())
                actual_opt = resumed_optimizer.state_dict()
                self.assertEqual(expected_opt["param_groups"], actual_opt["param_groups"])
                for pid in expected_opt["state"]:
                    self.assert_states_equal(expected_opt["state"][pid], actual_opt["state"][pid])
                self.assertEqual(random.random(), expected_draws[0])
                self.assertEqual(np.random.rand(), expected_draws[1])
                self.assertTrue(torch.equal(torch.rand(3), expected_draws[2]))
                self.assertTrue(torch.equal(resumed_sampler(6), expected_draws[3]))
                with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                    runner.restore_payload(saved, resumed, resumed_optimizer, resumed_sampler, {**identity, "changed": True})

    def test_microbatch_guards_and_separable_gradient_equivalence(self):
        for arch in ("tsae_paper", "txc_base"):
            with self.assertRaisesRegex(ValueError, "batch-dependent"):
                runner.validate_microbatch(arch, 1024, 256)
        for arch in ("topk_sae", "stacked_sae"):
            with self.subTest(arch=arch):
                full = make_model(arch)
                small = copy.deepcopy(full)
                data = runner.WindowSampler(self.acts, 18)(6)
                runner.one_update(full, _make_optimizer(full, self.cfg), data, self.cfg, 0,
                                  arch=arch, microbatch_size=6)
                runner.one_update(small, _make_optimizer(small, self.cfg), data, self.cfg, 0,
                                  arch=arch, microbatch_size=4)
                for name, weight in full.state_dict().items():
                    torch.testing.assert_close(weight, small.state_dict()[name], rtol=1e-5, atol=1e-7)

    def test_source_integrity_and_cache_gather(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "HISTORICAL_COMMIT").write_text(runner.HISTORICAL_COMMIT)
            for name in runner.HISTORICAL_FILES:
                dest = root / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes((SOURCE / name).read_bytes())
            self.assertEqual(len(runner.verify_historical_source(root)), 64)
            (root / "src/temp_bench/architectures/tsae.py").write_text("changed")
            with self.assertRaisesRegex(RuntimeError, "source SHA-256 mismatch"):
                runner.verify_historical_source(root)
        sampler = runner.WindowSampler(self.acts, 1)
        rng = np.random.default_rng(1)
        rows = rng.integers(0, 7, size=6)
        positions = rng.integers(0, 9 - 5 + 1, size=6)
        expected = torch.stack([self.acts[row, pos:pos + 5] for row, pos in zip(rows, positions)])
        self.assertTrue(torch.equal(expected, sampler(6)))


if __name__ == "__main__":
    unittest.main()
