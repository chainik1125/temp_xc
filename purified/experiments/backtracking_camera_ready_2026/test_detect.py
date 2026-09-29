"""CPU invariants for the matched C7 detector; no model/data downloads."""

from __future__ import annotations

import importlib.util
import ast
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

SPEC = importlib.util.spec_from_file_location("camera_ready_detect", Path(__file__).with_name("detect.py"))
det = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(det)


class TinyEncoder(torch.nn.Module):
    def __init__(self, d_in: int = 3, width: int = 4):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(1, d_in * width + 1).reshape(d_in, width).float() / 12)
        self.register_buffer("threshold", torch.tensor(0.5))
        self.seen_training_modes = []

    def encode(self, x):
        self.seen_training_modes.append(self.training)
        return torch.relu(x @ self.weight)


class TinyTSAE(TinyEncoder):
    def encode(self, x):
        self.seen_training_modes.append(self.training)
        z = torch.relu(x @ self.weight)
        if not self.training:
            return z * (z > self.threshold)
        flat = z.flatten()
        values, indices = flat.topk(x.shape[0] * x.shape[1])
        return torch.zeros_like(flat).scatter(0, indices, values).reshape(z.shape)


def valid_config(arch="topk_sae", width=32768):
    return {
        "arch": arch, "d_sae": width, "seed": 42, "historical_commit": det.HISTORICAL_COMMIT,
        "act_cache_key": det.ACT_CACHE_KEY, "protocol_version": "c7-camera-ready-300k-v1",
        "train_key": "test-fixture", "status": "complete", "n_steps_completed": 300000,
        "hparams": {"d_sae": width, "k_pos": 20, "T": 5},
        "training_cfg": {"batch_size": 1024, "n_steps": 300000},
    }


class DetectionTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(89)
        y = np.tile([0, 1], 30)
        groups = np.repeat([f"q{i:02}" for i in range(10)], 6)
        x = rng.exponential(0.7, size=(60, 12))
        x[rng.random(x.shape) < 0.6] = 0
        x[:, 3] += y * 0.7
        return sparse.csr_matrix(x), y, groups

    def test_duplicate_problem_text_ids_share_primary_fold_but_old_split_leaks(self):
        x, y, question_ids = self.fixture()
        prompts = [{"id": f"q{i:02}", "prompt": f"Solve problem {i}"} for i in range(10)]
        prompts[0]["prompt"] = "Solve x + 1 = 2"
        prompts[1]["prompt"] = "  SOLVE  x + 1\n= 2 "
        canonical, metadata = det.canonical_prompt_groups(question_ids, prompts)
        self.assertEqual(metadata["n_question_ids"], 10)
        self.assertEqual(metadata["n_canonical_prompt_groups"], 9)
        self.assertEqual(len(metadata["duplicate_text_groups"]), 1)
        self.assertEqual(set(next(iter(metadata["duplicate_text_groups"].values()))), {"q00", "q01"})
        audit, primary_folds, historical_folds = det.grouping_leakage_audit(question_ids, canonical)
        duplicate_rows = np.isin(question_ids, ["q00", "q01"])
        self.assertEqual(len(np.unique(primary_folds[duplicate_rows])), 1)
        self.assertGreater(len(np.unique(historical_folds[duplicate_rows])), 1)
        self.assertEqual(audit["canonical_prompt"]["n_prompt_groups_split_across_folds"], 0)
        self.assertEqual(audit["historical_question_id"]["n_prompt_groups_split_across_folds"], 1)
        self.assertEqual(audit["historical_question_id"]["n_test_sentences_with_identical_prompt_in_training"], 12)
        results, _, actual_folds = det.probe_sparse(x, y, canonical, width=12, s_grid=(1, 2))
        np.testing.assert_array_equal(primary_folds, actual_folds)
        self.assertEqual(sum(f["n_test_questions"] for f in results["historical_raw_C1"]["folds"]), 9)
        for fold in range(5):
            self.assertFalse(set(canonical[actual_folds == fold]) & set(canonical[actual_folds != fold]))

    def test_prompt_map_rejects_unknown_missing_and_repeated_ids(self):
        prompts = [{"id": "a", "prompt": "x + 1"}, {"id": "b", "prompt": "x - 1"}]
        self.assertNotEqual(det.normalized_prompt_hash(prompts[0]["prompt"]),
                            det.normalized_prompt_hash(prompts[1]["prompt"]))
        with self.assertRaisesRegex(ValueError, "unknown"):
            det.canonical_prompt_groups(np.asarray(["unknown"]), prompts)
        with self.assertRaisesRegex(ValueError, "cover every"):
            det.canonical_prompt_groups(np.asarray(["a"]), prompts)
        with self.assertRaisesRegex(ValueError, "unique IDs"):
            det.canonical_prompt_groups(np.asarray(["a", "b"]), prompts + [prompts[0]])

    def test_checkpoint_gates(self):
        cfg = valid_config()
        det.validate_checkpoint(cfg, arch="topk_sae", d_sae=32768, seed=42)
        cfg["n_steps_completed"] = 299999
        with self.assertRaisesRegex(ValueError, "300000"):
            det.validate_checkpoint(cfg, arch="topk_sae", d_sae=32768, seed=42)
        det.validate_checkpoint(cfg, arch="topk_sae", d_sae=32768, seed=42, allow_smoke=True)
        cfg["historical_commit"] = "different"
        with self.assertRaisesRegex(ValueError, "provenance"):
            det.validate_checkpoint(cfg, arch="topk_sae", d_sae=32768, seed=42, allow_smoke=True)
        with self.assertRaisesRegex(ValueError, "width"):
            det.validate_checkpoint(valid_config(width=16384), arch="topk_sae", d_sae=16384, seed=42)
        det.validate_checkpoint(valid_config(arch="tsae_paper", width=16384),
                                arch="tsae_paper", d_sae=16384, seed=42)

    def test_pooling_preserves_shared_feature_ids_and_stacked_positions(self):
        z = torch.zeros(1, 5, 4)
        z[0, 0, 1] = 2
        z[0, 4, 1] = 7
        views = det.code_views(z, "topk_sae", position_aware=True)
        self.assertEqual(views["max"][0, 1].item(), 7)
        self.assertAlmostEqual(views["mean"][0, 1].item(), 1.8, places=6)
        reversed_views = det.code_views(z.flip(1), "topk_sae")
        torch.testing.assert_close(views["max"], reversed_views["max"])
        torch.testing.assert_close(views["mean"], reversed_views["mean"])
        stacked = det.code_views(z, "stacked_sae")
        self.assertEqual(set(stacked), {"position_identity"})
        self.assertEqual(tuple(stacked["position_identity"].shape), (1, 20))
        self.assertEqual(stacked["position_identity"][0, 1].item(), 2)
        self.assertEqual(stacked["position_identity"][0, 17].item(), 7)
        self.assertEqual(det.feature_identity(np.array([1, 17]), width=4, position_aware=True),
                         [{"position": 0, "offset": -12, "feature": 1},
                          {"position": 4, "offset": -8, "feature": 1}])

    def test_encoding_is_eval_readonly_and_uses_exact_last_five(self):
        model = TinyEncoder()
        model.train()
        rng = np.random.default_rng(4)
        acts = rng.normal(size=(13, 6, 3)).astype(np.float32)
        acts[:, 0] = 1e6  # Excluded -13 position must not leak into any view.
        threshold = model.threshold.clone()
        views = det.encode_sparse_views(model, acts, "topk_sae", batch_size=3)
        self.assertFalse(model.training)
        self.assertFalse(any(model.seen_training_modes))
        torch.testing.assert_close(threshold, model.threshold)
        expected = torch.relu(torch.from_numpy(acts[:, -5:]) @ model.weight).detach()
        np.testing.assert_allclose(views["max"].toarray(), expected.amax(1).numpy(), atol=1e-6)
        np.testing.assert_allclose(views["mean"].toarray(), expected.mean(1).numpy(), atol=1e-6)
        self.assertTrue(all(sparse.isspmatrix_csr(value) for value in views.values()))

    def test_tsae_mode_audit_detects_batch_dependence_without_mutation(self):
        model = TinyTSAE()
        model.eval()
        acts = np.zeros((6, 6, 3), dtype=np.float32)
        acts[:3] = 0.1
        acts[3:] = 3
        before = model.threshold.clone()
        result = det.tsae_inference_diagnostic(model, acts, rows=6, batch_sizes=(1, 6))
        self.assertTrue(result["buffers_unchanged"])
        self.assertFalse(model.training)
        torch.testing.assert_close(before, model.threshold)
        self.assertGreater(result["encodings"]["batchtopk_training_mode_T5_B6"]
                           ["max_abs_difference_vs_first_batch_size"], 0)
        self.assertLess(result["encodings"]["threshold_eval_mode_T5_B6"]
                        ["max_abs_difference_vs_first_batch_size"], 1e-5)
        model.threshold.fill_(-1)
        with self.assertRaisesRegex(ValueError, "uninitialized"):
            det.encode_sparse_views(model, acts, "tsae_paper")

    def test_raw_probe_matches_independent_dense_reference(self):
        x, y, groups = self.fixture()
        result, oof, folds = det.probe_sparse(x, y, groups, width=12, s_grid=(1, 2, 4), n_folds=5)
        dense = x.toarray()
        expected = np.zeros((len(y), 3))
        scores = {s: [] for s in (1, 2, 4)}
        for train, test in GroupKFold(5).split(dense, y, groups):
            diff = np.abs(dense[train][y[train] == 1].mean(0) - dense[train][y[train] == 0].mean(0))
            for j, s in enumerate((1, 2, 4)):
                selected = np.argsort(diff)[-s:]
                clf = LogisticRegression(penalty="l1", C=1, solver="liblinear", max_iter=2000,
                                         random_state=42).fit(dense[train][:, selected], y[train])
                pred = clf.predict_proba(dense[test][:, selected])[:, 1]
                expected[test, j] = pred
                scores[s].append(average_precision_score(y[test], pred))
        np.testing.assert_allclose(oof["historical_raw_C1"], expected, atol=1e-12)
        for s, values in scores.items():
            self.assertAlmostEqual(result["historical_raw_C1"]["metrics"][str(s)]
                                   ["mean_fold_average_precision"], np.mean(values))
        for group in np.unique(groups):
            self.assertEqual(len(np.unique(folds[groups == group])), 1)

    def test_scaled_probe_is_invariant_to_feature_units(self):
        x, y, groups = self.fixture()
        _, before, _ = det.probe_sparse(x, y, groups, width=12, s_grid=(2, 4))
        multiplied = x.multiply(np.geomspace(0.001, 1000, 12)[None, :]).tocsr()
        _, after, _ = det.probe_sparse(multiplied, y, groups, width=12, s_grid=(2, 4))
        np.testing.assert_allclose(before["matched_scaled_C1"], after["matched_scaled_C1"], atol=1e-12)

    def test_test_fold_values_do_not_change_training_selection_or_scaling(self):
        x, y, groups = self.fixture()
        before, _, folds = det.probe_sparse(x, y, groups, width=12, s_grid=(2, 4))
        changed = x.toarray()
        changed[folds == 0] *= 10000
        after, _, _ = det.probe_sparse(sparse.csr_matrix(changed), y, groups, width=12, s_grid=(2, 4))
        for mode in det.PROBE_MODES:
            for s in (2, 4):
                a = before[mode]["folds"][0]["budgets"][str(s)]
                b = after[mode]["folds"][0]["budgets"][str(s)]
                self.assertEqual(a["selected_columns"], b["selected_columns"])
                self.assertEqual(a["selected_scale"], b["selected_scale"])
                self.assertEqual(a["coef"], b["coef"])

    def test_constant_features_remain_finite(self):
        x, y, groups = self.fixture()
        x = sparse.csr_matrix(np.zeros(x.shape))
        _, oof, _ = det.probe_sparse(x, y, groups, width=12, s_grid=(1, 2))
        self.assertTrue(all(np.isfinite(value).all() for value in oof.values()))

    def test_float32_legacy_diagnostic_executes_the_pinned_helper(self):
        historical = Path(os.environ.get("C7_HISTORICAL_ROOT", str(
            Path(__file__).resolve().parents[2] / "artifacts/camera_ready_2026/sources/extended-300k/purified")))
        source = historical / det.HISTORICAL_PROBE_FILE
        if not source.exists():
            self.skipTest("pinned helper source unavailable; set C7_HISTORICAL_ROOT")
        self.assertEqual(det.sha256(source), det.HISTORICAL_PROBE_SHA256)
        # Execute precisely the historical function without importing unrelated runtime dependencies.
        tree = ast.parse(source.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                        and node.name == "compute_probe_metrics_at_S")
        namespace = {"np": np, "DEFAULT_PR_AUC_S_GRID": det.S_GRID}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        helper = namespace["compute_probe_metrics_at_S"]
        x, y, groups = self.fixture()
        x = x.astype(np.float32)
        results, _, _ = det.probe_sparse(x, y, groups, width=12, s_grid=(1, 2, 4))
        replay = det.historical_probe_replay(x, y, groups, results["historical_raw_C1"],
                                             helper, s_grid=(1, 2, 4))
        independently = helper(x.toarray(), y, groups, S_grid=(1, 2, 4))
        for s in (1, 2, 4):
            self.assertEqual(replay["mean_fold_average_precision"][str(s)], independently["pr_auc"][s])
        self.assertEqual(replay["one_view_dense_bytes"], x.shape[0] * x.shape[1] * 4)
        self.assertEqual(len(replay["selected_features_by_fold"]), 5)

    @unittest.skipUnless(os.environ.get("C7_HISTORICAL_ROOT"), "set C7_HISTORICAL_ROOT for pinned architecture integration")
    def test_actual_historical_architecture_encoders(self):
        historical = Path(os.environ["C7_HISTORICAL_ROOT"]).resolve()
        os.environ["TEMP_BENCH_ROOT"] = str(historical)
        sys.path.insert(0, str(historical / "src"))
        from temp_bench.config import instantiate_arch, load_arch

        acts = np.random.default_rng(52).normal(size=(11, 6, 8)).astype(np.float32)
        for arch in det.ARCHS:
            spec = load_arch(arch, component="c7")
            spec = spec.model_copy(update={"hparams": {**spec.hparams, "d_sae": 64, "k_pos": 2}})
            model = instantiate_arch(spec, d_in=8)
            if arch == "tsae_paper":
                model.threshold.fill_(0.25)
            views = det.encode_sparse_views(model, acts, arch, batch_size=3, position_aware=True)
            self.assertFalse(model.training)
            with torch.inference_mode():
                expected = det.code_views(model.encode(torch.from_numpy(acts[:, -5:])),
                                          arch, position_aware=True)
            for name, value in views.items():
                np.testing.assert_allclose(value.toarray(), expected[name].numpy(), atol=1e-6)
            if arch == "stacked_sae":
                self.assertEqual(views["position_identity"].shape, (11, 320))
            if arch == "tsae_paper":
                diagnostic = det.tsae_inference_diagnostic(model, acts, batch_sizes=(1, 4))
                self.assertTrue(diagnostic["buffers_unchanged"])

    def test_smoke_cli_writes_compact_safe_oof_and_provenance(self):
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            historical = root / "historical"
            historical.mkdir()
            (historical / "HISTORICAL_COMMIT").write_text(det.HISTORICAL_COMMIT)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            save_file(TinyEncoder().state_dict(), checkpoint / "model.safetensors")
            cfg = valid_config(width=4)
            cfg["n_steps_completed"] = 2
            cfg["model_sha256"] = det.sha256(checkpoint / "model.safetensors")
            (checkpoint / "config.json").write_text(json.dumps(cfg))
            rng = np.random.default_rng(4)
            acts = rng.normal(size=(40, 6, 3)).astype(np.float32)
            keys = np.asarray([f"q{i // 4:02}|sentence{i}" for i in range(40)])
            prompts = [{"id": f"q{i:02}", "prompt": f"Problem {i}"} for i in range(10)]
            prompts[1]["prompt"] = "  PROBLEM\n0 "
            prompt_path = historical / det.STAGE_A_PROMPTS_FILE
            prompt_path.parent.mkdir(parents=True)
            prompt_path.write_text(json.dumps(prompts))
            data = root / "data.npz"
            np.savez(data, X=acts, keys=keys, is_bt=np.tile([0, 1], 20))
            fake = types.ModuleType("temp_bench.config")
            fake.load_arch = lambda *a, **k: types.SimpleNamespace(model_copy=lambda update: update)
            fake.instantiate_arch = lambda spec, d_in: TinyEncoder(d_in=d_in)
            result = root / "results" / "detection.json"
            with patch.dict(sys.modules, {"temp_bench": types.ModuleType("temp_bench"), "temp_bench.config": fake}):
                self.assertEqual(det.main([
                    "--historical-root", str(historical), "--checkpoint-dir", str(checkpoint),
                    "--sentence-acts", str(data), "--output", str(result), "--arch", "topk_sae",
                    "--seed", "42", "--d-sae", "4", "--device", "cpu", "--allow-smoke",
                    "--s-grid", "1", "2", "--position-aware", "--save-codes",
                ]), 0)
            payload = json.loads(result.read_text())
            self.assertTrue(payload["smoke"])
            self.assertEqual(payload["status"], "complete")
            self.assertEqual(payload["provenance"]["checkpoint_sha256"], cfg["model_sha256"])
            self.assertEqual(set(payload["views"]), {"last", "mean", "max", "position_identity"})
            self.assertEqual(payload["protocol"]["grouping"], "normalized_prompt_sha256")
            self.assertEqual(payload["data"]["n_questions"], 10)
            self.assertEqual(payload["data"]["n_prompt_groups"], 9)
            self.assertEqual(set(payload["diagnostics"]["historical_question_id"]["views"]), set(payload["views"]))
            with np.load(result.parent / payload["oof_predictions"]["path"], allow_pickle=False) as out:
                self.assertEqual(out["last__matched_scaled_C1"].shape, (40, 2))
                self.assertEqual(out["historical_question_id__last__matched_scaled_C1"].shape, (40, 2))
                self.assertEqual(out["question_ids"].dtype.kind, "U")
                self.assertEqual(out["canonical_group_ids"].dtype.kind, "U")
                self.assertTrue(np.all(out["fold_id"] >= 0))
                duplicate_rows = np.isin(out["question_ids"], ["q00", "q01"])
                self.assertEqual(len(np.unique(out["fold_id"][duplicate_rows])), 1)
                self.assertGreater(len(np.unique(out["historical_question_id_fold_id"][duplicate_rows])), 1)


if __name__ == "__main__":
    unittest.main()
