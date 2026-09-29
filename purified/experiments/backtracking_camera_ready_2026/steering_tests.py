"""CPU tests for C7 heldout selection, atom identity, and durable records."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from steering import (append, canonical_hash, choose_magnitude, complete_scores,
                      decoder_atom, digest, fresh_question_split, frozen_json,
                      limited_starts, make_split, mining_text_overlap,
                      normalized_problem_hash, phase1_command, read, reduce_codes, rows,
                      unlock_command, validate_checkpoint_receipt, SOURCE_COMMIT)


class SteeringProtocolTests(unittest.TestCase):
    def test_text_overlap_excluded_despite_different_question_id_namespace(self):
        prompts = [{"id": f"synthetic{i}", "prompt": f"Problem {i}"} for i in range(300)]
        traces = [{"question_id": row["id"], "prompt": row["prompt"]} for row in prompts]
        math500 = [{"unique_id": f"math{i}", "problem": f"Other problem {i}"} for i in range(500)]
        math500[0]["problem"] = "  PROBLEM\n0 "
        audit = mining_text_overlap(math500, prompts, traces)
        self.assertEqual(audit["normalized_exact_matches"], [{"math500_qid": "math0", "mining_qids": ["synthetic0"]}])
        split = fresh_question_split([r["unique_id"] for r in math500], [r["id"] for r in prompts], [], audit)
        self.assertEqual(split["mining_text_overlap_qids_excluded"], ["math0"])
        self.assertNotIn("math0", split["validation_qids"] + split["test_qids"])
        self.assertNotEqual(normalized_problem_hash("1+1"), normalized_problem_hash("1-1"))

    def test_checkpoint_requires_actual_completion_and_verified_provenance(self):
        from train import ACT_CACHE_KEY, ACT_CACHE_SHA256, PROTOCOL_VERSION
        config = {"arch": "txc_base", "status": "complete", "n_steps_completed": 300000,
            "protocol_version": PROTOCOL_VERSION, "historical_commit": SOURCE_COMMIT,
            "historical_source_sha256": "source", "model_sha256": "weight", "checkpoint_sha256": "weight",
            "act_cache_key": ACT_CACHE_KEY, "act_cache_sha256": ACT_CACHE_SHA256,
            "training_cfg": {"n_steps": 300000, "batch_size": 1024},
            "hparams": {"k_pos": 20, "T": 5}, "train_key": "key", "seed": 42, "d_sae": 32768}
        validate_checkpoint_receipt(config, "txc_base", "source", "weight")
        for key in ("status", "n_steps_completed", "historical_commit", "historical_source_sha256", "model_sha256", "checkpoint_sha256"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_checkpoint_receipt({k: v for k, v in config.items() if k != key}, "txc_base", "source", "weight")
        with self.assertRaises(ValueError):
            validate_checkpoint_receipt(config, "txc_base", "changed", "weight")

    def test_partial_batch_limit_keeps_full_protocol_batch_shape(self):
        self.assertEqual(list(limited_starts(120, 8, 1)), [0])
        self.assertEqual(list(limited_starts(112, 8, None)), list(range(0, 112, 8)))
        with self.assertRaises(ValueError):
            limited_starts(120, 8, 0)

    def test_partial_phase1_is_durable_resumable_and_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for qid in ["a", "b", "c"]:
                append(root / "math.jsonl", {"unique_id": qid, "problem": qid, "answer": "2"})
            frozen_json(root / "split.json", {"validation_qids": ["a"], "test_qids": ["b", "c"],
                                               "math500_sha256": digest(root / "math.jsonl")})
            args = SimpleNamespace(historical_root=root, split_file=root / "split.json",
                workspace=root / "shared", math500=root / "math.jsonl", batch_size=2,
                max_batches=1, max_new_tokens=1024, verify_zero=False)
            bt = SimpleNamespace(_build_prompt=lambda text: text, _fix_byte_decode=lambda text: text)
            tok = SimpleNamespace(decode=lambda ids, **kw: "2")
            with patch("steering.historical", return_value=bt), patch("steering.load_reasoning_model", return_value=(None, tok)), \
                 patch("steering.canonical_prompt_ids", return_value=[1, 2]), \
                 patch("steering.generate_token_panels", side_effect=lambda m, t, ids, budget: [[3, 4] for _ in ids]):
                phase1_command(args)
                self.assertEqual(len(rows(args.workspace / "phase1_unsteered.jsonl")), 2)
                self.assertFalse((args.workspace / "phase1_unsteered.json").exists())
                self.assertEqual(read(args.workspace / "phase1_progress.json")["status"], "partial")
                args.max_batches = None
                phase1_command(args)
            self.assertEqual(len(read(args.workspace / "phase1_unsteered.json")), 3)
            self.assertEqual(read(args.workspace / "phase1_progress.json")["status"], "complete")

    def test_fresh_split_excludes_all_published_and_mining_ids(self):
        qids = [f"q{i}" for i in range(500)]
        split = fresh_question_split(qids, qids[61:70], qids[:61])
        chosen = set(split["validation_qids"] + split["test_qids"])
        self.assertEqual(len(chosen), 120)
        self.assertEqual(len(split["test_qids"]), 100)
        self.assertFalse(chosen & set(qids[:70]))
        self.assertEqual(split, fresh_question_split(qids[::-1], qids[61:70], qids[:61]))

    def test_split_is_stratified_disjoint_and_order_invariant(self):
        wrong = [f"wrong{i}" for i in range(31)]
        correct = [f"correct{i}" for i in range(30)]
        mining = [f"mine{i}" for i in range(300)]
        a = make_split(wrong, correct, mining)
        b = make_split(wrong[::-1], correct[::-1], mining[::-1])
        self.assertEqual(a, b)
        self.assertEqual(len(a["validation_qids"]), 20)
        self.assertEqual(len(a["test_qids"]), 41)
        self.assertFalse(set(a["test_qids"]) & set(a["validation_qids"]))
        with self.assertRaises(ValueError):
            make_split(wrong, correct, ["wrong0"])

    def test_stacked_uses_actual_position_atom(self):
        model = SimpleNamespace(saes=[SimpleNamespace(W_dec=torch.tensor([[1., 2.], [3., 4.]])),
                                     SimpleNamespace(W_dec=torch.tensor([[5., 6.], [7., 8.]]))])
        atom, metadata = decoder_atom(model, "atoms", 2)
        self.assertTrue(torch.equal(atom, torch.tensor([5., 7.])))
        self.assertEqual(metadata, {"position": 1, "feature_id": 0, "flat_feature_id": 2})
        codes = torch.tensor([[[2., 0.], [1., 3.]]])
        self.assertTrue(torch.equal(reduce_codes(codes, "atoms"), torch.tensor([[2., 0., 1., 3.]])))
        self.assertTrue(torch.equal(reduce_codes(codes, "max"), torch.tensor([[2., 3.]])))
        self.assertTrue(torch.equal(reduce_codes(codes, "mean"), torch.tensor([[1.5, 1.5]])))

    def test_magnitude_selected_on_coherent_count_and_zero_wins_tie(self):
        scores = {(q, m): {"coherent_gc": value} for q in ("a", "b")
                  for m, value in [(0., 1), (-4., 0), (4., 3)]}
        magnitude, _ = choose_magnitude(["a", "b"], [-4., 0., 4.], scores)
        self.assertEqual(magnitude, 4.)
        for value in scores.values():
            value["coherent_gc"] = 1
        self.assertEqual(choose_magnitude(["a", "b"], [-4., 0., 4.], scores)[0], 0.)

    def test_resume_rejects_mismatch_and_judge_failure_never_becomes_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = root / "test"
            frozen_json(p / "generation_identity.json", {"qids": ["a"], "magnitudes": [0.]})
            frozen_json(p / "judge_identity.json", {"model": "mock"})
            with self.assertRaises(ValueError):
                frozen_json(p / "judge_identity.json", {"model": "different"})
            append(p / "generations.jsonl", {"question_id": "a", "magnitude": 0.,
                   "generation_sha256": "text-hash", "correct": True})
            common = {"question_id": "a", "magnitude": 0., "generation_sha256": "text-hash",
                      "judge_identity": canonical_hash({"model": "mock"})}
            append(p / "judgments.jsonl", {**common, "rubric": "backtracking", "label": -1})
            append(p / "judgments.jsonl", {**common, "rubric": "coherence", "label": 3})
            with self.assertRaises(ValueError):
                complete_scores(root, "test")
            append(p / "judgments.jsonl", {**common, "rubric": "backtracking", "label": 2})
            _, scores = complete_scores(root, "test")
            self.assertEqual(scores[("a", 0.)]["coherent_gc"], 2)

    def test_unlock_releases_only_selected_test_dose(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("generations.jsonl", "judgments.jsonl"):
                append(root / "validation" / name, {"fixture": name})
            selection = {"direction_sha256": "direction", "phase1_sha256": "phase1",
                "magnitude": -4., "validation_generations_sha256": digest(root / "validation/generations.jsonl"),
                "validation_judgments_sha256": digest(root / "validation/judgments.jsonl")}
            frozen_json(root / "selection.json", selection)
            frozen_json(root / "test_candidates/generation_identity.json", {
                "qids": ["a"], "magnitudes": [-4., 0., 4.],
                "direction_sha256": "direction", "phase1_sha256": "phase1"})
            for mag in [-4., 0., 4.]:
                append(root / "test_candidates/generations.jsonl", {"question_id": "a",
                    "magnitude": mag, "answer_text": "2", "ground_truth": "2"})
            fake_bt = SimpleNamespace(extract_boxed=lambda text: text, answers_match=lambda a, b: a == b)
            with patch("steering.historical", return_value=fake_bt):
                unlock_command(SimpleNamespace(workspace=root, historical_root=root))
            self.assertEqual({r["magnitude"] for r in rows(root / "test/generations.jsonl")}, {-4., 0.})
            self.assertTrue(all("correct" not in r for r in rows(root / "test_candidates/generations.jsonl")))


if __name__ == "__main__":
    unittest.main()
