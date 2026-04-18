"""Tests for sweeps.py — verify full sweep produces same results as old scripts."""

import math

import pytest
import torch

from src.utils.seed import set_seed
from src.data.toy.toy_model import ToyModel
from src.data.toy.markov import generate_markov_activations
from src.architectures.relu_sae import ReLUSAE, ReLUSAETrainingConfig, train_relu_sae
from src.training.train_tfa import create_tfa, train_tfa, TFATrainingConfig
from src.eval.feature_recovery import (
    feature_recovery_score, sae_decoder_directions, tfa_decoder_directions,
)

# Will be implemented:
from src.pipeline.toy_data import DataConfig, build_data_pipeline
from src.pipeline.toy_models import SAEModelSpec, TFAModelSpec, ModelEntry
from src.pipeline.toy_sweeps import run_topk_sweep, run_l1_sweep

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tiny config
N_FEAT = 4
D_HID = 8
SEQ_LEN = 16
DICT_WIDTH = 8
SEED = 42
STEPS = 100
K_VALUES = [1, 3]


DATA_CFG = DataConfig(
    num_features=N_FEAT,
    hidden_dim=D_HID,
    seq_len=SEQ_LEN,
    pi=[0.5] * N_FEAT,
    rho=[0.0, 0.3, 0.7, 0.9],
    dict_width=DICT_WIDTH,
    seed=SEED,
    eval_n_seq=20,
)


# ── Old-style inline loop (from run_auc_and_crosscoder.py) ───────────


def _run_old_topk_sweep_sae_tfa():
    """Reproduce the old inline loop for SAE + TFA at k=1,3."""
    set_seed(SEED)
    model = ToyModel(num_features=N_FEAT, hidden_dim=D_HID).to(DEVICE)
    model.eval()
    true_features = model.feature_directions
    pi_t = torch.tensor([0.5] * N_FEAT)
    rho_t = torch.tensor([0.0, 0.3, 0.7, 0.9])

    with torch.no_grad():
        acts, _ = generate_markov_activations(10000, SEQ_LEN, pi_t, rho_t, device=DEVICE)
        hidden = model(acts)
        sf = math.sqrt(D_HID) / hidden.reshape(-1, D_HID).norm(dim=-1).mean().item()

    def gen_flat(batch_size):
        n_seq = max(1, batch_size // SEQ_LEN)
        acts, _ = generate_markov_activations(n_seq, SEQ_LEN, pi_t, rho_t, device=DEVICE)
        return (model(acts) * sf).reshape(-1, D_HID)[:batch_size]

    def gen_seq(n_seq):
        acts, _ = generate_markov_activations(n_seq, SEQ_LEN, pi_t, rho_t, device=DEVICE)
        return model(acts) * sf

    set_seed(SEED + 100)
    acts_eval, _ = generate_markov_activations(20, SEQ_LEN, pi_t, rho_t, device=DEVICE)
    eval_hidden = model(acts_eval) * sf

    sae_results = []
    tfa_results = []

    for k in K_VALUES:
        # SAE
        set_seed(SEED)
        sae = ReLUSAE(D_HID, DICT_WIDTH, k=k).to(DEVICE)
        cfg = ReLUSAETrainingConfig(total_steps=STEPS, batch_size=32, lr=3e-4, log_every=STEPS)
        sae, _ = train_relu_sae(sae, gen_flat, cfg, DEVICE)
        sae.eval()
        flat = eval_hidden.reshape(-1, D_HID)
        n = flat.shape[0]
        total_se = total_signal = total_l0 = 0.0
        with torch.no_grad():
            for s in range(0, n, 4096):
                x = flat[s:min(s + 4096, n)].to(DEVICE)
                x_hat, z = sae(x)
                total_se += (x - x_hat).pow(2).sum().item()
                total_signal += x.pow(2).sum().item()
                total_l0 += (z > 0).float().sum(dim=-1).sum().item()
        nmse = total_se / total_signal
        l0 = total_l0 / n
        dd = sae_decoder_directions(sae).to(DEVICE)
        tf = true_features.T.to(DEVICE)
        auc = feature_recovery_score(dd, tf)["auc"]
        sae_results.append({"k": k, "nmse": nmse, "l0": l0, "auc": auc})
        del sae; torch.cuda.empty_cache()

        # TFA
        set_seed(SEED)
        tfa = create_tfa(dimin=D_HID, width=DICT_WIDTH, k=k, n_heads=1,
                         n_attn_layers=1, bottleneck_factor=1, device=DEVICE)
        tfa_cfg = TFATrainingConfig(total_steps=STEPS, batch_size=4, lr=1e-3, log_every=STEPS)
        tfa, _ = train_tfa(tfa, gen_seq, tfa_cfg, DEVICE)
        tfa.eval()
        n_seq = eval_hidden.shape[0]
        total_se = total_signal = total_novel_l0 = total_pred_l0 = 0.0
        n_tokens = 0
        with torch.no_grad():
            for s in range(0, n_seq, 256):
                x = eval_hidden[s:min(s + 256, n_seq)].to(DEVICE)
                recons, inter = tfa(x)
                B, T, D = x.shape
                xf, rf = x.reshape(-1, D), recons.reshape(-1, D)
                total_se += (xf - rf).pow(2).sum().item()
                total_signal += xf.pow(2).sum().item()
                total_novel_l0 += (inter["novel_codes"] > 0).float().sum(dim=-1).sum().item()
                total_pred_l0 += (inter["pred_codes"].abs() > 1e-8).float().sum(dim=-1).sum().item()
                n_tokens += B * T
        nmse = total_se / total_signal
        novel_l0 = total_novel_l0 / n_tokens
        pred_l0 = total_pred_l0 / n_tokens
        dd = tfa_decoder_directions(tfa).to(DEVICE)
        auc = feature_recovery_score(dd, tf)["auc"]
        tfa_results.append({"k": k, "nmse": nmse, "novel_l0": novel_l0,
                            "pred_l0": pred_l0, "total_l0": novel_l0 + pred_l0, "auc": auc})
        del tfa; torch.cuda.empty_cache()

    return sae_results, tfa_results


# ── Tests ────────────────────────────────────────────────────────────


class TestTopKSweepEquivalence:
    @pytest.fixture(scope="class")
    def old_results(self):
        return _run_old_topk_sweep_sae_tfa()

    @pytest.fixture(scope="class")
    def new_results(self):
        models = [
            ModelEntry("SAE", SAEModelSpec(), "flat",
                       training_overrides={"total_steps": STEPS, "batch_size": 32, "lr": 3e-4}),
            ModelEntry("TFA", TFAModelSpec(n_heads=1, n_attn_layers=1, bottleneck_factor=1),
                       "seq",
                       training_overrides={"total_steps": STEPS, "batch_size": 4, "lr": 1e-3}),
        ]
        return run_topk_sweep(
            models=models,
            k_values=K_VALUES,
            data_config=DATA_CFG,
            device=DEVICE,
        )

    def test_sae_nmse_matches(self, old_results, new_results):
        old_sae, _ = old_results
        for i, k in enumerate(K_VALUES):
            old_val = old_sae[i]["nmse"]
            new_val = new_results["SAE"][i].nmse
            assert abs(old_val - new_val) < 1e-5, f"k={k}: old={old_val}, new={new_val}"

    def test_sae_l0_matches(self, old_results, new_results):
        old_sae, _ = old_results
        for i, k in enumerate(K_VALUES):
            assert abs(old_sae[i]["l0"] - new_results["SAE"][i].novel_l0) < 1e-5

    def test_sae_auc_matches(self, old_results, new_results):
        old_sae, _ = old_results
        for i, k in enumerate(K_VALUES):
            assert abs(old_sae[i]["auc"] - new_results["SAE"][i].auc) < 1e-5

    def test_tfa_nmse_matches(self, old_results, new_results):
        _, old_tfa = old_results
        for i, k in enumerate(K_VALUES):
            old_val = old_tfa[i]["nmse"]
            new_val = new_results["TFA"][i].nmse
            assert abs(old_val - new_val) < 1e-5, f"k={k}: old={old_val}, new={new_val}"

    def test_tfa_novel_l0_matches(self, old_results, new_results):
        _, old_tfa = old_results
        for i, k in enumerate(K_VALUES):
            assert abs(old_tfa[i]["novel_l0"] - new_results["TFA"][i].novel_l0) < 1e-5

    def test_tfa_pred_l0_matches(self, old_results, new_results):
        _, old_tfa = old_results
        for i, k in enumerate(K_VALUES):
            assert abs(old_tfa[i]["pred_l0"] - new_results["TFA"][i].pred_l0) < 1e-5

    def test_tfa_total_l0_matches(self, old_results, new_results):
        _, old_tfa = old_results
        for i, k in enumerate(K_VALUES):
            assert abs(old_tfa[i]["total_l0"] - new_results["TFA"][i].total_l0) < 1e-5

    def test_tfa_auc_matches(self, old_results, new_results):
        _, old_tfa = old_results
        for i, k in enumerate(K_VALUES):
            assert abs(old_tfa[i]["auc"] - new_results["TFA"][i].auc) < 1e-5
