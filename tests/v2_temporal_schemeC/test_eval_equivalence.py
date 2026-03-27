"""Tests for eval_unified.py — verify numerical equivalence with old eval functions."""

import math

import pytest
import torch

from src.utils.seed import set_seed
from src.v2_temporal_schemeC.toy_model import ToyModel
from src.v2_temporal_schemeC.markov_data_generation import generate_markov_activations
from src.v2_temporal_schemeC.relu_sae import ReLUSAE, ReLUSAETrainingConfig, train_relu_sae
from src.v2_temporal_schemeC.train_tfa import create_tfa, train_tfa, TFATrainingConfig
from src.v2_temporal_schemeC.temporal_crosscoder import (
    TemporalCrosscoder, CrosscoderTrainingConfig, train_crosscoder,
)
from src.v2_temporal_schemeC.feature_recovery import (
    feature_recovery_score, sae_decoder_directions, tfa_decoder_directions,
)

# Will be implemented:
from src.v2_temporal_schemeC.experiment.model_specs import (
    SAEModelSpec, TFAModelSpec, TXCDRModelSpec,
)
from src.v2_temporal_schemeC.experiment.eval_unified import evaluate_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tiny config
N_FEAT = 4
D_HID = 8
SEQ_LEN = 16
SEED = 42
TRAIN_STEPS = 100


# ── Old eval functions (from run_auc_and_crosscoder.py) ──────────────


def _old_eval_sae(sae, eval_hidden, device):
    sae.eval()
    flat = eval_hidden.reshape(-1, D_HID)
    n = flat.shape[0]
    total_se = total_signal = total_l0 = 0.0
    with torch.no_grad():
        for s in range(0, n, 4096):
            x = flat[s:min(s + 4096, n)].to(device)
            x_hat, z = sae(x)
            total_se += (x - x_hat).pow(2).sum().item()
            total_signal += x.pow(2).sum().item()
            total_l0 += (z > 0).float().sum(dim=-1).sum().item()
    return {"nmse": total_se / total_signal, "l0": total_l0 / n}


def _old_eval_tfa(tfa, eval_hidden, device):
    tfa.eval()
    n_seq = eval_hidden.shape[0]
    total_se = total_signal = total_novel_l0 = 0.0
    n_tokens = 0
    with torch.no_grad():
        for s in range(0, n_seq, 256):
            x = eval_hidden[s:min(s + 256, n_seq)].to(device)
            recons, inter = tfa(x)
            B, T, D = x.shape
            xf, rf = x.reshape(-1, D), recons.reshape(-1, D)
            total_se += (xf - rf).pow(2).sum().item()
            total_signal += xf.pow(2).sum().item()
            total_novel_l0 += (inter["novel_codes"] > 0).float().sum(dim=-1).sum().item()
            n_tokens += B * T
    return {"nmse": total_se / total_signal, "novel_l0": total_novel_l0 / n_tokens}


def _old_eval_tfa_full(tfa, eval_hidden, device):
    """From run_tfa_l1_with_total_l0.py — includes pred_l0 and total_l0."""
    tfa.eval()
    n_seq = eval_hidden.shape[0]
    total_se = total_signal = total_novel_l0 = total_pred_l0 = 0.0
    n_tokens = 0
    with torch.no_grad():
        for s in range(0, n_seq, 256):
            x = eval_hidden[s:min(s + 256, n_seq)].to(device)
            recons, inter = tfa(x)
            B, T, D = x.shape
            xf, rf = x.reshape(-1, D), recons.reshape(-1, D)
            total_se += (xf - rf).pow(2).sum().item()
            total_signal += xf.pow(2).sum().item()
            total_novel_l0 += (inter["novel_codes"] > 0).float().sum(dim=-1).sum().item()
            total_pred_l0 += (inter["pred_codes"].abs() > 1e-8).float().sum(dim=-1).sum().item()
            n_tokens += B * T
    novel_l0 = total_novel_l0 / n_tokens
    pred_l0 = total_pred_l0 / n_tokens
    return {"nmse": total_se / total_signal, "novel_l0": novel_l0,
            "pred_l0": pred_l0, "total_l0": novel_l0 + pred_l0}


def _old_eval_crosscoder(txcdr, eval_hidden, device, T):
    txcdr.eval()
    total_se = total_signal = total_l0 = 0.0
    n_windows = 0
    with torch.no_grad():
        for s in range(0, eval_hidden.shape[0], 256):
            seqs = eval_hidden[s:min(s + 256, eval_hidden.shape[0])].to(device)
            for t in range(SEQ_LEN - T + 1):
                w = seqs[:, t:t + T, :]
                loss, x_hat, z = txcdr(w)
                total_se += (x_hat - w).pow(2).sum().item()
                total_signal += w.pow(2).sum().item()
                total_l0 += (z > 0).float().sum(dim=-1).sum().item()
                n_windows += w.shape[0]
    return {"nmse": total_se / total_signal, "l0": total_l0 / n_windows}


# ── Fixtures ─────────────────────────────────────────────────────────


def _build_eval_data():
    """Build eval data deterministically."""
    set_seed(SEED)
    model = ToyModel(num_features=N_FEAT, hidden_dim=D_HID).to(DEVICE)
    model.eval()
    pi_t = torch.tensor([0.5] * N_FEAT)
    rho_t = torch.tensor([0.0, 0.3, 0.7, 0.9])
    with torch.no_grad():
        sf_acts, _ = generate_markov_activations(10000, SEQ_LEN, pi_t, rho_t, device=DEVICE)
        sf_hidden = model(sf_acts)
        sf = math.sqrt(D_HID) / sf_hidden.reshape(-1, D_HID).norm(dim=-1).mean().item()
        set_seed(SEED + 100)
        acts_eval, _ = generate_markov_activations(20, SEQ_LEN, pi_t, rho_t, device=DEVICE)
        hidden = model(acts_eval) * sf
    return hidden.detach(), model, pi_t, rho_t, sf


@pytest.fixture(scope="module")
def eval_hidden():
    hidden, model, _, _, _ = _build_eval_data()
    return hidden, model


@pytest.fixture(scope="module")
def trained_sae():
    hidden, model, pi_t, rho_t, sf = _build_eval_data()
    set_seed(SEED)
    sae = ReLUSAE(D_HID, D_HID, k=2).to(DEVICE)
    flat = hidden.reshape(-1, D_HID).detach()
    flat_gen = lambda bs: flat[torch.randint(0, flat.shape[0], (bs,))]
    cfg = ReLUSAETrainingConfig(total_steps=TRAIN_STEPS, batch_size=32, lr=3e-4, log_every=TRAIN_STEPS)
    sae, _ = train_relu_sae(sae, flat_gen, cfg, DEVICE)
    return sae


@pytest.fixture(scope="module")
def trained_tfa():
    hidden, model, pi_t, rho_t, sf = _build_eval_data()
    set_seed(SEED)
    tfa = create_tfa(dimin=D_HID, width=D_HID, k=2, n_heads=1,
                     n_attn_layers=1, bottleneck_factor=1, device=DEVICE)
    def seq_gen(n):
        acts, _ = generate_markov_activations(n, SEQ_LEN, pi_t, rho_t, device=DEVICE)
        return model(acts) * sf
    cfg = TFATrainingConfig(total_steps=TRAIN_STEPS, batch_size=4, lr=1e-3, log_every=TRAIN_STEPS)
    tfa, _ = train_tfa(tfa, seq_gen, cfg, DEVICE)
    return tfa


@pytest.fixture(scope="module")
def trained_txcdr():
    hidden, model, pi_t, rho_t, sf = _build_eval_data()
    T = 2
    set_seed(SEED)
    txcdr = TemporalCrosscoder(D_HID, D_HID, T, k=2).to(DEVICE)
    def win_gen(bs):
        acts, _ = generate_markov_activations(max(1, bs // (SEQ_LEN - T + 1)) + 1, SEQ_LEN, pi_t, rho_t, device=DEVICE)
        h = model(acts) * sf
        windows = []
        for t in range(SEQ_LEN - T + 1):
            windows.append(h[:, t:t + T, :])
        all_w = torch.cat(windows, dim=0)
        idx = torch.randperm(all_w.shape[0], device=DEVICE)[:bs]
        return all_w[idx]
    cfg = CrosscoderTrainingConfig(total_steps=TRAIN_STEPS, batch_size=32, lr=3e-4, log_every=TRAIN_STEPS)
    txcdr, _ = train_crosscoder(txcdr, win_gen, cfg, DEVICE)
    return txcdr


# ── Tests ────────────────────────────────────────────────────────────


class TestSAEEvalEquivalence:
    def test_nmse_matches(self, eval_hidden, trained_sae):
        hidden, model = eval_hidden
        old = _old_eval_sae(trained_sae, hidden, DEVICE)
        new = evaluate_model(SAEModelSpec(), trained_sae, hidden, DEVICE,
                             seq_len=SEQ_LEN)
        assert abs(old["nmse"] - new.nmse) < 1e-10, f"old={old['nmse']}, new={new.nmse}"

    def test_l0_matches(self, eval_hidden, trained_sae):
        hidden, model = eval_hidden
        old = _old_eval_sae(trained_sae, hidden, DEVICE)
        new = evaluate_model(SAEModelSpec(), trained_sae, hidden, DEVICE,
                             seq_len=SEQ_LEN)
        assert abs(old["l0"] - new.novel_l0) < 1e-10

    def test_pred_l0_is_zero(self, eval_hidden, trained_sae):
        hidden, model = eval_hidden
        new = evaluate_model(SAEModelSpec(), trained_sae, hidden, DEVICE,
                             seq_len=SEQ_LEN)
        assert new.pred_l0 == 0.0

    def test_auc_matches(self, eval_hidden, trained_sae):
        hidden, model = eval_hidden
        true_feats = model.feature_directions
        dd = sae_decoder_directions(trained_sae).to(DEVICE)
        tf = true_feats.T.to(DEVICE)
        old_auc = feature_recovery_score(dd, tf)["auc"]
        new = evaluate_model(SAEModelSpec(), trained_sae, hidden, DEVICE,
                             true_features=true_feats, seq_len=SEQ_LEN)
        assert abs(old_auc - new.auc) < 1e-10


class TestTFAEvalEquivalence:
    def test_nmse_matches(self, eval_hidden, trained_tfa):
        hidden, model = eval_hidden
        old = _old_eval_tfa(trained_tfa, hidden, DEVICE)
        new = evaluate_model(TFAModelSpec(), trained_tfa, hidden, DEVICE,
                             seq_len=SEQ_LEN)
        assert abs(old["nmse"] - new.nmse) < 1e-10

    def test_novel_l0_matches(self, eval_hidden, trained_tfa):
        hidden, model = eval_hidden
        old = _old_eval_tfa(trained_tfa, hidden, DEVICE)
        new = evaluate_model(TFAModelSpec(), trained_tfa, hidden, DEVICE,
                             seq_len=SEQ_LEN)
        assert abs(old["novel_l0"] - new.novel_l0) < 1e-10

    def test_pred_l0_matches_full_eval(self, eval_hidden, trained_tfa):
        hidden, model = eval_hidden
        old = _old_eval_tfa_full(trained_tfa, hidden, DEVICE)
        new = evaluate_model(TFAModelSpec(), trained_tfa, hidden, DEVICE,
                             seq_len=SEQ_LEN)
        assert abs(old["pred_l0"] - new.pred_l0) < 1e-10
        assert abs(old["total_l0"] - new.total_l0) < 1e-10

    def test_auc_matches(self, eval_hidden, trained_tfa):
        hidden, model = eval_hidden
        true_feats = model.feature_directions
        dd = tfa_decoder_directions(trained_tfa).to(DEVICE)
        tf = true_feats.T.to(DEVICE)
        old_auc = feature_recovery_score(dd, tf)["auc"]
        new = evaluate_model(TFAModelSpec(), trained_tfa, hidden, DEVICE,
                             true_features=true_feats, seq_len=SEQ_LEN)
        assert abs(old_auc - new.auc) < 1e-10


class TestTXCDREvalEquivalence:
    def test_nmse_matches(self, eval_hidden, trained_txcdr):
        hidden, model = eval_hidden
        old = _old_eval_crosscoder(trained_txcdr, hidden, DEVICE, T=2)
        new = evaluate_model(TXCDRModelSpec(T=2), trained_txcdr, hidden, DEVICE,
                             seq_len=SEQ_LEN)
        assert abs(old["nmse"] - new.nmse) < 1e-10

    def test_l0_matches(self, eval_hidden, trained_txcdr):
        hidden, model = eval_hidden
        old = _old_eval_crosscoder(trained_txcdr, hidden, DEVICE, T=2)
        new = evaluate_model(TXCDRModelSpec(T=2), trained_txcdr, hidden, DEVICE,
                             seq_len=SEQ_LEN)
        assert abs(old["l0"] - new.novel_l0) < 1e-10

    def test_auc_matches(self, eval_hidden, trained_txcdr):
        hidden, model = eval_hidden
        true_feats = model.feature_directions
        # Decoder-averaging approach: average decoder matrices, then compute AUC
        tf = true_feats.T.to(DEVICE)
        dd0 = trained_txcdr.decoder_directions(0).to(DEVICE)
        dd1 = trained_txcdr.decoder_directions(1).to(DEVICE)
        avg_dd = (dd0 + dd1) / 2
        expected_auc = feature_recovery_score(avg_dd, tf)["auc"]
        new = evaluate_model(TXCDRModelSpec(T=2), trained_txcdr, hidden, DEVICE,
                             true_features=true_feats, seq_len=SEQ_LEN)
        assert abs(expected_auc - new.auc) < 1e-10
