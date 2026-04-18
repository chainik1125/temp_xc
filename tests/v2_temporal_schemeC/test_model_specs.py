"""Tests for model_specs.py — verify each spec creates, trains, and evals correctly."""

import pytest
import torch

from src.utils.seed import set_seed
from src.v2_temporal_schemeC.relu_sae import ReLUSAE
from src.bench.architectures._tfa_module import TemporalSAE
from src.bench.architectures.crosscoder import TemporalCrosscoder

# Will be implemented:
from src.v2_temporal_schemeC.experiment.model_specs import (
    SAEModelSpec,
    TFAModelSpec,
    TXCDRModelSpec,
    TXCDRv2ModelSpec,
    EvalOutput,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestSAEModelSpec:
    def setup_method(self):
        self.spec = SAEModelSpec()

    def test_data_format_is_flat(self):
        assert self.spec.data_format == "flat"

    def test_n_decoder_positions_is_none(self):
        assert self.spec.n_decoder_positions is None

    def test_create_returns_relu_sae(self):
        model = self.spec.create(d_in=8, d_sae=8, k=3, device=DEVICE)
        assert isinstance(model, ReLUSAE)

    def test_create_topk_mode(self):
        model = self.spec.create(d_in=8, d_sae=8, k=3, device=DEVICE)
        assert model.k == 3

    def test_create_relu_mode(self):
        model = self.spec.create(d_in=8, d_sae=8, k=None, device=DEVICE)
        assert model.k is None

    def test_decoder_directions_shape(self):
        model = self.spec.create(d_in=8, d_sae=12, k=3, device=DEVICE)
        dd = self.spec.decoder_directions(model)
        assert dd.shape == (8, 12)  # (d_in, d_sae)

    def test_eval_forward_returns_eval_output(self):
        set_seed(42)
        model = self.spec.create(d_in=8, d_sae=8, k=3, device=DEVICE)
        x = torch.randn(4, 8, device=DEVICE)
        out = self.spec.eval_forward(model, x)
        assert isinstance(out, EvalOutput)
        assert out.n_tokens == 4
        assert out.sum_se >= 0
        assert out.sum_pred_l0 == 0  # SAE has no predictable component


class TestTFAModelSpec:
    def setup_method(self):
        self.spec = TFAModelSpec()

    def test_data_format_is_seq(self):
        assert self.spec.data_format == "seq"

    def test_n_decoder_positions_is_none(self):
        assert self.spec.n_decoder_positions is None

    def test_create_returns_temporal_sae(self):
        model = self.spec.create(d_in=8, d_sae=8, k=3, device=DEVICE)
        assert isinstance(model, TemporalSAE)

    def test_create_default_no_pos_encoding(self):
        model = self.spec.create(d_in=8, d_sae=8, k=3, device=DEVICE)
        assert not model.use_pos_encoding

    def test_create_with_pos_encoding(self):
        spec = TFAModelSpec(use_pos_encoding=True)
        model = spec.create(d_in=8, d_sae=8, k=3, device=DEVICE)
        assert model.use_pos_encoding

    def test_l1_mode_creates_relu_type(self):
        model = self.spec.create(d_in=8, d_sae=8, k=None, device=DEVICE)
        assert model.sae_diff_type == "relu"

    def test_topk_mode_creates_topk_type(self):
        model = self.spec.create(d_in=8, d_sae=8, k=3, device=DEVICE)
        assert model.sae_diff_type == "topk"

    def test_decoder_directions_shape(self):
        model = self.spec.create(d_in=8, d_sae=12, k=3, device=DEVICE)
        dd = self.spec.decoder_directions(model)
        assert dd.shape == (8, 12)

    def test_eval_forward_returns_eval_output(self):
        set_seed(42)
        model = self.spec.create(d_in=8, d_sae=8, k=3, device=DEVICE)
        x = torch.randn(2, 10, 8, device=DEVICE)
        out = self.spec.eval_forward(model, x)
        assert isinstance(out, EvalOutput)
        assert out.n_tokens == 20  # 2 * 10
        assert out.sum_pred_l0 > 0  # TFA has predictable component

    def test_same_param_count_with_and_without_pos(self):
        spec_no = TFAModelSpec(use_pos_encoding=False)
        spec_yes = TFAModelSpec(use_pos_encoding=True)
        m1 = spec_no.create(d_in=8, d_sae=8, k=3, device=DEVICE)
        m2 = spec_yes.create(d_in=8, d_sae=8, k=3, device=DEVICE)
        p1 = sum(p.numel() for p in m1.parameters())
        p2 = sum(p.numel() for p in m2.parameters())
        assert p1 == p2


class TestTXCDRModelSpec:
    def setup_method(self):
        self.spec = TXCDRModelSpec(T=2)

    def test_data_format_is_window(self):
        assert self.spec.data_format == "window"

    def test_n_decoder_positions_equals_T(self):
        assert self.spec.n_decoder_positions == 2

    def test_create_returns_temporal_crosscoder(self):
        model = self.spec.create(d_in=8, d_sae=8, k=3, device=DEVICE)
        assert isinstance(model, TemporalCrosscoder)
        assert model.T == 2

    def test_decoder_directions_per_position(self):
        model = self.spec.create(d_in=8, d_sae=12, k=3, device=DEVICE)
        dd0 = self.spec.decoder_directions(model, pos=0)
        dd1 = self.spec.decoder_directions(model, pos=1)
        assert dd0.shape == (8, 12)
        assert dd1.shape == (8, 12)
        assert not torch.allclose(dd0, dd1)  # different per position

    def test_eval_forward_returns_eval_output(self):
        set_seed(42)
        model = self.spec.create(d_in=8, d_sae=8, k=3, device=DEVICE)
        x = torch.randn(4, 2, 8, device=DEVICE)  # (B, T=2, d)
        out = self.spec.eval_forward(model, x)
        assert isinstance(out, EvalOutput)
        assert out.n_tokens == 4  # 4 windows
        assert out.sum_pred_l0 == 0  # TXCDR has no pred component


class TestTXCDRv2ModelSpec:
    def test_k_times_T_effective(self):
        spec = TXCDRv2ModelSpec(T=2)
        model = spec.create(d_in=8, d_sae=8, k=3, device=DEVICE)
        # k_effective = 3 * 2 = 6, model.k should be 6
        assert model.k == 6

    def test_k_times_T_exceeds_d_sae_raises(self):
        spec = TXCDRv2ModelSpec(T=5)
        with pytest.raises(ValueError, match="k\\*T=45 exceeds d_sae=8"):
            spec.create(d_in=8, d_sae=8, k=9, device=DEVICE)

    def test_k_times_T_at_limit_succeeds(self):
        spec = TXCDRv2ModelSpec(T=5)
        model = spec.create(d_in=8, d_sae=40, k=8, device=DEVICE)
        assert model.k == 40
