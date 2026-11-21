"""The CTU network end to end on CPU, and the consistency of the two configuration files.

The ImageNet ResNet-50 trunk is replaced by a small stand-in with the same output width (2048 channels), so the
tests need no weights download. Everything above the trunk is the real CTU code.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

import ctu.models.ctu as ctu_model
from ctu.losses import BatchHardTriplet, ctu_total_loss


class TinyTrunk(nn.Module):
    """Stand-in for the ResNet-50 trunk: a stride-4 map with 2048 channels."""

    def __init__(self, last_stride: int = 1):
        super().__init__()
        self.net = nn.Conv2d(3, 2048, kernel_size=4, stride=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@pytest.fixture
def tiny_ctu(monkeypatch):
    monkeypatch.setattr(ctu_model, "Backbone", TinyTrunk)
    torch.manual_seed(0)
    return ctu_model.CTU(num_classes=5, dim=16, num_layers=2, num_heads=4, n_samples=2)


def test_forward_returns_features_logits_and_per_layer_losses(tiny_ctu):
    out = tiny_ctu(torch.randn(4, 3, 32, 16), train=True)
    assert out["feat"].shape == (4, 16)
    assert out["feat_bn"].shape == (4, 16)
    assert out["logits"].shape == (4, 5)
    assert len(out["layer_losses"]) == 2
    assert len(out["layer_losses"][0]["sample_logits"]) == 2  # one classifier output per sampled Fatt


def test_total_loss_is_finite_and_reaches_the_classifier(tiny_ctu):
    labels = torch.tensor([0, 0, 1, 1])
    out = tiny_ctu(torch.randn(4, 3, 32, 16), labels, train=True)
    loss = ctu_total_loss(out, labels, nn.CrossEntropyLoss(), BatchHardTriplet(margin=0.3))
    assert torch.isfinite(loss)
    loss.backward()
    assert tiny_ctu.classifier.weight.grad is not None


def test_eval_mode_uses_one_sample_per_layer(tiny_ctu):
    tiny_ctu.eval()
    with torch.no_grad():
        out = tiny_ctu(torch.randn(2, 3, 32, 16), train=False)
    assert out["logits"].shape == (2, 5)
    assert all(len(layer["sample_logits"]) == 1 for layer in out["layer_losses"])


def test_build_ctu_reads_the_model_section_of_a_config(monkeypatch, smoke_cfg):
    monkeypatch.setattr(ctu_model, "Backbone", TinyTrunk)
    model = ctu_model.build_ctu(smoke_cfg, num_classes=7)
    assert model.num_classes == 7
    assert model.num_layers == smoke_cfg["model"]["num_layers"]
    assert model.n_samples == smoke_cfg["model"]["n_samples"]
    assert model.beta == pytest.approx(smoke_cfg["model"]["beta"])


def test_both_configs_describe_the_same_method_and_dataset(default_cfg, smoke_cfg):
    assert default_cfg["paper"] == smoke_cfg["paper"]
    assert default_cfg["data"]["root"] == smoke_cfg["data"]["root"]
    assert default_cfg["data"]["height"] == smoke_cfg["data"]["height"] == 256
    assert default_cfg["data"]["width"] == smoke_cfg["data"]["width"] == 128
    assert default_cfg["model"]["dim"] == smoke_cfg["model"]["dim"] == 256


def test_smoke_run_is_a_reduced_version_of_the_full_run(default_cfg, smoke_cfg):
    assert smoke_cfg["train"]["epochs"] <= default_cfg["train"]["epochs"]
    assert smoke_cfg["train"]["batch_size"] <= default_cfg["train"]["batch_size"]
    assert smoke_cfg["model"]["n_samples"] <= default_cfg["model"]["n_samples"]


def test_default_beta_is_log_four_as_the_method_fixes_it(default_cfg):
    assert default_cfg["model"]["beta"] == pytest.approx(math.log(4.0), abs=1e-9)


def test_training_and_evaluation_settings_match_the_method(default_cfg):
    assert default_cfg["train"]["num_instances"] == 4  # PK sampling with four images per identity
    assert default_cfg["train"]["margin"] == pytest.approx(0.3)
    assert default_cfg["eval"]["ranks"] == [1, 5, 10]
