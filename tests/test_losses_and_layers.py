"""CTU losses and layers, checked against hand-computed values (Sec. III and Eqs. 10-18 of the paper)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from ctu.losses import BatchHardTriplet, ctu_layer_loss, ctu_total_loss
from ctu.models.ut import (
    CommonTransformer,
    MultiHeadUT,
    cascade_mask,
    head_diversity_loss,
    uncertainty_loss,
)


def test_batch_hard_triplet_matches_a_hand_computed_value():
    # Identity A: a0 = (1, 0) and a1 = (0, 1). Identity B: b0 = (0.6, 0.8) and b1 = (0.8, 0.6).
    feats = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8], [0.8, 0.6]])
    labels = torch.tensor([0, 0, 1, 1])
    hardest_positive_a = math.sqrt(2.0)  # a0 and a1 are orthogonal
    hardest_negative_a = math.sqrt(0.4)  # b1 is closest to a0, b0 is closest to a1
    term_a = max(0.0, hardest_positive_a - hardest_negative_a + 0.3)
    # For the B anchors the hardest positive is sqrt(0.08) and the hardest negative is sqrt(0.4): the hinge is 0.
    loss = BatchHardTriplet(margin=0.3)(feats, labels)
    assert loss.item() == pytest.approx(2 * term_a / 4, rel=1e-6)


def test_batch_hard_triplet_is_zero_when_identities_are_well_separated():
    feats = torch.tensor([[1.0, 0.0], [0.99, 0.14], [-1.0, 0.0], [-0.99, -0.14]])
    labels = torch.tensor([0, 0, 1, 1])
    assert BatchHardTriplet(margin=0.3)(feats, labels).item() == pytest.approx(0.0)


def test_batch_hard_triplet_is_zero_without_any_positive_pair():
    feats = torch.randn(3, 4)
    assert BatchHardTriplet(margin=0.3)(feats, torch.tensor([0, 1, 2])).item() == 0.0


def test_batch_hard_triplet_back_propagates():
    feats = torch.randn(6, 8, requires_grad=True)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    BatchHardTriplet(margin=0.3)(feats, labels).backward()
    assert feats.grad is not None and torch.isfinite(feats.grad).all()


def test_layer_loss_adds_sample_classification_and_both_regularisers():
    ce = nn.CrossEntropyLoss()
    labels = torch.tensor([0, 1])
    uniform = torch.zeros(2, 3)  # cross-entropy over 3 classes is log 3 for uniform logits
    layer = {
        "logits_cls": uniform,
        "sample_logits": [uniform, uniform],
        "lhd": torch.tensor(0.5),
        "lun": torch.tensor(0.25),
    }
    value = ctu_layer_loss(layer, labels, ce)
    assert value.item() == pytest.approx(2 * math.log(3) + 0.5 + 0.25, rel=1e-6)


def test_total_loss_averages_layers_and_adds_the_triplet_and_final_cross_entropy():
    ce = nn.CrossEntropyLoss()
    triplet = BatchHardTriplet(margin=0.3)
    labels = torch.tensor([0, 0, 1, 1])
    feats = torch.randn(4, 8)
    uniform = torch.zeros(4, 3)
    layer = {
        "logits_cls": uniform,
        "sample_logits": [uniform],
        "lhd": torch.tensor(0.0),
        "lun": torch.tensor(0.0),
    }
    outputs = {"layer_losses": [layer, layer], "feat": feats, "logits": uniform}
    value = ctu_total_loss(outputs, labels, ce, triplet)
    # Each layer gives 2 log 3 (sample + classifier); the mean over two identical layers is the same.
    expected = 2 * math.log(3) + triplet(feats, labels).item() + math.log(3)
    assert value.item() == pytest.approx(expected, rel=1e-6)


def test_uncertainty_loss_is_a_hinge_on_log_sigma():
    beta = math.log(4.0)
    assert uncertainty_loss(torch.ones(2, 3), beta).item() == pytest.approx(beta)  # log 1 = 0
    sigma_large = torch.full((2, 3), math.exp(beta + 1.0))
    assert uncertainty_loss(sigma_large, beta).item() == pytest.approx(0.0)
    assert math.isfinite(uncertainty_loss(torch.zeros(2, 3), beta).item())  # sigma is clamped before the log


def test_head_diversity_is_one_for_identical_heads_and_zero_for_orthogonal_ones():
    torch.manual_seed(0)
    base = torch.randn(2, 1, 1, 1, 4)
    identical = base.expand(2, 3, 1, 1, 4).clone()
    assert head_diversity_loss(identical).item() == pytest.approx(1.0, abs=1e-6)
    orthogonal = torch.zeros(2, 3, 1, 1, 4)
    for t in range(3):
        orthogonal[:, t, 0, 0, t] = 1.0
    assert head_diversity_loss(orthogonal).item() == pytest.approx(0.0, abs=1e-6)
    assert head_diversity_loss(torch.randn(2, 1, 1, 1, 4)).item() == 0.0  # a single head has no pairs


def test_cascade_mask_is_a_min_max_map_over_the_tokens():
    torch.manual_seed(0)
    sigma = torch.rand(2, 4, 6, 8) + 0.1
    k = torch.randn(2, 4, 6, 8)
    v = torch.randn(2, 4, 6, 8)
    cm = cascade_mask(sigma, k, v)
    assert cm.shape == (2, 6, 1)
    assert torch.all(cm >= -1e-6) and torch.all(cm <= 1 + 1e-6)
    # The 1e-6 in the min-max denominator leaves a residue of about 1e-6 / (score range) at the ends,
    # so the extremes are close to 0 and 1 rather than equal to them.
    assert torch.allclose(cm.amin(dim=1), torch.zeros(2, 1), atol=1e-3)
    assert torch.allclose(cm.amax(dim=1), torch.ones(2, 1), atol=1e-3)


def test_multihead_ut_returns_the_documented_shapes_and_positive_sigma():
    torch.manual_seed(0)
    ut = MultiHeadUT(dim=16, num_heads=4, n_samples=3)
    out = ut(torch.randn(2, 5, 16))
    assert out["out"].shape == (2, 5, 16)
    assert len(out["fatt_list"]) == 3
    assert out["heads"].shape == (2, 4, 3, 5, 4)
    assert out["sigma"].shape == (2, 4, 5, 4)
    assert out["k"].shape == out["v"].shape == (2, 4, 5, 4)
    assert torch.all(out["sigma"] > 0)


def test_multihead_ut_rejects_a_dimension_that_does_not_split_into_heads():
    with pytest.raises(AssertionError):
        MultiHeadUT(dim=10, num_heads=4)


def test_multihead_ut_sampling_is_reproducible_under_a_seed():
    ut = MultiHeadUT(dim=8, num_heads=2, n_samples=2)
    x = torch.randn(1, 3, 8)
    torch.manual_seed(1)
    first = ut(x)["out"]
    torch.manual_seed(1)
    second = ut(x)["out"]
    assert torch.equal(first, second)


def test_common_transformer_keeps_the_token_shape():
    torch.manual_seed(0)
    assert CommonTransformer(dim=16, num_heads=4)(torch.randn(2, 5, 16)).shape == (2, 5, 16)
