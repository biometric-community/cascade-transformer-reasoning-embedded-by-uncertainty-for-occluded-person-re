"""Re-ID evaluation (CMC and mAP, Market-1501 protocol), PK sampling and filename parsing."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
import torch

from ctu.data.market1501 import parse_market_name
from ctu.data.sampler import RandomIdentitySampler
from ctu.metrics.reid import evaluate_rank


class _Labelled:
    def __init__(self, labels):
        self.labels = labels


def test_rank_and_map_match_a_hand_computed_ranking():
    # The query lies on the x-axis. The gallery order by cosine distance is relevant, irrelevant,
    # relevant, irrelevant, so AP = (1/1 + 2/3) / 2 and the rank-1 hit is certain.
    query = torch.tensor([[1.0, 0.0]])
    angles = [0.0, 0.2, 1.0, 2.0]
    gallery = torch.tensor([[math.cos(a), math.sin(a)] for a in angles])
    out = evaluate_rank(
        query, np.array([1]), np.array([1]),
        gallery, np.array([1, 2, 1, 2]), np.array([2, 2, 2, 2]),
        ranks=(1, 2),
    )
    assert out["Rank-1"] == pytest.approx(100.0)
    assert out["mAP"] == pytest.approx(100.0 * (1.0 + 2.0 / 3.0) / 2.0)


def test_same_identity_same_camera_matches_are_removed_before_ranking():
    # The gallery item of the same identity from the query's own camera is junk and is dropped.
    query = torch.tensor([[1.0, 0.0]])
    gallery = torch.tensor([[1.0, 0.0], [math.cos(0.5), math.sin(0.5)], [math.cos(1.0), math.sin(1.0)]])
    out = evaluate_rank(
        query, np.array([1]), np.array([1]),
        gallery, np.array([1, 2, 1]), np.array([1, 2, 2]),
        ranks=(1, 2),
    )
    assert out["Rank-1"] == pytest.approx(0.0)  # the first kept item is the wrong identity
    assert out["Rank-2"] == pytest.approx(100.0)
    assert out["mAP"] == pytest.approx(50.0)


def test_queries_without_any_match_score_zero():
    query = torch.tensor([[1.0, 0.0]])
    gallery = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    out = evaluate_rank(
        query, np.array([9]), np.array([1]),
        gallery, np.array([1, 2]), np.array([2, 2]),
        ranks=(1,),
    )
    assert out == {"mAP": 0.0, "Rank-1": 0.0}


def test_ranks_past_a_short_gallery_are_certain_matches():
    # Three gallery items, one of them the query's identity on another camera. The match is ranked first, so
    # every rank beyond the three items is reached as well and scores 100 instead of raising an error.
    query = torch.tensor([[1.0, 0.0]])
    gallery = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    out = evaluate_rank(
        query, np.array([1]), np.array([1]),
        gallery, np.array([1, 2, 3]), np.array([2, 2, 2]),
        ranks=(1, 5, 10),
    )
    assert out == pytest.approx({"mAP": 100.0, "Rank-1": 100.0, "Rank-5": 100.0, "Rank-10": 100.0})


def test_identity_sampler_yields_blocks_of_instances_per_identity():
    random.seed(0)
    labels = [pid for pid in range(6) for _ in range(5)]
    sampler = RandomIdentitySampler(_Labelled(labels), num_instances=4, batch_size=8)
    indices = list(sampler)
    assert len(indices) == len(sampler) == (6 // 2) * 8
    for start in range(0, len(indices), 4):
        assert len({labels[i] for i in indices[start:start + 4]}) == 1


def test_identity_sampler_repeats_identities_that_are_too_small():
    random.seed(1)
    labels = [0, 0] + [pid for pid in range(1, 5) for _ in range(6)]
    sampler = RandomIdentitySampler(_Labelled(labels), num_instances=4, batch_size=8)
    indices = list(sampler)
    assert len(indices) == len(sampler)
    for start in range(0, len(indices), 4):
        assert len({labels[i] for i in indices[start:start + 4]}) == 1


def test_market_filenames_give_identity_and_camera():
    assert parse_market_name("0002_c1s1_000451_03.jpg") == (2, 1)
    assert parse_market_name("-1_c3s3_000123_00.jpg") == (-1, 3)  # distractor identity
    with pytest.raises(ValueError):
        parse_market_name("not_a_market_file.jpg")
