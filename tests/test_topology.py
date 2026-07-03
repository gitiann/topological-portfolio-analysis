"""Tests for the Gudhi H0 wrapper. Requires gudhi (skipped if absent)."""

from __future__ import annotations

import numpy as np
import pytest

from toporisk.topology import finite_pairs


def test_finite_pairs_drops_infinite_death():
    dgm = np.array([[0.0, 1.0], [0.0, np.inf]])
    assert finite_pairs(dgm).shape == (1, 2)


def test_finite_pairs_empty():
    assert finite_pairs(np.empty((0, 2))).shape == (0, 2)


def test_h0_two_clusters(two_clusters):
    pytest.importorskip("gudhi")
    from toporisk.topology import persistence_diagram_h0

    dgm = persistence_diagram_h0(two_clusters)
    finite = finite_pairs(dgm)
    # n points, connected Rips -> n-1 finite H0 merges, one infinite bar
    assert finite.shape[0] == two_clusters.shape[0] - 1
    # the two clusters are ~10 apart, so the largest merge death is ~10
    assert finite[:, 1].max() > 5.0
    # all H0 births are 0
    assert np.allclose(finite[:, 0], 0.0)
