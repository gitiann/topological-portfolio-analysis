"""Tests for landscapes. mean_landscape is implemented; the rest are stubs."""

from __future__ import annotations

import numpy as np
import pytest

from toporisk.landscape import lp_norm, mean_landscape, persistence_landscape

# --- mean_landscape: implemented -> pass -------------------------------------

def test_mean_landscape_pointwise():
    a = np.array([[0.0, 2.0, 4.0]])
    b = np.array([[2.0, 2.0, 0.0]])
    m = mean_landscape([a, b])
    assert np.allclose(m, [[1.0, 2.0, 2.0]])


def test_mean_landscape_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mean_landscape([np.zeros((1, 3)), np.zeros((1, 4))])


def test_mean_landscape_empty_raises():
    with pytest.raises(ValueError):
        mean_landscape([])


# --- lp_norm: stub -> xfail --------------------------------------------------


def test_lp_norm_triangle():
    # a single triangular tent, height 1 over 3 samples, dx=1  -> L1 area ~ 1.0
    ls = np.array([[0.0, 1.0, 0.0]])
    assert lp_norm(ls, dx=1.0, p=1.0) == pytest.approx(1.0)



def test_lp_norm_zero_landscape():
    assert lp_norm(np.zeros((1, 10)), dx=1.0, p=1.0) == 0.0

def test_lp_norm_scales_with_dx():
    ls = np.array([[0.0, 1.0, 0.0]])
    assert lp_norm(ls, dx=1.0, p=1.0) == pytest.approx(1.0)
    assert lp_norm(ls, dx=0.5, p=1.0) == pytest.approx(0.5)
    assert lp_norm(ls, dx=2.0, p=1.0) == pytest.approx(2.0)


# --- persistence_landscape: stub -> xfail ------------------------------------


def test_persistence_landscape_shape():
    dgm = np.array([[0.0, 1.0], [0.0, 2.0]])
    ls = persistence_landscape(dgm, num_layers=1, resolution=100, x_range=(0.0, 2.0))
    assert ls.shape == (1, 100)



def test_persistence_landscape_empty_diagram():
    ls = persistence_landscape(np.empty((0, 2)), num_layers=1, resolution=50, x_range=(0.0, 1.0))
    assert ls.shape == (1, 50)
    assert np.allclose(ls, 0.0)


def test_landscape_single_tent_values():
    # one pair (0,2) on grid [0,2] with 5 ticks: 0, 0.5, 1, 1.5, 2
    # tent(t) = max(0, min(t-0, 2-t)) -> peak 1.0 at t=1
    dgm = np.array([[0.0, 2.0]])
    ls = persistence_landscape(dgm, num_layers=1, resolution=5, x_range=(0.0, 2.0))
    assert np.allclose(ls[0], [0.0, 0.5, 1.0, 0.5, 0.0])


def test_landscape_two_tents_ranking_and_padding():
    # pairs (0,2) and (0,1). At t=1.0 only the first tent is non-zero,
    # so eta_2 must be 0 there -- this is the zero-padding path.
    dgm = np.array([[0.0, 2.0], [0.0, 1.0]])
    ls = persistence_landscape(dgm, num_layers=2, resolution=5, x_range=(0.0, 2.0))
    assert np.allclose(ls[0], [0.0, 0.5, 1.0, 0.5, 0.0])   # pointwise max
    assert np.allclose(ls[1], [0.0, 0.5, 0.0, 0.0, 0.0])   # pointwise 2nd largest

def test_landscape_matches_gudhi():
    # Gudhi uses the perpendicular-to-diagonal convention (peak = (d-b)/sqrt(2));
    # we use Bubenik's standard tent (peak = (d-b)/2). They differ by exactly sqrt(2).
    from gudhi.representations import Landscape
    dgm = np.array([[0.0, 2.0], [0.0, 1.0], [0.0, 1.5]])
    mine = persistence_landscape(dgm, num_layers=2, resolution=50, x_range=(0.0, 2.0))
    theirs = Landscape(num_landscapes=2, resolution=50, sample_range=[0.0, 2.0]).fit_transform([dgm])[0]
    assert np.allclose(mine.ravel() * np.sqrt(2), theirs, atol=1e-8)