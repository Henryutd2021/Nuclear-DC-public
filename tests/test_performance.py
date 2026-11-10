"""Tests for part-load performance curves.

VCC: relative COP shape applied to the model's Houston-derated full-load COP;
the IPLV hump (COP peaks near 40-50% load) makes electricity-vs-cooling
non-convex, hence the SOS2 piecewise breakpoints. NGCC: relative heat-rate
shape applied to full-load efficiency.
"""

import numpy as np
import pytest

from src.performance import (
    ngcc_efficiency_at_load,
    vcc_cop_at_load,
    vcc_pwl_points,
)


def test_vcc_cop_at_full_load_equals_baseline():
    assert vcc_cop_at_load(1.0, 3.2) == pytest.approx(3.2)


def test_vcc_cop_peaks_above_baseline_at_part_load():
    # IPLV hump: ~40-50% load gives the highest COP, above the full-load value.
    assert vcc_cop_at_load(0.45, 3.2) > vcc_cop_at_load(1.0, 3.2)


def test_vcc_cop_drops_at_very_low_load():
    assert vcc_cop_at_load(0.1, 3.2) < vcc_cop_at_load(1.0, 3.2)


def test_vcc_cop_accepts_array():
    out = vcc_cop_at_load(np.array([0.1, 0.45, 1.0]), 3.2)
    assert out.shape == (3,)
    assert out[1] > out[2] > out[0]


def test_vcc_pwl_points_endpoints():
    q_pts, p_pts = vcc_pwl_points(3.2, 160.0)
    assert q_pts[0] == 0.0 and p_pts[0] == 0.0
    assert q_pts[-1] == pytest.approx(160.0)
    # Full-load electricity = Q / COP_full = 160 / 3.2 = 50 MWe.
    assert p_pts[-1] == pytest.approx(50.0)


def test_vcc_pwl_is_non_convex():
    # Incremental electricity-per-cooling slope must dip then rise (the hump),
    # which is exactly why a convex LP relaxation is invalid and SOS2 is needed.
    q_pts, p_pts = vcc_pwl_points(3.2, 160.0)
    slopes = np.diff(p_pts) / np.diff(q_pts)
    assert slopes.min() < slopes[0]  # an interior segment is flatter than the first
    assert slopes[-1] > slopes.min()  # and it rises again toward full load


def test_vcc_pwl_points_match_cop_curve():
    q_pts, p_pts = vcc_pwl_points(3.2, 160.0)
    for q, p in zip(q_pts[1:], p_pts[1:]):
        cop_here = vcc_cop_at_load(q / 160.0, 3.2)
        assert p == pytest.approx(q / cop_here, rel=1e-9)


def test_ngcc_efficiency_full_load_equals_baseline():
    assert ngcc_efficiency_at_load(1.0, 0.495) == pytest.approx(0.495)


def test_ngcc_efficiency_worse_at_part_load():
    assert ngcc_efficiency_at_load(0.4, 0.495) < ngcc_efficiency_at_load(1.0, 0.495)


def test_ngcc_efficiency_accepts_array():
    out = ngcc_efficiency_at_load(np.array([0.4, 1.0]), 0.495)
    assert out[0] < out[1]
