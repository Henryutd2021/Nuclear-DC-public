"""Tests for per-component capital annualization and the Section 45Y
clean-electricity PTC levelization.

Each asset is amortized over its own engineering life via its own CRF,
instead of a single project-wide 20-year CRF.
Assets with an end-of-life salvage value get a sinking-fund credit, so the
annual capital charge is (P - F) recovered, i.e. EAC = P*CRF - F*SFF.
45Y: the technology-neutral clean-electricity PTC (26 U.S.C. 45Y) is a flat
per-MWh credit paid for a 10-year statutory window and levelized over the
longer TAC horizon by the ratio of annuity present-value factors.
"""

import pytest

from src.finance import crf, annualized_capex, levelized_ptc_usd_per_mwh


def test_crf_20yr_at_6_7pct_matches_paper_value():
    # CRF(i=6.7%, n=20) is the 0.0922 used as the global factor today.
    assert crf(0.067, 20) == pytest.approx(0.0922, abs=5e-4)


def test_crf_30yr_at_6_7pct():
    assert crf(0.067, 30) == pytest.approx(0.0782, abs=5e-4)


def test_crf_15yr_at_6_7pct():
    assert crf(0.067, 15) == pytest.approx(0.1077, abs=5e-4)


def test_shorter_life_gives_higher_recovery_factor():
    # A 15-year battery must recover its capital faster than a 30-year plant.
    assert crf(0.067, 15) > crf(0.067, 20) > crf(0.067, 30)


def test_annualized_capex_without_salvage_is_capex_times_crf():
    capex = 1_000_000.0
    assert annualized_capex(capex, 0.067, 25) == pytest.approx(capex * crf(0.067, 25))


def test_salvage_credit_reduces_annual_capital_charge():
    capex = 1_000_000.0
    with_salvage = annualized_capex(capex, 0.067, 15, salvage_fraction=0.05)
    without_salvage = annualized_capex(capex, 0.067, 15, salvage_fraction=0.0)
    assert with_salvage < without_salvage


def test_annualized_capex_salvage_matches_sinking_fund_formula():
    # EAC = P*CRF - F*SFF, F = sf*P, SFF = CRF - i  ->  P*(CRF*(1-sf) + sf*i)
    capex, i, n, sf = 1_000_000.0, 0.067, 15, 0.05
    expected = capex * (crf(i, n) * (1.0 - sf) + sf * i)
    assert annualized_capex(capex, i, n, salvage_fraction=sf) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Section 45Y clean-electricity PTC: flat $30/MWh (CY2025 prevailing-wage,
# inflation-adjusted) paid for the statutory 10-year window, levelized across
# the 20-year TAC horizon by A(i,10)/A(i,20).
# ---------------------------------------------------------------------------
def test_45y_levelized_rate_at_paper_baseline():
    # A(6.7%,10)/A(6.7%,20) = 0.6567 -> 30 * 0.6567 = 19.70 $/MWh.
    assert levelized_ptc_usd_per_mwh(30.0, 0.067, 10, 20) == pytest.approx(
        19.70, abs=0.01
    )


def test_45y_levelization_matches_annuity_ratio():
    i, d, w = 0.067, 10, 20
    a = lambda n: (1 - (1 + i) ** -n) / i  # noqa: E731
    expected = 30.0 * a(d) / a(w)
    assert levelized_ptc_usd_per_mwh(30.0, i, d, w) == pytest.approx(expected)


def test_45y_full_rate_when_duration_covers_window():
    assert levelized_ptc_usd_per_mwh(30.0, 0.067, 20, 20) == pytest.approx(30.0)
    assert levelized_ptc_usd_per_mwh(30.0, 0.067, 25, 20) == pytest.approx(30.0)


def test_45y_levelized_rate_scales_linearly_with_credit():
    half = levelized_ptc_usd_per_mwh(15.0, 0.067, 10, 20)
    full = levelized_ptc_usd_per_mwh(30.0, 0.067, 10, 20)
    assert full == pytest.approx(2.0 * half)


def test_45y_higher_discount_rate_lowers_levelized_value_share():
    # Discounting weights the credited early years more, so the levelized
    # share rises with the discount rate.
    low_i = levelized_ptc_usd_per_mwh(30.0, 0.03, 10, 20) / 30.0
    high_i = levelized_ptc_usd_per_mwh(30.0, 0.12, 10, 20) / 30.0
    assert high_i > low_i


def test_45y_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        levelized_ptc_usd_per_mwh(30.0, -0.01, 10, 20)
    with pytest.raises(ValueError):
        levelized_ptc_usd_per_mwh(30.0, 0.067, 0, 20)
    with pytest.raises(ValueError):
        levelized_ptc_usd_per_mwh(30.0, 0.067, 10, 0)
