"""Tests for src.data - year-aware ERCOT + Houston time series loader."""

from pathlib import Path

import pytest

from src.data import load_time_series

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_load_2023_168h_returns_aligned_series():
    ts = load_time_series(project_root=PROJECT_ROOT, year=2023, num_hours=168)
    assert ts.num_hours == 168
    assert ts.year == 2023
    assert len(ts.it_load_MW) == 168
    assert len(ts.wet_bulb_C) == 168
    assert len(ts.price_import_usd_per_mwh) == 168
    assert len(ts.carbon_intensity_g_per_kwh) == 168


def test_load_2023_full_year_returns_8760():
    ts = load_time_series(project_root=PROJECT_ROOT, year=2023, num_hours=8760)
    assert ts.num_hours == 8760


def test_no_nans_in_loaded_data():
    ts = load_time_series(project_root=PROJECT_ROOT, year=2023, num_hours=8760)
    assert not ts.it_load_MW.isna().any()
    assert not ts.wet_bulb_C.isna().any()
    assert not ts.price_import_usd_per_mwh.isna().any()
    assert not ts.carbon_intensity_g_per_kwh.isna().any()


def test_it_load_matches_nlr_vercellino_200mw_profile():
    """Memory nuclear-dc-data-state: NLR 200 MW DC, mean 94.7, max 142.4 MW."""
    ts = load_time_series(project_root=PROJECT_ROOT, year=2023, num_hours=8760)
    assert ts.it_load_MW.min() > 0
    assert 140.0 < ts.it_load_MW.max() < 150.0
    assert 90.0 < ts.it_load_MW.mean() < 100.0


def test_wet_bulb_in_houston_climate_range():
    ts = load_time_series(project_root=PROJECT_ROOT, year=2023, num_hours=8760)
    assert -15.0 < ts.wet_bulb_C.min() < 5.0
    assert 25.0 < ts.wet_bulb_C.max() < 32.0


def test_carbon_intensity_in_ercot_realistic_range():
    """Memory: 2023 ERCOT AEF mean = 333 gCO2/kWh."""
    ts = load_time_series(project_root=PROJECT_ROOT, year=2023, num_hours=8760)
    assert 100.0 < ts.carbon_intensity_g_per_kwh.min()
    assert ts.carbon_intensity_g_per_kwh.max() < 1000.0
    assert 280.0 < ts.carbon_intensity_g_per_kwh.mean() < 400.0


def test_2024_cleaner_grid_than_2022():
    """ERCOT decarbonization trend: 2022 mean ≈ 350, 2024 mean ≈ 320 gCO2/kWh."""
    ts22 = load_time_series(project_root=PROJECT_ROOT, year=2022, num_hours=8760)
    ts24 = load_time_series(project_root=PROJECT_ROOT, year=2024, num_hours=8760)
    assert ts24.carbon_intensity_g_per_kwh.mean() < ts22.carbon_intensity_g_per_kwh.mean()


def test_2022_volatile_higher_price_than_2024_low():
    """Memory: 2022 DA mean ≈ $80 (Uri + heatwave), 2024 DA mean ≈ $28."""
    ts22 = load_time_series(project_root=PROJECT_ROOT, year=2022, num_hours=8760)
    ts24 = load_time_series(project_root=PROJECT_ROOT, year=2024, num_hours=8760)
    assert ts22.price_import_usd_per_mwh.mean() > ts24.price_import_usd_per_mwh.mean()


def test_start_hour_slicing_returns_correct_window():
    ts_full = load_time_series(project_root=PROJECT_ROOT, year=2023, num_hours=8760)
    ts_slice = load_time_series(
        project_root=PROJECT_ROOT, year=2023, num_hours=168, start_hour=4000
    )
    assert ts_slice.it_load_MW.iloc[0] == pytest.approx(ts_full.it_load_MW.iloc[4000])
    assert ts_slice.it_load_MW.iloc[-1] == pytest.approx(ts_full.it_load_MW.iloc[4167])


def test_invalid_year_rejected():
    with pytest.raises(ValueError, match=r"year"):
        load_time_series(project_root=PROJECT_ROOT, year=2025, num_hours=168)
    with pytest.raises(ValueError, match=r"year"):
        load_time_series(project_root=PROJECT_ROOT, year=2021, num_hours=168)


def test_slice_exceeds_available_raises():
    with pytest.raises(ValueError, match=r"num_hours|start_hour"):
        load_time_series(
            project_root=PROJECT_ROOT, year=2023, num_hours=200, start_hour=8700
        )
