"""Tests for src.cases.case3 - NGCC on-site off-grid baseline."""

from pathlib import Path

import pytest

from src.cases.case3 import solve_case3
from src.config import load_config
from src.data import load_time_series

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cfg():
    return load_config(case_id=3, project_root=PROJECT_ROOT)


@pytest.fixture(scope="module")
def ts_2023_168h():
    return load_time_series(project_root=PROJECT_ROOT, year=2023, num_hours=168)


def test_case3_loads_with_ngcc_enabled(cfg):
    assert cfg.case.case_id == 3
    assert cfg.case.equipment.ngcc_enabled is True
    assert cfg.case.equipment.grid_import_enabled is False
    assert cfg.case.ngcc is not None
    assert cfg.case.vcc is not None


def test_case3_smoke_returns_result(cfg, ts_2023_168h):
    result = solve_case3(cfg, ts_2023_168h)
    assert result.tac_usd_per_yr > 0


def test_case3_off_grid_balance(cfg, ts_2023_168h):
    """P_NGCC == P_IT + P_VCC every hour; no grid component."""
    result = solve_case3(cfg, ts_2023_168h)
    residual = (
        result.P_NGCC_elec_MW - result.P_IT_MW - result.P_VCC_elec_MW
    ).abs().max()
    assert residual < 1e-9


def test_case3_fuel_consumption_matches_partload_efficiency(cfg, ts_2023_168h):
    """Fuel [MMBtu/h] = P_NGCC [MW] * 3.412 / η(load), where η falls below the
    design point per the part-load heat-rate curve."""
    from src.performance import ngcc_efficiency_at_load

    result = solve_case3(cfg, ts_2023_168h)
    ngcc_cap = cfg.case.capacities.ngcc_capacity_MWe
    eta_load = ngcc_efficiency_at_load(
        result.P_NGCC_elec_MW / ngcc_cap, cfg.case.ngcc.net_efficiency_hhv
    )
    expected = result.P_NGCC_elec_MW * 3.412 / eta_load
    diff = (result.fuel_consumption_MMBtu_per_h - expected).abs().max()
    assert diff < 1e-6


def test_case3_fuel_cost_uses_hourly_henry_hub(cfg, ts_2023_168h):
    """Fuel cost = (HH_hourly + basis) × MMBtu/h, broadcasting daily HH."""
    result = solve_case3(cfg, ts_2023_168h)
    delivered_hourly = (
        ts_2023_168h.henry_hub_usd_per_mmbtu_hourly
        + cfg.case.ngcc.henry_hub_basis_usd_per_mmbtu
    )
    expected = result.fuel_consumption_MMBtu_per_h * delivered_hourly
    diff = (result.fuel_cost_usd_per_h - expected).abs().max()
    assert diff < 1e-6


def test_case3_annual_mean_delivered_fuel_is_reported(cfg, ts_2023_168h):
    """delivered_fuel_usd_per_mmbtu scalar = mean of the hourly delivered series."""
    result = solve_case3(cfg, ts_2023_168h)
    delivered_hourly = (
        ts_2023_168h.henry_hub_usd_per_mmbtu_hourly
        + cfg.case.ngcc.henry_hub_basis_usd_per_mmbtu
    )
    assert result.delivered_fuel_usd_per_mmbtu == pytest.approx(
        float(delivered_hourly.mean()), rel=1e-6
    )


def test_case3_2022_fuel_more_expensive_than_2024(cfg):
    """S2: 2022 HH spike → Case 3 fuel cost ≈ 3× 2024."""
    ts22 = load_time_series(project_root=PROJECT_ROOT, year=2022, num_hours=8760)
    ts24 = load_time_series(project_root=PROJECT_ROOT, year=2024, num_hours=8760)
    r22 = solve_case3(cfg, ts22)
    r24 = solve_case3(cfg, ts24)
    # 2022 mean HH $6.45 vs 2024 $2.19 → ratio ~ 2.9x
    ratio = r22.fuel_annual_usd / r24.fuel_annual_usd
    assert 2.0 < ratio < 4.0


def test_case3_direct_emissions_match_partload_emission_factor(cfg, ts_2023_168h):
    """Direct CO2 [kg/h] = P_NGCC × 1h × CO2_direct[g/kWh] × heat-rate multiplier.
    CO2 tracks fuel burned, so the full-load emission factor is scaled by the
    same part-load heat-rate penalty applied to fuel."""
    from src.performance import ngcc_hr_multiplier

    result = solve_case3(cfg, ts_2023_168h)
    ngcc_cap = cfg.case.capacities.ngcc_capacity_MWe
    hr_mult = ngcc_hr_multiplier(result.P_NGCC_elec_MW / ngcc_cap)
    expected = (
        result.P_NGCC_elec_MW
        * cfg.base.time.delta_t
        * cfg.case.ngcc.co2_direct_g_per_kwh_e
        * hr_mult
    )
    diff = (result.direct_emissions_kg_co2_per_h - expected).abs().max()
    assert diff < 1e-6


def test_case3_lifecycle_emissions_include_upstream_ch4(cfg, ts_2023_168h):
    """Lifecycle CO2 includes upstream methane leakage (Alvarez 2018, +60 g/kWh)."""
    result = solve_case3(cfg, ts_2023_168h)
    assert result.co2_lifecycle_annual_tonnes > result.co2_direct_annual_tonnes
    ratio = result.co2_lifecycle_annual_tonnes / result.co2_direct_annual_tonnes
    # 360 direct + 60 upstream → lifecycle 420 → ratio 420/360 = 1.167
    assert 1.10 < ratio < 1.25


def test_case3_tac_components_sum_correctly(cfg, ts_2023_168h):
    result = solve_case3(cfg, ts_2023_168h)
    parts = (
        result.capex_annual_usd
        + result.fom_annual_usd
        + result.vom_annual_usd
        + result.fuel_annual_usd
        + result.carbon_annual_usd
    )
    assert parts == pytest.approx(result.tac_usd_per_yr, rel=1e-9)


def test_case3_pue_setpoint_does_not_change_fuel_or_tac(cfg, ts_2023_168h):
    """Option A: PUE is decoupled from the cooling load, so the pue argument must
    not change Case 3 fuel burn or TAC (the heat the chiller rejects is fixed by
    P_IT / eta_chain regardless of the reported PUE)."""
    r110 = solve_case3(cfg, ts_2023_168h, pue=1.10)
    r150 = solve_case3(cfg, ts_2023_168h, pue=1.50)
    assert r150.fuel_annual_usd == pytest.approx(r110.fuel_annual_usd)
    assert r150.tac_usd_per_yr == pytest.approx(r110.tac_usd_per_yr)
