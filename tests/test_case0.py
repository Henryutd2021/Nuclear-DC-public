"""Tests for src.cases.case0 - grid-only baseline dispatch + TAC."""

from pathlib import Path

import pytest

from src.cases.case0 import solve_case0
from src.config import load_config
from src.data import load_time_series

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def cfg():
    return load_config(case_id=0, project_root=PROJECT_ROOT)


@pytest.fixture(scope="module")
def ts_2023_168h():
    return load_time_series(
        project_root=PROJECT_ROOT, year=2023, num_hours=168
    )


def test_case0_smoke_returns_result(cfg, ts_2023_168h):
    result = solve_case0(cfg, ts_2023_168h)
    assert result.tac_usd_per_yr > 0


def test_case0_dispatch_length_matches_horizon(cfg, ts_2023_168h):
    result = solve_case0(cfg, ts_2023_168h)
    assert len(result.P_grid_buy_MW) == 168
    assert len(result.Q_cool_MWth) == 168
    assert len(result.P_VCC_elec_MW) == 168
    assert len(result.grid_cost_usd_per_h) == 168
    assert len(result.grid_emissions_kg_co2_per_h) == 168


def test_case0_energy_balance_closes(cfg, ts_2023_168h):
    """P_grid_buy == P_IT + P_VCC for every hour."""
    result = solve_case0(cfg, ts_2023_168h)
    residual = (
        result.P_grid_buy_MW - result.P_IT_MW - result.P_VCC_elec_MW
    ).abs().max()
    assert residual < 1e-9


def test_case0_cooling_load_matches_eta_chain_definition(cfg, ts_2023_168h):
    """Q_cool(t) = P_IT(t) / eta_chain (Option A: the heat the chiller rejects is
    the full IT draw scaled by distribution losses; PUE is a reported outcome,
    not a driver of the cooling load)."""
    result = solve_case0(cfg, ts_2023_168h, pue=1.30)
    eta = cfg.base.physics.cooling_chain_efficiency
    expected = result.P_IT_MW / eta
    diff = (result.Q_cool_MWth - expected).abs().max()
    assert diff < 1e-9


def test_case0_vcc_power_uses_houston_derated_partload_cop(cfg, ts_2023_168h):
    """P_VCC = Q_cool / (COP(load) × relief(T_wb)): the design-point COP
    evaluated at the realized cooling-load fraction (the IPLV part-load
    curve) times the hourly wet-bulb relief, not a flat full-load value."""
    from src.performance import vcc_cop_at_load, vcc_wet_bulb_relief

    result = solve_case0(cfg, ts_2023_168h)
    q_max = cfg.case.capacities.electric_chiller_capacity_MWth
    vcc = cfg.case.vcc
    relief = vcc_wet_bulb_relief(
        ts_2023_168h.wet_bulb_C.to_numpy(),
        vcc.cop_design_wet_bulb_C,
        vcc.cop_wet_bulb_relief_per_K,
        vcc.cop_wet_bulb_relief_cap,
    )
    cop_load = vcc_cop_at_load(result.Q_cool_MWth / q_max, vcc.cop_houston) * relief
    expected = result.Q_cool_MWth / cop_load
    diff = (result.P_VCC_elec_MW - expected).abs().max()
    assert diff < 1e-9


def test_case0_tac_decomposition_sums_to_total(cfg, ts_2023_168h):
    result = solve_case0(cfg, ts_2023_168h)
    parts = (
        result.capex_annual_usd
        + result.fom_annual_usd
        + result.vom_annual_usd
        + result.grid_annual_usd
        + result.carbon_annual_usd
    )
    assert parts == pytest.approx(result.tac_usd_per_yr, rel=1e-9)


def test_case0_pue_setpoint_does_not_change_dispatch(cfg, ts_2023_168h):
    """Option A regression guard: PUE is decoupled from the cooling load, so the
    pue argument must not alter dispatch. Grid consumption, cost, and emissions
    are identical regardless of the PUE setpoint (S1 now sweeps cooling COP)."""
    r110 = solve_case0(cfg, ts_2023_168h, pue=1.10)
    r150 = solve_case0(cfg, ts_2023_168h, pue=1.50)
    assert r150.P_grid_buy_MW.sum() == pytest.approx(r110.P_grid_buy_MW.sum())
    assert r150.grid_annual_usd == pytest.approx(r110.grid_annual_usd)
    assert r150.co2_annual_tonnes == pytest.approx(r110.co2_annual_tonnes)


def test_case0_grid_cost_per_buy_equals_lmp(cfg, ts_2023_168h):
    """cost / P_grid_buy must equal LMP × dt for every hour."""
    result = solve_case0(cfg, ts_2023_168h)
    cost_per_buy = result.grid_cost_usd_per_h / result.P_grid_buy_MW
    expected = ts_2023_168h.price_import_usd_per_mwh * cfg.base.time.delta_t
    diff = (cost_per_buy - expected).abs().max()
    assert diff < 1e-6


def test_case0_emissions_unit_check(cfg, ts_2023_168h):
    """emissions [kg] = P_grid_buy [MW] × dt [h] × carbon [g/kWh] (units: g/kWh × MWh × kWh/MWh / 1000)."""
    result = solve_case0(cfg, ts_2023_168h)
    expected = (
        result.P_grid_buy_MW
        * cfg.base.time.delta_t
        * ts_2023_168h.carbon_intensity_g_per_kwh
    )
    diff = (result.grid_emissions_kg_co2_per_h - expected).abs().max()
    assert diff < 1e-6


def test_case0_2022_grid_cost_exceeds_2024_full_year(cfg):
    """Memory: 2022 mean DA ≈ $80; 2024 mean DA ≈ $28 - full-year cost must reflect."""
    ts22 = load_time_series(project_root=PROJECT_ROOT, year=2022, num_hours=8760)
    ts24 = load_time_series(project_root=PROJECT_ROOT, year=2024, num_hours=8760)
    r22 = solve_case0(cfg, ts22)
    r24 = solve_case0(cfg, ts24)
    assert r22.grid_annual_usd > r24.grid_annual_usd


def test_case0_zero_load_hour_yields_zero_vcc_power_not_nan(cfg, ts_2023_168h):
    """An idle hour (P_IT = 0) must give P_VCC = 0, not 0/0 = NaN: the
    relative-COP curve passes through (0, 0), so the unguarded division
    would silently propagate NaN into grid cost and emissions."""
    import dataclasses

    it_load = ts_2023_168h.it_load_MW.copy()
    it_load.iloc[10] = 0.0
    ts_idle = dataclasses.replace(ts_2023_168h, it_load_MW=it_load)

    result = solve_case0(cfg, ts_idle)
    assert not result.P_VCC_elec_MW.isna().any()
    assert not result.P_grid_buy_MW.isna().any()
    assert result.P_VCC_elec_MW.iloc[10] == 0.0
    assert result.grid_cost_usd_per_h.iloc[10] == 0.0
