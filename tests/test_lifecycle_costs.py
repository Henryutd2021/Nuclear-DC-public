"""Integration tests for lifecycle costing in the solvers: per-component capital
recovery, BESS salvage credit and absorber maintenance availability."""

from pathlib import Path

import pyomo.environ as pyo
import pytest

from src.cases.case0 import solve_case0
from src.cases.case3 import solve_case3
from src.config import load_config, with_bess
from src.data import load_time_series
from src.finance import annualized_capex, crf
from src.milp.builder import build_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def ts():
    return load_time_series(project_root=PROJECT_ROOT, year=2023, num_hours=168)


# ----------------------------------------------------------------------------
# Each asset amortized over its own lifetime_years, not a global 20-yr CRF
# ----------------------------------------------------------------------------
def test_case0_vcc_capex_amortized_over_vcc_lifetime(ts):
    cfg = load_config(case_id=0, project_root=PROJECT_ROOT)
    res = solve_case0(cfg, ts)
    vcc = cfg.case.vcc
    cap = cfg.case.capacities.electric_chiller_capacity_MWth
    wacc = cfg.financial.WACC_nominal
    expected = annualized_capex(
        vcc.capex_usd_per_kWth * cap * 1000.0, wacc, vcc.lifetime_years
    )
    assert res.capex_annual_usd == pytest.approx(expected, rel=1e-9)


def test_case3_capex_uses_component_lifetimes(ts):
    cfg = load_config(case_id=3, project_root=PROJECT_ROOT)
    res = solve_case3(cfg, ts)
    ngcc, vcc = cfg.case.ngcc, cfg.case.vcc
    caps = cfg.case.capacities
    wacc = cfg.financial.WACC_nominal
    expected = annualized_capex(
        ngcc.capex_usd_per_kWe * caps.ngcc_capacity_MWe * 1000.0,
        wacc,
        ngcc.lifetime_years,
    ) + annualized_capex(
        vcc.capex_usd_per_kWth * caps.electric_chiller_capacity_MWth * 1000.0,
        wacc,
        vcc.lifetime_years,
    )
    assert res.capex_annual_usd == pytest.approx(expected, rel=1e-9)


def test_case3_ngcc_30yr_life_is_cheaper_than_old_global_20yr(ts):
    cfg = load_config(case_id=3, project_root=PROJECT_ROOT)
    ngcc = cfg.case.ngcc
    caps = cfg.case.capacities
    wacc = cfg.financial.WACC_nominal
    ngcc_capex = ngcc.capex_usd_per_kWe * caps.ngcc_capacity_MWe * 1000.0
    assert ngcc.lifetime_years == 30
    assert ngcc_capex * crf(wacc, 30) < ngcc_capex * crf(wacc, 20)


def test_case1_reactor_capex_at_40yr_and_vcc_at_25yr(ts):
    cfg = load_config(case_id=1, project_root=PROJECT_ROOT)
    m = build_model(cfg, ts, pue=1.30)
    rx, vcc = cfg.case.reactor, cfg.case.vcc
    wacc = cfg.financial.WACC_nominal
    cap_vcc = cfg.case.capacities.electric_chiller_capacity_MWth
    # Case 1 = reactor + turbine(capex 0) + VCC, no absorption, no BESS.
    expected = annualized_capex(
        rx.capex_usd_per_kWe * rx.electric_power_net_MWe * 1000.0,
        wacc,
        rx.lifetime_years,
    ) + annualized_capex(
        vcc.capex_usd_per_kWth * cap_vcc * 1000.0, wacc, vcc.lifetime_years
    )
    assert rx.lifetime_years == 40  # NLR ATB 2024 SMR amortization convention
    assert pyo.value(m.capex_annual) == pytest.approx(expected, rel=1e-9)


# ----------------------------------------------------------------------------
# BESS carries an end-of-life salvage credit
# ----------------------------------------------------------------------------
def test_bess_capex_includes_salvage_credit(ts):
    cfg = with_bess(load_config(case_id=1, project_root=PROJECT_ROOT), enabled=True)
    m = build_model(cfg, ts, pue=1.30)
    bs = cfg.case.bess
    caps = cfg.case.capacities
    wacc = cfg.financial.WACC_nominal
    sf = bs.end_of_life_credit_pct / 100.0
    assert sf > 0.0  # BESS must declare a salvage fraction
    bess_capex = bs.capex_usd_per_kwh * caps.bess_capacity_MWh * 1000.0
    bess_with_salvage = annualized_capex(bess_capex, wacc, bs.lifetime_years, sf)
    bess_no_salvage = annualized_capex(bess_capex, wacc, bs.lifetime_years, 0.0)
    # Whole-plant capex with BESS must reflect the salvage-credited BESS term.
    cfg_nobess = load_config(case_id=1, project_root=PROJECT_ROOT)
    m_nobess = build_model(cfg_nobess, ts, pue=1.30)
    bess_contribution = pyo.value(m.capex_annual) - pyo.value(m_nobess.capex_annual)
    assert bess_contribution == pytest.approx(bess_with_salvage, rel=1e-9)
    assert bess_contribution < bess_no_salvage


# ----------------------------------------------------------------------------
# Absorption chiller maintenance coincides with the refueling outage
# ----------------------------------------------------------------------------
def test_absorption_availability_is_unity_with_coincident_maintenance():
    # Absorber maintenance is scheduled inside the reactor refueling outage
    # window (it has no steam source then), so no separate continuous
    # availability derate is applied to its hourly yield.
    cfg = load_config(case_id=2, project_root=PROJECT_ROOT)
    assert cfg.case.absorption.availability == pytest.approx(1.0)


def test_absorption_available_param_is_gate_times_availability(ts):
    cfg = load_config(case_id=2, project_root=PROJECT_ROOT)
    m = build_model(cfg, ts, pue=1.30)
    avail = cfg.case.absorption.availability
    # In hours when the crystallization gate is open (and outside the outage
    # window) the availability param equals the configured factor; otherwise 0.
    vals = {pyo.value(m.absorption_available[t]) for t in m.T}
    assert vals <= {0.0, avail}
    assert avail in vals  # at least some hours are crystallization-open
