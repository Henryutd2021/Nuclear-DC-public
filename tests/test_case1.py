"""Tests for src.cases.case1 - Pyomo MILP for nuclear-only, no heat recovery."""

from pathlib import Path

import pytest

from src.cases.case1 import solve_case1
from src.config import load_config
from src.data import load_time_series

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Skip the whole module if no Gurobi license is available locally
gurobi = pytest.importorskip("pyomo.opt").SolverFactory("gurobi")
if not gurobi.available(exception_flag=False):
    pytest.skip(
        "Gurobi solver unavailable; Case 1+ tests need it",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def cfg():
    return load_config(case_id=1, project_root=PROJECT_ROOT)


@pytest.fixture(scope="module")
def ts_2023_168h():
    return load_time_series(project_root=PROJECT_ROOT, year=2023, num_hours=168)


def test_case1_24h_solve_returns_optimal(cfg):
    ts = load_time_series(project_root=PROJECT_ROOT, year=2023, num_hours=24)
    result = solve_case1(cfg, ts)
    assert result.tac_usd_per_yr > 0


def test_case1_electric_balance_closes(cfg, ts_2023_168h):
    """Reactor + grid_buy = IT + VCC + grid_sell every hour."""
    result = solve_case1(cfg, ts_2023_168h)
    residual = (
        result.P_turb_net_MW
        + result.P_grid_buy_MW
        - result.P_IT_MW
        - result.P_VCC_elec_MW
        - result.P_grid_sell_MW
    ).abs().max()
    assert residual < 1e-4


def test_case1_cooling_balance_via_vcc_only(cfg, ts_2023_168h):
    """No absorption in Case 1 → Q_VCC_cool == Q_cool_demand every hour."""
    result = solve_case1(cfg, ts_2023_168h)
    residual = (result.Q_VCC_cool_MWth - result.Q_cool_demand_MWth).abs().max()
    assert residual < 1e-4


def test_case1_reactor_respects_min_load(cfg, ts_2023_168h):
    """BWRX-300 min-load constraint: P_rx ≥ 50% × 870 = 435 MWth every hour."""
    result = solve_case1(cfg, ts_2023_168h)
    assert result.P_rx_MWth.min() >= 435.0 - 1e-4


def test_case1_reactor_respects_ramp(cfg, ts_2023_168h):
    """Vendor ramp: |dP_rx/dt| ≤ 0.5%/min × 60 × 870 = 261 MWth/h."""
    result = solve_case1(cfg, ts_2023_168h)
    deltas = result.P_rx_MWth.diff().abs().dropna()
    assert deltas.max() <= 261.0 + 1e-3


def test_case1_capex_dominated_by_reactor(cfg, ts_2023_168h):
    """BWRX OCC $7,615/kWe × 300 MWe, amortized over 40 years, ≈ $165M/yr."""
    result = solve_case1(cfg, ts_2023_168h)
    assert result.capex_annual_usd > 1.5e8  # at least $150M
    # Most of TAC is CAPEX + FOM at modest production
    assert (result.capex_annual_usd + result.fom_annual_usd) > 0.5 * result.tac_usd_per_yr


def test_case1_tac_far_higher_than_case0_at_atb_mid(cfg, ts_2023_168h):
    """At ATB-Mid CAPEX, building nuclear w/o cogen loses badly to grid-only."""
    from src.cases.case0 import solve_case0
    cfg0 = load_config(case_id=0, project_root=PROJECT_ROOT)
    r1 = solve_case1(cfg, ts_2023_168h)
    r0 = solve_case0(cfg0, ts_2023_168h)
    assert r1.tac_usd_per_yr > 2.0 * r0.tac_usd_per_yr
