"""Tests for src.cases.case2 - cascaded HP extraction + absorption + VCC."""

from pathlib import Path

import pytest

from src.cases.case2 import solve_case2
from src.config import load_config
from src.data import load_time_series

PROJECT_ROOT = Path(__file__).resolve().parent.parent

gurobi = pytest.importorskip("pyomo.opt").SolverFactory("gurobi")
if not gurobi.available(exception_flag=False):
    pytest.skip("Gurobi unavailable", allow_module_level=True)


@pytest.fixture(scope="module")
def cfg():
    return load_config(case_id=2, project_root=PROJECT_ROOT)


@pytest.fixture(scope="module")
def ts_2023_168h():
    return load_time_series(project_root=PROJECT_ROOT, year=2023, num_hours=168)


def test_case2_solve_returns_optimal(cfg, ts_2023_168h):
    r = solve_case2(cfg, ts_2023_168h)
    assert r.tac_usd_per_yr > 0
    assert r.case_id == 2


def test_case2_electric_balance_closes(cfg, ts_2023_168h):
    """P_turb_net + P_grid_buy = P_IT + P_VCC + abs_parasitic + P_grid_sell.

    No ORC stream; turbine Willans-line already nets the extraction
    penalty into P_turb_net, so it appears once in the balance.
    """
    r = solve_case2(cfg, ts_2023_168h)
    parasitic = cfg.case.absorption.parasitic_kWe_per_kWth * r.Q_abs_cool_MWth
    residual = (
        r.P_turb_net_MW
        + r.P_grid_buy_MW
        - r.P_IT_MW
        - r.P_VCC_elec_MW
        - parasitic
        - r.P_grid_sell_MW
    ).abs().max()
    assert residual < 1e-4


def test_case2_cooling_balance_combines_abs_and_vcc(cfg, ts_2023_168h):
    """Q_abs_cool + Q_VCC_cool == Q_cool_demand every hour."""
    r = solve_case2(cfg, ts_2023_168h)
    residual = (r.Q_abs_cool_MWth + r.Q_VCC_cool_MWth - r.Q_cool_demand_MWth).abs().max()
    assert residual < 1e-4


def test_case2_reactor_min_load_and_ramp(cfg, ts_2023_168h):
    r = solve_case2(cfg, ts_2023_168h)
    assert r.P_rx_MWth.min() >= 435.0 - 1e-4
    deltas = r.P_rx_MWth.diff().abs().dropna()
    assert deltas.max() <= 261.0 + 1e-3


def test_case2_absorption_active_in_normal_hours(cfg, ts_2023_168h):
    """Absorption dispatches where it is economic, not unconditionally.

    Under the flat 45Y generation credit, diverting steam forfeits the
    credit on lost output while the VCC's draw only costs the LMP, so the
    chillers split by price: the VCC serves cheap-LMP hours and absorption
    takes over as the LMP (and the VCC's part-load marginal) rises. For a
    mixed-price week the aggregate share sits well inside (0, 1).
    """
    r = solve_case2(cfg, ts_2023_168h)
    abs_share = r.Q_abs_cool_MWth.sum() / r.Q_cool_demand_MWth.sum()
    assert 0.05 < abs_share < 0.95
    # Absorption actually runs in a meaningful number of hours...
    assert (r.Q_abs_cool_MWth > 1.0).sum() >= 24
    # ...and delivered cooling respects the installed nameplate.
    assert r.Q_abs_cool_MWth.max() <= 160.0 + 1e-6


def test_case2_capex_includes_absorption(cfg, ts_2023_168h):
    """Case 2 capex exceeds Case 1 by exactly the absorption chiller's
    own-lifetime annualized CAPEX. The VCC capacity is identical in both cases
    (160 MWth) and the reactor/turbine are shared, so every other capital line
    cancels in the difference."""
    from src.cases.case1 import solve_case1
    from src.finance import annualized_capex

    cfg1 = load_config(case_id=1, project_root=PROJECT_ROOT)
    r1 = solve_case1(cfg1, ts_2023_168h)
    r2 = solve_case2(cfg, ts_2023_168h)
    ab = cfg.case.absorption
    abs_cap = cfg.case.capacities.absorption_capacity_MWth
    expected = annualized_capex(
        ab.capex_usd_per_kWth * abs_cap * 1000.0,
        cfg.financial.WACC_nominal,
        ab.lifetime_years,
    )
    delta = r2.capex_annual_usd - r1.capex_annual_usd
    assert delta == pytest.approx(expected, rel=1e-9)
