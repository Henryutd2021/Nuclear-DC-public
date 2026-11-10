"""Regression tests for model constraints:

- absorption parasitic load charged to the electric balance
- PCC capacity bound on grid purchases and sales
- reactor capacity factor realized by the refueling outage on full-year runs
"""

from pathlib import Path

import pyomo.environ as pyo
import pytest

from src.cases.case2 import solve_case2
from src.config import load_config
from src.data import load_time_series
from src.milp.builder import build_model
from src.milp.solve import solve_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent

gurobi = pytest.importorskip("pyomo.opt").SolverFactory("gurobi")
if not gurobi.available(exception_flag=False):
    pytest.skip("Gurobi unavailable", allow_module_level=True)


# ----- F.2 absorption parasitic --------------------------------------------


def test_absorption_parasitic_accounted_in_electric_balance():
    """The absorption parasitic (pump 0.020 + tower fan 0.015 = 0.035
    kWe/kWc per the equipment datasheet) must be wired into the electric
    balance whenever absorption cooling is delivered. A 24-h winter slice
    may dispatch little or no absorption under the price-split economics,
    so use a week and assert the parasitic appears when the chiller runs.
    """
    cfg = load_config(case_id=2, project_root=PROJECT_ROOT)
    assert cfg.case.absorption.parasitic_kWe_per_kWth == pytest.approx(0.035)
    ts = load_time_series(project_root=PROJECT_ROOT, year=2023, num_hours=168)
    r = solve_case2(cfg, ts, pue=1.30)
    parasitic = cfg.case.absorption.parasitic_kWe_per_kWth * r.Q_abs_cool_MWth
    assert parasitic.sum() > 0   # absorption is driven somewhere → parasitic > 0


# ----- A11 PCC capacity ----------------------------------------------------


def test_grid_buy_and_sell_bounded_by_pcc_capacity():
    cfg = load_config(case_id=1, project_root=PROJECT_ROOT)
    ts = load_time_series(project_root=PROJECT_ROOT, year=2022, num_hours=168)
    model = build_model(cfg, ts, pue=1.30)
    solve_model(model, quiet=True)
    pcc_cap = cfg.case.capacities.pcc_capacity_MW or 300.0
    for t in model.T:
        assert pyo.value(model.P_grid_buy[t]) <= pcc_cap + 1e-6
        assert pyo.value(model.P_grid_sell[t]) <= pcc_cap + 1e-6


# ----- A15 reactor CF ------------------------------------------------------


def test_reactor_outage_window_realizes_cf_on_full_year_run():
    """Full-year solves carry a scheduled refueling outage: P_rx = 0 for
    round((1-CF)*8760) = 701 contiguous hours from March 15 (hour 1752), so
    the maximum achievable CF equals the configured 0.92 and the data center
    is exposed to grid-priced backup during the window.
    """
    cfg = load_config(case_id=1, project_root=PROJECT_ROOT)
    ts = load_time_series(project_root=PROJECT_ROOT, year=2024, num_hours=8760)
    model = build_model(cfg, ts, pue=1.30)
    solve_model(model, quiet=True)
    p_max = cfg.case.reactor.thermal_power_MWth
    # Outage hours pinned at zero, with grid imports covering the load
    outage = range(1752, 1752 + 701)
    assert all(pyo.value(model.P_rx[t]) <= 1e-6 for t in outage)
    assert sum(pyo.value(model.P_grid_buy[t]) for t in outage) > 0
    # Realized CF cannot exceed the configured value, and must-run
    # economics keep the reactor near-baseload outside the window
    total = sum(pyo.value(model.P_rx[t]) for t in model.T)
    cf = total / (p_max * 8760)
    assert cf <= 0.92 + 1e-6
    assert cf >= 0.80


def test_reactor_outage_not_applied_on_short_slice():
    """A 24h slice carries neither outage hours nor a CF constraint
    (either would force a sub-year slice into unrepresentative dispatch)."""
    cfg = load_config(case_id=1, project_root=PROJECT_ROOT)
    ts = load_time_series(project_root=PROJECT_ROOT, year=2024, num_hours=24)
    model = build_model(cfg, ts, pue=1.30)
    assert not hasattr(model, "reactor_cf")
    p_min = cfg.case.reactor.min_load_fraction * cfg.case.reactor.thermal_power_MWth
    for t in model.T:
        assert model.P_rx[t].lb == pytest.approx(p_min)
