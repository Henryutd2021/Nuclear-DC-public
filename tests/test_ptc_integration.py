"""Integration tests for the Section 45Y clean-electricity PTC wired into the
Cases 1-2 MILP. The credit is a flat per-MWh reduction on net nuclear
generation, levelized over the TAC window, default-on in the baseline.
"""

from pathlib import Path

import pyomo.environ as pyo
import pytest

from src.config import load_config, with_nuclear_ptc
from src.data import load_time_series
from src.finance import levelized_ptc_usd_per_mwh
from src.milp.builder import build_model
from src.milp.solve import solve_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_GUROBI = pyo.SolverFactory("gurobi")
_HAS_GUROBI = _GUROBI.available(exception_flag=False)
_solver_skip = pytest.mark.skipif(not _HAS_GUROBI, reason="Gurobi not available")


@pytest.fixture(scope="module")
def ts():
    return load_time_series(project_root=PROJECT_ROOT, year=2023, num_hours=72)


def test_ptc_default_enabled():
    cfg = load_config(case_id=1, project_root=PROJECT_ROOT)
    assert cfg.financial.nuclear_ptc_enabled is True
    assert cfg.financial.ptc_usd_per_mwh_assumed == pytest.approx(30.0)
    assert cfg.financial.ptc_credit_duration_years == 10


def test_ptc_expression_present_on_model(ts):
    cfg = load_config(case_id=1, project_root=PROJECT_ROOT)
    m = build_model(cfg, ts, pue=1.30)
    assert hasattr(m, "ptc_annual")


@_solver_skip
def test_ptc_credit_matches_levelized_flat_rate(ts):
    cfg = load_config(case_id=1, project_root=PROJECT_ROOT)
    m = build_model(cfg, ts, pue=1.30)
    solve_model(m, solver_config=cfg.base.solver)
    fin = cfg.financial
    dt = cfg.base.time.delta_t
    annual_scale = 8760.0 / ts.num_hours
    rate = levelized_ptc_usd_per_mwh(
        fin.ptc_usd_per_mwh_assumed,
        cfg.financial.WACC_nominal,
        fin.ptc_credit_duration_years,
        fin.project_lifetime_years,
    )
    expected = (
        -sum(rate * pyo.value(m.P_turb_net[t]) for t in m.T) * dt * annual_scale
    )
    assert pyo.value(m.ptc_annual) == pytest.approx(expected, rel=1e-9)
    assert pyo.value(m.ptc_annual) < 0.0  # it is a credit


@_solver_skip
def test_ptc_lowers_tac_vs_disabled(ts):
    cfg_on = load_config(case_id=1, project_root=PROJECT_ROOT)
    cfg_off = with_nuclear_ptc(cfg_on, False)
    m_on = build_model(cfg_on, ts, pue=1.30)
    m_off = build_model(cfg_off, ts, pue=1.30)
    solve_model(m_on, solver_config=cfg_on.base.solver)
    solve_model(m_off, solver_config=cfg_off.base.solver)
    assert pyo.value(m_on.objective) < pyo.value(m_off.objective)


def test_ptc_disabled_zeroes_credit(ts):
    cfg = with_nuclear_ptc(load_config(case_id=1, project_root=PROJECT_ROOT), False)
    m = build_model(cfg, ts, pue=1.30)
    assert pyo.value(m.ptc_annual) == pytest.approx(0.0)


@_solver_skip
def test_result_surfaces_ptc_and_tac_decomposes(ts):
    from src.cases.case1 import solve_case1

    cfg = load_config(case_id=1, project_root=PROJECT_ROOT)
    res = solve_case1(cfg, ts, pue=1.30)
    assert res.ptc_annual_usd < 0.0  # a credit
    total = (
        res.capex_annual_usd
        + res.fom_annual_usd
        + res.vom_annual_usd
        + res.fuel_annual_usd
        + res.grid_annual_usd
        + res.carbon_annual_usd
        + res.ptc_annual_usd
    )
    assert total == pytest.approx(res.tac_usd_per_yr, rel=1e-9)
