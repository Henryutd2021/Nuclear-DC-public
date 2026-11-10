"""Tests for the S4 reactor CAPEX sensitivity helper."""

from pathlib import Path

import pytest

from src.cases.case0 import solve_case0
from src.cases.case2 import solve_case2
from src.config import load_config, with_reactor_capex
from src.data import load_time_series
from src.kpi import heat_recovery_premium

PROJECT_ROOT = Path(__file__).resolve().parent.parent

gurobi = pytest.importorskip("pyomo.opt").SolverFactory("gurobi")
if not gurobi.available(exception_flag=False):
    pytest.skip("Gurobi unavailable", allow_module_level=True)


def test_with_reactor_capex_foak_sets_14700():
    cfg = load_config(case_id=2, project_root=PROJECT_ROOT)
    cfg_foak = with_reactor_capex(cfg, "FOAK", PROJECT_ROOT)
    assert cfg_foak.case.reactor.capex_usd_per_kWe == pytest.approx(14700.0, abs=1)


def test_with_reactor_capex_noak_sets_2250():
    cfg = load_config(case_id=2, project_root=PROJECT_ROOT)
    cfg_noak = with_reactor_capex(cfg, "NOAK", PROJECT_ROOT)
    assert cfg_noak.case.reactor.capex_usd_per_kWe == pytest.approx(2250.0, abs=1)


def test_with_reactor_capex_rejects_unknown_scenario():
    cfg = load_config(case_id=2, project_root=PROJECT_ROOT)
    with pytest.raises(ValueError, match=r"scenario"):
        with_reactor_capex(cfg, "Bargain_Bin", PROJECT_ROOT)


def test_with_reactor_capex_rejects_case_without_reactor():
    """Case 0 has no reactor block; helper must reject."""
    cfg0 = load_config(case_id=0, project_root=PROJECT_ROOT)
    with pytest.raises(ValueError, match=r"reactor block"):
        with_reactor_capex(cfg0, "NOAK", PROJECT_ROOT)


def test_noak_tac_lower_than_atb_mid_tac():
    """NOAK ($2,250) reactor capex should drop Case 2 TAC vs ATB-Mid ($7,615)."""
    cfg = load_config(case_id=2, project_root=PROJECT_ROOT)
    ts = load_time_series(project_root=PROJECT_ROOT, year=2023, num_hours=168)
    r_mid = solve_case2(cfg, ts, pue=1.30)
    r_noak = solve_case2(
        with_reactor_capex(cfg, "NOAK", PROJECT_ROOT), ts, pue=1.30
    )
    assert r_noak.tac_usd_per_yr < r_mid.tac_usd_per_yr


def test_noak_premium_closer_to_zero_than_atb_mid():
    """S4: NOAK closes the gap toward Premium=0 (or positive).

    At ATB-Mid Case 2 premium is roughly -230%; at NOAK it should be much
    less negative (still likely negative at 200 MW DC scale, but the
    direction must be correct).
    """
    cfg = load_config(case_id=2, project_root=PROJECT_ROOT)
    cfg0 = load_config(case_id=0, project_root=PROJECT_ROOT)
    ts = load_time_series(project_root=PROJECT_ROOT, year=2023, num_hours=168)
    r0 = solve_case0(cfg0, ts, pue=1.30)
    r_mid = solve_case2(cfg, ts, pue=1.30)
    r_noak = solve_case2(
        with_reactor_capex(cfg, "NOAK", PROJECT_ROOT), ts, pue=1.30
    )
    prem_mid = heat_recovery_premium(r0.tac_usd_per_yr, r_mid.tac_usd_per_yr)
    prem_noak = heat_recovery_premium(r0.tac_usd_per_yr, r_noak.tac_usd_per_yr)
    assert prem_noak > prem_mid, (
        f"Expected NOAK Premium ({prem_noak:.3f}) > ATB-Mid Premium "
        f"({prem_mid:.3f})"
    )
