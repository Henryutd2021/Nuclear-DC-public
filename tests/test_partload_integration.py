"""Integration tests for part-load efficiency wired into the solvers.

VCC in the nuclear MILP (Cases 1-2) is an SOS2 piecewise map P_vcc = f(Q_vcc),
so the solved electricity must lie on the non-convex IPLV curve rather than a
single full-load COP ratio. Cases 0 and 3 are closed-form and look the COP up
at the realized cooling-load fraction. NGCC fuel in Case 3 follows the
part-load heat-rate curve.
"""

from pathlib import Path

import numpy as np
import pyomo.environ as pyo
import pytest

from src.cases.case0 import solve_case0
from src.cases.case3 import solve_case3
from src.config import load_config
from src.data import load_time_series
from src.milp.builder import build_model
from src.milp.solve import solve_model
from src.performance import (
    ngcc_efficiency_at_load,
    vcc_cop_at_load,
    vcc_pwl_points,
    vcc_wet_bulb_relief,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _relief(cfg, ts, t):
    """Hourly wet-bulb relief multiplier on the design-point VCC COP."""
    vcc = cfg.case.vcc
    return float(
        vcc_wet_bulb_relief(
            float(ts.wet_bulb_C.iloc[t]),
            vcc.cop_design_wet_bulb_C,
            vcc.cop_wet_bulb_relief_per_K,
            vcc.cop_wet_bulb_relief_cap,
        )
    )

_GUROBI = pyo.SolverFactory("gurobi")
_HAS_GUROBI = _GUROBI.available(exception_flag=False)
_solver_skip = pytest.mark.skipif(not _HAS_GUROBI, reason="Gurobi not available")


@pytest.fixture(scope="module")
def ts():
    # 72 h keeps the SOS2 MILP small while still spanning a real load swing.
    return load_time_series(project_root=PROJECT_ROOT, year=2023, num_hours=72)


# ----------------------------------------------------------------------------
# Builder (Cases 1-2): VCC is an SOS2 piecewise, solved P_vcc follows the curve
# ----------------------------------------------------------------------------
@_solver_skip
def test_case1_vcc_solves_and_follows_partload_curve(ts):
    cfg = load_config(case_id=1, project_root=PROJECT_ROOT)
    m = build_model(cfg, ts, pue=1.30)
    # The SOS2 piecewise makes this a MILP, not the old pure LP.
    assert hasattr(m, "vcc_pwl")
    solve_model(m, solver_config=cfg.base.solver)

    vcc = cfg.case.vcc
    q_max = cfg.case.capacities.electric_chiller_capacity_MWth
    # The MILP follows the SOS2 piecewise (linear interpolation of the same
    # breakpoints), an approximation of the smooth q/COP(load) relation that is
    # exact at the breakpoints. Verify the solution lies on that piecewise map.
    q_pts, p_pts = vcc_pwl_points(vcc.cop_houston, q_max)
    checked = 0
    for t in m.T:
        q = pyo.value(m.Q_vcc_cool[t])
        p = pyo.value(m.P_vcc[t])
        if q > 1e-6:
            expected = float(np.interp(q, q_pts, p_pts)) / _relief(cfg, ts, t)
            assert p == pytest.approx(expected, abs=1e-4, rel=1e-4)
            checked += 1
    assert checked > 0  # VCC actually carried cooling in some hour


@_solver_skip
def test_case1_partload_vcc_uses_less_power_than_flat_cop(ts):
    # Economic consequence: at part load the IPLV hump gives a higher COP, so
    # total VCC electricity is below the naive full-load-COP estimate.
    cfg = load_config(case_id=1, project_root=PROJECT_ROOT)
    m = build_model(cfg, ts, pue=1.30)
    solve_model(m, solver_config=cfg.base.solver)
    vcc = cfg.case.vcc
    p_actual = sum(pyo.value(m.P_vcc[t]) for t in m.T)
    q_total = sum(pyo.value(m.Q_vcc_cool[t]) for t in m.T)
    p_flat = q_total / vcc.cop_houston
    assert p_actual < p_flat


# ----------------------------------------------------------------------------
# Case 0 (closed-form): VCC electricity uses the load-dependent COP
# ----------------------------------------------------------------------------
def test_case0_vcc_power_uses_partload_cop(ts):
    cfg = load_config(case_id=0, project_root=PROJECT_ROOT)
    res = solve_case0(cfg, ts)
    vcc = cfg.case.vcc
    q_max = cfg.case.capacities.electric_chiller_capacity_MWth
    for t, (q, p) in enumerate(zip(res.Q_cool_MWth, res.P_VCC_elec_MW)):
        if q > 1e-9:
            cop_here = float(vcc_cop_at_load(q / q_max, vcc.cop_houston))
            cop_here *= _relief(cfg, ts, t)
            assert p == pytest.approx(q / cop_here, rel=1e-9)


def test_case0_partload_lowers_vcc_energy_vs_flat_cop(ts):
    cfg = load_config(case_id=0, project_root=PROJECT_ROOT)
    res = solve_case0(cfg, ts)
    vcc = cfg.case.vcc
    p_flat = res.Q_cool_MWth.sum() / vcc.cop_houston
    assert res.P_VCC_elec_MW.sum() < p_flat


# ----------------------------------------------------------------------------
# Case 3 (closed-form): VCC COP + NGCC heat rate both load-dependent
# ----------------------------------------------------------------------------
def test_case3_vcc_power_uses_partload_cop(ts):
    cfg = load_config(case_id=3, project_root=PROJECT_ROOT)
    res = solve_case3(cfg, ts)
    vcc = cfg.case.vcc
    q_max = cfg.case.capacities.electric_chiller_capacity_MWth
    for t, (q, p) in enumerate(zip(res.Q_cool_MWth, res.P_VCC_elec_MW)):
        if q > 1e-9:
            cop_here = float(vcc_cop_at_load(q / q_max, vcc.cop_houston))
            cop_here *= _relief(cfg, ts, t)
            assert p == pytest.approx(q / cop_here, rel=1e-9)


def test_case3_ngcc_fuel_uses_partload_heat_rate(ts):
    cfg = load_config(case_id=3, project_root=PROJECT_ROOT)
    res = solve_case3(cfg, ts)
    ngcc = cfg.case.ngcc
    ngcc_cap = cfg.case.capacities.ngcc_capacity_MWe
    eta_full = ngcc.net_efficiency_hhv
    _MMBTU_PER_MWh = 3.412
    for p_ngcc, fuel in zip(res.P_NGCC_elec_MW, res.fuel_consumption_MMBtu_per_h):
        if p_ngcc > 1e-9:
            eta_here = float(ngcc_efficiency_at_load(p_ngcc / ngcc_cap, eta_full))
            assert fuel == pytest.approx(p_ngcc * _MMBTU_PER_MWh / eta_here, rel=1e-9)


def test_case3_partload_raises_fuel_vs_flat_efficiency(ts):
    # NGCC runs below design most hours, so part-load heat rate must raise total
    # fuel above the flat full-load-efficiency estimate.
    cfg = load_config(case_id=3, project_root=PROJECT_ROOT)
    res = solve_case3(cfg, ts)
    ngcc = cfg.case.ngcc
    _MMBTU_PER_MWh = 3.412
    fuel_flat = res.P_NGCC_elec_MW.sum() * _MMBTU_PER_MWh / ngcc.net_efficiency_hhv
    assert res.fuel_consumption_MMBtu_per_h.sum() > fuel_flat


def test_case3_emissions_track_fuel_burned(ts):
    # CO2 comes from burning fuel: direct emissions must stay proportional to
    # fuel consumed (constant carbon-per-MMBtu), so the part-load heat-rate
    # penalty raises emissions and fuel together rather than decoupling them.
    cfg = load_config(case_id=3, project_root=PROJECT_ROOT)
    res = solve_case3(cfg, ts)
    mask = res.fuel_consumption_MMBtu_per_h > 1e-9
    ratio = (
        res.direct_emissions_kg_co2_per_h[mask]
        / res.fuel_consumption_MMBtu_per_h[mask]
    )
    assert ratio.std() == pytest.approx(0.0, abs=1e-9)
    assert ratio.mean() > 0.0
