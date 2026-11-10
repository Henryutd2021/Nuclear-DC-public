"""Tests for the BESS S3 binary sensitivity layer in src.milp.builder."""

from pathlib import Path

import pytest

from src.cases.case2 import solve_case2
from src.config import load_config, with_bess
from src.data import load_time_series
from src.milp.builder import build_model
from src.milp.solve import solve_model
import pyomo.environ as pyo

PROJECT_ROOT = Path(__file__).resolve().parent.parent

gurobi = pytest.importorskip("pyomo.opt").SolverFactory("gurobi")
if not gurobi.available(exception_flag=False):
    pytest.skip("Gurobi unavailable", allow_module_level=True)


@pytest.fixture(scope="module")
def cfg2_no_bess():
    return load_config(case_id=2, project_root=PROJECT_ROOT)


@pytest.fixture(scope="module")
def cfg2_bess(cfg2_no_bess):
    return with_bess(cfg2_no_bess, enabled=True)


@pytest.fixture(scope="module")
def ts_168h():
    return load_time_series(project_root=PROJECT_ROOT, year=2023, num_hours=168)


def _model(cfg, ts, pue=1.30):
    m = build_model(cfg, ts, pue=pue)
    solve_model(m, quiet=True)
    return m


def test_with_bess_helper_flips_flag(cfg2_no_bess, cfg2_bess):
    assert cfg2_no_bess.case.equipment.bess_enabled is False
    assert cfg2_bess.case.equipment.bess_enabled is True


def test_with_bess_requires_bess_block_in_yaml():
    """Calling with_bess on a case without a bess block must raise."""
    cfg0 = load_config(case_id=0, project_root=PROJECT_ROOT)
    with pytest.raises(ValueError, match=r"bess block"):
        with_bess(cfg0, enabled=True)


def test_bess_soc_dynamics_per_hour(cfg2_bess, ts_168h):
    """B_soc[t] = B_soc[t-1] × (1 - loss) + sqrt(η)·B_charge[t]·dt − B_discharge[t]/sqrt(η)·dt."""
    m = _model(cfg2_bess, ts_168h)
    bs = cfg2_bess.case.bess
    sqrt_eta = bs.round_trip_efficiency ** 0.5
    loss = bs.self_discharge_per_hour
    dt = m._dt
    for t in list(m.T)[1:]:
        prev = pyo.value(m.B_soc[t - 1])
        expected = prev * (1 - loss) + (
            sqrt_eta * pyo.value(m.B_charge[t]) - pyo.value(m.B_discharge[t]) / sqrt_eta
        ) * dt
        assert abs(pyo.value(m.B_soc[t]) - expected) < 1e-6


def test_bess_initial_soc_from_yaml(cfg2_bess, ts_168h):
    """B_soc[0] dynamics use initial_soc × Cap_E as the 'previous' value."""
    m = _model(cfg2_bess, ts_168h)
    bs = cfg2_bess.case.bess
    Cap_E = cfg2_bess.case.capacities.bess_capacity_MWh
    soc_init = bs.initial_soc_fraction * Cap_E
    sqrt_eta = bs.round_trip_efficiency ** 0.5
    loss = bs.self_discharge_per_hour
    dt = m._dt
    expected_first = soc_init * (1 - loss) + (
        sqrt_eta * pyo.value(m.B_charge[0]) - pyo.value(m.B_discharge[0]) / sqrt_eta
    ) * dt
    assert abs(pyo.value(m.B_soc[0]) - expected_first) < 1e-6


def test_bess_soc_within_bounds(cfg2_bess, ts_168h):
    m = _model(cfg2_bess, ts_168h)
    bs = cfg2_bess.case.bess
    Cap_E = cfg2_bess.case.capacities.bess_capacity_MWh
    lo = bs.soc_min_fraction * Cap_E - 1e-6
    hi = bs.soc_max_fraction * Cap_E + 1e-6
    for t in m.T:
        soc = pyo.value(m.B_soc[t])
        assert lo <= soc <= hi


def test_bess_no_simultaneous_charge_and_discharge(cfg2_bess, ts_168h):
    """LP formulation relies on round-trip loss to avoid simultaneity; verify it holds."""
    m = _model(cfg2_bess, ts_168h)
    for t in m.T:
        product = pyo.value(m.B_charge[t]) * pyo.value(m.B_discharge[t])
        # Allow tiny numerical noise but no meaningful overlap
        assert product < 1.0   # MW × MW; well-behaved Gurobi solutions should be much smaller


def test_bess_raises_tac_by_capex(cfg2_no_bess, cfg2_bess, ts_168h):
    """Adding BESS adds ~$5.6M/yr CAPEX (NLR ATB 2-hr 529 $/kWh × 100 MWh,
    15-yr life with 5% salvage credit: $52.9M × 0.1057 ≈ $5.59M/yr)."""
    r_no = solve_case2(cfg2_no_bess, ts_168h, pue=1.30)
    r_yes = solve_case2(cfg2_bess, ts_168h, pue=1.30)
    delta_capex = r_yes.capex_annual_usd - r_no.capex_annual_usd
    assert 5.0e6 < delta_capex < 6.5e6


def test_bess_arbitrage_in_volatile_year_lowers_grid_cost():
    """In a high-volatility ERCOT slice, BESS should reduce net grid cost
    via off-peak charge / on-peak discharge - i.e., grid_annual cheaper with BESS.
    Use a 168h slice from August 2022 (cap-frequent hours)."""
    cfg2 = load_config(case_id=2, project_root=PROJECT_ROOT)
    ts = load_time_series(
        project_root=PROJECT_ROOT, year=2022, num_hours=168, start_hour=5088
    )  # Aug 1 00:00
    r_no = solve_case2(cfg2, ts, pue=1.30)
    r_yes = solve_case2(with_bess(cfg2), ts, pue=1.30)
    # With BESS, net grid expense (buy − sell × LMP) should drop
    assert r_yes.grid_annual_usd <= r_no.grid_annual_usd + 1e-3
