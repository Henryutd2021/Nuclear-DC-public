"""Physical and accounting regressions for the reactor, outage and carbon accounting."""

from pathlib import Path

import pyomo.environ as pyo
import pytest

from src.config import load_config
from src.data import load_time_series
from src.finance import annualized_capex
from src.milp.builder import build_model
from src.milp.result import extract_result
from src.milp.solve import solve_model

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def inputs():
    cfg = load_config(case_id=2, project_root=ROOT)
    cfg.base.solver.threads = 1
    ts = load_time_series(project_root=ROOT, year=2023, num_hours=24)
    return cfg, ts


@pytest.mark.parametrize("case_id", [1, 2])
def test_full_load_rating_and_costs_use_300_mwe_net(case_id, inputs):
    _, ts = inputs
    cfg = load_config(case_id=case_id, project_root=ROOT)
    cfg.base.solver.threads = 1
    m = build_model(cfg, ts)
    for t in m.T:
        m.P_rx[t].fix(870.0)
        if case_id == 2:
            m.Q_to_abs[t].fix(0.0)
    solve_model(m, solver_config=cfg.base.solver)
    # Rounding of the inherited Willans coefficients leaves <0.01 MW.
    assert pyo.value(m.P_turb_gross[0]) == pytest.approx(316.0, abs=0.01)
    assert pyo.value(m.P_turb_net[0]) == pytest.approx(300.0, abs=0.01)
    assert pyo.value(m.fuel_annual) == pytest.approx(26_280_000.0)
    equipment_capex = annualized_capex(250 * 160_000, 0.067, 25)
    equipment_fom = 8 * 160_000
    if case_id == 2:
        equipment_capex += annualized_capex(750 * 160_000, 0.067, 25)
        equipment_fom += 20 * 160_000
    assert pyo.value(m.capex_annual) - equipment_capex == pytest.approx(
        annualized_capex(7615 * 300_000, 0.067, 40)
    )
    assert pyo.value(m.fom_annual) - equipment_fom == pytest.approx(36_300_000)


def test_explicit_refueling_and_separate_absorber_maintenance(inputs):
    cfg, ts = inputs
    # Two disjoint schedules represent refueling in one year and chiller
    # maintenance in the other; the short horizon keeps this regression cheap.
    cfg.case.reactor = cfg.case.reactor.model_copy(
        update={"scheduled_outage_hours": [4, 5]}
    )
    cfg.case.absorption = cfg.case.absorption.model_copy(
        update={"scheduled_maintenance_hours": [12, 13]}
    )
    m = build_model(cfg, ts)
    assert m.P_rx[4].ub == 0.0
    assert m.P_rx[12].lb == 435.0
    for t in (4, 5, 12, 13):
        assert pyo.value(m.absorption_available[t]) == 0.0
        assert m.Q_to_abs[t].ub == 0.0
    solve_model(m, solver_config=cfg.base.solver)
    for t in (4, 5):
        assert pyo.value(m.P_turb_net[t]) == pytest.approx(0.0)
        assert pyo.value(m.P_grid_buy[t]) > 0.0
    for t in (12, 13):
        assert pyo.value(m.Q_vcc_cool[t]) == pytest.approx(
            pyo.value(m.Q_cool_demand[t])
        )


@pytest.mark.parametrize("field", ["reactor", "absorption"])
def test_explicit_schedule_outside_horizon_is_rejected(field, inputs):
    cfg, ts = inputs
    key = "scheduled_outage_hours" if field == "reactor" else "scheduled_maintenance_hours"
    block = getattr(cfg.case, field)
    setattr(cfg.case, field, block.model_copy(update={key: [24]}))
    with pytest.raises(ValueError, match="outside.*horizon"):
        build_model(cfg, ts)


def test_empty_outage_override_preserves_online_operation(inputs):
    cfg, ts = inputs
    cfg.case.reactor = cfg.case.reactor.model_copy(
        update={"scheduled_outage_hours": []}
    )
    m = build_model(cfg, ts)
    assert all(m.P_rx[t].lb == 435.0 for t in m.T)


def test_nuclear_carbon_exemption_preserves_lifecycle_reporting(inputs):
    cfg, ts = inputs
    cfg.base.physics.carbon_price_usd_per_tco2 = 100.0
    cfg.case.reactor = cfg.case.reactor.model_copy(
        update={"co2_billed_g_per_kwh_e": 0.0}
    )
    m = build_model(cfg, ts)
    solve_model(m, solver_config=cfg.base.solver)
    result = extract_result(m, cfg, ts)
    assert result.co2_rx_lifecycle_tonnes > 0.0
    expected_billed = (
        result.co2_grid_import_tonnes - result.co2_export_credit_tonnes
    ) * 100.0
    assert result.carbon_annual_usd == pytest.approx(expected_billed)
    assert result.co2_annual_tonnes == pytest.approx(
        result.co2_rx_lifecycle_tonnes
        + result.co2_grid_import_tonnes
        - result.co2_export_credit_tonnes
    )


def test_default_carbon_charge_matches_lifecycle_accounting(inputs):
    cfg, ts = inputs
    cfg.base.physics.carbon_price_usd_per_tco2 = 100.0
    m = build_model(cfg, ts)
    solve_model(m, solver_config=cfg.base.solver)
    result = extract_result(m, cfg, ts)
    assert result.carbon_annual_usd == pytest.approx(result.co2_annual_tonnes * 100)


def test_reactor_load_following_uses_vendor_ramp_limit(inputs):
    cfg, ts = inputs
    m = build_model(cfg, ts)
    # 0.5%/min of 870 MWth permits at most 261 MWth between hourly points.
    m.P_rx[0].set_value(435.0)
    m.P_rx[1].set_value(870.0)
    assert pyo.value(m.ramp_up[1].body) > pyo.value(m.ramp_up[1].upper)
    m.P_rx[1].set_value(696.0)
    assert pyo.value(m.ramp_up[1].body) == pytest.approx(pyo.value(m.ramp_up[1].upper))
