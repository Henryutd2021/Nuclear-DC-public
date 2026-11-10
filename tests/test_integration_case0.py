"""Phase 1 integration smoke test: Case 0 end-to-end on real ERCOT data.

Marked ``integration`` so it can be filtered out by ``pytest -m 'not integration'``
during fast unit-test loops. The default ``pytest`` invocation includes it.
"""

from pathlib import Path

import pytest

from src.cases.case0 import solve_case0
from src.config import load_config
from src.data import load_time_series
from src.kpi import compute_kpis_case0, heat_recovery_premium

PROJECT_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def cfg():
    return load_config(case_id=0, project_root=PROJECT_ROOT)


def test_case0_168h_2023_smoke(cfg):
    ts = load_time_series(project_root=PROJECT_ROOT, year=2023, num_hours=168)
    result = solve_case0(cfg, ts)
    kpis = compute_kpis_case0(result, ts, cfg)

    # No NaN/inf anywhere in dispatch
    assert result.P_grid_buy_MW.notna().all()
    assert result.Q_cool_MWth.notna().all()
    assert result.grid_cost_usd_per_h.notna().all()
    assert result.grid_emissions_kg_co2_per_h.notna().all()

    # TAC for a 200 MW DC grid-only baseline on 2023 ERCOT should be in
    # the $20M–$300M/yr range (very loose smoke bounds).
    assert 2.0e7 < kpis.tac_usd_per_yr < 3.0e8

    # LCOE in ERCOT 2023 ($55 DA mean) should land roughly $30–$200/MWh.
    assert 30 < kpis.lcoe_usd_per_mwh_e < 200


def test_case0_pue_setpoint_decoupled_from_tac_2023(cfg):
    """Option A: PUE is a reported outcome, not a driver, so sweeping the PUE
    setpoint ∈ {1.10, 1.30, 1.50} leaves TAC and CO2 unchanged. S1 now sweeps the
    cooling COP instead of PUE."""
    ts = load_time_series(project_root=PROJECT_ROOT, year=2023, num_hours=168)
    tacs = []
    co2s = []
    for pue in (1.10, 1.30, 1.50):
        r = solve_case0(cfg, ts, pue=pue)
        kpis = compute_kpis_case0(r, ts, cfg)
        tacs.append(kpis.tac_usd_per_yr)
        co2s.append(kpis.co2_annual_tonnes)
    assert tacs[1] == pytest.approx(tacs[0])
    assert tacs[2] == pytest.approx(tacs[0])
    assert co2s[1] == pytest.approx(co2s[0])
    assert co2s[2] == pytest.approx(co2s[0])


def test_case0_three_year_regime_full_8760(cfg):
    """S2: 2022 volatile > 2023 high > 2024 low → grid_annual monotone."""
    tacs = {}
    for year in (2022, 2023, 2024):
        ts = load_time_series(project_root=PROJECT_ROOT, year=year, num_hours=8760)
        r = solve_case0(cfg, ts)
        kpis = compute_kpis_case0(r, ts, cfg)
        tacs[year] = kpis.tac_usd_per_yr
    # Monotone ordering across the three market years
    assert tacs[2022] > tacs[2023] > tacs[2024]


def test_heat_recovery_premium_against_self_is_zero(cfg):
    """Sanity: Premium of baseline against itself must be zero."""
    ts = load_time_series(project_root=PROJECT_ROOT, year=2023, num_hours=168)
    r = solve_case0(cfg, ts)
    kpis = compute_kpis_case0(r, ts, cfg)
    premium = heat_recovery_premium(kpis.tac_usd_per_yr, kpis.tac_usd_per_yr)
    assert premium == pytest.approx(0.0, abs=1e-12)
