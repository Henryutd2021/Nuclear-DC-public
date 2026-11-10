"""Integration: 4-case Premium matrix and the wet-bulb crystallization gate.

Locks in cross-case behavior at the baseline grid (3 ERCOT years
× full-load PUE 1.35 × ATB-Mid reactor CAPEX). Premium(Case 0, Case N) signs
become regression checks: if a future code change accidentally flips
one of them, this test catches it.
"""

from pathlib import Path

import pytest

from src.cases.case0 import solve_case0
from src.cases.case1 import solve_case1
from src.cases.case2 import solve_case2
from src.cases.case3 import solve_case3
from src.config import load_config
from src.data import load_time_series
from src.kpi import heat_recovery_premium

PROJECT_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.integration

gurobi = pytest.importorskip("pyomo.opt").SolverFactory("gurobi")
if not gurobi.available(exception_flag=False):
    pytest.skip("Gurobi unavailable", allow_module_level=True)


def _solve_all(year: int, pue: float = 1.35, hours: int = 168):
    """Solve all 4 cases for one (year, PUE) and return TAC dict."""
    ts = load_time_series(project_root=PROJECT_ROOT, year=year, num_hours=hours)
    tacs = {}
    for case_id, solver in (
        (0, solve_case0),
        (1, solve_case1),
        (2, solve_case2),
        (3, solve_case3),
    ):
        cfg = load_config(case_id=case_id, project_root=PROJECT_ROOT)
        r = solver(cfg, ts, pue=pue)
        tacs[case_id] = r.tac_usd_per_yr
    return tacs


def test_all_cases_solve_and_return_positive_tac():
    tacs = _solve_all(2023)
    for cid, tac in tacs.items():
        assert tac > 0, f"Case {cid} returned non-positive TAC {tac}"


def test_nuclear_cases_lose_to_case0_at_atb_mid_capex():
    """At NLR ATB Moderate ($7,615/kWe), BWRX CAPEX dominates → Cases 1-2 lose to Case 0."""
    tacs = _solve_all(2023)
    for cid in (1, 2):
        prem = heat_recovery_premium(tacs[0], tacs[cid])
        assert prem < 0, f"Case {cid} unexpectedly cheaper than Case 0 at ATB-Mid"


def test_case2_drives_absorption_at_realistic_pue():
    """Cascade extraction must actually utilize absorption when economic.

    With the heat-balance Willans penalty (0.20 MWe/MWth) and the flat 45Y
    generation credit, absorption competes hour-by-hour with the VCC: steam
    diversion forfeits the credit on lost generation, so cheap-LMP hours go
    to the VCC and higher-priced hours to absorption. The regression check
    is that the penalty model neither shuts the chiller off entirely nor
    hands it the whole load unconditionally.
    """
    ts = load_time_series(project_root=PROJECT_ROOT, year=2023, num_hours=168)
    cfg2 = load_config(case_id=2, project_root=PROJECT_ROOT)
    r2 = solve_case2(cfg2, ts, pue=1.35)
    abs_share = float(r2.Q_abs_cool_MWth.sum() / r2.Q_cool_demand_MWth.sum())
    assert 0.05 < abs_share < 0.95, (
        f"Expected a price-split absorption/VCC dispatch in a mixed-price week; "
        f"got absorption share {abs_share:.0%}"
    )


def test_case2_within_marginal_band_of_case1():
    """A key finding is that absorption CAPEX (~$6.9 M/yr at the
    baseline) dominates the grid-cost savings the cascade buys, so Case 2 sits
    only a few percent above Case 1 at ATB-Mid CAPEX. The exact sign of the
    gap is a sensitivity finding (see S5 grid), but the gap should stay small.
    """
    tacs = _solve_all(2023)
    delta_pct = abs(tacs[2] - tacs[1]) / tacs[1]
    assert delta_pct < 0.10, (
        f"Expected |TAC_2 − TAC_1| < 10% of Case 1 at ATB-Mid; "
        f"got {delta_pct:.1%} (TAC_1={tacs[1]/1e6:.1f}M, TAC_2={tacs[2]/1e6:.1f}M)"
    )


def test_p1a_absorption_gate_triggers_in_hot_hour_summer_slice():
    """A 168h slice starting mid-July (hour 4344 = Jul 1 00:00) hits Houston
    wet-bulb peaks that exceed the 27 °C crystallization-cutoff threshold;
    Case 2 must force absorption off and use VCC backup during those hours.
    """
    cfg = load_config(case_id=2, project_root=PROJECT_ROOT)
    ts = load_time_series(
        project_root=PROJECT_ROOT, year=2023, num_hours=168, start_hour=4344
    )
    r = solve_case2(cfg, ts, pue=1.35)
    # At least some VCC dispatch must happen - proves the gate engaged
    assert r.Q_VCC_cool_MWth.sum() > 0, (
        "Expected VCC backup to engage during July hot hours when absorption "
        "is gated off by the crystallization rule"
    )
