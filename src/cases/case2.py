"""Case 2 - BWRX-300 + main turbine + cascaded HP extraction + absorption.

Main steam (287 °C / 7.17 MPa) enters the HP
turbine and does electrical work first. A mid-pressure tap (5-7 barg,
~160 °C) bleeds part of the flow to drive a double-effect LiBr-H2O
absorption chiller; the rest continues through the LP turbine and
condenser. The Willans-line linearization in src.milp.builder charges
the electricity loss per MWth of extracted heat - cascaded, NOT parallel,
so high-grade steam is no longer wasted on a 6 °C cooling duty.

The absorber is unavailable when the wet-bulb temperature reaches the LiBr
crystallization gate, and its COP varies with the hourly wet-bulb temperature.
"""

from __future__ import annotations

from typing import Optional

from src.config import RunConfig
from src.data import TimeSeries
from src.milp.builder import build_model
from src.milp.result import NuclearCaseResult, extract_result
from src.milp.solve import solve_model


def solve_case2(
    cfg: RunConfig,
    ts: TimeSeries,
    pue: Optional[float] = None,
) -> NuclearCaseResult:
    if cfg.case.case_id != 2:
        raise ValueError(f"solve_case2 requires case_id=2, got {cfg.case.case_id}")
    model = build_model(cfg, ts, pue=pue)
    solve_model(model, solver_config=cfg.base.solver)
    return extract_result(model, cfg, ts)
