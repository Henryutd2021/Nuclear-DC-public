"""Gurobi solver invocation for the Nuclear-DC Pyomo models.

The builder is mostly a continuous LP, but the vapor-compression chiller
is an SOS2 piecewise map of its non-convex part-load COP curve (see
`src/milp/builder.py`), which makes the model a MILP. Gurobi is the only
supported solver: a missing or unlicensed installation raises an error.
"""

from __future__ import annotations

import math
from typing import Optional

import pyomo.environ as pyo

from src.config import SolverConfig


def _gurobi():
    """Return the Pyomo Gurobi interface, or raise if it cannot be used.

    Pyomo's file interface can report installed Gurobi bindings as available
    even when no license is usable, so the license is checked separately.
    """
    solver = pyo.SolverFactory("gurobi")
    if not solver.available(exception_flag=False):
        raise RuntimeError("Gurobi is not available; install gurobipy and a Gurobi license")
    if not solver.license_is_valid():
        raise RuntimeError("Gurobi is installed but no valid license was found")
    return solver


def solve_model(
    model: pyo.ConcreteModel,
    solver_config: Optional[SolverConfig] = None,
    mip_gap: float = 1e-4,
    time_limit: Optional[float] = None,
    quiet: bool = True,
) -> None:
    """Solve ``model`` in-place with Gurobi. Raises if the solve is not optimal.

    Args:
        model: a Pyomo ConcreteModel built by :func:`src.milp.builder.build_model`.
        solver_config: optional ``SolverConfig`` (from ``cfg.base.solver``)
            whose fields override the positional fallbacks below. Wires
            Threads / Method / Seed / MIPGap / TimeLimit / OutputFlag
            through to Gurobi so the yaml-declared values actually take
            effect.
        mip_gap: optimality gap fallback (used only when ``solver_config``
            is None); matches the config/base.yaml default so ad-hoc callers
            get the same tolerance as the production grid. LPs ignore this.
        time_limit: optional hard timeout in seconds (fallback only).
        quiet: suppress solver console output.
    """
    solver_name = "gurobi"
    solver = _gurobi()

    opts = solver.options
    if solver_config is not None:
        opts["MIPGap"] = solver_config.mip_gap
        opts["Threads"] = max(0, solver_config.threads)
        opts["Seed"] = solver_config.seed
        opts["Method"] = solver_config.method
        opts["OutputFlag"] = 0 if quiet else 1
        if solver_config.time_limit is not None:
            opts["TimeLimit"] = solver_config.time_limit
    else:
        opts["MIPGap"] = mip_gap
        if time_limit is not None:
            opts["TimeLimit"] = time_limit

    results = solver.solve(model, tee=not quiet)
    term = results.solver.termination_condition
    if term != pyo.TerminationCondition.optimal:
        raise RuntimeError(
            f"Solver terminated with non-optimal condition: {term}"
        )

    # Record achieved solve quality on the model so extract_result can
    # propagate it into summary.json (the results object is otherwise
    # discarded and the realized gap would be unrecoverable from artifacts).
    lb: float | None = None
    ub: float | None = None
    gap: float | None = None
    try:
        lb = float(results.problem.lower_bound)
        ub = float(results.problem.upper_bound)
        if math.isfinite(lb) and math.isfinite(ub):
            gap = abs(ub - lb) / max(1.0, abs(ub))
        else:
            lb = ub = None
    except (AttributeError, TypeError, ValueError):
        lb = ub = None
    model._solver_quality = {
        "solver_name": solver_name,
        "lower_bound": lb,
        "upper_bound": ub,
        "mip_gap_achieved": gap,
    }
