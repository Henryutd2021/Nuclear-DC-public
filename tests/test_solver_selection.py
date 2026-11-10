"""The solver wrapper uses Gurobi only and never masks a failed solve."""

import pyomo.environ as pyo
import pytest

from src.milp.solve import solve_model


def _small_model():
    model = pyo.ConcreteModel()
    model.x = pyo.Var(bounds=(2.0, None))
    model.objective = pyo.Objective(expr=model.x)
    return model


def test_gurobi_solves_and_records_quality():
    model = _small_model()
    solve_model(model)
    assert pyo.value(model.x) == pytest.approx(2.0)
    assert model._solver_quality["solver_name"] == "gurobi"


def test_missing_gurobi_license_raises(monkeypatch):
    class UnlicensedGurobi:
        def available(self, exception_flag=False):
            return True  # Installed bindings do not imply a usable license.

        def license_is_valid(self):
            return False

    monkeypatch.setattr(pyo, "SolverFactory", lambda name: UnlicensedGurobi())
    with pytest.raises(RuntimeError, match="license"):
        solve_model(_small_model())


def test_nonoptimal_solve_raises():
    model = _small_model()
    model.impossible = pyo.Constraint(expr=model.x <= 1.0)
    with pytest.raises(RuntimeError, match="non-optimal"):
        solve_model(model)
