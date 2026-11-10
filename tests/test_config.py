"""Tests for src.config - Pydantic schemas + configured values."""

from pathlib import Path

import pytest

from src.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_load_case0_returns_run_config():
    cfg = load_config(case_id=0, project_root=PROJECT_ROOT)
    assert cfg.case.case_id == 0
    assert cfg.case.label == "Grid-only baseline"


def test_case0_equipment_flags_match_v25_locked_spec():
    cfg = load_config(case_id=0, project_root=PROJECT_ROOT)
    eq = cfg.case.equipment
    # Case 0 = pure grid + VCC
    assert eq.reactor_enabled is False
    assert eq.turbine_enabled is False
    assert eq.orc_enabled is False
    assert eq.absorption_chiller_enabled is False
    assert eq.electric_chiller_enabled is True
    assert eq.tes_enabled is False
    assert eq.bess_enabled is False
    assert eq.ngcc_enabled is False
    assert eq.grid_import_enabled is True
    assert eq.grid_export_enabled is False


def test_case0_capacities_cover_peak_it_heat_rejection():
    """Case 0 chiller must cover P_IT_max / eta_chain at the peak IT hour."""
    cfg = load_config(case_id=0, project_root=PROJECT_ROOT)
    # P_IT_max ≈ 142.4 MW (NLR Vercellino dataset), eta_chain=0.9
    # gives Q_cool_max ≈ 158.3 MWc.
    assert cfg.case.capacities.electric_chiller_capacity_MWth >= 158.3


def test_case0_vcc_houston_derated_cop_is_below_nameplate():
    cfg = load_config(case_id=0, project_root=PROJECT_ROOT)
    vcc = cfg.case.vcc
    assert vcc is not None
    assert vcc.cop_houston < vcc.cop_baseline   # derating applied
    assert vcc.cop_baseline == pytest.approx(3.5, abs=0.01)
    assert vcc.cop_houston == pytest.approx(3.2, abs=0.05)


def test_financial_params_match_v25_locked_values():
    """WACC=6.7%, n=20 → CRF=0.0922."""
    cfg = load_config(case_id=0, project_root=PROJECT_ROOT)
    assert cfg.financial.WACC_nominal == pytest.approx(0.067, abs=1e-4)
    assert cfg.financial.project_lifetime_years == 20
    assert cfg.financial.capital_recovery_factor == pytest.approx(0.0922, abs=1e-4)


def test_crf_self_consistent_with_formula():
    """CRF = i(1+i)^n / ((1+i)^n - 1) must match the stored value."""
    cfg = load_config(case_id=0, project_root=PROJECT_ROOT)
    i = cfg.financial.WACC_nominal
    n = cfg.financial.project_lifetime_years
    computed = i * (1 + i) ** n / ((1 + i) ** n - 1)
    assert computed == pytest.approx(cfg.financial.capital_recovery_factor, abs=1e-4)


def test_base_time_and_solver_loaded():
    cfg = load_config(case_id=0, project_root=PROJECT_ROOT)
    assert cfg.base.time.num_hours > 0
    assert cfg.base.time.delta_t == pytest.approx(1.0)
    assert cfg.base.solver.name == "gurobi"
    assert 0 < cfg.base.solver.mip_gap < 1


def test_invalid_case_id_rejected():
    """Valid set is {0, 1, 2, 3}; 4 and 5 must both be rejected."""
    with pytest.raises(ValueError, match=r"case_id"):
        load_config(case_id=4, project_root=PROJECT_ROOT)
    with pytest.raises(ValueError, match=r"case_id"):
        load_config(case_id=5, project_root=PROJECT_ROOT)
    with pytest.raises(ValueError, match=r"case_id"):
        load_config(case_id=-1, project_root=PROJECT_ROOT)


def test_missing_yaml_raises_filenotfound():
    """Bogus project root must raise FileNotFoundError on the first yaml read."""
    with pytest.raises(FileNotFoundError):
        load_config(case_id=0, project_root="/nonexistent/path/Nuclear-DC")
