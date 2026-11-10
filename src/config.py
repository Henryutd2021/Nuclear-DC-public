"""Configuration loading for Nuclear-DC.

Three sources of truth merged into a single ``RunConfig``:

1. ``config/base.yaml``           - runtime knobs (time horizon, solver, physics).
2. ``config/plant_case{N}.yaml``  - per-case equipment toggles + capacities.
3. ``data/economics/financial_parameters.yaml`` - financial constants (WACC, CRF).

Equipment cost and performance parameters (BWRX-300, absorption, VCC, BESS,
NGCC) are set in the case yaml files; the S4 reactor capital scenarios are read
from ``data/reactor/bwrx300_economic.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import yaml  # noqa: F401  (used by with_reactor_capex)
from pydantic import BaseModel, ConfigDict, Field

CaseId = Literal[0, 1, 2, 3]
_VALID_CASE_IDS: set[int] = {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# config/plant_case{N}.yaml
# ---------------------------------------------------------------------------
class CaseEquipment(BaseModel):
    reactor_enabled: bool = False
    turbine_enabled: bool = False
    orc_enabled: bool = False
    absorption_chiller_enabled: bool = False
    electric_chiller_enabled: bool = False
    tes_enabled: bool = False
    bess_enabled: bool = False
    ngcc_enabled: bool = False
    grid_import_enabled: bool = True
    grid_export_enabled: bool = False


class CaseCapacities(BaseModel):
    model_config = ConfigDict(extra="allow")

    reactor_thermal_capacity_MWth: Optional[float] = None
    turbine_capacity_MWe: Optional[float] = None
    orc_capacity_MWe: Optional[float] = None
    absorption_capacity_MWth: Optional[float] = None
    electric_chiller_capacity_MWth: Optional[float] = None
    tes_capacity_MWh_th: Optional[float] = None
    bess_capacity_MWh: Optional[float] = None
    bess_power_MW: Optional[float] = None
    ngcc_capacity_MWe: Optional[float] = None
    pcc_capacity_MW: Optional[float] = None       # PCC interconnect limit


class VccConfig(BaseModel):
    cop_baseline: float = Field(gt=1.0)
    cop_houston: float = Field(gt=1.0)
    fixed_om_usd_per_kWth_year: float = Field(ge=0.0)
    variable_om_usd_per_mwh_th: float = Field(ge=0.0)
    capex_usd_per_kWth: float = Field(ge=0.0)
    lifetime_years: int = Field(gt=0)
    # Wet-bulb relief on the design-point system COP, mirroring the relative
    # linear response the absorption chiller already carries (0.015/1.10 ≈
    # 1.36% per K) so neither cooling technology is asymmetrically favored.
    # relief(t) = clip(1 + relief_per_K * (T_design - T_wb(t)), ..., cap)
    cop_design_wet_bulb_C: float = Field(26.0)
    cop_wet_bulb_relief_per_K: float = Field(0.0, ge=0.0)
    cop_wet_bulb_relief_cap: float = Field(1.0, ge=1.0)


class NgccConfig(BaseModel):
    capex_usd_per_kWe: float = Field(gt=0.0)
    fixed_om_usd_per_kWe_year: float = Field(ge=0.0)
    variable_om_usd_per_mwh_e: float = Field(ge=0.0)
    net_efficiency_lhv: float = Field(gt=0.0, lt=1.0)
    net_efficiency_hhv: float = Field(gt=0.0, lt=1.0)
    co2_direct_g_per_kwh_e: float = Field(ge=0.0)
    co2_upstream_ch4_g_per_kwh_e: float = Field(ge=0.0)
    henry_hub_basis_usd_per_mmbtu: float
    lifetime_years: int = Field(gt=0)


class ReactorConfig(BaseModel):
    """BWRX-300 economic + operational parameters used by Cases 1-3."""

    thermal_power_MWth: float = Field(gt=0.0)
    electric_power_net_MWe: float = Field(gt=0.0)
    thermal_efficiency: float = Field(gt=0.0, lt=1.0)
    capex_usd_per_kWe: float = Field(gt=0.0)        # S4: FOAK/ATB-Mid/NOAK
    fixed_om_usd_per_kWe_year: float = Field(ge=0.0)
    variable_om_usd_per_mwh_e: float = Field(ge=0.0)
    # Fuel is priced on the net-electric basis the fleet statistics use
    # ($/MWh_e of net generation); the builder converts to a thermal-basis
    # rate via the plant's net efficiency so fuel spend stays proportional
    # to thermal energy actually consumed (extraction does not burn less).
    fuel_cost_usd_per_mwh_e: float = Field(ge=0.0)
    co2_lifecycle_g_per_kwh_e: float = Field(ge=0.0)
    # None prices the lifecycle factor, preserving the baseline convention.
    # Zero exempts direct nuclear operation from the carbon charge without
    # removing lifecycle emissions from the separately reported footprint.
    co2_billed_g_per_kwh_e: Optional[float] = Field(None, ge=0.0)
    # BWRX-300 operating limits
    min_load_fraction: float = Field(gt=0.0, lt=1.0)
    ramp_rate_pct_per_min: float = Field(gt=0.0)
    capacity_factor: float = Field(gt=0.0, le=1.0)
    # Explicit zero-based horizon indices, for nonannual refueling schedules.
    # None uses the annual capacity-factor-derived outage; [] means no outage.
    scheduled_outage_hours: Optional[list[int]] = None
    lifetime_years: int = Field(gt=0)


class TurbineConfig(BaseModel):
    """Main HP+LP steam turbine fed by reactor (Cases 1-2).

    Gross output follows a Willans line in reactor thermal input (fitted to
    data/perf/turbine_hr.csv part-load heat rates), and the cascaded
    extraction at the HP/LP crossover (7 bar abs, ~165 °C) charges the lost
    LP expansion work per MWth of steam diverted to the double-effect
    absorption chiller:

        P_turb_gross = main_willans_slope * P_rx - main_willans_intercept
                     - extraction_willans_slope_MWe_per_MWth * Q_to_absorption

    The extraction slope comes from the steam-table heat balance documented
    in config/plant_case2.yaml (lost LP work net of the hot-condensate-return
    credit, ~0.20 MWe/MWth for the 7-bar tap).
    """

    rated_efficiency: float = Field(gt=0.0, lt=1.0)   # at full main-steam load, zero extraction
    capex_usd_per_kWe: float = Field(ge=0.0)          # 0 if bundled into reactor CAPEX
    fixed_om_usd_per_kWe_year: float = Field(ge=0.0)
    variable_om_usd_per_mwh_e: float = Field(ge=0.0)
    aux_load_fraction: float = Field(ge=0.0, lt=1.0)  # parasitic loads as fraction of gross
    # Main-turbine Willans line (gross output vs reactor thermal input),
    # fitted to data/perf/turbine_hr.csv then scaled by 316/300 to retain the
    # inherited relative part-load shape at the corrected design rating
    # (870 MWth -> 316 MWe gross). This scaling is a modeling approximation,
    # not a new vendor-validated part-load map. The no-load intercept captures the
    # part-load efficiency droop a constant rated_efficiency misses (~7.6%
    # output overstatement at the 50% minimum load). When the intercept is 0
    # the line reduces to the constant-efficiency model with slope
    # rated_efficiency.
    main_willans_slope_MWe_per_MWth: float = Field(gt=0.0, lt=1.0)
    main_willans_intercept_MWe: float = Field(0.0, ge=0.0)
    # Cascaded extraction: electricity penalty per unit of heat diverted
    # to absorption at the mid-pressure tap. 0 means "no extraction allowed"
    # (Case 1) and the absorption block must be disabled.
    extraction_willans_slope_MWe_per_MWth: float = Field(0.0, ge=0.0, le=1.0)


class OrcConfig(BaseModel):
    """ORC bottoming cycle (Case 2)."""

    capex_usd_per_kWe: float = Field(gt=0.0)
    fixed_om_usd_per_kWe_year: float = Field(ge=0.0)
    variable_om_usd_per_mwh_e: float = Field(ge=0.0)
    net_efficiency: float = Field(gt=0.0, lt=1.0)     # at design T_hot (120 °C)
    extraction_temperature_C: float = Field(gt=0.0)
    lifetime_years: int = Field(gt=0)


class AbsorptionConfig(BaseModel):
    """Double-effect LiBr-H2O absorption chiller (Cases 2, 3).

    Wet-bulb-driven dynamic COP: COP_effective(t) = nameplate × derate(T_wb).
    Crystallization cutoff: forces VCC backup when cooling-water-in > critical.
    """

    capex_usd_per_kWth: float = Field(gt=0.0)
    fixed_om_usd_per_kWth_year: float = Field(ge=0.0)
    variable_om_usd_per_mwh_th: float = Field(ge=0.0)
    cop_nameplate: float = Field(gt=0.0)              # 1.30 for double-effect
    cop_houston_baseline: float = Field(gt=0.0)       # 1.10 at T_wb = 26 °C (paper baseline)
    derate_per_celsius_above_baseline: float = Field(gt=0.0)
    cooling_water_approach_K: float = Field(gt=0.0)
    crystallization_cw_inlet_C: float = Field(gt=0.0) # 32 °C critical
    parasitic_kWe_per_kWth: float = Field(ge=0.0)
    lifetime_years: int = Field(gt=0)
    availability: float = Field(1.0, gt=0.0, le=1.0)  # maintenance availability
    # Independent maintenance can be retained in non-refueling years. None
    # coincides with the reactor outage; explicit hours additionally gate steam.
    scheduled_maintenance_hours: Optional[list[int]] = None


class BessConfig(BaseModel):
    """Lithium-ion BESS for the S3 binary sensitivity.

    Capital is a single all-in $/kWh at the system's own duration (NLR ATB
    2-hour utility-scale battery), so the energy rating alone carries the
    capital charge - no separate $/kW power-block term to double-count.
    """

    capex_usd_per_kwh: float = Field(gt=0.0)
    fixed_om_usd_per_kw_year: float = Field(ge=0.0)
    variable_om_usd_per_mwh: float = Field(ge=0.0)
    round_trip_efficiency: float = Field(gt=0.0, le=1.0)
    self_discharge_per_hour: float = Field(ge=0.0)
    soc_min_fraction: float = Field(ge=0.0, lt=1.0)
    soc_max_fraction: float = Field(gt=0.0, le=1.0)
    initial_soc_fraction: float = Field(ge=0.0, le=1.0)
    lifetime_years: int = Field(gt=0)
    end_of_life_credit_pct: float = Field(0.0, ge=0.0, lt=100.0)  # salvage at EOL


class CaseConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_id: CaseId
    label: str
    equipment: CaseEquipment
    capacities: CaseCapacities
    vcc: Optional[VccConfig] = None
    ngcc: Optional[NgccConfig] = None
    reactor: Optional[ReactorConfig] = None
    turbine: Optional[TurbineConfig] = None
    orc: Optional[OrcConfig] = None
    absorption: Optional[AbsorptionConfig] = None
    bess: Optional[BessConfig] = None


# ---------------------------------------------------------------------------
# config/base.yaml
# ---------------------------------------------------------------------------
class TimeConfig(BaseModel):
    num_hours: int = Field(gt=0)
    delta_t: float = Field(gt=0.0)


class SolverConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: Literal["gurobi"]
    mip_gap: float = Field(gt=0.0, lt=1.0)
    time_limit: Optional[float] = None
    # threads: 0 = Gurobi default (use all cores). Set explicitly when
    # solving in parallel processes to avoid SMT oversubscription.
    threads: int = 0
    log_to_console: bool = True
    log_to_file: bool = True
    seed: int = 42
    # LP method: -1 auto, 0 primal simplex, 1 dual simplex, 2 barrier,
    # 3 concurrent, 4 deterministic concurrent. Barrier (2) is the default.
    method: int = Field(2, ge=-1, le=5)


class PhysicsConfig(BaseModel):
    # forbid: an undeclared physics key in base.yaml means dead config - a
    # knob with no model behind it must fail loudly, not validate silently.
    model_config = ConfigDict(extra="forbid")

    pue_default: float = Field(1.35, ge=1.0)
    cooling_chain_efficiency: float = Field(0.9, gt=0.0, le=1.0)
    # S6: optional carbon price applied to net annual CO2 in the TAC
    # objective. Default 0 leaves the baseline unpriced; S6 sensitivity
    # drives this via with_carbon_price().
    carbon_price_usd_per_tco2: float = Field(0.0, ge=0.0)


class BaseConfig(BaseModel):
    time: TimeConfig
    solver: SolverConfig
    physics: PhysicsConfig


# ---------------------------------------------------------------------------
# data/economics/financial_parameters.yaml
# ---------------------------------------------------------------------------
class FinancialParams(BaseModel):
    model_config = ConfigDict(extra="allow")

    WACC_nominal: float = Field(gt=0.0, lt=1.0)
    WACC_real: float = Field(gt=0.0, lt=1.0)
    inflation_rate: float
    project_lifetime_years: int = Field(gt=0)
    capital_recovery_factor: float = Field(gt=0.0, lt=1.0)
    capital_recovery_factor_formula: str
    reactor_physical_life_years: int = Field(gt=0)
    # Section 45Y clean-electricity PTC (26 U.S.C. 45Y; technology-neutral,
    # zero-GHG facilities placed in service after 2024). Flat per-MWh credit
    # at the prevailing-wage rate, no gross-receipts phaseout, paid for the
    # first ptc_credit_duration_years (statutorily 10) and levelized over the
    # TAC window in builder.py via finance.levelized_ptc_usd_per_mwh.
    # Default-on for Cases 1-2.
    ptc_usd_per_mwh_assumed: float
    ptc_credit_duration_years: int = Field(10, gt=0)
    nuclear_ptc_enabled: bool = True


class RunConfig(BaseModel):
    case: CaseConfig
    base: BaseConfig
    financial: FinancialParams


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_config(case_id: int, project_root: Union[Path, str]) -> RunConfig:
    """Load and validate the configuration for a specific case.

    Args:
        case_id: One of {0, 1, 2, 3}.
        project_root: Path to the Nuclear-DC repo root.

    Returns:
        A validated ``RunConfig`` bundling case, base, and financial parameters.

    Raises:
        ValueError: if ``case_id`` is not in {0, 1, 2, 3} or the yaml's
            internal ``case_id`` does not match the caller's.
        FileNotFoundError: if the per-case yaml does not exist on disk.
    """
    if case_id not in _VALID_CASE_IDS:
        raise ValueError(
            f"case_id must be one of {sorted(_VALID_CASE_IDS)}, got {case_id!r}"
        )

    project_root = Path(project_root)
    base_path = project_root / "config" / "base.yaml"
    case_path = project_root / "config" / f"plant_case{case_id}.yaml"
    fin_path = project_root / "data" / "economics" / "financial_parameters.yaml"

    base = BaseConfig(**_load_yaml(base_path))
    case = CaseConfig(**_load_yaml(case_path))
    financial_raw = _load_yaml(fin_path)
    financial = FinancialParams(**financial_raw["financial"])

    if case.case_id != case_id:
        raise ValueError(
            f"Case ID mismatch: yaml {case_path} declares case_id={case.case_id}, "
            f"caller requested {case_id}"
        )

    return RunConfig(case=case, base=base, financial=financial)


def with_bess(cfg: RunConfig, enabled: bool = True) -> RunConfig:
    """Return a copy of ``cfg`` with ``equipment.bess_enabled`` set.

    Lets the S3 sensitivity flip BESS on/off without editing yaml files.
    The case yaml must already carry a ``bess`` block - the helper only
    toggles the enablement flag.
    """
    if enabled and cfg.case.bess is None:
        raise ValueError(
            f"Case {cfg.case.case_id} has no bess block in yaml; cannot enable"
        )
    new_equipment = cfg.case.equipment.model_copy(update={"bess_enabled": enabled})
    new_case = cfg.case.model_copy(update={"equipment": new_equipment})
    return cfg.model_copy(update={"case": new_case})


_REACTOR_CAPEX_SCENARIOS: dict[str, str] = {
    "FOAK": "FOAK",
    "ATB_Mid": "ATB_Mid",
    "NOAK": "NOAK",
}


def with_carbon_price(cfg: RunConfig, price_usd_per_tco2: float) -> RunConfig:
    """Return a copy of ``cfg`` with the carbon price set to ``price_usd_per_tco2``.

    Used by the S6 sensitivity to sweep the policy lever {$0, $50, $100}
    without editing yamls. The price enters the TAC objective in builder.py
    (Cases 1-2) and the post-solve TAC in case0.py / case3.py.

    Raises ValueError if the price is negative.
    """
    if price_usd_per_tco2 < 0:
        raise ValueError(
            f"carbon_price_usd_per_tco2 must be ≥ 0, got {price_usd_per_tco2!r}"
        )
    new_physics = cfg.base.physics.model_copy(
        update={"carbon_price_usd_per_tco2": price_usd_per_tco2}
    )
    new_base = cfg.base.model_copy(update={"physics": new_physics})
    return cfg.model_copy(update={"base": new_base})


def with_nuclear_ptc(cfg: RunConfig, enabled: bool) -> RunConfig:
    """Return a copy of ``cfg`` with the Section 45Y clean-electricity PTC on/off.

    The credit is default-on in the baseline (Cases 1-2). This toggle lets a
    sensitivity run the policy-off counterfactual without editing yamls; the
    credit enters the TAC objective in builder.py as a flat per-MWh reduction
    on net nuclear generation, levelized over the TAC window (see
    ``finance.levelized_ptc_usd_per_mwh``).
    """
    new_financial = cfg.financial.model_copy(
        update={"nuclear_ptc_enabled": bool(enabled)}
    )
    return cfg.model_copy(update={"financial": new_financial})


def with_wacc(cfg: RunConfig, wacc: float) -> RunConfig:
    """Return a copy of ``cfg`` with the WACC overridden and CRF recomputed.

    S7 mini-sensitivity: sweep WACC ∈ {5%, 6.7%, 10%} on Case 2 to
    test whether financing cost is a larger lever on Premium than the 6.5×
    FOAK→NOAK CAPEX learning trajectory. Each WACC value re-computes the
    Capital Recovery Factor over the 20-year project life:

        CRF(i, n) = i * (1+i)^n / ((1+i)^n - 1)

    so every CAPEX×CRF term in cases/builder picks up the new annualization
    automatically (no further code paths need to read WACC directly).
    """
    if wacc <= 0.0 or wacc >= 1.0:
        raise ValueError(f"wacc must be in (0, 1), got {wacc!r}")
    n = cfg.financial.project_lifetime_years
    factor = (1.0 + wacc) ** n
    crf = wacc * factor / (factor - 1.0)
    new_financial = cfg.financial.model_copy(
        update={
            "WACC_nominal": wacc,
            "capital_recovery_factor": crf,
        }
    )
    return cfg.model_copy(update={"financial": new_financial})


def with_reactor_capex(
    cfg: RunConfig,
    scenario: str,
    project_root: Union[Path, str],
) -> RunConfig:
    """Override the reactor CAPEX with an S4 scenario.

    Reads ``data/reactor/bwrx300_economic.yaml`` and substitutes the chosen
    scenario's ``overnight_capital_cost_usd_per_kWe`` into the case config.
    Used to drive the S4 sensitivity (FOAK $14,700 / ATB_Mid $7,615 /
    NOAK $2,250 per kWe) without editing yamls.

    Args:
        cfg: the base ``RunConfig`` for one of Cases 1-3.
        scenario: ``"FOAK"``, ``"ATB_Mid"``, or ``"NOAK"``.
        project_root: repo root path so the helper can read the data yaml.

    Raises:
        ValueError: if the case has no reactor block, or scenario is unknown.
    """
    if cfg.case.reactor is None:
        raise ValueError(
            f"Case {cfg.case.case_id} has no reactor block; nothing to override"
        )
    if scenario not in _REACTOR_CAPEX_SCENARIOS:
        raise ValueError(
            f"scenario must be one of {sorted(_REACTOR_CAPEX_SCENARIOS)}, "
            f"got {scenario!r}"
        )
    path = (
        Path(project_root) / "data" / "reactor" / "bwrx300_economic.yaml"
    )
    if not path.exists():
        raise FileNotFoundError(f"BWRX-300 economic yaml not found: {path}")
    with path.open() as f:
        econ = yaml.safe_load(f)
    occ = float(
        econ["scenarios"][scenario]["overnight_capital_cost_usd_per_kWe"]
    )
    new_reactor = cfg.case.reactor.model_copy(update={"capex_usd_per_kWe": occ})
    new_case = cfg.case.model_copy(update={"reactor": new_reactor})
    return cfg.model_copy(update={"case": new_case})


def with_cooling_cop(cfg: RunConfig, cop_houston: float) -> RunConfig:
    """Return a copy of ``cfg`` with the electric-chiller (VCC) COP overridden.

    S1 recasts the old "PUE sweep" as a cooling-efficiency sweep. With the heat
    load fixed at P_IT / eta_chain, the data-center cooling overhead is set by
    the chiller COP: reported PUE = 1 + 1 / (eta_chain * COP). Sweeping COP over
    {11.11, 3.70, 2.22} reproduces effective PUE {1.10, 1.30, 1.50}. The
    absorption COP is a physical property of the LiBr cycle and is left as-is.

    Raises ValueError if the COP is non-positive or the case has no vcc block.
    """
    if cop_houston <= 0.0:
        raise ValueError(f"cop_houston must be > 0, got {cop_houston!r}")
    if cfg.case.vcc is None:
        raise ValueError(
            f"Case {cfg.case.case_id} has no vcc block; cannot override COP"
        )
    new_vcc = cfg.case.vcc.model_copy(update={"cop_houston": cop_houston})
    new_case = cfg.case.model_copy(update={"vcc": new_vcc})
    return cfg.model_copy(update={"case": new_case})
