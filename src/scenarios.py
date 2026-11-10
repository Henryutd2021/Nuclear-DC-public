"""Scenario definitions and shared run helpers for the 100-record production grid.

The grid (``build_run_grid``) covers the baseline and eight one- or two-way
sensitivities; ``scripts/run_scenarios.py`` extends it with the additional
scenario groups reported in the paper and executes everything.

  main_baseline       4 runs   Cases 0-3  | year 2023 | full-load PUE 1.35 | reactor ATB-Mid
  s1_pue              6 runs   Cases 1-2  | year 2023 | effective PUE {1.10,1.30,1.50}
                                                      | via chiller COP {11.1,3.70,2.22}
                                                      | reactor ATB-Mid
  s2_price           12 runs   Cases 0-3  | year in {2022, 2023, 2024} | full-load PUE 1.35
                                                      | reactor ATB-Mid
  s3_battery          8 runs   Cases 0-3  | year 2023 | full-load PUE 1.35
                                                      | bess in {off, on}
                                                      | (Cases 0/3 have no BESS block;
                                                      rows duplicated for table shape)
  s4_capex            6 runs   Cases 1-2  | year 2023 | full-load PUE 1.35
                                                      | reactor in {FOAK, ATB_Mid, NOAK}
  s5_feasibility_2d  25 runs   Case 2     | year 2023 | full-load PUE 1.35
                                                      | (SMR, absorption) CAPEX 5x5 grid
                                                      driven by config/capex_grid_s5.yaml
  s6_carbon_price    12 runs   Cases 0-3  | carbon price in {0, 50, 100} $/tCO2
  s7_wacc             3 runs   Case 2     | WACC in {5%, 6.7%, 10%}
  s8_size_matching   24 runs   Cases 0-3  | IT-load multiplier in
                                                      {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

import pandas as pd
import yaml

from src.cases.case0 import Case0Result, solve_case0
from src.cases.case1 import solve_case1
from src.cases.case2 import solve_case2
from src.cases.case3 import Case3NgccResult, solve_case3
from src.config import RunConfig
from src.data import TimeSeries
from src.kpi import (
    epbt_years,
    heat_recovery_premium,
    water_footprint_l_per_mwh,
)
from src.milp.result import NuclearCaseResult

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_FULL_LOAD_PUE = 1.35

CASE_SOLVERS: dict[int, Callable] = {
    0: solve_case0,
    1: solve_case1,
    2: solve_case2,
    3: solve_case3,
}

# ---------------------------------------------------------------------------
# Result -> common DataFrame / scalar dict
# ---------------------------------------------------------------------------


def result_to_dispatch_df(
    r: Union[Case0Result, Case3NgccResult, NuclearCaseResult],
) -> pd.DataFrame:
    """Stack the hourly Series carried by every result type into one frame."""
    cols: dict[str, pd.Series] = {}
    # Common fields
    cols["P_IT_MW"] = r.P_IT_MW
    cols["Q_cool_MWth"] = getattr(
        r, "Q_cool_MWth", getattr(r, "Q_cool_demand_MWth", None)
    )
    cols["P_VCC_elec_MW"] = r.P_VCC_elec_MW
    # Case-specific
    if isinstance(r, Case0Result):
        cols["P_grid_buy_MW"] = r.P_grid_buy_MW
        cols["grid_cost_usd_per_h"] = r.grid_cost_usd_per_h
        cols["grid_emissions_kg_co2_per_h"] = r.grid_emissions_kg_co2_per_h
    elif isinstance(r, Case3NgccResult):
        cols["P_NGCC_elec_MW"] = r.P_NGCC_elec_MW
        cols["fuel_consumption_MMBtu_per_h"] = r.fuel_consumption_MMBtu_per_h
        cols["fuel_cost_usd_per_h"] = r.fuel_cost_usd_per_h
        cols["direct_emissions_kg_co2_per_h"] = r.direct_emissions_kg_co2_per_h
        cols["upstream_emissions_kg_co2_per_h"] = r.upstream_emissions_kg_co2_per_h
    else:  # NuclearCaseResult
        cols["P_rx_MWth"] = r.P_rx_MWth
        cols["P_turb_net_MW"] = r.P_turb_net_MW
        cols["Q_to_abs_MWth"] = r.Q_to_abs_MWth
        cols["Q_abs_cool_MWth"] = r.Q_abs_cool_MWth
        cols["Q_VCC_cool_MWth"] = r.Q_VCC_cool_MWth
        cols["P_grid_buy_MW"] = r.P_grid_buy_MW
        cols["P_grid_sell_MW"] = r.P_grid_sell_MW
    df = pd.DataFrame(cols)
    df.index.name = "hour"
    return df


def result_to_scalars(
    r: Union[Case0Result, Case3NgccResult, NuclearCaseResult],
    ts: TimeSeries,
    cfg: RunConfig,
) -> dict[str, Any]:
    """Scalar KPIs, common across result types.

    LCOE denominator = IT energy delivered (MWh_e/yr).
    LCOC denominator = cooling delivered (MWh_c/yr).
    LCOC numerator (Case 0): CAPEX+FOM+VOM(VCC) + grid cost attributable to VCC.
    LCOC numerator (Cases 1-3): cooling-side equipment cost only (we don't try
    to split shared electricity-sourcing cost across IT vs VCC, so the LCOC
    here is the cooling-equipment levelized cost - the comparable cross-case
    number is total TAC and the Heat-Recovery Premium).
    """
    dt = cfg.base.time.delta_t
    n = r.P_IT_MW.shape[0]
    annual_scale = 8760.0 / n
    it_energy_annual_MWh = float(r.P_IT_MW.sum() * dt * annual_scale)
    cool_energy_annual_MWh = float(
        (r.Q_cool_MWth if hasattr(r, "Q_cool_MWth") else r.Q_cool_demand_MWth).sum()
        * dt
        * annual_scale
    )

    tac = float(r.tac_usd_per_yr)
    lcoe = tac / it_energy_annual_MWh if it_energy_annual_MWh > 0 else float("nan")

    out: dict[str, Any] = {
        "tac_usd_per_yr": tac,
        "lcoe_usd_per_mwh_e": lcoe,
        "it_energy_annual_MWh": it_energy_annual_MWh,
        "cool_energy_annual_MWh": cool_energy_annual_MWh,
        "capex_annual_usd": float(r.capex_annual_usd),
        "fom_annual_usd": float(r.fom_annual_usd),
        "vom_annual_usd": float(r.vom_annual_usd),
        "carbon_annual_usd": float(r.carbon_annual_usd),
        "carbon_price_usd_per_tco2": float(
            cfg.base.physics.carbon_price_usd_per_tco2
        ),
        "pue": float(r.pue),
    }

    # Annual energies needed for KPI #7 (EPBT) and KPI #8 (water footprint).
    p_turb_net_annual = 0.0
    p_ngcc_annual = 0.0
    p_grid_buy_annual = 0.0
    q_abs_cool_annual = 0.0     # needed for the 3-tier water split (Case 2)
    installed_mwe_for_epbt = 0.0
    epbt_tech: Optional[str] = None

    if isinstance(r, Case0Result):
        out["co2_annual_tonnes"] = float(r.co2_annual_tonnes)
        out["grid_annual_usd"] = float(r.grid_annual_usd)
        out["fuel_annual_usd"] = 0.0
        # LCOC for Case 0: cooling-attributable cost
        grid_cost_cooling_annual = float(
            (ts.price_import_usd_per_mwh * r.P_VCC_elec_MW * dt).sum() * annual_scale
        )
        cost_cooling = (
            r.capex_annual_usd + r.fom_annual_usd + r.vom_annual_usd
            + grid_cost_cooling_annual
        )
        out["lcoc_usd_per_mwh_c"] = (
            cost_cooling / cool_energy_annual_MWh
            if cool_energy_annual_MWh > 0 else float("nan")
        )
        out["co2_per_mwh_kg"] = (
            r.co2_annual_tonnes * 1000.0 / it_energy_annual_MWh
            if it_energy_annual_MWh > 0 else float("nan")
        )
        p_grid_buy_annual = float(r.P_grid_buy_MW.sum() * dt * annual_scale)
        # EPBT undefined for grid-only case - no on-site plant to amortize.
    elif isinstance(r, Case3NgccResult):
        out["co2_annual_tonnes"] = float(r.co2_lifecycle_annual_tonnes)
        out["co2_direct_annual_tonnes"] = float(r.co2_direct_annual_tonnes)
        out["co2_lifecycle_annual_tonnes"] = float(r.co2_lifecycle_annual_tonnes)
        out["fuel_annual_usd"] = float(r.fuel_annual_usd)
        out["grid_annual_usd"] = 0.0
        out["ngcc_capacity_MWe"] = float(r.ngcc_capacity_MWe)
        out["delivered_fuel_usd_per_mmbtu"] = float(r.delivered_fuel_usd_per_mmbtu)
        out["lcoc_usd_per_mwh_c"] = float("nan")  # not directly split
        out["co2_per_mwh_kg"] = (
            r.co2_lifecycle_annual_tonnes * 1000.0 / it_energy_annual_MWh
            if it_energy_annual_MWh > 0 else float("nan")
        )
        p_ngcc_annual = float(r.P_NGCC_elec_MW.sum() * dt * annual_scale)
        installed_mwe_for_epbt = float(r.ngcc_capacity_MWe)
        epbt_tech = "ngcc"
    else:  # NuclearCaseResult
        out["co2_annual_tonnes"] = float(r.co2_annual_tonnes)
        # CO2 breakdown so the net (which is negative whenever the overbuilt
        # reactor's export credit dominates) is self-documenting in the table:
        # net = reactor-lifecycle + grid-import debit - grid-export credit.
        out["co2_rx_lifecycle_tonnes"] = float(r.co2_rx_lifecycle_tonnes)
        out["co2_grid_import_tonnes"] = float(r.co2_grid_import_tonnes)
        out["co2_export_credit_tonnes"] = float(r.co2_export_credit_tonnes)
        out["fuel_annual_usd"] = float(r.fuel_annual_usd)
        out["grid_annual_usd"] = float(r.grid_annual_usd)
        # Section 45Y clean-electricity PTC (negative = a credit, already in tac_usd_per_yr).
        out["ptc_annual_usd"] = float(r.ptc_annual_usd)
        out["Q_abs_cool_annual_MWh"] = float(
            r.Q_abs_cool_MWth.sum() * dt * annual_scale
        )
        out["Q_to_abs_annual_MWh"] = float(
            r.Q_to_abs_MWth.sum() * dt * annual_scale
        )
        out["P_grid_buy_annual_MWh"] = float(
            r.P_grid_buy_MW.sum() * dt * annual_scale
        )
        out["P_grid_sell_annual_MWh"] = float(
            r.P_grid_sell_MW.sum() * dt * annual_scale
        )
        out["lcoc_usd_per_mwh_c"] = float("nan")
        out["co2_per_mwh_kg"] = (
            r.co2_annual_tonnes * 1000.0 / it_energy_annual_MWh
            if it_energy_annual_MWh > 0 else float("nan")
        )
        p_turb_net_annual = float(r.P_turb_net_MW.sum() * dt * annual_scale)
        p_grid_buy_annual = float(r.P_grid_buy_MW.sum() * dt * annual_scale)
        q_abs_cool_annual = float(r.Q_abs_cool_MWth.sum() * dt * annual_scale)
        installed_mwe_for_epbt = float(cfg.case.reactor.electric_power_net_MWe)
        epbt_tech = "nuclear_bwr"

    # ---- KPI #7 - EPBT (years) ---------------------------------------------
    if epbt_tech is not None:
        annual_e_out = (
            p_turb_net_annual if epbt_tech == "nuclear_bwr" else p_ngcc_annual
        )
        out["epbt_years"] = epbt_years(
            installed_mwe=installed_mwe_for_epbt,
            annual_electric_output_mwh=annual_e_out,
            technology=epbt_tech,
        )
    else:
        out["epbt_years"] = None

    # ---- KPI #8 - Water footprint, 3-tier -----------------------------------
    wf = water_footprint_l_per_mwh(
        it_energy_annual_mwh=it_energy_annual_MWh,
        p_turb_net_annual_mwh=p_turb_net_annual,
        p_ngcc_annual_mwh=p_ngcc_annual,
        p_grid_buy_annual_mwh=p_grid_buy_annual,
        q_cool_annual_mwh=cool_energy_annual_MWh,
        q_abs_cool_annual_mwh=q_abs_cool_annual,
    )
    if wf is None:
        out["water_total_l_per_mwh_e"] = None
        out["water_direct_site_l_per_mwh_e"] = None
        out["water_indirect_generation_l_per_mwh_e"] = None
        out["water_scarcity_m3_world_eq_per_mwh_e"] = None
    else:
        out["water_total_l_per_mwh_e"] = wf.total_l_per_mwh_e
        out["water_direct_site_l_per_mwh_e"] = wf.direct_site_l_per_mwh_e
        out["water_indirect_generation_l_per_mwh_e"] = wf.indirect_generation_l_per_mwh_e
        out["water_scarcity_m3_world_eq_per_mwh_e"] = (
            wf.scarcity_weighted_m3_world_eq_per_mwh_e
        )

    return out


# ---------------------------------------------------------------------------
# S5 - joint SMR × absorption CAPEX override for the 2D feasibility grid
# ---------------------------------------------------------------------------


def _load_s5_grid() -> dict[str, Any]:
    """Read config/capex_grid_s5.yaml once and return the parsed dict."""
    with (PROJECT_ROOT / "config" / "capex_grid_s5.yaml").open() as f:
        return yaml.safe_load(f)


def with_capex_pair(
    cfg: RunConfig,
    smr_capex_usd_per_kWe: float,
    absorption_capex_usd_per_kWth: float,
) -> RunConfig:
    """Apply a (SMR, absorption) CAPEX pair to a Case 2 config.

    The two costs are set independently so the heatmap axes are physically
    meaningful CAPEX numbers rather than dimensionless multipliers.
    """
    if cfg.case.case_id != 2:
        raise ValueError("S5 2D feasibility scan only applies to Case 2")
    new_reactor = cfg.case.reactor.model_copy(
        update={"capex_usd_per_kWe": smr_capex_usd_per_kWe}
    )
    new_abs = cfg.case.absorption.model_copy(
        update={"capex_usd_per_kWth": absorption_capex_usd_per_kWth}
    )
    new_case = cfg.case.model_copy(
        update={"reactor": new_reactor, "absorption": new_abs}
    )
    return cfg.model_copy(update={"case": new_case})


# ---------------------------------------------------------------------------
# S8 - data-center size matching against a fixed commercial BWRX-300
# ---------------------------------------------------------------------------


def scale_load_time_series(ts: TimeSeries, load_multiplier: float) -> TimeSeries:
    """Scale only the IT-load trace while preserving market/weather inputs."""
    if abs(load_multiplier - 1.0) < 1e-12:
        return ts
    return TimeSeries(
        year=ts.year,
        num_hours=ts.num_hours,
        it_load_MW=ts.it_load_MW * load_multiplier,
        wet_bulb_C=ts.wet_bulb_C,
        price_import_usd_per_mwh=ts.price_import_usd_per_mwh,
        carbon_intensity_g_per_kwh=ts.carbon_intensity_g_per_kwh,
        henry_hub_usd_per_mmbtu=ts.henry_hub_usd_per_mmbtu,
        henry_hub_usd_per_mmbtu_hourly=ts.henry_hub_usd_per_mmbtu_hourly,
    )


def with_data_center_scale(cfg: RunConfig, load_multiplier: float) -> RunConfig:
    """Scale data-center-side equipment capacities for the S8 size sweep.

    The commercial BWRX-300 block is deliberately not resized here. S8 asks how
    a fixed 300 MWe-net SMR matches campuses of different size, so only chillers,
    the NGCC comparator, and the grid interconnect scale with the load. The
    PCC interconnect is data-center-side infrastructure: a larger campus
    contracts a proportionally larger connection, and at the bigger sizes the
    fixed reactor covers a shrinking fraction so the campus relies more on the
    grid. We only scale the interconnect up (multiplier > 1) so the oversized
    export headroom of the small-campus cases is left untouched.
    """
    if abs(load_multiplier - 1.0) < 1e-12:
        return cfg

    cap = cfg.case.capacities
    updates: dict[str, float] = {}
    if cap.electric_chiller_capacity_MWth is not None:
        updates["electric_chiller_capacity_MWth"] = (
            cap.electric_chiller_capacity_MWth * load_multiplier
        )
    if cfg.case.case_id == 2 and cap.absorption_capacity_MWth is not None:
        updates["absorption_capacity_MWth"] = (
            cap.absorption_capacity_MWth * load_multiplier
        )
    if cfg.case.case_id == 3 and cap.ngcc_capacity_MWe is not None:
        updates["ngcc_capacity_MWe"] = cap.ngcc_capacity_MWe * load_multiplier
    if load_multiplier > 1.0:
        base_pcc = (
            cap.pcc_capacity_MW if cap.pcc_capacity_MW is not None else 300.0
        )
        updates["pcc_capacity_MW"] = base_pcc * load_multiplier

    if not updates:
        return cfg
    new_cap = cap.model_copy(update=updates)
    new_case = cfg.case.model_copy(update={"capacities": new_cap})
    return cfg.model_copy(update={"case": new_case})


# ---------------------------------------------------------------------------
# Run specification and grid
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunSpec:
    """One scenario of the production grid."""

    group: str        # 'main_baseline' | 's1_pue' | ... | 's6_carbon_price' | 's7_wacc'
    run_id: str       # filesystem-safe key, unique within group
    case_id: int
    year: int
    pue: float
    reactor_scenario: Optional[str] = None   # 'FOAK' | 'ATB_Mid' | 'NOAK' | None
    bess_on: bool = False                    # only meaningful for Cases 1-2
    smr_capex_usd_per_kWe: Optional[float] = None         # S5 only
    absorption_capex_usd_per_kWth: Optional[float] = None # S5 only
    smr_capex_tag: Optional[str] = None                   # S5 grid label
    absorption_capex_tag: Optional[str] = None            # S5 grid label
    carbon_price_usd_per_tco2: float = 0.0                # S6 sensitivity
    wacc_override: Optional[float] = None                 # S7 sensitivity
    load_multiplier: float = 1.0                          # S8 sensitivity
    cooling_cop_override: Optional[float] = None          # S1 effective-COP sweep


_ALL_CASES = (0, 1, 2, 3)
_NUCLEAR_CASES = (1, 2)


def build_run_grid() -> list[RunSpec]:
    specs: list[RunSpec] = []

    # ---- main_baseline: 4 cases, 2023, full-load PUE 1.35, ATB-Mid --------
    for cid in _ALL_CASES:
        specs.append(
            RunSpec(
                group="main_baseline",
                run_id=f"case{cid}",
                case_id=cid,
                year=2023,
                pue=BASELINE_FULL_LOAD_PUE,
                reactor_scenario="ATB_Mid" if cid in _NUCLEAR_CASES else None,
            )
        )

    # ---- S1 cooling efficiency: Cases 1-2 × effective PUE {1.10,1.30,1.50} ----
    # The heat load is fixed (P_IT / eta_chain); cooling efficiency is the lever.
    # Each target PUE maps to an electric-chiller COP via
    #   PUE = 1 + 1 / (eta_chain * COP)  =>  COP = 1 / ((PUE - 1) * eta_chain),
    # with eta_chain = 0.9 (config/base.yaml), giving COP {11.11, 3.70, 2.22}.
    _ETA_CHAIN = 0.9
    for cid in _NUCLEAR_CASES:
        for pue in (1.10, 1.30, 1.50):
            pue_tag = f"{int(round(pue * 100)):03d}"
            specs.append(
                RunSpec(
                    group="s1_pue",
                    run_id=f"case{cid}_pue{pue_tag}",
                    case_id=cid,
                    year=2023,
                    pue=pue,
                    reactor_scenario="ATB_Mid",
                    cooling_cop_override=1.0 / ((pue - 1.0) * _ETA_CHAIN),
                )
            )

    # ---- S2 ERCOT year regime: 4 cases × {2022, 2023, 2024} ---------------
    for year in (2022, 2023, 2024):
        for cid in _ALL_CASES:
            specs.append(
                RunSpec(
                    group="s2_price",
                    run_id=f"case{cid}_year{year}",
                    case_id=cid,
                    year=year,
                    pue=BASELINE_FULL_LOAD_PUE,
                    reactor_scenario="ATB_Mid" if cid in _NUCLEAR_CASES else None,
                )
            )

    # ---- S3 BESS: 4 cases × {off, on}, 2023, ATB-Mid ----------------------
    for cid in _ALL_CASES:
        for bess_on in (False, True):
            tag = "on" if bess_on else "off"
            specs.append(
                RunSpec(
                    group="s3_battery",
                    run_id=f"case{cid}_bess_{tag}",
                    case_id=cid,
                    year=2023,
                    pue=BASELINE_FULL_LOAD_PUE,
                    reactor_scenario="ATB_Mid" if cid in _NUCLEAR_CASES else None,
                    bess_on=bess_on,
                )
            )

    # ---- S4 Reactor CAPEX: Cases 1-2 × {FOAK, ATB_Mid, NOAK}, 2023 -------
    for cid in _NUCLEAR_CASES:
        for scen in ("FOAK", "ATB_Mid", "NOAK"):
            specs.append(
                RunSpec(
                    group="s4_capex",
                    run_id=f"case{cid}_{scen}",
                    case_id=cid,
                    year=2023,
                    pue=BASELINE_FULL_LOAD_PUE,
                    reactor_scenario=scen,
                )
            )

    # ---- S5 SMR × absorption 2D feasibility grid: 5 × 5 = 25 runs ---------
    grid = _load_s5_grid()
    smr_axis = grid["axes"]["smr_capex_usd_per_kWe"]
    abs_axis = grid["axes"]["absorption_capex_usd_per_kWth"]
    for smr in smr_axis:
        for ab in abs_axis:
            specs.append(
                RunSpec(
                    group="s5_feasibility_2d",
                    run_id=f"case2_smr_{smr['tag']}_abs_{ab['tag']}",
                    case_id=2,
                    year=2023,
                    pue=BASELINE_FULL_LOAD_PUE,
                    # S5 sets SMR CAPEX directly (overrides the S4 scenario path)
                    smr_capex_usd_per_kWe=float(smr["value"]),
                    absorption_capex_usd_per_kWth=float(ab["value"]),
                    smr_capex_tag=smr["tag"],
                    absorption_capex_tag=ab["tag"],
                )
            )

    # ---- S6 Carbon price: Cases 0-3 × {$0, $50, $100}/tCO2 ----------------
    # Locates the carbon price at which the grid-cost margin turns positive.
    # The $0 row duplicates main_baseline numbers for table shape.
    for price in (0.0, 50.0, 100.0):
        price_tag = f"co2_{int(round(price))}"
        for cid in _ALL_CASES:
            specs.append(
                RunSpec(
                    group="s6_carbon_price",
                    run_id=f"case{cid}_{price_tag}",
                    case_id=cid,
                    year=2023,
                    pue=BASELINE_FULL_LOAD_PUE,
                    reactor_scenario="ATB_Mid" if cid in _NUCLEAR_CASES else None,
                    carbon_price_usd_per_tco2=price,
                )
            )

    # ---- S7 WACC mini-scan: Case 2 only × {5%, 6.7%, 10%} ------------------
    # Tests the policy-leverage claim: is the ΔPremium from a 5%→10% WACC
    # spread larger or smaller than the ΔPremium from FOAK→NOAK CAPEX?
    # 5%   → DOE LPO Section 1703 loan guarantee / regulated asset base
    # 6.7% -> NLR ATB 2024 baseline (matches main_baseline)
    # 10%  → merchant project risk premium (no LPO, private financing)
    for wacc in (0.05, 0.067, 0.10):
        wacc_tag = f"wacc_{int(round(wacc * 1000))}"  # e.g. wacc_50, wacc_67, wacc_100
        specs.append(
            RunSpec(
                group="s7_wacc",
                run_id=f"case2_{wacc_tag}",
                case_id=2,
                year=2023,
                pue=BASELINE_FULL_LOAD_PUE,
                reactor_scenario="ATB_Mid",
                wacc_override=wacc,
            )
        )

    # ---- S8 DC-size matching: 4 cases × 6 IT-load multipliers --------------
    # Fixed BWRX-300 nameplate; data-center-side cooling and NGCC comparator
    # capacities scale with the IT-load trace. This is the deployment-realistic
    # counterpart to the exploratory load-matched-reactor counterfactual.
    for mult in (0.50, 1.00, 1.50, 2.00, 2.50, 3.00):
        mult_tag = f"x{int(round(mult * 100)):03d}"
        for cid in _ALL_CASES:
            specs.append(
                RunSpec(
                    group="s8_size_matching",
                    run_id=f"case{cid}_load_{mult_tag}",
                    case_id=cid,
                    year=2023,
                    pue=BASELINE_FULL_LOAD_PUE,
                    reactor_scenario="ATB_Mid" if cid in _NUCLEAR_CASES else None,
                    load_multiplier=mult,
                )
            )

    return specs


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def add_premium_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Heat-Recovery Premium + Carbon-Abatement Cost vs Case 0.

    Both metrics use the same (year, pue, load_multiplier, bess_applied) matching to Case 0,
    falling back to (year, pue, False) and finally to main_baseline Case 0.

    Heat-Recovery Premium = (TAC_case0 - TAC_case) / TAC_case0
    Carbon-Abatement Cost ($/tCO2) = (TAC_case - TAC_case0) / (CO2_case0 - CO2_case)
    """
    # Include carbon_price in the matching key so S6 rows pair with the
    # Case-0 baseline at the same carbon price (Premium and Abatement Cost both
    # need a price-matched denominator to be meaningful). S7 WACC scan compares
    # Case 2 against the constant ATB-baseline Case 0 (different WACC's per row
    # would mean Case 0 also needs a WACC scan - out of scope for the mini-scan).
    key_cols = [
        "year",
        "pue",
        "load_multiplier",
        "bess_applied",
        "carbon_price_usd_per_tco2",
    ]
    case0_rows = df[df["case_id"] == 0].set_index(key_cols)
    # Several groups carry an identical Case-0 control row (same key, same
    # deterministic solve). to_dict() keeps the last duplicate, which is only
    # safe while duplicates agree - assert that instead of relying on it.
    for col in ("tac_usd_per_yr", "co2_annual_tonnes"):
        spread = case0_rows.groupby(level=key_cols)[col].agg(
            lambda v: v.max() - v.min()
        )
        if (spread > 1e-6 * case0_rows[col].abs().max()).any():
            raise RuntimeError(
                f"Case-0 duplicate-key rows disagree on {col}; "
                "the baseline lookup would be order-dependent."
            )
    baseline_tac = case0_rows["tac_usd_per_yr"].to_dict()
    baseline_co2 = case0_rows["co2_annual_tonnes"].to_dict()
    main_case0 = df[(df["group"] == "main_baseline") & (df["case_id"] == 0)].iloc[0]
    main_tac = float(main_case0["tac_usd_per_yr"])
    main_co2 = float(main_case0["co2_annual_tonnes"])

    def _lookup(row, table: dict, main_fallback: float) -> float:
        key = (
            row["year"], row["pue"], row["load_multiplier"], row["bess_applied"],
            row["carbon_price_usd_per_tco2"],
        )
        # Fallbacks: (1) same year/pue at carbon=0, (2) main baseline value.
        v = table.get(key)
        if v is None:
            v = table.get((row["year"], row["pue"], row["load_multiplier"], False, 0.0))
        if v is None:
            v = table.get((row["year"], row["pue"], 1.0, False, 0.0))
        if v is None:
            v = main_fallback
        return float(v)

    df = df.copy()
    df["tac_case0_baseline_usd_per_yr"] = df.apply(
        lambda r: _lookup(r, baseline_tac, main_tac), axis=1
    )
    df["co2_case0_baseline_tonnes"] = df.apply(
        lambda r: _lookup(r, baseline_co2, main_co2), axis=1
    )
    df["heat_recovery_premium"] = df.apply(
        lambda r: heat_recovery_premium(
            r["tac_case0_baseline_usd_per_yr"], r["tac_usd_per_yr"]
        ),
        axis=1,
    )
    return df


def environment_stamp() -> dict[str, Any]:
    """Code and package provenance recorded in the run manifest."""
    import platform
    import subprocess

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],  # noqa: S607
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        sha, dirty = None, None

    versions: dict[str, Any] = {"python": platform.python_version()}
    for mod in ("pyomo", "pandas", "numpy", "pydantic"):
        try:
            versions[mod] = __import__(mod).__version__
        except ImportError:
            versions[mod] = None
    try:
        import gurobipy

        versions["gurobi"] = ".".join(str(v) for v in gurobipy.gurobi.version())
    except ImportError:
        versions["gurobi"] = None
    return {"git_sha": sha, "git_dirty": dirty, "package_versions": versions}
