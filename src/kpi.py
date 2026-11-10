"""Headline KPIs for Nuclear-DC.

  1. TAC (passthrough)
  2. Net levelized cost of IT supply ($/MWh_e delivered to IT; net TAC over
     IT energy - includes cooling capital, carbon cost, PTC and export
     netting, so it is NOT a generation LCOE)
  3. LCOC ($/MWh_c delivered as chilled water)
  4. Heat-Recovery Premium (free function - needs two TACs)
  5. Annual CO2 (passthrough)
  6. EPBT (years, energy payback time - Lenzen 2008 / IAEA 2018 method)
  7. Water footprint, three-tier:
        7a. Direct site water (L/MWh_e_IT) - cooling-tower makeup at the DC
            (VCC + absorption-chiller Q_reject side)
        7b. Indirect generation water (L/MWh_e_IT) - Macknick 2012 weighted
            by the power-source mix
        7c. Scarcity-weighted total (m3 world-eq/MWh_e_IT) - multiplied by
            the AWARE / Aqueduct scarcity factor for the host basin (ERCOT
            South Hub baseline 0.65; world-mean reference = 1.0)

The static carbon-abatement-cost KPI was retired: with merchant exports
credited at the grid AEF its denominator is dominated by off-site
displacement rather than load decarbonization, and it is algebraically the
same quantity as the carbon-price crossover the paper already reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.cases.case0 import Case0Result
from src.config import RunConfig
from src.data import TimeSeries

# ---------------------------------------------------------------------------
# Literature factors for KPIs 7-8 (one source of truth, citable in Methods).
# ---------------------------------------------------------------------------
# Embodied primary energy per kWe installed (MWh_e-equivalent), Lenzen 2008
# Table 5 + IAEA 2018 INPRO update; values consistent with a ~0.5 yr EPBT
# at 92% CF for BWR and ~0.2 yr for NGCC.
EMBODIED_ENERGY_MWH_PER_KWE: dict[str, float] = {
    "nuclear_bwr": 4.5,   # BWRX-300 proxy - Lenzen 2008 PWR/BWR midpoint
    "ngcc": 1.0,          # NREL ATB 2024 LCI screening value
}

# Water-consumption factors (all entries L per kWh of the named basis).
# Generation factors are NLR Macknick 2012; the data-center cooling-duty
# factors are authors' estimates (Macknick covers power generation only).
# "Consumption" not "withdrawal" - once-through cooling withdraws much more
# but returns most of it; cooling-tower consumption is the true footprint.
WATER_L_PER_KWH: dict[str, float] = {
    "nuclear_cooling_tower": 2.54,   # L/kWh_e - Macknick 2012 median nuclear w/ tower
    "ngcc_cooling_tower":    0.78,   # L/kWh_e - Macknick 2012 median NGCC w/ tower
    "ercot_grid_blend":      1.42,   # L/kWh_e - Macknick 2012 ERCOT generation mix
    "vcc_hybrid_cooling":    0.10,   # L/kWh_c - VCC tower water for DC duty (authors' estimate)
    # Per MWh_c of cooling produced by the absorption chiller, the
    # cooling tower must reject Q_cool + Q_input ≈ Q_cool * (1 + 1/COP_abs).
    # With double-effect COP_abs ≈ 1.2 (Houston annual mean), that's ~1.83×
    # the heat rejected per unit cooling vs ~1.0× for VCC - but absorption
    # uses a wet-cooling tower at lower delta-T so its makeup water per MWh
    # rejected matches the VCC factor. Net: ~1.83× the site water per MWh_c
    # delivered. We capture this as a higher per-MWh_c factor:
    "absorption_cooling_reject": 0.183,  # = 0.10 * (1 + 1/1.2)
}

# Aqueduct / AWARE-style basin scarcity factor for the Case 2 anchor
# location (ERCOT South Hub / Houston). World average is 1.0; ERCOT South
# baseline ~0.65 (moderate scarcity, conservative midpoint between Aqueduct
# baseline-water-stress and AWARE). Used only by KPI #8c.
AQUEDUCT_SCARCITY_FACTOR_ERCOT_SOUTH = 0.65


@dataclass(frozen=True)
class KPIBundle:
    tac_usd_per_yr: float
    lcoe_usd_per_mwh_e: float
    lcoc_usd_per_mwh_c: float
    co2_annual_tonnes: float
    co2_per_mwh_kg: float


def compute_kpis_case0(
    result: Case0Result, ts: TimeSeries, cfg: RunConfig
) -> KPIBundle:
    """Compute Case 0 KPIs from a solver result + the time series + config.

    LCOE denominator = IT energy delivered (MWh_e) - the useful work.
    LCOC numerator   = cooling-attributable annualized cost (capex+fom+vom of
                       VCC + the grid energy cost specifically for running VCC).
    LCOC denominator = cooling delivered (MWh_c).
    """
    dt = cfg.base.time.delta_t
    annual_scale = 8760.0 / result.P_IT_MW.shape[0]

    # ---- Energies (annualized) --------------------------------------------
    it_energy_annual_MWh = result.P_IT_MW.sum() * dt * annual_scale
    cool_energy_annual_MWh = result.Q_cool_MWth.sum() * dt * annual_scale

    # ---- LCOE / CO2 intensity -------------------------------------------
    lcoe = result.tac_usd_per_yr / it_energy_annual_MWh
    co2_per_mwh_kg = result.co2_annual_tonnes * 1000.0 / it_energy_annual_MWh

    # ---- LCOC: split grid cost between IT and VCC ------------------------
    grid_cost_cooling_annual = (
        ts.price_import_usd_per_mwh * result.P_VCC_elec_MW * dt
    ).sum() * annual_scale
    cost_cooling_annual = (
        result.capex_annual_usd
        + result.fom_annual_usd
        + result.vom_annual_usd
        + grid_cost_cooling_annual
    )
    lcoc = cost_cooling_annual / cool_energy_annual_MWh

    return KPIBundle(
        tac_usd_per_yr=result.tac_usd_per_yr,
        lcoe_usd_per_mwh_e=lcoe,
        lcoc_usd_per_mwh_c=lcoc,
        co2_annual_tonnes=result.co2_annual_tonnes,
        co2_per_mwh_kg=co2_per_mwh_kg,
    )


def heat_recovery_premium(tac_baseline: float, tac_case: float) -> float:
    """Heat-Recovery Premium (reported as the grid-cost margin).

        Premium = (TAC_baseline - TAC_case) / TAC_baseline

    Positive  → ``case`` is cheaper than the grid-only baseline.
    Zero      → break-even.
    Negative  → ``case`` is more expensive (heat recovery hurts).

    Raises ValueError if ``tac_baseline`` is not strictly positive.
    """
    if tac_baseline <= 0:
        raise ValueError(
            f"tac_baseline must be positive, got {tac_baseline!r}"
        )
    return (tac_baseline - tac_case) / tac_baseline


def epbt_years(
    installed_mwe: float,
    annual_electric_output_mwh: float,
    technology: str,
) -> Optional[float]:
    """KPI #7 - energy payback time (years) following Lenzen 2008 method.

        EPBT = (embodied_energy_per_kWe × installed_kWe) / annual_output_kWh

    The embodied factor is the cumulative primary-energy demand of plant
    construction + decommissioning, expressed as MWh_e-equivalent per kWe
    installed. ``technology`` keys into ``EMBODIED_ENERGY_MWH_PER_KWE`` -
    only nuclear_bwr and ngcc are populated; pass anything else to get None.
    """
    factor = EMBODIED_ENERGY_MWH_PER_KWE.get(technology)
    if factor is None or annual_electric_output_mwh <= 0:
        return None
    # factor [MWh/kWe] × installed_kWe = total embodied MWh; divide by the
    # annual output MWh to get years. Both sides are MWh, so no extra ×1000.
    embodied_mwh = factor * installed_mwe * 1000.0
    return embodied_mwh / annual_electric_output_mwh


@dataclass(frozen=True)
class WaterFootprint:
    """Three-tier water KPI.

    All three layers normalised to IT energy delivered so they're directly
    comparable across cases. The L→m3 conversion happens only in
    ``scarcity_weighted_m3_world_eq_per_mwh_e`` (so the scarcity tier is
    in m3-world-eq while the other two are in litres).
    """

    direct_site_l_per_mwh_e: float
    indirect_generation_l_per_mwh_e: float
    total_l_per_mwh_e: float
    scarcity_weighted_m3_world_eq_per_mwh_e: float


def water_footprint_l_per_mwh(
    it_energy_annual_mwh: float,
    p_turb_net_annual_mwh: float = 0.0,
    p_ngcc_annual_mwh: float = 0.0,
    p_grid_buy_annual_mwh: float = 0.0,
    q_cool_annual_mwh: float = 0.0,
    q_abs_cool_annual_mwh: float = 0.0,
    scarcity_factor: float = AQUEDUCT_SCARCITY_FACTOR_ERCOT_SOUTH,
) -> Optional[WaterFootprint]:
    """KPI #8 (three-tier) - water consumption per MWh_e_IT delivered.

    Decomposes the footprint into three policy-readable layers:

      W_direct_site      = factor_vcc * (Q_cool_total - Q_abs_cool_total)
                         + factor_absorption_reject * Q_abs_cool_total
      W_indirect_gen     = factor_nuclear * P_turb_net_total
                         + factor_ngcc    * P_NGCC_total
                         + factor_grid    * P_grid_buy_total
      W_total            = W_direct_site + W_indirect_gen
      W_scarcity_weighted = (W_total / 1000) * scarcity_factor   # m3-world-eq

    The split between VCC and absorption on the cooling side is the key
    Note: an absorption chiller forced to reject Q_cool*(1+1/COP)
    consumes more cooling-tower makeup per MWh_c than a VCC of the same
    duty, so Case 2 may *raise* site water even while it lowers indirect
    generation water by displacing grid imports.

    Exporting nuclear electricity *still consumes water at the tower* even
    when the kWh goes to ERCOT instead of IT, so we charge the full
    P_turb_net rather than the IT-share. Returns ``None`` if
    ``it_energy_annual_mwh`` is non-positive.
    """
    if it_energy_annual_mwh <= 0:
        return None

    q_vcc_only = max(0.0, q_cool_annual_mwh - q_abs_cool_annual_mwh)

    # Direct site water: VCC tower + absorption-chiller Q_reject tower.
    # Macknick factors are L/kWh, so multiply MWh by 1000.
    w_direct_l = (
        WATER_L_PER_KWH["vcc_hybrid_cooling"]         * q_vcc_only             * 1000.0
        + WATER_L_PER_KWH["absorption_cooling_reject"] * q_abs_cool_annual_mwh * 1000.0
    )

    # Indirect generation water: charge the full plant output, then the
    # IT-energy denominator below converts to a per-IT-MWh basis.
    w_indirect_l = (
        WATER_L_PER_KWH["nuclear_cooling_tower"] * p_turb_net_annual_mwh  * 1000.0
        + WATER_L_PER_KWH["ngcc_cooling_tower"]  * p_ngcc_annual_mwh      * 1000.0
        + WATER_L_PER_KWH["ercot_grid_blend"]    * p_grid_buy_annual_mwh  * 1000.0
    )

    w_direct_per_mwh = w_direct_l / it_energy_annual_mwh
    w_indirect_per_mwh = w_indirect_l / it_energy_annual_mwh
    w_total_per_mwh = w_direct_per_mwh + w_indirect_per_mwh
    w_scarcity_per_mwh = (w_total_per_mwh / 1000.0) * scarcity_factor

    return WaterFootprint(
        direct_site_l_per_mwh_e=w_direct_per_mwh,
        indirect_generation_l_per_mwh_e=w_indirect_per_mwh,
        total_l_per_mwh_e=w_total_per_mwh,
        scarcity_weighted_m3_world_eq_per_mwh_e=w_scarcity_per_mwh,
    )
