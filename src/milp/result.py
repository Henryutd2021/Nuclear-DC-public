"""Shared result dataclass + extractor for Cases 1-2 (nuclear MILP outputs)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pyomo.environ as pyo

from src.config import RunConfig
from src.data import TimeSeries


@dataclass(frozen=True)
class NuclearCaseResult:
    """Outputs of a nuclear-case MILP solve (Cases 1 or 2).

    Absorption fields are zero-filled for Case 1 (which has no cogen) so
    cross-case comparisons can use the same attributes.
    """

    case_id: int

    # Hourly dispatch
    P_IT_MW: pd.Series
    Q_cool_demand_MWth: pd.Series
    P_rx_MWth: pd.Series
    P_turb_net_MW: pd.Series
    Q_to_abs_MWth: pd.Series
    Q_abs_cool_MWth: pd.Series
    P_VCC_elec_MW: pd.Series
    Q_VCC_cool_MWth: pd.Series
    P_grid_buy_MW: pd.Series
    P_grid_sell_MW: pd.Series

    # Scalars
    pue: float
    capex_annual_usd: float
    fom_annual_usd: float
    vom_annual_usd: float
    fuel_annual_usd: float
    grid_annual_usd: float
    carbon_annual_usd: float
    # Section 45Y clean-electricity PTC, negative (a credit). Zero when the
    # credit is disabled. Already included in tac_usd_per_yr via the objective.
    ptc_annual_usd: float
    tac_usd_per_yr: float
    co2_annual_tonnes: float

    # CO2 breakdown (tonnes/yr). co2_annual_tonnes is the NET of these three:
    # net = reactor-lifecycle + grid-import debit - grid-export credit.
    # The net goes negative whenever the export credit exceeds on-site +
    # import emissions (i.e. when the overbuilt reactor exports heavily).
    co2_rx_lifecycle_tonnes: float = 0.0
    co2_grid_import_tonnes: float = 0.0
    co2_export_credit_tonnes: float = 0.0

    # Realized relative MIP gap reported by the solver (None when the
    # solver interface does not expose bounds).
    mip_gap_achieved: float | None = None


def _to_series(component, T) -> pd.Series:
    """Pull a Pyomo variable or parameter index over T into a pandas Series."""
    return pd.Series([pyo.value(component[t]) for t in T], index=list(T))


def extract_result(
    model: pyo.ConcreteModel, cfg: RunConfig, ts: TimeSeries
) -> NuclearCaseResult:
    """Convert a solved Pyomo model into a NuclearCaseResult."""
    T = list(model.T)

    P_IT = pd.Series(ts.it_load_MW.values, index=T)
    Q_cool = _to_series(model.Q_cool_demand, T)
    P_rx = _to_series(model.P_rx, T)
    P_turb_net = _to_series(model.P_turb_net, T)
    Q_to_abs = _to_series(model.Q_to_abs, T)
    Q_abs_cool = _to_series(model.Q_abs_cool, T)
    P_vcc = _to_series(model.P_vcc, T)
    Q_vcc = _to_series(model.Q_vcc_cool, T)
    P_grid_buy = _to_series(model.P_grid_buy, T)
    P_grid_sell = _to_series(model.P_grid_sell, T)

    annual_scale = model._annual_scale
    dt = model._dt

    # Reactor lifecycle CO2 (UNECE 2022, 12 g/kWh_e) + grid AEF on imports
    # minus offset on exports. There is no ORC, so the only nuclear-electric
    # stream is P_turb_net.
    rx = cfg.case.reactor
    aef = _to_series(model.AEF, T)
    co2_rx_kg = (P_turb_net.sum() * dt * rx.co2_lifecycle_g_per_kwh_e) * annual_scale
    co2_grid_kg = (P_grid_buy * dt * aef).sum() * annual_scale
    co2_offset_kg = (P_grid_sell * dt * aef).sum() * annual_scale
    co2_tonnes = (co2_rx_kg + co2_grid_kg - co2_offset_kg) / 1000.0

    return NuclearCaseResult(
        case_id=int(cfg.case.case_id),
        P_IT_MW=P_IT,
        Q_cool_demand_MWth=Q_cool,
        P_rx_MWth=P_rx,
        P_turb_net_MW=P_turb_net,
        Q_to_abs_MWth=Q_to_abs,
        Q_abs_cool_MWth=Q_abs_cool,
        P_VCC_elec_MW=P_vcc,
        Q_VCC_cool_MWth=Q_vcc,
        P_grid_buy_MW=P_grid_buy,
        P_grid_sell_MW=P_grid_sell,
        pue=model._pue,
        capex_annual_usd=float(pyo.value(model.capex_annual)),
        fom_annual_usd=float(pyo.value(model.fom_annual)),
        vom_annual_usd=float(pyo.value(model.vom_annual)),
        fuel_annual_usd=float(pyo.value(model.fuel_annual)),
        grid_annual_usd=float(pyo.value(model.grid_annual)),
        carbon_annual_usd=float(pyo.value(model.carbon_annual)),
        ptc_annual_usd=float(pyo.value(model.ptc_annual)),
        tac_usd_per_yr=float(pyo.value(model.objective)),
        co2_annual_tonnes=co2_tonnes,
        co2_rx_lifecycle_tonnes=co2_rx_kg / 1000.0,
        co2_grid_import_tonnes=co2_grid_kg / 1000.0,
        co2_export_credit_tonnes=co2_offset_kg / 1000.0,
        mip_gap_achieved=getattr(model, "_solver_quality", {}).get(
            "mip_gap_achieved"
        ),
    )
