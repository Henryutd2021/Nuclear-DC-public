"""Absorption chiller value decomposition.

Decomposes the Heat-Recovery Premium of Case 2 vs Case 1 into a waterfall of
six components so the paper can answer "where does absorption chiller value
come from / go to?":

    + Gross VCC electricity saved    (positive, absorption displaces Case 1 VCC kWh)
    - Turbine power lost             (negative, extraction reduces P_turb_net)
    - Absorption CAPEX + FOM         (negative, equipment overhead)
    - Case 2 VCC backup/top-up (negative, steam-limited or hot-day fallback)
    - 45Y credit forgone             (negative, diverted steam reduces creditable output)
    = Net contribution to Premium   (= TAC_Case1 - TAC_Case2)

The extra cooling-tower water cost (Q_reject grows by 1+1/COP) is reported as
a MEMO externality column only: water carries no price in the MILP objective,
so it is excluded from the component sum and the residual - otherwise the
residual would absorb a non-TAC term by construction.

The "matching pair" semantics: each row pairs a Case 2 run with the Case 1
run sharing the same (year, pue, reactor_scenario, bess, carbon_price). The
script also computes the residual (= measured ΔTAC − sum of TAC components) so
any modeling gap surfaces explicitly rather than being hidden.

Output: ``results/value_decomp_case2.csv`` (one row per matched Case 1/Case 2
pair), written by ``scripts/analyze_results.py`` from the stored run results;
no solver is needed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data import load_time_series
from src.finance import annualized_capex

# Texas industrial water cost - Macknick 2012 + TWDB 2024 industrial tariff
# midpoint, used as the externality price for the ΔWater component. Honest
# tradeoff cost rather than a strict market price; kept explicit so users
# can scale it themselves.
WATER_COST_USD_PER_M3 = 0.50


@dataclass(frozen=True)
class ValueDecompositionRow:
    """One waterfall row for a (Case 1, Case 2) matched pair."""

    matching_key: str
    year: int
    pue: float
    reactor_scenario: Optional[str]
    bess_applied: bool
    carbon_price_usd_per_tco2: float

    tac_case1_usd_per_yr: float
    tac_case2_usd_per_yr: float
    net_value_of_absorption_usd_per_yr: float          # = TAC1 - TAC2

    # Waterfall components (positive = adds value, negative = subtracts)
    vcc_elec_saved_usd_per_yr: float                     # gross Case 1 VCC electricity cost
    net_vcc_elec_saved_usd_per_yr: float                 # gross saved less Case 2 VCC backup/top-up
    turbine_gen_lost_usd_per_yr: float
    absorption_capex_fom_usd_per_yr: float
    # Memo externality (not priced in the MILP TAC; excluded from the sum)
    extra_water_cost_usd_per_yr: float
    crystal_cutoff_backup_usd_per_yr: float            # legacy name: all Case 2 VCC backup/top-up
    ptc_forgone_usd_per_yr: float                        # less 45Y credit (Case 2 generates less)
    sum_of_components_usd_per_yr: float
    residual_usd_per_yr: float                          # net - sum_components


def _summary(case_dir: Path) -> dict:
    with (case_dir / "summary.json").open() as f:
        return json.load(f)


def _dispatch(case_dir: Path) -> pd.DataFrame:
    return pd.read_csv(case_dir / "dispatch.csv.gz", compression="gzip")


def _match_pair(
    summaries: dict[str, dict], group: str, case1_id: str, case2_id: str
) -> Optional[tuple[Path, Path]]:
    """Return (case1_dir, case2_dir) if both exist for ``group``."""
    s1 = summaries.get((group, case1_id))
    s2 = summaries.get((group, case2_id))
    if s1 is None or s2 is None:
        return None
    return s1, s2


def _component_value_usd(
    case1_disp: pd.DataFrame,
    case2_disp: pd.DataFrame,
    price_import: pd.Series,
    price_export: pd.Series,
    absorption_capex_annual_usd: float,
    absorption_fom_annual_usd: float,
    crystal_backup_usd: float,
    extraction_slope_MWe_per_MWth: float,
    dt: float,
    annual_scale: float,
) -> dict[str, float]:
    """Translate hourly dispatch deltas into annualized $/yr components.

    Gross VCC electricity displaced is priced at *import* LMP (counterfactual:
    Case 1 would have bought that kWh from ERCOT). Case 2 backup/top-up VCC is
    kept as a separate negative component so the waterfall does not hide it in
    the savings bar. Turbine power lost is priced at *export* LMP
    (counterfactual: Case 2 would have sold those kWh to ERCOT). The asymmetry
    is deliberate - it mirrors how Case 1 vs Case 2 actually sees the market.
    """
    # MW values aligned by integer hour index.
    p_vcc_1 = case1_disp["P_VCC_elec_MW"].to_numpy()
    p_vcc_2 = case2_disp["P_VCC_elec_MW"].to_numpy()
    q_to_abs = case2_disp["Q_to_abs_MWth"].to_numpy()
    p_import = price_import.to_numpy()
    p_export = price_export.to_numpy()

    gross_vcc_displaced_mw = p_vcc_1
    vcc_saved_usd = float((gross_vcc_displaced_mw * p_import).sum() * dt * annual_scale)

    p_turb_lost_mw = extraction_slope_MWe_per_MWth * q_to_abs
    turbine_lost_usd = float((p_turb_lost_mw * p_export).sum() * dt * annual_scale)

    return {
        "vcc_elec_saved_usd_per_yr": vcc_saved_usd,
        "turbine_gen_lost_usd_per_yr": -turbine_lost_usd,
        "absorption_capex_fom_usd_per_yr": -(
            absorption_capex_annual_usd + absorption_fom_annual_usd
        ),
        "crystal_cutoff_backup_usd_per_yr": -crystal_backup_usd,
    }


def _absorption_capex_annual(s2: dict, project_root: Path) -> tuple[float, float]:
    """Return (annualized absorption CAPEX, FOM) for the Case 2 run.

    The summary.json stores aggregated CAPEX/FOM across all equipment, so we
    isolate the absorption block by reading the case-2 yaml + the financial
    factor. This keeps the decomposition robust to future CAPEX changes.
    """
    import yaml

    with (project_root / "config" / "plant_case2.yaml").open() as f:
        case2_yaml = yaml.safe_load(f)
    with (project_root / "data" / "economics" / "financial_parameters.yaml").open() as f:
        fin = yaml.safe_load(f)["financial"]

    # S5 may have overridden absorption CAPEX; fall back to yaml default.
    abs_capex_usd_per_kWth = s2["metadata"].get("absorption_capex_usd_per_kWth")
    if abs_capex_usd_per_kWth is None:
        abs_capex_usd_per_kWth = case2_yaml["absorption"]["capex_usd_per_kWth"]
    abs_fom_usd_per_kWth_yr = case2_yaml["absorption"]["fixed_om_usd_per_kWth_year"]
    abs_cap_MWth = case2_yaml["capacities"]["absorption_capacity_MWth"]
    abs_lifetime_years = case2_yaml["absorption"]["lifetime_years"]

    # Amortize over the chiller's own life (matches builder per-component CRF).
    # S7 may have overridden WACC; read the effective WACC, else the baseline.
    wacc = s2["metadata"].get("wacc_effective") or fin["WACC_nominal"]

    capex_annual = annualized_capex(
        abs_capex_usd_per_kWth * abs_cap_MWth * 1000.0, wacc, abs_lifetime_years
    )
    fom_annual = abs_fom_usd_per_kWth_yr * abs_cap_MWth * 1000.0
    return capex_annual, fom_annual


def _extra_water_cost(
    s1: dict, s2: dict, water_cost_usd_per_m3: float = WATER_COST_USD_PER_M3
) -> float:
    """ΔDirect site water (Case 2 − Case 1) × IT energy × water cost."""
    site_2 = s2.get("water_direct_site_l_per_mwh_e")
    site_1 = s1.get("water_direct_site_l_per_mwh_e")
    it_mwh_2 = s2["it_energy_annual_MWh"]
    if site_2 is None or site_1 is None:
        return 0.0
    extra_l = (site_2 - site_1) * it_mwh_2          # litres
    extra_m3 = extra_l / 1000.0
    return float(extra_m3 * water_cost_usd_per_m3)


def compute_value_decomposition(
    outputs_dir: Path,
    project_root: Path,
    extraction_slope_MWe_per_MWth: Optional[float] = None,
) -> pd.DataFrame:
    """Build the absorption-chiller waterfall across every matched Case 1/2 pair.

    The extraction slope defaults to the live plant_case2.yaml value (0.20,
    the 7-bar crossover heat-balance derivation), net of the turbine
    auxiliary-load fraction so the lost-generation component is valued at
    the salable (net) output the diverted steam would have produced.
    Override only for sensitivity.
    """
    import yaml as _yaml

    if extraction_slope_MWe_per_MWth is None:
        with (project_root / "config" / "plant_case2.yaml").open() as f:
            _tb = _yaml.safe_load(f)["turbine"]
        extraction_slope_MWe_per_MWth = float(
            _tb["extraction_willans_slope_MWe_per_MWth"]
        ) * (1.0 - float(_tb["aux_load_fraction"]))

    rows: list[ValueDecompositionRow] = []

    # Iterate over every group that contains both case1 and case2 runs.
    for group_dir in sorted(outputs_dir.iterdir()):
        if not group_dir.is_dir() or group_dir.name == "figures":
            continue
        case1_runs = sorted(group_dir.glob("case1*"))
        for c1_dir in case1_runs:
            run_id = c1_dir.name
            c2_dir = group_dir / run_id.replace("case1", "case2", 1)
            if not c2_dir.exists():
                continue

            s1 = _summary(c1_dir)
            s2 = _summary(c2_dir)
            d1 = _dispatch(c1_dir)
            d2 = _dispatch(c2_dir)

            year = int(s1["metadata"]["year"])
            ts = load_time_series(project_root=project_root, year=year, num_hours=8760)
            # The MILP uses the same series for both buy & sell; reuse it for
            # both legs of the value decomposition.
            price_import = ts.price_import_usd_per_mwh
            price_export = ts.price_import_usd_per_mwh

            abs_capex, abs_fom = _absorption_capex_annual(s2, project_root)
            extra_water = _extra_water_cost(s1, s2)

            # Case 2 VCC backup/top-up proxy: all hours where the VCC is
            # firing in the absorption case, including steam-limited top-up
            # and full crystallization-gate fallback. Charge those kWh at the
            # hourly import LMP.
            crystal_mask = d2["P_VCC_elec_MW"] > 1e-3
            crystal_backup_mw = d2["P_VCC_elec_MW"][crystal_mask]
            crystal_price = price_import.iloc[crystal_backup_mw.index]
            dt = 1.0  # matches base.yaml time.delta_t = 1.0 h
            annual_scale = 8760.0 / len(d2)
            crystal_backup_usd = float(
                (crystal_backup_mw.values * crystal_price.values).sum()
                * dt
                * annual_scale
            )

            comp = _component_value_usd(
                d1, d2,
                price_import=price_import,
                price_export=price_export,
                absorption_capex_annual_usd=abs_capex,
                absorption_fom_annual_usd=abs_fom,
                crystal_backup_usd=crystal_backup_usd,
                extraction_slope_MWe_per_MWth=extraction_slope_MWe_per_MWth,
                dt=dt,
                annual_scale=annual_scale,
            )

            # Section 45Y credit forgone by adding absorption: diverting steam to
            # the absorber lowers turbine output, so Case 2 earns slightly less
            # production credit than Case 1. Both ptc values are negative (a
            # credit), so ptc_1 - ptc_2 < 0 and the term subtracts value.
            ptc_1 = float(s1.get("ptc_annual_usd", 0.0) or 0.0)
            ptc_2 = float(s2.get("ptc_annual_usd", 0.0) or 0.0)
            comp["ptc_forgone_usd_per_yr"] = ptc_1 - ptc_2

            tac_1 = float(s1["tac_usd_per_yr"])
            tac_2 = float(s2["tac_usd_per_yr"])
            net = tac_1 - tac_2
            sum_components = sum(comp.values())

            rows.append(
                ValueDecompositionRow(
                    matching_key=f"{group_dir.name}/{run_id.split('case1')[1] or 'base'}",
                    year=year,
                    pue=float(s1["metadata"]["pue"]),
                    reactor_scenario=s1["metadata"].get("reactor_scenario"),
                    bess_applied=bool(s1["metadata"].get("bess_applied", False)),
                    carbon_price_usd_per_tco2=float(
                        s1["metadata"].get("carbon_price_usd_per_tco2", 0.0)
                    ),
                    tac_case1_usd_per_yr=tac_1,
                    tac_case2_usd_per_yr=tac_2,
                    net_value_of_absorption_usd_per_yr=net,
                    net_vcc_elec_saved_usd_per_yr=(
                        comp["vcc_elec_saved_usd_per_yr"]
                        + comp["crystal_cutoff_backup_usd_per_yr"]
                    ),
                    sum_of_components_usd_per_yr=sum_components,
                    residual_usd_per_yr=net - sum_components,
                    extra_water_cost_usd_per_yr=-extra_water,
                    **comp,
                )
            )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([asdict(r) for r in rows])


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    outputs_dir = project_root / "results"

    df = compute_value_decomposition(outputs_dir, project_root)
    out_path = outputs_dir / "value_decomp_case2.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} matched (Case 1, Case 2) pairs → {out_path}")
    if len(df) > 0:
        # Print the main_baseline row to terminal for sanity.
        print(df[df["matching_key"].str.startswith("main_baseline")].T)


if __name__ == "__main__":
    main()
