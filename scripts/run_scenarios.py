"""Run every scenario reported in the paper and write the run-level results.

    python scripts/run_scenarios.py --plan     # print the execution plan only
    python scripts/run_scenarios.py            # solve and write results/

The 197 scenario records comprise the 100-record production grid
(``src.scenarios.build_run_grid``), nine reactor-capital/market-year records,
72 records for the 40-year production-tax-credit levelization window, and 16
targeted checks (flat load, biennial refueling, nuclear carbon-billing
exemption, extraction Willans slope).

Records whose operational configuration is identical share one solve. Reactor
and absorber capital costs are fixed objective constants, so they are left out
of the solve key and applied afterwards as an exact annuity shift. Everything
else -- operating conditions, credit windows, outages -- is part of the key.

Each parallel worker runs one Gurobi solve with one thread; all other solver
settings come from ``config/base.yaml``.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
import psutil  # noqa: E402

from src.config import (  # noqa: E402
    load_config,
    with_bess,
    with_carbon_price,
    with_cooling_cop,
    with_reactor_capex,
    with_wacc,
)
from src.data import load_time_series  # noqa: E402
from src.finance import annualized_capex  # noqa: E402
from src.milp.builder import _outage_hours  # noqa: E402
from src.scenarios import (  # noqa: E402
    BASELINE_FULL_LOAD_PUE,
    CASE_SOLVERS,
    RunSpec,
    add_premium_columns,
    build_run_grid,
    environment_stamp,
    result_to_dispatch_df,
    result_to_scalars,
    scale_load_time_series,
    with_capex_pair,
    with_data_center_scale,
)

OUT = ROOT / "results"


@dataclasses.dataclass(frozen=True)
class ScenarioSpec:
    run: RunSpec
    ptc_window: int = 20
    flat_load: bool = False
    no_reactor_outage: bool = False
    retained_background_outage: bool = False
    zero_nuclear_billing: bool = False
    extraction_slope: float | None = None

    @property
    def path(self):
        return OUT / self.run.group / self.run.run_id


def build_specs():
    production = [ScenarioSpec(s) for s in build_run_grid()]
    specs = list(production)
    # Reactor capital x market year.
    for cid in (1, 2):
        for year in (2022, 2024):
            for capital in (2250, 5000):
                specs.append(ScenarioSpec(RunSpec(
                    "capital_year", f"case{cid}_USD{capital}_{year}",
                    cid, year, BASELINE_FULL_LOAD_PUE,
                    reactor_scenario=f"CUSTOM:{capital}")))
    specs.append(ScenarioSpec(RunSpec(
        "capital_year", "case1_USD5000_2023", 1, 2023,
        BASELINE_FULL_LOAD_PUE, reactor_scenario="CUSTOM:5000")))
    # Production grid repeated with a 40-year credit levelization window.
    specs += [dataclasses.replace(s, run=dataclasses.replace(
        s.run, group="ptc40_" + s.run.group), ptc_window=40)
        for s in production if s.run.case_id in (1, 2)]
    for capital in (3500, 4500):
        specs.append(ScenarioSpec(RunSpec(
            "ptc40_capital_extra", f"case2_USD{capital}", 2, 2023,
            BASELINE_FULL_LOAD_PUE, reactor_scenario=f"CUSTOM:{capital}"),
            ptc_window=40))
    # Flat IT load with the same annual energy.
    for cid in (0, 1, 2, 3):
        base = RunSpec("flat_load", f"case{cid}_flat", cid, 2023,
                       BASELINE_FULL_LOAD_PUE,
                       reactor_scenario="ATB_Mid" if cid in (1, 2) else None)
        specs.append(ScenarioSpec(base, flat_load=True))
    for cid in (1, 2):
        # Biennial refueling: the second year keeps background outages only.
        specs.append(ScenarioSpec(RunSpec(
            "biennial", f"case{cid}_nonrefueling_year", cid, 2023,
            BASELINE_FULL_LOAD_PUE, reactor_scenario="ATB_Mid"),
            retained_background_outage=True))
        # Upper bound: the second year has no reactor outage at all.
        specs.append(ScenarioSpec(RunSpec(
            "biennial_no_outage", f"case{cid}_nonrefueling_year", cid, 2023,
            BASELINE_FULL_LOAD_PUE, reactor_scenario="ATB_Mid"),
            no_reactor_outage=True))
        # Carbon price applied with nuclear lifecycle emissions exempt from billing.
        for price in (50, 100):
            specs.append(ScenarioSpec(RunSpec(
                "nuclear_carbon_exempt", f"case{cid}_co2_{price}", cid, 2023,
                BASELINE_FULL_LOAD_PUE, reactor_scenario="ATB_Mid",
                carbon_price_usd_per_tco2=price), zero_nuclear_billing=True))
    # Extraction Willans-slope sweep.
    for slope in (0.08, 0.15, 0.20, 0.25):
        specs.append(ScenarioSpec(RunSpec(
            "willans_slope", f"case2_alpha_{slope:.2f}", 2, 2023,
            BASELINE_FULL_LOAD_PUE, reactor_scenario="ATB_Mid"),
            extraction_slope=slope))
    return specs


def prepare(spec):
    s = spec.run
    cfg = load_config(s.case_id, ROOT)
    solver = cfg.base.solver.model_copy(update={"threads": 1})
    cfg = cfg.model_copy(update={"base": cfg.base.model_copy(update={"solver": solver})})
    if s.reactor_scenario and s.case_id in (1, 2):
        if s.reactor_scenario.startswith("CUSTOM:"):
            rx = cfg.case.reactor.model_copy(update={
                "capex_usd_per_kWe": float(s.reactor_scenario.split(":")[1])})
            cfg = cfg.model_copy(update={"case": cfg.case.model_copy(update={"reactor": rx})})
        else:
            cfg = with_reactor_capex(cfg, s.reactor_scenario, ROOT)
    if s.bess_on and s.case_id in (1, 2):
        cfg = with_bess(cfg, True)
    if s.smr_capex_usd_per_kWe is not None:
        cfg = with_capex_pair(cfg, s.smr_capex_usd_per_kWe, s.absorption_capex_usd_per_kWth)
    if s.carbon_price_usd_per_tco2 > 0:
        cfg = with_carbon_price(cfg, s.carbon_price_usd_per_tco2)
    if s.wacc_override is not None:
        cfg = with_wacc(cfg, s.wacc_override)
    cfg = with_data_center_scale(cfg, s.load_multiplier)
    if s.cooling_cop_override is not None:
        cfg = with_cooling_cop(cfg, s.cooling_cop_override)
    if s.case_id in (1, 2):
        cfg = cfg.model_copy(update={"financial": cfg.financial.model_copy(
            update={"project_lifetime_years": spec.ptc_window})})
        updates = {}
        rx_updates = {}
        if spec.no_reactor_outage or spec.retained_background_outage:
            original = sorted(_outage_hours(cfg.case.reactor.capacity_factor, 8760))
            # The 701-h annual outage proxy is split into 336 h of refueling and
            # 365 h of background outages. The non-refueling year drops only the
            # refueling component; the upper-bound variant drops all outages.
            rx_updates["scheduled_outage_hours"] = original[336:] if spec.retained_background_outage else []
            if s.case_id == 2:
                # The absorber keeps its full annual maintenance window.
                updates["absorption"] = cfg.case.absorption.model_copy(update={
                    "scheduled_maintenance_hours": sorted(_outage_hours(
                        cfg.case.reactor.capacity_factor, 8760))})
        if spec.zero_nuclear_billing:
            rx_updates["co2_billed_g_per_kwh_e"] = 0.0
        if rx_updates:
            updates["reactor"] = cfg.case.reactor.model_copy(update=rx_updates)
        if spec.extraction_slope is not None:
            updates["turbine"] = cfg.case.turbine.model_copy(update={
                "extraction_willans_slope_MWe_per_MWth": spec.extraction_slope})
        cfg = cfg.model_copy(update={"case": cfg.case.model_copy(update=updates)})
    return cfg


def dispatch_key(spec, cfg):
    data = cfg.model_dump()
    for component, field in (("reactor", "capex_usd_per_kWe"),
                             ("absorption", "capex_usd_per_kWth")):
        if data["case"].get(component):
            data["case"][component][field] = None
    payload = {"config": data, "year": spec.run.year,
               "load_multiplier": spec.run.load_multiplier,
               "pue": spec.run.pue, "flat_load": spec.flat_load}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def variable_capital(cfg):
    """The two fixed objective constants deliberately omitted from solve keys."""
    value = 0.0
    if cfg.case.reactor is not None and cfg.case.equipment.reactor_enabled:
        rx = cfg.case.reactor
        value += annualized_capex(rx.capex_usd_per_kWe * rx.electric_power_net_MWe * 1000,
                                  cfg.financial.WACC_nominal, rx.lifetime_years)
    if cfg.case.absorption is not None and cfg.case.equipment.absorption_chiller_enabled:
        ab = cfg.case.absorption
        value += annualized_capex(ab.capex_usd_per_kWth *
                                  cfg.case.capacities.absorption_capacity_MWth * 1000,
                                  cfg.financial.WACC_nominal, ab.lifetime_years)
    return value


def metadata(spec, cfg, elapsed):
    s = spec.run
    return {**dataclasses.asdict(s), "ptc_window_years": spec.ptc_window,
            "flat_load": spec.flat_load, "no_reactor_outage": spec.no_reactor_outage,
            "retained_background_outage": spec.retained_background_outage,
            "zero_nuclear_billing": spec.zero_nuclear_billing,
            "extraction_slope_override": spec.extraction_slope,
            "bess_on_requested": s.bess_on, "bess_supported": s.case_id in (1, 2),
            "bess_applied": s.bess_on and s.case_id in (1, 2),
            "wacc_effective": cfg.financial.WACC_nominal,
            "crf_effective": cfg.financial.capital_recovery_factor,
            "num_hours": 8760, "solve_seconds": elapsed,
            "solver": cfg.base.solver.model_dump()}


def solve_one(spec):
    cfg = prepare(spec)
    key = dispatch_key(spec, cfg)
    existing = spec.path / "summary.json"
    if existing.exists():
        stored = json.loads(existing.read_text())
        stored_cfg = json.loads((spec.path / "config.json").read_text())
        if stored["metadata"].get("dispatch_key") == key and stored_cfg == cfg.model_dump():
            return spec, stored
        raise RuntimeError(f"Config changed for existing result {existing}; choose a new output root")
    start = time.time()
    ts = load_time_series(ROOT, spec.run.year, 8760)
    ts = scale_load_time_series(ts, spec.run.load_multiplier)
    if spec.flat_load:
        ts = dataclasses.replace(ts, it_load_MW=pd.Series(
            ts.it_load_MW.mean(), index=ts.it_load_MW.index))
    result = CASE_SOLVERS[spec.run.case_id](cfg, ts, pue=spec.run.pue)
    dispatch = result_to_dispatch_df(result)
    summary = result_to_scalars(result, ts, cfg)
    if spec.run.case_id in (1, 2):
        heat = float(dispatch.P_rx_MWth.sum())
        electric = float(dispatch.P_turb_net_MW.sum())
        steam = float(dispatch.Q_to_abs_MWth.sum())
        summary.update({"P_rx_annual_MWh": heat, "P_turb_net_annual_MWh": electric,
                        "reactor_net_electric_efficiency": electric / heat,
                        "reactor_steam_utilization": (electric + steam) / heat,
                        "absorption_cooling_share": float(dispatch.Q_abs_cool_MWth.sum()
                            / dispatch.Q_cool_MWth.sum())})
    summary["metadata"] = {**metadata(spec, cfg, time.time() - start),
                           "dispatch_key": key,
                           "mip_gap_achieved": getattr(result, "mip_gap_achieved", None),
                           "calculation": "optimized" if spec.run.case_id in (1, 2) else "deterministic"}
    spec.path.mkdir(parents=True, exist_ok=True)
    dispatch.to_csv(spec.path / "dispatch.csv.gz", index=True, compression="gzip")
    (spec.path / "config.json").write_text(json.dumps(cfg.model_dump(), indent=2))
    existing.write_text(json.dumps(summary, indent=2))
    return spec, summary


def materialize(spec, representative, summary):
    """Write a record that reuses another record's dispatch."""
    cfg, source_cfg = prepare(spec), prepare(representative)
    delta = variable_capital(cfg) - variable_capital(source_cfg)
    data = dict(summary)
    data["tac_usd_per_yr"] += delta
    data["capex_annual_usd"] += delta
    data["lcoe_usd_per_mwh_e"] = data["tac_usd_per_yr"] / data["it_energy_annual_MWh"]
    # The relative MIP gap is rescaled for the additive objective shift.
    source_gap = summary["metadata"].get("mip_gap_achieved")
    adjusted_gap = (None if source_gap is None else source_gap *
                    max(1.0, abs(summary["tac_usd_per_yr"])) /
                    max(1.0, abs(data["tac_usd_per_yr"])))
    data["metadata"] = {**metadata(spec, cfg, 0.0),
                        "dispatch_key": dispatch_key(spec, cfg),
                        "mip_gap_achieved": adjusted_gap,
                        "calculation": "exact_fixed_capital_postprocess" if delta else "identical_config_reuse",
                        "dispatch_source": str(representative.path.relative_to(OUT)),
                        "capital_delta_usd_per_yr": delta}
    spec.path.mkdir(parents=True, exist_ok=True)
    link = spec.path / "dispatch.csv.gz"
    if not link.exists():
        link.symlink_to(os.path.relpath(representative.path / "dispatch.csv.gz", spec.path))
    (spec.path / "config.json").write_text(json.dumps(cfg.model_dump(), indent=2))
    (spec.path / "summary.json").write_text(json.dumps(data, indent=2))
    return data


def scalar_row(summary):
    row = {**summary["metadata"], **{k: v for k, v in summary.items() if k != "metadata"}}
    row.pop("solver", None)
    return row


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", action="store_true", help="print the execution plan only")
    parser.add_argument("--workers", type=int, default=psutil.cpu_count(logical=False) or 1,
                        help="parallel worker processes (capped at the physical core count)")
    parser.add_argument("--out", type=Path, default=OUT,
                        help="output directory (default: results/)")
    args = parser.parse_args()
    OUT = args.out.resolve()
    workers = min(args.workers, psutil.cpu_count(logical=False) or 1)
    specs = build_specs()
    representatives = {}
    for s in specs:
        representatives.setdefault(dispatch_key(s, prepare(s)), s)
    plan = {"num_records": len(specs), "distinct_operational_configs": len(representatives),
            "distinct_nuclear_solves": sum(s.run.case_id in (1, 2) for s in representatives.values()),
            "workers": workers, "threads_per_solve": 1,
            "specs": [dataclasses.asdict(s) for s in specs],
            "environment": environment_stamp()}
    OUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({k: v for k, v in plan.items() if k not in ("specs", "environment")}), flush=True)
    if args.plan:
        return
    # Check Gurobi availability before any expensive annual solve.
    import gurobipy as gp
    with gp.Env(empty=True) as env:
        env.setParam("OutputFlag", 0)
        env.start()
    start = time.time()
    solved = {}
    # Workers resolve result paths through OUT; the initializer keeps a
    # non-default --out valid under the spawn start method as well.
    with ProcessPoolExecutor(max_workers=workers, initializer=_set_out, initargs=(OUT,)) as pool:
        futures = {pool.submit(solve_one, s): key for key, s in representatives.items()}
        for n, future in enumerate(as_completed(futures), 1):
            spec, summary = future.result()
            solved[futures[future]] = summary
            print(f"[{n}/{len(futures)}] {spec.run.group}/{spec.run.run_id} "
                  f"TAC={summary['tac_usd_per_yr']/1e6:.6f}M "
                  f"{summary['metadata']['solve_seconds']:.1f}s", flush=True)
    rows = []
    for s in specs:
        key = dispatch_key(s, prepare(s))
        representative = representatives[key]
        summary = solved[key] if s == representative else materialize(s, representative, solved[key])
        rows.append(scalar_row(summary))
    all_df = pd.DataFrame(rows)
    all_df.to_csv(OUT / "all_records.csv", index=False)
    base_groups = {s.group for s in build_run_grid()}
    base = add_premium_columns(all_df[all_df.group.isin(base_groups)])
    base.to_csv(OUT / "master_kpi_table.csv", index=False)
    ledger = add_premium_columns(all_df[all_df.group.isin(base_groups | {"capital_year"})])
    ledger.to_csv(OUT / "paper_ledger.csv", index=False)
    ptc40 = all_df[all_df.group.str.startswith("ptc40_")].copy()
    controls = base[base.case_id.isin([0, 3])]
    add_premium_columns(pd.concat([controls, ptc40])).to_csv(OUT / "ptc40_ledger.csv", index=False)
    plan.update({"completed_at_utc": pd.Timestamp.utcnow().isoformat(),
                 "wall_seconds": time.time() - start,
                 "calculation_counts": all_df.calculation.value_counts().to_dict()})
    (OUT / "manifest.json").write_text(json.dumps(plan, indent=2))
    print(f"Completed {len(rows)} records in {time.time()-start:.1f}s", flush=True)


def _set_out(path):
    global OUT
    OUT = path


if __name__ == "__main__":
    main()
