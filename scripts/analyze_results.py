"""Derive the paper metrics from the run results and validate the ledgers.

    python scripts/analyze_results.py

Reads results/ (written by scripts/run_scenarios.py) and writes
results/paper_metrics.json, results/verification.json,
results/value_decomp_case2.csv and results/source_hashes.json.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from src.config import load_config
from src.finance import annualized_capex
from src.milp.builder import _outage_hours
from src.results.value_decomposition import compute_value_decomposition

OUT = ROOT / "results"


def crossing(x, y):
    for x0, x1, y0, y1 in zip(x[:-1], x[1:], y[:-1], y[1:]):
        if y0 == 0:
            return float(x0)
        if y0 * y1 <= 0:
            return float(x0 - y0 * (x1 - x0) / (y1 - y0))
    return None


def main():
    all_rows = pd.read_csv(OUT / "all_records.csv")
    grid = pd.read_csv(OUT / "master_kpi_table.csv")
    ledger = pd.read_csv(OUT / "paper_ledger.csv")
    def row(group, name):
        vals = all_rows[(all_rows.group == group) & (all_rows.run_id == name)]
        assert len(vals) == 1, (group, name)
        return vals.iloc[0]
    def summary(group, name):
        return json.loads((OUT / group / name / "summary.json").read_text())
    def dispatch(group, name):
        return pd.read_csv(OUT / group / name / "dispatch.csv.gz")
    baseline = {str(cid): summary("main_baseline", f"case{cid}") for cid in range(4)}
    c0 = baseline["0"]["tac_usd_per_yr"]
    report = {"baseline": baseline}
    report["baseline_headline"] = {
        "grid_cost_margins_pct": {str(c):100*(1-baseline[str(c)]["tac_usd_per_yr"]/c0) for c in range(4)},
        "case2_minus_case1_musd": (baseline["2"]["tac_usd_per_yr"]-baseline["1"]["tac_usd_per_yr"])/1e6,
        "case2_absorption_share_pct": 100*baseline["2"]["absorption_cooling_share"],
    }
    def carbon_thresholds(prefix="", exempt=False):
        vals = {}
        for c in (1,2):
            nuclear = [baseline[str(c)]["tac_usd_per_yr"] if exempt else
                       row(prefix+"s6_carbon_price",f"case{c}_co2_0").tac_usd_per_yr]
            nuclear += [row("nuclear_carbon_exempt" if exempt else prefix+"s6_carbon_price",
                            f"case{c}_co2_{p}").tac_usd_per_yr for p in (50,100)]
            grid_cost = [row("s6_carbon_price",f"case0_co2_{p}").tac_usd_per_yr for p in (0,50,100)]
            vals[str(c)] = crossing([0,50,100],list(np.array(nuclear)-grid_cost))
        return vals
    report["carbon_crossovers_usd_per_tco2"] = carbon_thresholds()
    report["capital_parity_usd_per_kwe"] = {}
    for cid in (1,2):
        cfg = load_config(cid, ROOT)
        coefficient = annualized_capex(1000*cfg.case.reactor.electric_power_net_MWe,
                                      cfg.financial.WACC_nominal,cfg.case.reactor.lifetime_years)
        report["capital_parity_usd_per_kwe"][str(cid)] = cfg.case.reactor.capex_usd_per_kWe + (
            c0-baseline[str(cid)]["tac_usd_per_yr"])/coefficient
    report["size_matching"] = grid[(grid.group=="s8_size_matching") & (grid.case_id==2)].assign(
        net_export_twh=lambda x:(x.P_grid_sell_annual_MWh-x.P_grid_buy_annual_MWh)/1e6,
        gap_musd=lambda x:(x.tac_usd_per_yr-x.tac_case0_baseline_usd_per_yr)/1e6,
        excess_usd_per_mwh=lambda x:(x.tac_usd_per_yr-x.tac_case0_baseline_usd_per_yr)/x.it_energy_annual_MWh,
        margin_pct=lambda x:100*x.heat_recovery_premium).to_dict("records")
    report["capital_year"] = ledger[ledger.group=="capital_year"].assign(
        margin_pct=lambda x:100*x.heat_recovery_premium).to_dict("records")
    report["year_results"] = grid[grid.group=="s2_price"].to_dict("records")
    report["wacc_results"] = grid[grid.group=="s7_wacc"].to_dict("records")
    report["capital_results"] = grid[grid.group=="s4_capex"].to_dict("records")
    report["battery_increment_musd"] = {str(c):(row("s3_battery",f"case{c}_bess_on").tac_usd_per_yr-
        baseline[str(c)]["tac_usd_per_yr"])/1e6 for c in (1,2)}
    report["willans"] = all_rows[all_rows.group=="willans_slope"].assign(
        case2_minus_case1_musd=lambda x:(x.tac_usd_per_yr-baseline["1"]["tac_usd_per_yr"])/1e6,
        absorption_share_pct=lambda x:100*x.absorption_cooling_share).to_dict("records")
    report["flat_load"] = []
    for cid in range(4):
        s = row("flat_load",f"case{cid}_flat")
        report["flat_load"].append({"case_id":cid, "tac_musd":s.tac_usd_per_yr/1e6,
            "baseline_tac_musd":baseline[str(cid)]["tac_usd_per_yr"]/1e6,
            "margin_pct":100*(1-s.tac_usd_per_yr/row("flat_load","case0_flat").tac_usd_per_yr),
            "annual_IT_MWh":s.it_energy_annual_MWh,
            "absorption_share_pct":100*s.get("absorption_cooling_share",np.nan)})
    for group, key, second_year_outage in (("biennial", "biennial", 365),
                                          ("biennial_no_outage", "biennial_upper_bound", 0)):
        report[key] = []
        for cid in (1,2):
            s = row(group,f"case{cid}_nonrefueling_year")
            mean = (s.tac_usd_per_yr+baseline[str(cid)]["tac_usd_per_yr"])/2
            report[key].append({"case_id":cid,"nonrefueling_year_tac_musd":s.tac_usd_per_yr/1e6,
                "two_year_mean_tac_musd":mean/1e6,"mean_margin_pct":100*(1-mean/c0),
                "baseline_reduction_musd":(baseline[str(cid)]["tac_usd_per_yr"]-mean)/1e6,
                "mean_absorption_share_pct":50*(s.absorption_cooling_share+baseline[str(cid)]["absorption_cooling_share"]),
                "annual_background_outage_hours":second_year_outage,
                "mean_scheduled_reactor_availability":1-(701+second_year_outage)/(2*8760)})
    report["lifetime"] = []
    for cid in (1,2):
        cfg=load_config(cid,ROOT)
        overnight=cfg.case.reactor.capex_usd_per_kWe*cfg.case.reactor.electric_power_net_MWe*1000
        old=annualized_capex(overnight,cfg.financial.WACC_nominal,40)
        for life in (40,60,80):
            capital=annualized_capex(overnight,cfg.financial.WACC_nominal,life)
            tac=baseline[str(cid)]["tac_usd_per_yr"]+capital-old
            report["lifetime"].append({"case_id":cid,"reactor_amortization_years":life,
                "ptc_levelization_window_years":20,"tac_musd":tac/1e6,
                "capital_reduction_musd":(old-capital)/1e6,"margin_pct":100*(1-tac/c0)})
    report["zero_nuclear_billing"]={"crossovers":carbon_thresholds(exempt=True),"runs":[]}
    for cid in (1,2):
        for p in (50,100):
            s=row("nuclear_carbon_exempt",f"case{cid}_co2_{p}")
            base=row("s6_carbon_price",f"case{cid}_co2_{p}")
            report["zero_nuclear_billing"]["runs"].append({"case_id":cid,"price":p,
                "tac_musd":s.tac_usd_per_yr/1e6,"cost_reduction_musd":(base.tac_usd_per_yr-s.tac_usd_per_yr)/1e6,
                "lifecycle_rx_tonnes":s.co2_rx_lifecycle_tonnes,
                "lifecycle_balance_tonnes":s.co2_annual_tonnes})
    report["ptc40"]={"carbon_crossovers":carbon_thresholds("ptc40_"),"baseline":{},"capital_parity":{}}
    for cid in (1,2):
        s=row("ptc40_main_baseline",f"case{cid}")
        report["ptc40"]["baseline"][str(cid)]={"tac_musd":s.tac_usd_per_yr/1e6,
            "margin_pct":100*(1-s.tac_usd_per_yr/c0),
            "margin_change_pp":100*(baseline[str(cid)]["tac_usd_per_yr"]-s.tac_usd_per_yr)/c0}
        cfg=load_config(cid,ROOT)
        coeff=annualized_capex(300000,cfg.financial.WACC_nominal,40)
        report["ptc40"]["capital_parity"][str(cid)]=cfg.case.reactor.capex_usd_per_kWe+(c0-s.tac_usd_per_yr)/coeff
    vd=compute_value_decomposition(OUT,ROOT)
    vd.to_csv(OUT/"value_decomp_case2.csv",index=False)
    cfg=load_config(2,ROOT)
    coeff=annualized_capex(cfg.case.capacities.absorption_capacity_MWth*1000,
                           cfg.financial.WACC_nominal,cfg.case.absorption.lifetime_years)
    vd["break_even_absorber_capital_usd_per_kwc"]=750+vd.net_value_of_absorption_usd_per_yr/coeff
    report["value_decomposition"] = vd[vd.matching_key.isin(
        ["main_baseline/base", "s1_pue/_pue110","s1_pue/_pue130","s1_pue/_pue150"])].to_dict("records")
    report["year_absorption_penalty_musd"] = {str(y):(row("s2_price",f"case2_year{y}").tac_usd_per_yr-
        row("s2_price",f"case1_year{y}").tac_usd_per_yr)/1e6 for y in (2022,2023,2024)}
    d=dispatch("main_baseline","case2")
    mask=d.hour.isin(_outage_hours(0.92,8760))
    report["outage"]={"hours":int(mask.sum()),"mean_grid_import_MW":float(d.loc[mask,"P_grid_buy_MW"].mean()),
                      "cooling_GWh":float(d.loc[mask,"Q_VCC_cool_MWth"].sum()/1000)}
    report["sensitivity_spans"]=[]
    axes=[("Reactor capital","s4_capex","case2_NOAK","case2_FOAK"),
          ("Market year","s2_price","case2_year2022","case2_year2024"),
          ("WACC","s7_wacc","case2_wacc_50","case2_wacc_100"),
          ("Carbon price","s6_carbon_price","case2_co2_0","case2_co2_100"),
          ("Data-center size","s8_size_matching","case2_load_x050","case2_load_x300"),
          ("Absorber capital","s5_feasibility_2d","case2_smr_ATB_Mid_abs_Bare_Low","case2_smr_ATB_Mid_abs_Turnkey_High"),
          ("PUE","s1_pue","case2_pue110","case2_pue150"),
          ("BESS","s3_battery","case2_bess_off","case2_bess_on")]
    for label,g,a,b in axes:
        x=grid[(grid.group==g)&grid.run_id.isin([a,b])]
        report["sensitivity_spans"].append({"axis":label,
            "margin_span_pp":100*(x.heat_recovery_premium.max()-x.heat_recovery_premium.min()),
            "tac_span_musd":(x.tac_usd_per_yr.max()-x.tac_usd_per_yr.min())/1e6})
    # Numerical checks: cost identity, solve quality, full horizon, and
    # preserved lifecycle emissions under the carbon-billing exemption.
    checks={"records":len(all_rows),"paper_ledger_records":len(ledger),
            "max_reported_mip_gap":float(all_rows.mip_gap_achieved.max()),
            "fixed_nuclear_capital_field":300.0,"failures":[]}
    cost_fields=["capex_annual_usd","fom_annual_usd","vom_annual_usd","fuel_annual_usd",
                 "grid_annual_usd","carbon_annual_usd","ptc_annual_usd"]
    checks["max_cost_identity_error_usd"]=float((all_rows[cost_fields].fillna(0).sum(axis=1)-all_rows.tac_usd_per_yr).abs().max())
    assert len(all_rows)==197 and len(ledger)==109 and len(grid)==100
    assert checks["max_cost_identity_error_usd"]<1e-4
    assert checks["max_reported_mip_gap"]<=1e-4+1e-10
    for s in report["flat_load"]:
        assert abs(s["annual_IT_MWh"]-baseline["0"]["it_energy_annual_MWh"])<1e-6
    for s in report["zero_nuclear_billing"]["runs"]:
        assert s["lifecycle_rx_tonnes"]>0
    report["verification"]=checks
    # Undefined technology-specific endpoints are JSON null, not non-standard NaN.
    report = json.loads(json.dumps(report), parse_constant=lambda value: None)
    (OUT/"paper_metrics.json").write_text(json.dumps(report,indent=2,allow_nan=False))
    (OUT/"verification.json").write_text(json.dumps(checks,indent=2))
    paths=list((ROOT/"src").rglob("*.py"))+list((ROOT/"config").glob("*.yaml"))
    paths += [ROOT/"scripts/run_scenarios.py",ROOT/"scripts/analyze_results.py",
              ROOT/"scripts/make_figures.py",ROOT/"notebooks/figures.ipynb",
              ROOT/"data/reactor/bwrx300_economic.yaml",
              ROOT/"data/economics/financial_parameters.yaml",
              ROOT/"data/economics/henry_hub_daily_2022_2024.csv",
              ROOT/"data/workload/dc_200mw_real_60u_2018.csv"]
    paths += list((ROOT/"data/perf").glob("*.csv"))
    for year in (2022,2023,2024):
        paths += [ROOT/f"data/weather/houston_ambient_{year}.csv",
                  ROOT/f"data/ercot/{year}_dam_lmp_houston.csv",
                  ROOT/f"data/environmental/ercot_carbon_intensity_hourly_{year}.csv"]
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT/"source_hashes.json").write_text(json.dumps(hashes,indent=2))
    print(json.dumps({k:report[k] for k in ["baseline_headline","capital_parity_usd_per_kwe",
        "carbon_crossovers_usd_per_tco2","ptc40","verification"]},indent=2))


if __name__=="__main__":
    main()
