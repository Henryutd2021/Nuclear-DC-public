"""Build the hourly IT load profiles from the NLR generative-AI workload dataset.

Source: Vercellino et al. (2026), Measurement of Generative AI Workload Power
Profiles for Whole-Facility Data Center Infrastructure Planning,
arXiv:2604.07345; dataset doi:10.7799/3025227. The raw files are not
redistributed here; data/workload/raw_nlr_colocation/README.md explains where
to put them.

The 1-minute profiles of the 10 MW colocation facility (2,840 nodes, measured
year 2018) are scaled by 20 to a 200 MW IT nameplate and averaged to hourly
values. The 60% utilization variant (dc_200mw_real_60u_2018.csv) is the model
input; the 20/40/80% variants and profile_stats.yaml are written alongside.
"""
from pathlib import Path
import pandas as pd
import yaml

OUT = Path(__file__).resolve().parents[1] / "workload"
OUT.mkdir(parents=True, exist_ok=True)
DATA_ROOT = OUT.parent

NLR_DIR = Path(__file__).resolve().parents[1] / "workload/raw_nlr_colocation"
NAMEPLATE_TARGET_MW = 200    # paper target (10 MW × 20 = 200 MW)

results = {}
for u in ["20u", "40u", "60u", "80u"]:
    src = NLR_DIR / f"colocation_10MW_2840nodes_{u}_power.csv"
    df = pd.read_csv(src, parse_dates=["timestamp"])
    # Scale 10 MW → 200 MW by ×20
    df["IT_load_MW"] = (df["power_W"] / 1e6) * 20
    # Aggregate to hourly
    df = df.set_index("timestamp")
    hourly = df["IT_load_MW"].resample("1h").mean()
    hourly = hourly.iloc[:8760]   # ensure 8760
    out_df = pd.DataFrame({
        "hour": range(len(hourly)),
        "timestamp_local": hourly.index,
        "IT_load_MW": hourly.values.round(2),
    })
    out_path = OUT / f"dc_200mw_real_{u}_2018.csv"
    out_df.to_csv(out_path, index=False)
    results[u] = {
        "mean_MW": float(hourly.mean()),
        "min_MW": float(hourly.min()),
        "max_MW": float(hourly.max()),
        "std_MW": float(hourly.std()),
        "rows": int(len(hourly)),
        "csv": out_path.name,
    }
    print(f"  {u}: rows={len(hourly)}, mean={hourly.mean():.1f} MW, max={hourly.max():.1f} MW")

# Summary statistics (60u is the paper baseline)
stats = {
    "nameplate_it_capacity_MW": NAMEPLATE_TARGET_MW,
    "source_facility": "NLR 10 MW colocation facility, 2840 nodes, scaled by 20",
    "measurement_year": 2018,
    "aggregation": "hourly mean of 1-minute values",
    "baseline_utilization": "60u",
    "statistics_per_utilization": results,
}

with open(OUT / "profile_stats.yaml", "w") as f:
    yaml.dump(stats, f, sort_keys=False, default_flow_style=False)
print("Wrote workload/profile_stats.yaml")
