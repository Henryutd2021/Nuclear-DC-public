"""Hourly average carbon intensity of ERCOT generation, 2022-2024.

The intensity in each hour is the generation-weighted mean of lifecycle
emission factors (gCO2e/kWh, UNECE 2022 and EPA eGRID 2022) over the EIA-930
fuel mix written by fetch_eia930_erco.py:

    coal 980, natural gas 430, petroleum 810, nuclear 12, solar 43, wind 11,
    hydro 24, geothermal 38, battery storage 0, other 500

These are average, not marginal, emission factors. Writes
data/environmental/ercot_carbon_intensity_hourly_{year}.csv plus annual and
hour-of-day summaries.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ERCOT = ROOT / "ercot"
OUT = ROOT / "environmental"
OUT.mkdir(parents=True, exist_ok=True)

# Lifecycle emission factors (gCO2e/kWh) - UNECE 2022 + EPA eGRID 2022 weighted
EF = {
    "Coal": 980,
    "NaturalGas": 430,
    "Petroleum": 810,
    "Nuclear": 12,
    "Solar": 43,
    "Wind": 11,
    "Hydro": 24,
    "Geothermal": 38,
    "BatteryStorage": 0,
    "Other": 500,
    "Unknown": 400,
}

# Records for annual summary
summary = []

for year in [2022, 2023, 2024]:
    inp = ERCOT / f"{year}_genmix_erco.csv"
    df = pd.read_csv(inp, parse_dates=["timestamp_utc"])
    fuel_cols = [c for c in df.columns if c.startswith("gen_") and c.endswith("_MW")]
    print(f"[CI] {year}: {len(df):,} rows, fuels={[c[4:-3] for c in fuel_cols]}", flush=True)

    # Fill NaN with 0 (no generation)
    for c in fuel_cols:
        df[c] = df[c].fillna(0).clip(lower=0)

    # Hourly emissions (g) and hourly generation (MWh)
    df["co2_emissions_g"] = 0.0
    for c in fuel_cols:
        fuel = c[4:-3]   # gen_X_MW -> X
        ef = EF.get(fuel, 400)
        df["co2_emissions_g"] += df[c] * 1000 * ef   # MW × 1000 kWh/h × g/kWh = g/h
    df["total_gen_MW"] = df[fuel_cols].sum(axis=1)
    df["carbon_intensity_g_per_kwh"] = df["co2_emissions_g"] / (df["total_gen_MW"] * 1000).clip(lower=1e-3)

    # Save hourly
    out_cols = ["hour", "timestamp_utc", "total_gen_MW", "co2_emissions_g", "carbon_intensity_g_per_kwh"] + fuel_cols
    out_path = OUT / f"ercot_carbon_intensity_hourly_{year}.csv"
    df[out_cols].to_csv(out_path, index=False)
    print(f"  saved {out_path.name}: mean CI = {df['carbon_intensity_g_per_kwh'].mean():.0f} gCO2/kWh", flush=True)

    # Hour-of-day shape (for use in optimizer when WattTime not available)
    df["hour_of_day"] = pd.to_datetime(df["timestamp_utc"]).dt.hour
    hod_shape = df.groupby("hour_of_day")["carbon_intensity_g_per_kwh"].mean()

    summary.append({
        "year": year,
        "annual_mean_g_per_kwh": float(df["carbon_intensity_g_per_kwh"].mean()),
        "annual_min_g_per_kwh": float(df["carbon_intensity_g_per_kwh"].min()),
        "annual_max_g_per_kwh": float(df["carbon_intensity_g_per_kwh"].max()),
        "annual_std_g_per_kwh": float(df["carbon_intensity_g_per_kwh"].std()),
        "q1_mean": float(df.loc[df["timestamp_utc"].dt.month <= 3, "carbon_intensity_g_per_kwh"].mean()),
        "q2_mean": float(df.loc[df["timestamp_utc"].dt.month.between(4,6), "carbon_intensity_g_per_kwh"].mean()),
        "q3_mean": float(df.loc[df["timestamp_utc"].dt.month.between(7,9), "carbon_intensity_g_per_kwh"].mean()),
        "q4_mean": float(df.loc[df["timestamp_utc"].dt.month >= 10, "carbon_intensity_g_per_kwh"].mean()),
        **{f"hod_{i:02d}_mean": float(hod_shape.loc[i]) for i in range(24)}
    })

summary_df = pd.DataFrame(summary)
summary_df.to_csv(OUT / "ercot_carbon_intensity_summary.csv", index=False)
print("\n=== ANNUAL CARBON INTENSITY (gCO2e/kWh, AEF) ===")
print(summary_df[["year", "annual_mean_g_per_kwh", "annual_min_g_per_kwh", "annual_max_g_per_kwh", "annual_std_g_per_kwh"]].round(0).to_string(index=False))

# Compute paper-ready hour-of-day shape (mean across 3 years, relative to annual mean)
all_hourly = []
for year in [2022, 2023, 2024]:
    d = pd.read_csv(OUT / f"ercot_carbon_intensity_hourly_{year}.csv", parse_dates=["timestamp_utc"])
    d["hour_of_day"] = d["timestamp_utc"].dt.hour
    d["year"] = year
    all_hourly.append(d)
joined = pd.concat(all_hourly, ignore_index=True)
annual_mean_3yr = joined["carbon_intensity_g_per_kwh"].mean()
hod_3yr = joined.groupby("hour_of_day")["carbon_intensity_g_per_kwh"].mean()
hod_multiplier = (hod_3yr / annual_mean_3yr).round(4)
print("\n=== Hour-of-day multiplier (3-year mean) ===")
print(hod_multiplier.to_string())
print(f"\n3-yr annual mean CI = {annual_mean_3yr:.1f} gCO2e/kWh")

# Save HOD shape as small CSV the YAML can reference
hod_df = pd.DataFrame({"hour_of_day": hod_3yr.index,
                       "ci_g_per_kwh": hod_3yr.values,
                       "multiplier_vs_annual": hod_multiplier.values})
hod_df.to_csv(OUT / "ercot_carbon_intensity_hour_of_day_shape.csv", index=False)
