"""Download hourly Houston weather for 2022-2024 and derive wet-bulb temperature.

Source: Open-Meteo historical weather API (ERA5 reanalysis, Hersbach et al.
2020, doi:10.1002/qj.3803), CC BY 4.0, no key required. Location 29.7604 N,
95.3698 W, local time America/Chicago to match ERCOT settlement hours.

Wet-bulb temperature follows Stull (2011):

    Tw = T*atan(0.151977*(RH+8.313659)^0.5) + atan(T+RH) - atan(RH-1.676331)
         + 0.00391838*RH^1.5*atan(0.023101*RH) - 4.686035

with T the dry-bulb temperature in C and RH the relative humidity in %. The
model reads data/weather/houston_ambient_{year}.csv.
"""
from pathlib import Path
import urllib.parse
import urllib.request
import json
import math
import pandas as pd
import time

OUT = Path(__file__).resolve().parents[1] / "weather"
OUT.mkdir(parents=True, exist_ok=True)
ROOT = OUT.parent

LAT, LON = 29.7604, -95.3698     # Houston (downtown)
TZ = "America/Chicago"

BASE = "https://archive-api.open-meteo.com/v1/archive"
VARS = ",".join([
    "temperature_2m", "dew_point_2m", "relative_humidity_2m",
    "precipitation", "wind_speed_10m", "surface_pressure"
])


def fetch_year(year):
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "hourly": VARS,
        "timezone": TZ,
    }
    url = BASE + "?" + urllib.parse.urlencode(params)
    print(f"  fetching {year} ({len(VARS.split(','))} variables)...", flush=True)
    t = time.time()
    with urllib.request.urlopen(url, timeout=60) as r:
        d = json.loads(r.read().decode())
    df = pd.DataFrame(d["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    print(f"    got {len(df)} rows in {time.time()-t:.1f}s", flush=True)
    return df


def stull_wet_bulb(T, RH):
    """Stull (2011) empirical wet-bulb formula, valid 5% < RH < 99%, -20 < T < 50."""
    RH = max(min(RH, 99.0), 5.0)
    try:
        Tw = (T * math.atan(0.151977 * math.sqrt(RH + 8.313659))
              + math.atan(T + RH)
              - math.atan(RH - 1.676331)
              + 0.00391838 * (RH ** 1.5) * math.atan(0.023101 * RH)
              - 4.686035)
    except (ValueError, ZeroDivisionError):
        return T - 5.0  # fallback approximation
    return Tw


all_years = []
for year in [2022, 2023, 2024]:
    df = fetch_year(year)
    df["wet_bulb_C"] = df.apply(lambda r: stull_wet_bulb(r["temperature_2m"], r["relative_humidity_2m"]), axis=1)
    year_out = OUT / f"houston_hourly_{year}.csv"
    df.to_csv(year_out, index=False)
    print(f"  saved {year_out.name}")
    all_years.append(df)

# Combined 3-year file
combined = pd.concat(all_years, ignore_index=True)
combined = combined.sort_values("time").reset_index(drop=True)
combined["hour"] = range(len(combined))
combined.to_csv(OUT / "houston_wet_bulb_combined.csv", index=False)
print(f"\nCombined 3-yr file: {len(combined)} rows")
print(f"  dry-bulb range: {combined['temperature_2m'].min():.1f} to {combined['temperature_2m'].max():.1f} °C")
print(f"  wet-bulb range: {combined['wet_bulb_C'].min():.1f} to {combined['wet_bulb_C'].max():.1f} °C")
print(f"  wet-bulb summer p95: {combined.loc[combined['time'].dt.month.isin([6,7,8]), 'wet_bulb_C'].quantile(0.95):.1f} °C")

# Year-specific copies in weather/
for year in [2022, 2023, 2024]:
    df = combined[combined["time"].dt.year == year].copy()
    df = df.iloc[:8760].reset_index(drop=True)
    df["hour"] = range(len(df))
    out = OUT / f"houston_ambient_{year}.csv"
    df[["hour", "time", "temperature_2m", "dew_point_2m", "relative_humidity_2m",
        "wet_bulb_C"]].to_csv(out, index=False)
    print(f"  saved {out.name}")
