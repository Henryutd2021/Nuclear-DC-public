"""Download EIA daily Henry Hub spot prices (series NG.RNGWHHD.D) and keep 2022-2024.

Writes data/economics/henry_hub_daily_2022_2024.csv and a monthly summary.
"""
from pathlib import Path
import urllib.request
import pandas as pd

OUT_DIR = Path(__file__).resolve().parents[1] / "economics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# EIA public Excel for Henry Hub spot price (no API key needed)
URL = "https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDd.xls"
LOCAL = Path(__file__).resolve().parent / "RNGWHHDd.xls"

if not LOCAL.exists():
    print("Downloading Henry Hub daily spot prices from EIA...")
    urllib.request.urlretrieve(URL, LOCAL)
    print(f"  saved {LOCAL.name} ({LOCAL.stat().st_size/1024:.0f} KB)")
else:
    print(f"Already have {LOCAL.name}")

# Read the EIA xls - Data 1 sheet has the historical series
df = pd.read_excel(LOCAL, sheet_name="Data 1", skiprows=2)
df.columns = ["date", "price_usd_per_mmbtu"]
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Filter to 2022-2024
mask = (df["date"] >= "2022-01-01") & (df["date"] < "2025-01-01")
period = df[mask].copy()
period.to_csv(OUT_DIR / "henry_hub_daily_2022_2024.csv", index=False)
print(f"Saved {len(period)} daily HH prices")

# Monthly summary
period["year_month"] = period["date"].dt.to_period("M")
monthly = period.groupby("year_month")["price_usd_per_mmbtu"].agg(["mean", "min", "max", "std"])
monthly.to_csv(OUT_DIR / "henry_hub_monthly_summary.csv")

print("\n=== Henry Hub annual mean ===")
print(period.groupby(period["date"].dt.year)["price_usd_per_mmbtu"].agg(["mean", "min", "max", "std"]).round(2))
