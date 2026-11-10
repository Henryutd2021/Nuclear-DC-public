"""Download EIA-930 hourly balancing-authority data and extract ERCOT (ERCO).

Source: U.S. Energy Information Administration, Hourly Electric Grid Monitor,
six-month BALANCE files at https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/
(public, no key). EIA split some fuel columns in mid-2024; they are summed
back into the earlier fuel categories. The generation mix written to
data/ercot/{year}_genmix_erco.csv feeds build_real_carbon_intensity.py.
"""
from pathlib import Path
import urllib.request
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "ercot"
RAW_DIR  = Path(__file__).resolve().parent
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_{year}_{half}.csv"
HALVES = {"H1": "Jan_Jun", "H2": "Jul_Dec"}

# Canonical fuel buckets (post-aggregation)
FUEL_BUCKETS = {
    "Coal": ["Net Generation (MW) from Coal"],
    "NaturalGas": ["Net Generation (MW) from Natural Gas"],
    "Nuclear": ["Net Generation (MW) from Nuclear"],
    "Petroleum": ["Net Generation (MW) from All Petroleum Products"],
    "Hydro": ["Net Generation (MW) from Hydropower and Pumped Storage",
              "Net Generation (MW) from Hydropower Excluding Pumped Storage",
              "Net Generation (MW) from Pumped Storage"],
    "Solar": ["Net Generation (MW) from Solar",
              "Net Generation (MW) from Solar without Integrated Battery Storage",
              "Net Generation (MW) from Solar with Integrated Battery Storage"],
    "Wind": ["Net Generation (MW) from Wind",
             "Net Generation (MW) from Wind without Integrated Battery Storage",
             "Net Generation (MW) from Wind with Integrated Battery Storage"],
    "Geothermal": ["Net Generation (MW) from Geothermal"],
    "BatteryStorage": ["Net Generation (MW) from Battery Storage",
                       "Net Generation (MW) from Other Energy Storage"],
    "Other": ["Net Generation (MW) from Other Fuel Sources"],
    "Unknown": ["Net Generation (MW) from Unknown Fuel Sources"],
}

CORE_COLS = ["Balancing Authority", "Data Date", "Hour Number",
             "Local Time at End of Hour", "UTC Time at End of Hour",
             "Demand Forecast (MW)", "Demand (MW)", "Net Generation (MW)",
             "Total Interchange (MW)", "Demand (MW) (Adjusted)",
             "Net Generation (MW) (Adjusted)", "Region"]


def download(year, half_key):
    fname = BASE.format(year=year, half=HALVES[half_key])
    local = RAW_DIR / f"EIA930_BALANCE_{year}_{HALVES[half_key]}.csv"
    if local.exists() and local.stat().st_size > 5_000_000:
        print(f"  already have {local.name} ({local.stat().st_size/1e6:.0f} MB), skip", flush=True)
        return local
    print(f"  downloading {fname}", flush=True)
    urllib.request.urlretrieve(fname, local)
    print(f"  saved {local.name} ({local.stat().st_size/1e6:.0f} MB)", flush=True)
    return local


def filter_erco(local_path):
    """Read CSV, filter for ERCO, aggregate fuel buckets, return DataFrame."""
    # Get header to know which columns exist
    header_df = pd.read_csv(local_path, nrows=0)
    available_cols = set(header_df.columns)
    keep_cols = [c for c in CORE_COLS if c in available_cols]
    for fuels in FUEL_BUCKETS.values():
        keep_cols.extend([c for c in fuels if c in available_cols])

    chunks = []
    for chunk in pd.read_csv(local_path, usecols=keep_cols,
                             dtype={"Balancing Authority": "string"},
                             chunksize=200_000):
        chunks.append(chunk[chunk["Balancing Authority"] == "ERCO"].copy())
    df = pd.concat(chunks, ignore_index=True)

    # Aggregate fuel buckets
    for bucket, src_cols in FUEL_BUCKETS.items():
        cols_present = [c for c in src_cols if c in df.columns]
        df[f"gen_{bucket}_MW"] = df[cols_present].sum(axis=1, min_count=1) if cols_present else 0
    return df


for year in [2022, 2023, 2024]:
    print(f"\n[EIA930] year {year}", flush=True)
    halves = []
    for h in ["H1", "H2"]:
        local = download(year, h)
        halves.append(filter_erco(local))
    erco = pd.concat(halves, ignore_index=True)
    erco["timestamp_utc"] = pd.to_datetime(erco["UTC Time at End of Hour"])
    erco = erco.sort_values("timestamp_utc").reset_index(drop=True)
    erco["hour"] = range(len(erco))
    print(f"  ERCO rows: {len(erco):,}", flush=True)

    # Full file with aggregated buckets
    full_out = DATA_DIR / f"{year}_eia930_erco_full.csv"
    erco.to_csv(full_out, index=False)
    print(f"  saved {full_out.name}: {full_out.stat().st_size/1e6:.1f} MB", flush=True)

    # Slim load file
    load_cols = ["hour", "timestamp_utc", "Demand (MW)", "Demand (MW) (Adjusted)"]
    load_cols_present = [c for c in load_cols if c in erco.columns]
    erco[load_cols_present].to_csv(DATA_DIR / f"{year}_load_system_actual.csv", index=False)

    # Generation mix file
    gen_cols = ["hour", "timestamp_utc", "Net Generation (MW) (Adjusted)"] + \
               [f"gen_{b}_MW" for b in FUEL_BUCKETS.keys()]
    gen_cols_present = [c for c in gen_cols if c in erco.columns]
    erco[gen_cols_present].to_csv(DATA_DIR / f"{year}_genmix_erco.csv", index=False)
    print("  saved load + genmix CSVs", flush=True)

print("\n[EIA930] Done.")
