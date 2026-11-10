"""Year-aware hourly time series loader for Nuclear-DC."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import yaml

_VALID_YEARS: set[int] = {2022, 2023, 2024}
_HOURS_PER_YEAR: int = 8760


@dataclass(frozen=True)
class TimeSeries:
    """Aligned hourly inputs for a single ERCOT operating year.

    All Series are 0-indexed of length ``num_hours`` after any slicing.
    ``henry_hub_usd_per_mmbtu`` is the annual-mean Henry Hub spot price
    kept for backward compatibility; Case 3 now drives NGCC fuel cost
    from the hourly broadcast ``henry_hub_usd_per_mmbtu_hourly`` (EIA
    daily HH ffilled to 24-h blocks), which captures within-year fuel
    volatility - most notably the Jan 2024 cold-snap spike to $13.20.
    """

    year: int
    num_hours: int
    it_load_MW: pd.Series
    wet_bulb_C: pd.Series
    price_import_usd_per_mwh: pd.Series
    carbon_intensity_g_per_kwh: pd.Series
    henry_hub_usd_per_mmbtu: float
    henry_hub_usd_per_mmbtu_hourly: pd.Series


_IT_TRACE_YEAR: int = 2018  # measurement year of the NLR colocation trace


def _align_weekday_phase(load: pd.Series, year: int) -> pd.Series:
    """Shift the 2018 IT-load trace so weekdays line up with the market year.

    The workload trace was measured in calendar 2018 (Jan 1 = Monday) while
    prices/weather/carbon belong to 2022-2024. Joining them positionally puts
    weekend troughs against weekday peaks in two of the three years, which
    moves annual grid cost by more than the Case2-Case1 gap. A circular
    day-shift aligns day-of-week and hour-of-day while preserving the trace's
    seasonal position: hour i of the market year takes the 2018 hour with the
    same weekday, ``(i + 24*offset) mod 8760`` where ``offset`` is the
    market year's Jan-1 weekday (Monday = 0).
    """
    offset_days = date(year, 1, 1).weekday() - date(_IT_TRACE_YEAR, 1, 1).weekday()
    offset_days %= 7
    if offset_days == 0:
        return load
    rolled = np.roll(load.to_numpy(), -24 * offset_days)
    return pd.Series(rolled, index=load.index)


def _read_column(path: Path, col: str, start: int, n: int) -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"Time-series file not found: {path}")
    df = pd.read_csv(path)
    if col not in df.columns:
        raise ValueError(f"{path.name} missing required column {col!r}")
    available = len(df) - start
    if available < n:
        raise ValueError(
            f"{path.name}: only {available} rows available from start={start}, "
            f"need {n}"
        )
    return df[col].iloc[start : start + n].reset_index(drop=True).astype(float)


def load_time_series(
    project_root: Union[Path, str],
    year: int,
    num_hours: int,
    start_hour: int = 0,
) -> TimeSeries:
    """Load aligned hourly time series for one ERCOT operating year.

    Args:
        project_root: Repo root containing ``data/`` and ``config/``.
        year: One of {2022, 2023, 2024}.
        num_hours: How many consecutive hours to load (1..8760).
        start_hour: First hour to include (0-indexed).

    Returns:
        A frozen ``TimeSeries`` with IT load, wet-bulb temperature, DAM-LMP
        import price, and hourly ERCOT carbon intensity.

    Raises:
        ValueError: invalid year, non-positive num_hours, negative start_hour,
            or slice extending past hour 8760.
        FileNotFoundError: any of the required CSV files are missing.
    """
    if year not in _VALID_YEARS:
        raise ValueError(
            f"year must be one of {sorted(_VALID_YEARS)}, got {year!r}"
        )
    if num_hours <= 0:
        raise ValueError(f"num_hours must be positive, got {num_hours!r}")
    if start_hour < 0:
        raise ValueError(f"start_hour must be non-negative, got {start_hour!r}")
    if start_hour + num_hours > _HOURS_PER_YEAR:
        raise ValueError(
            f"start_hour ({start_hour}) + num_hours ({num_hours}) exceeds "
            f"{_HOURS_PER_YEAR}"
        )

    root = Path(project_root)
    it_path = root / "data" / "workload" / "dc_200mw_real_60u_2018.csv"
    wb_path = root / "data" / "weather" / f"houston_ambient_{year}.csv"
    price_path = root / "data" / "ercot" / f"{year}_dam_lmp_houston.csv"
    carbon_path = (
        root
        / "data"
        / "environmental"
        / f"ercot_carbon_intensity_hourly_{year}.csv"
    )

    it_full = _read_column(it_path, "IT_load_MW", 0, _HOURS_PER_YEAR)
    it_aligned = (
        _align_weekday_phase(it_full, year)
        .iloc[start_hour : start_hour + num_hours]
        .reset_index(drop=True)
    )

    return TimeSeries(
        year=year,
        num_hours=num_hours,
        it_load_MW=it_aligned,
        wet_bulb_C=_read_column(wb_path, "wet_bulb_C", start_hour, num_hours),
        price_import_usd_per_mwh=_read_column(
            price_path, "price_usd_per_mwh", start_hour, num_hours
        ),
        carbon_intensity_g_per_kwh=_read_column(
            carbon_path, "carbon_intensity_g_per_kwh", start_hour, num_hours
        ),
        henry_hub_usd_per_mmbtu=load_henry_hub_annual_mean(root, year),
        henry_hub_usd_per_mmbtu_hourly=_load_henry_hub_hourly(
            root, year, start_hour, num_hours
        ),
    )


def _load_henry_hub_hourly(
    project_root: Path, year: int, start_hour: int, num_hours: int
) -> pd.Series:
    """Broadcast EIA Henry Hub daily spot prices to an 8760 h hourly series.

    The raw CSV (``data/economics/henry_hub_daily_2022_2024.csv``) has only
    business days, so weekends and US federal holidays are forward-filled
    (use the most recent traded day). Each daily value then repeats 24 times
    to form an hourly series. Slice [start_hour, start_hour+num_hours) to
    align with the LMP series.
    """
    path = project_root / "data" / "economics" / "henry_hub_daily_2022_2024.csv"
    if not path.exists():
        raise FileNotFoundError(f"Henry Hub daily file not found: {path}")
    raw = pd.read_csv(path, parse_dates=["date"])
    raw = raw.set_index("date")["price_usd_per_mmbtu"].sort_index()

    full_days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    daily = raw.reindex(full_days).ffill().bfill()  # ffill weekends/holidays
    if daily.isna().any():
        raise ValueError(f"Henry Hub daily series has gaps after ffill for {year}")

    hourly = daily.repeat(24).reset_index(drop=True).astype(float)
    if len(hourly) < _HOURS_PER_YEAR:
        raise ValueError(
            f"Henry Hub hourly broadcast produced {len(hourly)} values, "
            f"expected at least {_HOURS_PER_YEAR}"
        )
    return hourly.iloc[start_hour : start_hour + num_hours].reset_index(drop=True)


def load_henry_hub_annual_mean(
    project_root: Union[Path, str], year: int
) -> float:
    """Return the Henry Hub annual mean spot price for ``year`` in $/MMBtu.

    Reads ``data/economics/henry_hub_summary.yaml`` (EIA NG.RNGWHHD.D series,
    2022-2024). Used by Case 3 to scale NGCC fuel cost across the S2
    year-regime sensitivity.

    Raises:
        ValueError: if ``year`` is not in {2022, 2023, 2024}.
        FileNotFoundError: if the summary yaml is missing.
    """
    if year not in _VALID_YEARS:
        raise ValueError(
            f"year must be one of {sorted(_VALID_YEARS)}, got {year!r}"
        )
    path = Path(project_root) / "data" / "economics" / "henry_hub_summary.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Henry Hub summary not found: {path}")
    with path.open() as f:
        summary = yaml.safe_load(f)
    return float(summary["annual_summary"][year]["mean_usd_per_mmbtu"])
