# Input data

The model reads the files below. Each hourly file starts at local midnight
(America/Chicago) on 1 January and the model uses its first 8760 rows, so the
last day of the 2024 leap year is not used. Timestamps are kept as in the
source: UTC for the ERCOT and EIA-930 series, local time for the weather.

## Hourly inputs (one file per year 2022, 2023, 2024)

`ercot/{year}_dam_lmp_houston.csv`
ERCOT day-ahead settlement point price at the HB_HOUSTON hub ($/MWh), used as
both the import and export price. Source: ERCOT Market Information System,
report NP4-183-CD, retrieved with the gridstatus package
(`_raw/fetch_ercot_dam.py`). `ercot/dam_summary.csv` gives annual statistics
for all hubs.

`environmental/ercot_carbon_intensity_hourly_{year}.csv`
Hourly average lifecycle carbon intensity of ERCOT generation (gCO2e/kWh).
It is the generation-weighted mean of fuel-specific lifecycle factors (UNECE
2022, EPA eGRID 2022) over the EIA-930 hourly fuel mix for the ERCO balancing
authority (`_raw/fetch_eia930_erco.py`, then
`_raw/build_real_carbon_intensity.py`). The two summary CSVs in the same
folder are written by the same script.

`weather/houston_ambient_{year}.csv`
Hourly dry-bulb temperature, dew point, relative humidity and wet-bulb
temperature for Houston (29.7604 N, 95.3698 W). Source: Open-Meteo historical
weather API, based on ERA5 reanalysis (Hersbach et al., 2020). Wet-bulb
temperature is computed with the Stull (2011) formula
(`_raw/fetch_open_meteo_houston.py`).

`workload/dc_200mw_real_60u_2018.csv`
Hourly IT load (MW) of the data center. It is the 10 MW colocation-facility
profile at 60% utilization from the NLR generative-AI workload dataset
(Vercellino et al., 2026; arXiv:2604.07345; doi:10.7799/3025227), measured in
2018, scaled by 20 and averaged from 1-minute to hourly values
(`_raw/build_real_ai_workload.py`). The model shifts the trace by whole days
so that weekdays line up with each market year. `workload/profile_stats.yaml`
summarizes the four utilization variants. The raw NLR files are not included;
see `workload/raw_nlr_colocation/README.md`.

`economics/henry_hub_daily_2022_2024.csv`
Henry Hub natural gas spot price ($/MMBtu) on trading days, from EIA series
NG.RNGWHHD.D (`_raw/fetch_henry_hub.py`). Case 3 carries each price forward
over weekends and holidays. `economics/henry_hub_summary.yaml` holds the annual
statistics.

## Parameters

`economics/financial_parameters.yaml` holds the discount rate, capital recovery
and Section 45Y production tax credit settings. `reactor/bwrx300_economic.yaml`
holds the three reactor overnight-cost scenarios of the S4 sensitivity. All
other equipment parameters are in `../config/plant_case*.yaml`; their sources
are given in the Supplementary Information of the paper.

`perf/` contains the part-load curves for the turbine heat rate, the absorption
chiller COP and the electric chiller COP (`_raw/build_performance_curves.py`).

## Rebuilding the data

`make data` (or the scripts in `_raw/` run individually) re-downloads the ERCOT,
EIA, Henry Hub and Open-Meteo data and rebuilds the carbon intensity series.
This needs network access and the optional packages in `requirements.txt`
(`gridstatus`, `xlrd`). Rebuilding the IT load needs the raw NLR files.
Intermediate downloads are kept out of version control by `.gitignore`.

## Terms of use

- ERCOT market data: public, ERCOT terms of use.
- EIA data (EIA-930, Henry Hub): U.S. government work, public domain.
- Open-Meteo / ERA5 weather data: CC BY 4.0. Cite Open-Meteo and Hersbach et
  al. (2020), doi:10.1002/qj.3803.
- NLR generative-AI workload profiles: public dataset from the NLR Data
  Catalog. Cite Vercellino et al. (2026) and doi:10.7799/3025227.
- UNECE (2022) and EPA eGRID emission factors and the NLR Annual Technology
  Baseline 2024: public reports, cited in the paper.
