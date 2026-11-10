# Raw NLR workload profiles (not included)

`../dc_200mw_real_60u_2018.csv` is built from the 1-minute whole-facility
power profiles of the NLR generative-AI workload dataset:

Vercellino, R., Willard, J., Campos, G., da Silva Pereira, W., Hull, O.,
Selensky, M., Mueller, J. (2026). Measurement of Generative AI Workload Power
Profiles for Whole-Facility Data Center Infrastructure Planning.
arXiv:2604.07345. Dataset: National Laboratory of the Rockies Data Catalog,
doi:10.7799/3025227, https://data.nlr.gov/submissions/312

The raw files (about 1 GB zipped) are not redistributed here. To rebuild the
IT load profiles, download the dataset, copy the four colocation files from
`03_whole-facility_profiles/colocation/simulated_data/` into this folder

```text
colocation_10MW_2840nodes_20u_power.csv
colocation_10MW_2840nodes_40u_power.csv
colocation_10MW_2840nodes_60u_power.csv
colocation_10MW_2840nodes_80u_power.csv
```

and run `python data/_raw/build_real_ai_workload.py` from the repository root.
