"""Write the part-load performance curves in data/perf/.

turbine_hr.csv is the BWRX-300 turbine heat rate versus output, ab_cop.csv the
double-effect absorption chiller COP versus heat input, and ec_cop.csv the
electric chiller COP versus load fraction. Sources are noted next to each
curve below.
"""
from pathlib import Path
import pandas as pd
import numpy as np

PERF = Path(__file__).resolve().parents[1] / "perf"
PERF.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 1. BWRX-300 Turbine Heat Rate
# =============================================================================
# BWRX-300: 870 MWth → 300 MWe (gross). Saturated-steam BWR turbine.
# Heat rate (HR) = Q_thermal_in / W_electric_out, units MWth/MWe.
# At nominal: HR = 870/300 = 2.90 → η = 1/2.90 = 0.345.
#
# Part-load HR shape: typical large-steam-turbine curve.  At ~30% load HR rises
# to ~3.4 (η drops to 0.29); at full load reaches design point.
#
# Source: GE-Hitachi BWRX-300 General Description (HR @ design point);
# Black & Veatch steam-turbine off-design factor library (HR vs load shape).

P_design = 300.0
HR_design = 2.90
# Off-design factors (relative HR multiplier at fractional load)
# At 0.30 load: HR_factor = 1.17; smooth fit to 1.0 at design
loads = np.array([0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05])
hr_factors = np.array([1.17, 1.115, 1.075, 1.045, 1.025, 1.012, 1.007, 1.004, 1.001, 1.000, 1.001])
p_mw = loads * P_design
hr_mwth_per_mwe = HR_design * hr_factors

turbine_df = pd.DataFrame({"P_MW": p_mw.round(1), "HR_MWth_per_MWe": hr_mwth_per_mwe.round(4)})
turbine_df.to_csv(PERF / "turbine_hr.csv", index=False)
print(f"saved turbine_hr.csv: {len(turbine_df)} rows, range "
      f"P=[{p_mw.min():.0f},{p_mw.max():.0f}] MW, HR=[{hr_mwth_per_mwe.min():.3f},{hr_mwth_per_mwe.max():.3f}]")

# =============================================================================
# 2. Absorption chiller COP (double-effect, Houston-corrected)
# =============================================================================
# COP_thermal = Q_cooling_out / Q_heat_in
# Nameplate: 1.30 for double-effect.  Houston wet-bulb correction → 1.10.
# Part-load: COP degrades slightly at low load (kw curve flattens at high load).
#
# Source: Shahzad 2021 (MDPI Energies) + oemof.thermal Kühn-Ziegler model +
# manufacturer datasheets (Shuangliang, Thermax) for part-load curves.

Q_design_ab = 20.0   # 20 MWth heat input @ design (Case 2 nominal)
# Houston-effective COP @ design = 1.10
COP_design_eff = 1.10
q_ab = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
load_ab = q_ab / Q_design_ab
# Part-load shape: COP rises slightly from low-load valley
# At 25% load COP = 0.85 × design; at 60-100% load = 1.00 × design (small degradation > 100%)
rel_cop_ab = np.interp(load_ab,
                       [0.00, 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90, 1.00],
                       [0.00, 0.78, 0.85, 0.91, 0.95, 0.98, 1.00, 1.00, 0.99])
cop_ab = (rel_cop_ab * COP_design_eff).round(4)
ab_df = pd.DataFrame({"Q_in_MWth": q_ab, "COP": cop_ab})
ab_df.to_csv(PERF / "ab_cop.csv", index=False)
print(f"saved ab_cop.csv: {len(ab_df)} rows, COP range [{cop_ab.min():.4f},{cop_ab.max():.4f}]")

# =============================================================================
# 3. Electric chiller COP (vapor-compression, water-cooled centrifugal)
# =============================================================================
# Used in Case 1 (no heat recovery - DC cooled by electric chiller running on
# nuclear electricity) and as backup in Case 2/3.
# Reference machine: 1500 RT water-cooled centrifugal (~5 MWth).
# COP at full load (AHRI-550/590): ~6.1; IPLV ~6.8.
#
# Source: ASHRAE Handbook (Refrigeration) + AHRI standard test data 2024 +
# Trane CenTraVac / York YMC2 datasheets (free PDFs).

# At Houston ambient (32°C cooling tower water vs ASHRAE 29°C), COP derates ~6%
# Effective full-load COP = 6.1 × 0.94 ≈ 5.75
# Part-load IPLV-style curve
COP_design_ec = 5.75
load_ec = np.linspace(0, 1.0, 11)
# Vapor-compression chillers actually IMPROVE at part-load (down to ~30%)
rel_cop_ec = np.array([0.00, 0.70, 0.95, 1.10, 1.18, 1.18, 1.14, 1.08, 1.04, 1.01, 1.00])
cop_ec = (rel_cop_ec * COP_design_ec).round(3)
ec_df = pd.DataFrame({"Load_fraction": load_ec.round(3), "COP": cop_ec})
ec_df.to_csv(PERF / "ec_cop.csv", index=False)
print(f"saved ec_cop.csv: {len(ec_df)} rows, COP range [{cop_ec.min():.3f},{cop_ec.max():.3f}]")

print("\nDone.")
