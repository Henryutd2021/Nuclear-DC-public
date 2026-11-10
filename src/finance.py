"""Capital annualization helpers and the Section 45Y clean-electricity PTC.

Each asset is amortized over its own engineering life rather than a single
project-wide window, and assets with end-of-life value receive a sinking-fund
salvage credit. Both reduce to the textbook equivalent-annual-cost identity

    EAC = P * CRF(i, n) - F * SFF(i, n),

with capital P, salvage F = salvage_fraction * P, recovery factor
CRF(i, n) = i (1+i)^n / ((1+i)^n - 1), and sinking-fund factor
SFF(i, n) = CRF(i, n) - i = i / ((1+i)^n - 1).
"""

from __future__ import annotations


def crf(wacc: float, lifetime_years: int) -> float:
    """Capital recovery factor at discount rate ``wacc`` over ``lifetime_years``."""
    if wacc <= 0.0:
        raise ValueError(f"wacc must be > 0, got {wacc!r}")
    if lifetime_years <= 0:
        raise ValueError(f"lifetime_years must be > 0, got {lifetime_years!r}")
    growth = (1.0 + wacc) ** lifetime_years
    return wacc * growth / (growth - 1.0)


def annualized_capex(
    capex: float,
    wacc: float,
    lifetime_years: int,
    salvage_fraction: float = 0.0,
) -> float:
    """Annual capital charge for ``capex`` recovered over its own life.

    ``salvage_fraction`` is the fraction of overnight capital recovered at end
    of life and credited via the sinking-fund factor. With no salvage this is
    simply ``capex * crf(wacc, lifetime_years)``.
    """
    if not 0.0 <= salvage_fraction < 1.0:
        raise ValueError(
            f"salvage_fraction must be in [0, 1), got {salvage_fraction!r}"
        )
    recovery = crf(wacc, lifetime_years)
    return capex * (recovery * (1.0 - salvage_fraction) + salvage_fraction * wacc)


def levelized_ptc_usd_per_mwh(
    full_credit_usd_per_mwh: float,
    wacc: float,
    credit_duration_years: int,
    window_years: int,
) -> float:
    """Section 45Y clean-electricity PTC levelized over the TAC window.

    The Section 45Y credit (26 U.S.C. 45Y; the technology-neutral successor
    to the renewable-electricity PTC for zero-GHG facilities placed in
    service after 2024) pays a flat per-MWh amount with no gross-receipts
    phaseout, but only for the first ``credit_duration_years`` (statutorily
    10) after commissioning. The TAC compares costs over a longer
    ``window_years`` annualization horizon, so the credit is converted to a
    window-equivalent levelized rate by the ratio of annuity present-value
    factors:

        rate_eff = rate * A(i, d) / A(i, w),   A(i, n) = (1 - (1+i)^-n) / i

    At i = 6.7%, d = 10, w = 20 the factor is 0.657, i.e. a $30/MWh 10-year
    credit is worth $19.70/MWh levelized across a 20-year window.
    """
    if credit_duration_years <= 0 or window_years <= 0:
        raise ValueError(
            "credit_duration_years and window_years must be positive, got "
            f"{credit_duration_years!r}, {window_years!r}"
        )
    if wacc <= 0.0:
        raise ValueError(f"wacc must be > 0, got {wacc!r}")
    if credit_duration_years >= window_years:
        return float(full_credit_usd_per_mwh)

    def annuity(n: int) -> float:
        return (1.0 - (1.0 + wacc) ** -n) / wacc

    return float(
        full_credit_usd_per_mwh
        * annuity(credit_duration_years)
        / annuity(window_years)
    )
