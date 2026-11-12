"""Pure pandas transforms for the startup-success pipeline.

Kept separate from the DAG so each step can be unit-tested without Airflow.

Problem statement
-----------------
At the moment a startup raises its FIRST funding round, can we predict whether it
will eventually exit successfully (acquired or IPO)?

Only information known at first funding is used as features (no total funding,
no later rounds) to avoid look-ahead leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Crunchbase snapshot ends ~Dec 2015. Only keep startups first funded 2000-2011
# so every company has had 4+ years to reach an outcome.
COHORT_START = 2000
COHORT_END = 2011
TOP_CATEGORIES = 15
ALREADY_PUBLIC_ROUND_TYPES = {"post_ipo_equity", "post_ipo_debt", "secondary_market"}

FEATURE_COLUMNS = [
    "first_round_type",
    "log_first_round_usd",
    "first_round_amount_missing",
    "primary_category",
    "is_usa",
    "founded_year_missing",
    "months_founded_to_funding",
    "funding_year",
    "fed_funds_rate",
    "nasdaq_12m_return",
]
TARGET = "is_success"


# ---------------------------------------------------------------------------
# Crunchbase
# ---------------------------------------------------------------------------
def clean_companies(df: pd.DataFrame) -> pd.DataFrame:
    """One row per company with a clean label and founding info."""
    out = df.copy()
    out = out.dropna(subset=["permalink", "status"])
    out = out.drop_duplicates(subset="permalink")
    out["founded_at"] = pd.to_datetime(out["founded_at"], errors="coerce")
    out["first_funding_at"] = pd.to_datetime(out["first_funding_at"], errors="coerce")
    # Bad historic dates in the raw file (e.g. year 0011) -> treat as missing
    out.loc[out["founded_at"].dt.year < 1900, "founded_at"] = pd.NaT
    out["primary_category"] = (
        out["category_list"].fillna("Unknown").str.split("|").str[0].str.strip()
    )
    out["is_usa"] = (out["country_code"] == "USA").astype(int)
    out[TARGET] = out["status"].isin(["acquired", "ipo"]).astype(int)
    return out[
        ["permalink", "name", "status", TARGET, "primary_category", "is_usa", "founded_at"]
    ].rename(columns={"permalink": "company_permalink"})


def clean_rounds(df: pd.DataFrame) -> pd.DataFrame:
    """Keep each company's FIRST funding round only."""
    out = df.copy()
    out["funded_at"] = pd.to_datetime(out["funded_at"], errors="coerce")
    out = out.dropna(subset=["company_permalink", "funded_at"])
    out = out[out["funded_at"].dt.year >= 1990]
    out = out.sort_values(["company_permalink", "funded_at"])
    first = out.groupby("company_permalink", as_index=False).first()
    first = first.rename(
        columns={
            "funded_at": "first_funded_at",
            "funding_round_type": "first_round_type",
            "raised_amount_usd": "first_round_usd",
        }
    )
    return first[["company_permalink", "first_funded_at", "first_round_type", "first_round_usd"]]


def build_startup_table(companies: pd.DataFrame, rounds: pd.DataFrame) -> pd.DataFrame:
    """Join companies to their first round and engineer startup-level features."""
    df = companies.merge(rounds, on="company_permalink", how="inner")
    # A "first round" of post-IPO equity/debt or secondary-market sale means the company
    # was already public/mature when it entered Crunchbase -> not a startup, and it leaks
    # the IPO outcome into the features. Exclude them.
    df = df[~df["first_round_type"].isin(ALREADY_PUBLIC_ROUND_TYPES)]
    df["funding_year"] = df["first_funded_at"].dt.year
    df = df[df["funding_year"].between(COHORT_START, COHORT_END)].copy()

    df["first_round_amount_missing"] = df["first_round_usd"].isna().astype(int)
    df["log_first_round_usd"] = np.log10(df["first_round_usd"].fillna(0).clip(lower=0) + 1)
    months = (df["first_funded_at"] - df["founded_at"]).dt.days / 30.44
    df["founded_year_missing"] = df["founded_at"].isna().astype(int)
    df["months_founded_to_funding"] = months.clip(lower=0, upper=240).fillna(months.median())

    top = df["primary_category"].value_counts().nlargest(TOP_CATEGORIES).index
    df["primary_category"] = df["primary_category"].where(df["primary_category"].isin(top), "Other")
    df["funding_month"] = df["first_funded_at"].dt.to_period("M").dt.to_timestamp()
    return df


# ---------------------------------------------------------------------------
# FRED macro data
# ---------------------------------------------------------------------------
def clean_fred_series(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """FRED CSV -> monthly series. First column is the date, second the value."""
    out = df.copy()
    out.columns = ["date", value_name]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[value_name] = pd.to_numeric(out[value_name], errors="coerce")  # FRED uses '.' for missing
    out = out.dropna()
    monthly = out.set_index("date")[value_name].resample("MS").last()
    return monthly.reset_index().rename(columns={"date": "month"})


def build_macro_table(fedfunds: pd.DataFrame, nasdaq: pd.DataFrame) -> pd.DataFrame:
    macro = fedfunds.merge(nasdaq, on="month", how="inner").sort_values("month")
    macro["nasdaq_12m_return"] = macro["nasdaq"].pct_change(12)
    macro = macro.rename(columns={"fedfunds": "fed_funds_rate"})
    return macro[["month", "fed_funds_rate", "nasdaq", "nasdaq_12m_return"]].dropna()


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------
def merge_startups_macro(startups: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Attach the macro environment in the month of each startup's first round."""
    df = startups.merge(macro, left_on="funding_month", right_on="month", how="left")
    df = df.dropna(subset=["fed_funds_rate", "nasdaq_12m_return"])
    keep = ["company_permalink", "name", "status", "first_funded_at"] + FEATURE_COLUMNS + [TARGET]
    return df[keep].reset_index(drop=True)
