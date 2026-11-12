import numpy as np
import pandas as pd

from pipeline.transforms import (
    TARGET, build_macro_table, build_startup_table, clean_companies,
    clean_fred_series, clean_rounds, merge_startups_macro,
)


def _companies():
    return pd.DataFrame({
        "permalink": ["/a", "/b", "/b", "/c"],
        "name": ["A", "B", "B", "C"],
        "category_list": ["Mobile|Apps", "Biotech", "Biotech", None],
        "status": ["acquired", "closed", "closed", "ipo"],
        "country_code": ["USA", "GBR", "GBR", "USA"],
        "founded_at": ["2004-01-01", "0011-01-01", "0011-01-01", "2005-06-01"],
        "first_funding_at": ["2005-01-01"] * 4,
    })


def _rounds():
    return pd.DataFrame({
        "company_permalink": ["/a", "/a", "/b", "/c"],
        "funded_at": ["2007-01-01", "2005-03-01", "2006-02-01", "2009-05-01"],
        "funding_round_type": ["venture", "seed", "angel", "venture"],
        "raised_amount_usd": [5e6, 1e5, None, 2e6],
    })


def test_clean_companies_label_and_dedup():
    c = clean_companies(_companies())
    assert len(c) == 3
    assert c.set_index("company_permalink")[TARGET].to_dict() == {"/a": 1, "/b": 0, "/c": 1}
    assert c.loc[c.company_permalink == "/b", "founded_at"].isna().all()  # year 0011 -> missing


def test_clean_rounds_keeps_first_round_only():
    r = clean_rounds(_rounds())
    a = r[r.company_permalink == "/a"].iloc[0]
    assert a.first_round_type == "seed" and a.first_round_usd == 1e5


def test_full_merge_has_no_missing_features():
    s = build_startup_table(clean_companies(_companies()), clean_rounds(_rounds()))
    d = pd.date_range("2003-01-01", "2010-12-31", freq="D")
    ff = pd.DataFrame({"DATE": d, "FEDFUNDS": 2.0})
    nq = pd.DataFrame({"DATE": d, "NASDAQCOM": np.linspace(1000, 2000, len(d)).astype(str)})
    nq.iloc[3, 1] = "."  # FRED missing marker
    macro = build_macro_table(clean_fred_series(ff, "fedfunds"), clean_fred_series(nq, "nasdaq"))
    out = merge_startups_macro(s, macro)
    assert len(out) == 3
    assert out.drop(columns=["first_funded_at"]).isna().sum().sum() == 0
