"""Startup Success Pipeline (Airflow).

Question: at a startup's FIRST funding round, can we predict whether it will
eventually be acquired or IPO? Does the macro environment at funding time matter?

Sources (two distinct datasets):
  1. Crunchbase open snapshot (companies + funding rounds)
  2. FRED macro data (Fed Funds rate + NASDAQ Composite)

Graph:
  [crunchbase TaskGroup] ─┐
                          ├─> merge -> load_to_postgres -> [eda_chart, baseline_model] -> publish_for_agents
  [fred TaskGroup]      ──┘
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from airflow.decorators import dag, task, task_group

REPO = Path("/repo")
STAGING = REPO / "data" / "staging"
ARTIFACTS = REPO / "artifacts" / "pipeline"
# top-level datasets/ so the agent UI dataset dropdown lists it
PUBLISH = REPO / "datasets" / "startup_success.csv"
TABLE = "startup_success"
WAREHOUSE_URL = os.environ.get(
    "WAREHOUSE_URL", "postgresql+psycopg2://startups:startups@warehouse-db:5432/startups"
)

CRUNCHBASE = "https://raw.githubusercontent.com/notpeter/crunchbase-data/master/{name}.csv"
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
HEADERS = {"User-Agent": "Mozilla/5.0 (startup-success-pipeline)"}

DEFAULT_ARGS = {"retries": 2, "retry_delay": timedelta(seconds=20)}


def _download(url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return str(dest)


def _engine():
    from sqlalchemy import create_engine

    return create_engine(WAREHOUSE_URL)


@dag(
    dag_id="startup_success_pipeline",
    description="Crunchbase + FRED -> clean -> merge -> Postgres -> analysis",
    start_date=datetime(2026, 1, 1),
    schedule="@monthly",  # refresh monthly (macro data updates monthly)
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["startups", "crunchbase", "fred", "assignment"],
)
def startup_success_pipeline():
    # ------------------------------------------------------------------
    # Dataset 1: Crunchbase  (extract both files in parallel, then clean)
    # ------------------------------------------------------------------
    @task_group(group_id="crunchbase")
    def crunchbase():
        @task
        def extract_companies() -> str:
            return _download(CRUNCHBASE.format(name="companies"), STAGING / "raw_companies.csv")

        @task
        def extract_rounds() -> str:
            return _download(CRUNCHBASE.format(name="rounds"), STAGING / "raw_rounds.csv")

        @task
        def clean_companies(path: str) -> str:
            from pipeline.transforms import clean_companies as fn

            out = STAGING / "companies_clean.csv"
            fn(pd.read_csv(path)).to_csv(out, index=False)
            return str(out)

        @task
        def clean_rounds(path: str) -> str:
            from pipeline.transforms import clean_rounds as fn

            out = STAGING / "first_rounds.csv"
            fn(pd.read_csv(path)).to_csv(out, index=False)
            return str(out)

        @task
        def build_startups(companies_path: str, rounds_path: str) -> str:
            from pipeline.transforms import build_startup_table

            companies = pd.read_csv(companies_path, parse_dates=["founded_at"])
            rounds = pd.read_csv(rounds_path, parse_dates=["first_funded_at"])
            out = STAGING / "startups.csv"
            build_startup_table(companies, rounds).to_csv(out, index=False)
            return str(out)

        return build_startups(
            clean_companies(extract_companies()), clean_rounds(extract_rounds())
        )

    # ------------------------------------------------------------------
    # Dataset 2: FRED macro  (both series in parallel, then combine)
    # ------------------------------------------------------------------
    @task_group(group_id="fred")
    def fred():
        @task
        def extract_fedfunds() -> str:
            return _download(FRED.format(series="FEDFUNDS"), STAGING / "raw_fedfunds.csv")

        @task
        def extract_nasdaq() -> str:
            return _download(FRED.format(series="NASDAQCOM"), STAGING / "raw_nasdaq.csv")

        @task
        def build_macro(fedfunds_path: str, nasdaq_path: str) -> str:
            from pipeline.transforms import build_macro_table, clean_fred_series

            macro = build_macro_table(
                clean_fred_series(pd.read_csv(fedfunds_path), "fedfunds"),
                clean_fred_series(pd.read_csv(nasdaq_path), "nasdaq"),
            )
            out = STAGING / "macro_monthly.csv"
            macro.to_csv(out, index=False)
            return str(out)

        return build_macro(extract_fedfunds(), extract_nasdaq())

    # ------------------------------------------------------------------
    # Merge + load
    # ------------------------------------------------------------------
    @task
    def merge(startups_path: str, macro_path: str) -> str:
        from pipeline.transforms import merge_startups_macro

        startups = pd.read_csv(startups_path, parse_dates=["first_funded_at", "funding_month"])
        macro = pd.read_csv(macro_path, parse_dates=["month"])
        merged = merge_startups_macro(startups, macro)
        out = STAGING / "merged.csv"
        merged.to_csv(out, index=False)
        print(f"merged rows={len(merged)} success_rate={merged['is_success'].mean():.3f}")
        return str(out)

    @task
    def load_to_postgres(merged_path: str) -> int:
        df = pd.read_csv(merged_path, parse_dates=["first_funded_at"])
        df.to_sql(TABLE, _engine(), if_exists="replace", index=False, chunksize=5000)
        return len(df)

    # ------------------------------------------------------------------
    # Analysis (both read from the database, run in parallel)
    # ------------------------------------------------------------------
    @task
    def eda_chart(_rows: int) -> str:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        df = pd.read_sql(f"SELECT * FROM {TABLE}", _engine())
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        df["rate_bucket"] = pd.cut(
            df["fed_funds_rate"], [-0.01, 0.5, 2, 4, 10], labels=["<0.5%", "0.5-2%", "2-4%", ">4%"]
        )
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
        df.groupby("funding_year")["is_success"].mean().plot(ax=axes[0], marker="o")
        axes[0].set_title("Exit rate by first-funding year")
        df.groupby("rate_bucket", observed=True)["is_success"].mean().plot.bar(ax=axes[1])
        axes[1].set_title("Exit rate by Fed Funds rate at funding")
        df.groupby("first_round_type")["is_success"].mean().sort_values().tail(8).plot.barh(ax=axes[2])
        axes[2].set_title("Exit rate by first round type")
        for ax in axes:
            ax.set_ylabel("share acquired or IPO")
        fig.tight_layout()
        out = ARTIFACTS / "eda_exit_rates.png"
        fig.savefig(out, dpi=120)
        return str(out)

    @task
    def baseline_model(_rows: int) -> dict:
        from sklearn.compose import ColumnTransformer
        from sklearn.dummy import DummyClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        from pipeline.transforms import FEATURE_COLUMNS, TARGET

        df = pd.read_sql(f"SELECT * FROM {TABLE}", _engine())
        X, y = df[FEATURE_COLUMNS], df[TARGET]
        cat = ["first_round_type", "primary_category"]
        num = [c for c in FEATURE_COLUMNS if c not in cat]
        model = Pipeline(
            [
                ("prep", ColumnTransformer([
                    ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
                    ("num", StandardScaler(), num),
                ])),
                ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        )
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
        model.fit(X_tr, y_tr)
        majority = DummyClassifier(strategy="most_frequent").fit(X_tr, y_tr)
        metrics = {
            "rows": int(len(df)),
            "success_rate": round(float(y.mean()), 4),
            "logreg_accuracy": round(accuracy_score(y_te, model.predict(X_te)), 4),
            "logreg_roc_auc": round(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]), 4),
            "majority_class_accuracy": round(accuracy_score(y_te, majority.predict(X_te)), 4),
        }
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
        print(metrics)
        return metrics

    @task
    def publish_for_agents(_chart: str, _metrics: dict) -> str:
        """Export the cleaned table from Postgres for the LangGraph agent loop."""
        df = pd.read_sql(f"SELECT * FROM {TABLE}", _engine())
        model_cols = [c for c in df.columns if c not in {"company_permalink", "name", "status", "first_funded_at"}]
        PUBLISH.parent.mkdir(parents=True, exist_ok=True)
        df[model_cols].to_csv(PUBLISH, index=False)
        return str(PUBLISH)

    rows = load_to_postgres(merge(crunchbase(), fred()))
    publish_for_agents(eda_chart(rows), baseline_model(rows))


startup_success_pipeline()
