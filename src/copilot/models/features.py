"""Feature tables for the survival and propensity models, built from the raw synthetic tables."""
from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

PLAN_IDX = {"Starter": 0, "Team": 1, "Business": 2, "Enterprise": 3}
LAST_MONTH = pd.Timestamp("2025-12-01")


def load(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    out = {t: con.execute(f"SELECT * FROM {t}").df() for t in ("accounts", "subscriptions_monthly", "plan_changes", "usage_monthly")}
    for t in ("subscriptions_monthly", "plan_changes", "usage_monthly"):
        out[t]["month"] = pd.to_datetime(out[t]["month"])
    out["accounts"]["signup_month"] = pd.to_datetime(out["accounts"]["signup_month"])
    return out


def survival_frame(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per account: duration in months, churn event flag, and covariates observed in the
    account's first two months only (so the model predicts from early-life signals, no leakage)."""
    acc = t["accounts"].copy()
    churn = t["plan_changes"].query("change_type == 'churn'")[["account_id", "month"]].rename(columns={"month": "churn_month"})
    acc = acc.merge(churn, on="account_id", how="left")
    acc["event"] = acc["churn_month"].notna().astype(int)
    end = acc["churn_month"].fillna(LAST_MONTH + pd.DateOffset(months=1))
    acc["duration"] = ((end.dt.year - acc["signup_month"].dt.year) * 12 + (end.dt.month - acc["signup_month"].dt.month)).astype(float)

    u = t["usage_monthly"].merge(acc[["account_id", "signup_month"]], on="account_id")
    u["age"] = (u["month"].dt.year - u["signup_month"].dt.year) * 12 + (u["month"].dt.month - u["signup_month"].dt.month)
    early = u[u["age"] <= 1].groupby("account_id").agg(
        early_utilization=("seat_utilization", "mean"),
        early_tickets=("support_tickets", "mean"),
        early_calls_per_user=("api_calls", "sum"),
        early_users=("active_users", "sum"),
    )
    early["early_calls_per_user"] = early["early_calls_per_user"] / early["early_users"].clip(lower=1)
    early = early.drop(columns="early_users")

    first = t["plan_changes"].query("change_type == 'new'")[["account_id", "new_plan", "new_seats"]]
    df = acc.merge(early, on="account_id").merge(first, on="account_id")
    df["log_seats"] = np.log(df["new_seats"])
    df["plan_idx"] = df["new_plan"].map(PLAN_IDX)
    df["seg_mid_market"] = (df["segment"] == "Mid-Market").astype(int)
    df["seg_enterprise"] = (df["segment"] == "Enterprise").astype(int)
    df["log_calls_per_user"] = np.log1p(df["early_calls_per_user"])
    df = df[df["duration"] > 0]
    return df[
        ["account_id", "segment", "duration", "event", "early_utilization", "early_tickets",
         "log_calls_per_user", "log_seats", "plan_idx", "seg_mid_market", "seg_enterprise"]
    ].reset_index(drop=True)


PANEL_FEATURES = [
    "seat_utilization", "support_tickets", "log_calls_per_user", "calls_mom_change", "util_mom_change",
    "tickets_3m", "tenure", "log_seats", "plan_idx", "seg_mid_market", "seg_enterprise",
]


def panel_frame(t: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Account-month panel. Features are as of month m; labels are what happens at m+1:
    ``churn_next`` (account churns) and ``expand_next`` (account's MRR goes up)."""
    u = t["usage_monthly"].sort_values(["account_id", "month"]).copy()
    s = t["subscriptions_monthly"][["account_id", "month", "plan", "seats", "mrr"]]
    acc = t["accounts"][["account_id", "segment", "signup_month"]]
    df = u.merge(s, on=["account_id", "month"], how="inner").merge(acc, on="account_id")
    g = df.groupby("account_id")
    df["calls_per_user"] = df["api_calls"] / df["active_users"].clip(lower=1)
    df["log_calls_per_user"] = np.log1p(df["calls_per_user"])
    df["calls_mom_change"] = g["api_calls"].pct_change().fillna(0).clip(-1, 5)
    df["util_mom_change"] = g["seat_utilization"].diff().fillna(0)
    df["tickets_3m"] = g["support_tickets"].transform(lambda x: x.rolling(3, min_periods=1).sum())
    df["tenure"] = (df["month"].dt.year - df["signup_month"].dt.year) * 12 + (df["month"].dt.month - df["signup_month"].dt.month)
    df["log_seats"] = np.log(df["seats"])
    df["plan_idx"] = df["plan"].map(PLAN_IDX)
    df["seg_mid_market"] = (df["segment"] == "Mid-Market").astype(int)
    df["seg_enterprise"] = (df["segment"] == "Enterprise").astype(int)

    pc = t["plan_changes"]
    nxt = df[["account_id", "month"]].assign(month=df["month"] + pd.DateOffset(months=1))
    churn_keys = set(map(tuple, pc.query("change_type == 'churn'")[["account_id", "month"]].itertuples(index=False)))
    exp_keys = set(map(tuple, pc.query("change_type == 'expansion'")[["account_id", "month"]].itertuples(index=False)))
    keys = list(map(tuple, nxt.itertuples(index=False)))
    df["churn_next"] = [int(k in churn_keys) for k in keys]
    df["expand_next"] = [int(k in exp_keys) for k in keys]
    return df[df["month"] < LAST_MONTH].reset_index(drop=True)
