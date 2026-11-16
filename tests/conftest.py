import duckdb
import pandas as pd
import pytest

from copilot.data.generate import GeneratorConfig, generate, write_duckdb


@pytest.fixture(scope="session")
def gen_db(tmp_path_factory):
    """A small generated SYNTHETIC dataset (fast) written to DuckDB."""
    path = tmp_path_factory.mktemp("data") / "small.duckdb"
    write_duckdb(generate(GeneratorConfig(seed=1, initial_accounts=150, new_per_month_mean=12)), path)
    return str(path)


@pytest.fixture()
def tiny_con():
    """Hand-built 2-month dataset with metric values worked out by hand (see test_metrics.py)."""
    con = duckdb.connect(":memory:")
    accounts = pd.DataFrame({
        "account_id": [1, 2, 3, 4], "account_name": ["A", "B", "C", "D"],
        "segment": ["SMB", "SMB", "Enterprise", "SMB"], "region": ["NA"] * 4, "industry": ["Software"] * 4,
        "signup_month": pd.to_datetime(["2024-01-01"] * 3 + ["2024-02-01"]),
    })
    subs = pd.DataFrame({
        "account_id": [1, 2, 3, 1, 3, 4],
        "month": pd.to_datetime(["2024-01-01"] * 3 + ["2024-02-01"] * 3),
        "plan": ["Team", "Starter", "Enterprise", "Team", "Enterprise", "Business"],
        "seats": [1, 1, 1, 1, 1, 1],
        "mrr": [100.0, 50.0, 1000.0, 150.0, 900.0, 200.0],
    })
    usage = pd.DataFrame({
        "account_id": [1, 2, 3, 1, 3, 4], "month": subs["month"],
        "active_users": [5, 2, 40, 6, 35, 8], "api_calls": [1000, 100, 9000, 1200, 8000, 500],
        "feature_events": [10] * 6, "support_tickets": [0, 3, 1, 0, 2, 1], "seat_utilization": [0.5] * 6,
    })
    for name, df in [("accounts", accounts), ("subscriptions_monthly", subs), ("usage_monthly", usage)]:
        con.register("t", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM t")
        con.unregister("t")
    for t in ("subscriptions_monthly", "usage_monthly"):
        con.execute(f"ALTER TABLE {t} ALTER month TYPE DATE")
    return con
