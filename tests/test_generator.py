import pandas as pd

from copilot.data.generate import PRICE_PER_SEAT, GeneratorConfig, generate


def test_deterministic():
    a = generate(GeneratorConfig(seed=3, initial_accounts=40, new_per_month_mean=4))
    b = generate(GeneratorConfig(seed=3, initial_accounts=40, new_per_month_mean=4))
    for k in a:
        pd.testing.assert_frame_equal(a[k], b[k])


def test_invariants():
    t = generate(GeneratorConfig(seed=5, initial_accounts=80, new_per_month_mean=6))
    s = t["subscriptions_monthly"]
    assert (s["mrr"] == s["seats"] * s["plan"].map(PRICE_PER_SEAT)).all()
    assert s["month"].nunique() == 24
    assert t["accounts"]["account_name"].str.startswith("SYNTH-").all()
    churn = t["plan_changes"].query("change_type == 'churn'")
    # no subscription snapshot at or after an account's churn month
    m = s.merge(churn[["account_id", "month"]].rename(columns={"month": "churn_month"}), on="account_id")
    assert (m["month"] < m["churn_month"]).all()
    assert not s.duplicated(["account_id", "month"]).any()
