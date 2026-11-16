import duckdb
import numpy as np
from scipy.stats import norm

from copilot.models import experiment, propensity, survival
from copilot.models.features import load, panel_frame, survival_frame


def test_sample_size_matches_closed_form():
    p0, mde = 0.02, 0.2
    n = experiment.sample_size(p0, mde)
    h = 2 * np.arcsin(np.sqrt(p0 * 1.2)) - 2 * np.arcsin(np.sqrt(p0))
    expected = 2 * ((norm.ppf(0.975) + norm.ppf(0.8)) / h) ** 2  # two-sample, equal arms
    assert abs(n - np.ceil(expected)) <= 1


def test_survival_and_propensity_smoke(gen_db, tmp_path):
    t = load(duckdb.connect(gen_db, read_only=True))
    sdf = survival_frame(t)
    assert (sdf["duration"] > 0).all() and set(sdf["event"].unique()) <= {0, 1}
    r = survival.run(sdf, tmp_path)
    assert 0.5 < r["cox_concordance_test"] <= 1.0
    assert (tmp_path / "km_by_segment.png").exists()
    panel = panel_frame(t)
    assert panel["month"].max() < np.datetime64("2025-12-01")
    res, _ = propensity.fit_one(panel, "churn_next")
    assert 0.5 < res["xgb_auc"] <= 1.0


def test_ab_simulation_controls_error_rates(gen_db):
    t = load(duckdb.connect(gen_db, read_only=True))
    r = experiment.run(t["plan_changes"], t["subscriptions_monthly"], n_sims=400)
    assert abs(r["empirical_power"] - 0.8) < 0.07
    assert r["empirical_false_positive_rate"] < 0.09
