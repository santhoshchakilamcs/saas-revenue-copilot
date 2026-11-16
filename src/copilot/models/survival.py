"""Survival analysis of account churn: Kaplan-Meier curves by segment, a log-rank test, and a Cox
proportional-hazards model on early-life covariates, scored by concordance on a holdout set."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from lifelines import CoxPHFitter, KaplanMeierFitter  # noqa: E402
from lifelines.statistics import multivariate_logrank_test  # noqa: E402
from lifelines.utils import concordance_index  # noqa: E402

COX_COVARIATES = [
    "early_utilization", "early_tickets", "log_calls_per_user", "log_seats", "plan_idx",
    "seg_mid_market", "seg_enterprise",
]


def run(df: pd.DataFrame, out_dir: Path, seed: int = 42) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    test_mask = rng.random(len(df)) < 0.25
    train, test = df[~test_mask], df[test_mask]

    # Kaplan-Meier by segment
    fig, ax = plt.subplots(figsize=(7, 4.5))
    km_stats = {}
    for seg in ["SMB", "Mid-Market", "Enterprise"]:
        d = df[df["segment"] == seg]
        km = KaplanMeierFitter(label=seg).fit(d["duration"], d["event"])
        km.plot_survival_function(ax=ax, ci_show=True)
        km_stats[seg] = {
            "n": int(len(d)),
            "events": int(d["event"].sum()),
            "survival_12m": float(km.survival_function_at_times(12).iloc[0]),
            "median_months": float(km.median_survival_time_) if np.isfinite(km.median_survival_time_) else None,
        }
    ax.set_xlabel("Months since signup")
    ax.set_ylabel("Share of accounts still subscribed")
    ax.set_title("Kaplan-Meier survival by segment (SYNTHETIC data)")
    fig.tight_layout()
    km_path = out_dir / "km_by_segment.png"
    fig.savefig(km_path, dpi=130)
    plt.close(fig)

    lr = multivariate_logrank_test(df["duration"], df["segment"], df["event"])

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(train[["duration", "event", *COX_COVARIATES]], duration_col="duration", event_col="event")
    # higher partial hazard -> shorter survival, so negate for concordance
    test_c = concordance_index(test["duration"], -cph.predict_partial_hazard(test[COX_COVARIATES]), test["event"])
    summary = cph.summary[["coef", "exp(coef)", "p"]].round(4)
    summary.to_csv(out_dir / "cox_summary.csv")

    return {
        "n_accounts": int(len(df)),
        "n_events": int(df["event"].sum()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "cox_concordance_train": float(cph.concordance_index_),
        "cox_concordance_test": float(test_c),
        "logrank_p": float(lr.p_value),
        "km": km_stats,
        "hazard_ratios": {k: float(v) for k, v in summary["exp(coef)"].items()},
        "artifacts": [str(km_path), str(out_dir / "cox_summary.csv")],
    }
