"""Run every model, log each as an MLflow run, and write artifacts/results.json.

    python -m copilot.models.run_all --db data/saas.duckdb
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import duckdb  # noqa: E402
import mlflow  # noqa: E402

from . import experiment, propensity, survival  # noqa: E402
from .features import load, panel_frame, survival_frame  # noqa: E402

ART = Path("artifacts")


def _flat(prefix: str, d: dict) -> dict[str, float]:
    out = {}
    for k, v in d.items():
        if isinstance(v, bool) or k.startswith("_"):
            continue
        if isinstance(v, (int, float)):
            out[f"{prefix}{k}"] = float(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/saas.duckdb")
    ap.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("saas-revenue-copilot")
    con = duckdb.connect(args.db, read_only=True)
    t = load(con)
    results: dict = {}

    with mlflow.start_run(run_name="survival_cox_km"):
        sdf = survival_frame(t)
        r = survival.run(sdf, ART / "survival")
        mlflow.log_params({"model": "CoxPHFitter", "penalizer": 0.01, "covariates": ",".join(survival.COX_COVARIATES), "holdout": 0.25})
        mlflow.log_metrics(_flat("", {k: v for k, v in r.items() if k != "km"}))
        for seg, s in r["km"].items():
            mlflow.log_metric(f"km_survival_12m_{seg.replace('-', '_')}", s["survival_12m"])
        for a in r["artifacts"]:
            mlflow.log_artifact(a)
        results["survival"] = r

    panel = panel_frame(t)
    prop = propensity.run(panel)
    for key in ("churn", "expansion"):
        with mlflow.start_run(run_name=f"xgb_{key}_propensity"):
            mlflow.log_params({"model": "XGBClassifier", **{k: v for k, v in propensity.XGB_PARAMS.items()}, "split": str(propensity.SPLIT.date())})
            mlflow.log_metrics(_flat("", prop[key]))
            mlflow.xgboost.log_model(prop["_models"][key], name=f"xgb_{key}")
        results[f"propensity_{key}"] = prop[key]

    with mlflow.start_run(run_name="upgrade_flow_ab_test"):
        e = experiment.run(t["plan_changes"], t["subscriptions_monthly"])
        mlflow.log_params({"relative_mde": e["relative_mde"], "alpha": e["alpha"], "power": e["planned_power"], "n_sims": e["n_sims"]})
        mlflow.log_metrics(_flat("", e))
        results["experiment"] = e

    ART.mkdir(exist_ok=True)
    (ART / "results.json").write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
