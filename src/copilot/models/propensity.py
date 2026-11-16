"""XGBoost propensity models: P(churn next month) and P(expansion next month) from the
account-month panel. Time-based split (train on months up to 2025-06, test on 2025-07..2025-11),
compared against a logistic-regression baseline on the same features."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import PANEL_FEATURES

SPLIT = pd.Timestamp("2025-07-01")
XGB_PARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=5, eval_metric="auc", random_state=42, n_jobs=4,
)


def _top_decile_lift(y: np.ndarray, p: np.ndarray) -> float:
    cut = np.quantile(p, 0.9)
    top = y[p >= cut]
    return float(top.mean() / y.mean()) if y.mean() > 0 else float("nan")


def fit_one(panel: pd.DataFrame, label: str) -> tuple[dict, xgb.XGBClassifier]:
    train, test = panel[panel["month"] < SPLIT], panel[panel["month"] >= SPLIT]
    Xtr, ytr = train[PANEL_FEATURES], train[label].to_numpy()
    Xte, yte = test[PANEL_FEATURES], test[label].to_numpy()

    pos_w = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    model = xgb.XGBClassifier(**XGB_PARAMS, scale_pos_weight=pos_w)
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]

    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    base.fit(Xtr, ytr)
    pb = base.predict_proba(Xte)[:, 1]

    imp = pd.Series(model.feature_importances_, index=PANEL_FEATURES).sort_values(ascending=False)
    return {
        "label": label,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "positive_rate_test": float(yte.mean()),
        "xgb_auc": float(roc_auc_score(yte, p)),
        "xgb_pr_auc": float(average_precision_score(yte, p)),
        "logreg_auc": float(roc_auc_score(yte, pb)),
        "top_decile_lift": _top_decile_lift(yte, p),
        "top_features": {k: round(float(v), 4) for k, v in imp.head(5).items()},
    }, model


def run(panel: pd.DataFrame) -> dict:
    churn, churn_model = fit_one(panel, "churn_next")
    expand, expand_model = fit_one(panel, "expand_next")
    return {"churn": churn, "expansion": expand, "_models": {"churn": churn_model, "expansion": expand_model}}
