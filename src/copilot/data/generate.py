"""SYNTHETIC SaaS dataset generator.

Everything this module writes is synthetic. No real company, customer or person is in it.

It simulates 24 months (2024-01 .. 2025-12) of a B2B SaaS business with seat-based
plans and metered API consumption:

* ``accounts``               one row per customer account
* ``subscriptions_monthly``  month-end subscription snapshot per active account (plan, seats, MRR)
* ``plan_changes``           one row per new / expansion / contraction / churn / reactivation event
* ``usage_monthly``          product usage events aggregated to account-month (active users, API calls,
                             feature events, support tickets)

The generating process is documented inline so the models downstream have a known ground truth:
churn hazard rises when an account's latent engagement falls, when seat utilization is low and when
support tickets pile up; expansion is driven by high seat utilization and high engagement.

Run:  python -m copilot.data.generate --seed 42 --out data/saas.duckdb
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

START = pd.Timestamp("2024-01-01")
N_MONTHS = 24

SEGMENTS = {
    # segment: (share of new accounts, seat range, base monthly churn hazard)
    "SMB": (0.60, (3, 25), 0.022),
    "Mid-Market": (0.30, (20, 150), 0.010),
    "Enterprise": (0.10, (150, 1200), 0.004),
}
PLANS = ["Starter", "Team", "Business", "Enterprise"]
PRICE_PER_SEAT = {"Starter": 15.0, "Team": 35.0, "Business": 70.0, "Enterprise": 110.0}
SEGMENT_PLAN_PROBS = {
    "SMB": [0.60, 0.35, 0.05, 0.00],
    "Mid-Market": [0.15, 0.50, 0.35, 0.00],
    "Enterprise": [0.00, 0.10, 0.45, 0.45],
}
REGIONS = ["NA", "EMEA", "APAC", "LATAM"]
REGION_P = [0.50, 0.28, 0.15, 0.07]
INDUSTRIES = ["Software", "Fintech", "Healthcare", "Retail", "Manufacturing", "Media"]


@dataclass
class GeneratorConfig:
    seed: int = 42
    initial_accounts: int = 500
    new_per_month_mean: float = 45.0
    months: int = N_MONTHS


def _month(i: int) -> pd.Timestamp:
    return START + pd.DateOffset(months=i)


def generate(cfg: GeneratorConfig | None = None) -> dict[str, pd.DataFrame]:
    """Return the four synthetic tables as DataFrames. Deterministic for a given seed."""
    cfg = cfg or GeneratorConfig()
    rng = np.random.default_rng(cfg.seed)

    accounts: list[dict] = []
    subs: list[dict] = []
    changes: list[dict] = []
    usage: list[dict] = []

    # live state per account id
    state: dict[int, dict] = {}
    next_id = 1

    def new_account(month_idx: int) -> None:
        nonlocal next_id
        seg = rng.choice(list(SEGMENTS), p=[v[0] for v in SEGMENTS.values()])
        lo, hi = SEGMENTS[seg][1]
        seats = int(rng.integers(lo, hi + 1))
        plan = rng.choice(PLANS, p=SEGMENT_PLAN_PROBS[seg])
        acc_id = next_id
        next_id += 1
        accounts.append(
            {
                "account_id": acc_id,
                "account_name": f"SYNTH-{acc_id:05d}",
                "segment": seg,
                "region": rng.choice(REGIONS, p=REGION_P),
                "industry": rng.choice(INDUSTRIES),
                "signup_month": _month(month_idx),
            }
        )
        state[acc_id] = {
            "segment": seg,
            "plan": plan,
            "seats": seats,
            # latent engagement: random walk, the hidden driver of churn and expansion
            "engagement": float(rng.normal(0.0, 1.0)),
            "active": True,
            "tenure": 0,
        }
        mrr = seats * PRICE_PER_SEAT[plan]
        changes.append(
            {
                "account_id": acc_id,
                "month": _month(month_idx),
                "change_type": "new",
                "old_plan": None,
                "new_plan": plan,
                "old_seats": 0,
                "new_seats": seats,
                "old_mrr": 0.0,
                "new_mrr": mrr,
            }
        )

    for _ in range(cfg.initial_accounts):
        new_account(0)

    for m in range(cfg.months):
        month = _month(m)
        if m > 0:
            for _ in range(int(rng.poisson(cfg.new_per_month_mean))):
                new_account(m)

        for acc_id, s in state.items():
            if not s["active"]:
                continue
            # usage this month, driven by engagement and seats
            e = s["engagement"]
            util = float(np.clip(0.68 + 0.15 * e + rng.normal(0, 0.08), 0.05, 1.0))
            active_users = max(1, int(round(s["seats"] * util)))
            api_calls = int(rng.poisson(max(1.0, active_users * 180 * np.exp(0.35 * e))))
            feature_events = int(rng.poisson(max(1.0, active_users * 40 * np.exp(0.25 * e))))
            tickets = int(rng.poisson(max(0.05, 0.4 + 0.6 * max(0.0, -e) + 0.002 * s["seats"] ** 0.5)))
            usage.append(
                {
                    "account_id": acc_id,
                    "month": month,
                    "active_users": active_users,
                    "api_calls": api_calls,
                    "feature_events": feature_events,
                    "support_tickets": tickets,
                    "seat_utilization": round(active_users / s["seats"], 4),
                }
            )

            # month-end decisions (accounts that signed up this month are not at risk yet)
            if s["tenure"] > 0:
                base = SEGMENTS[s["segment"]][2]
                logit = np.log(base / (1 - base)) - 0.9 * e + 1.2 * (util < 0.45) + 0.25 * tickets
                p_churn = 1 / (1 + np.exp(-logit))
                if rng.random() < p_churn:
                    old_mrr = s["seats"] * PRICE_PER_SEAT[s["plan"]]
                    changes.append(
                        {
                            "account_id": acc_id,
                            "month": month,
                            "change_type": "churn",
                            "old_plan": s["plan"],
                            "new_plan": None,
                            "old_seats": s["seats"],
                            "new_seats": 0,
                            "old_mrr": old_mrr,
                            "new_mrr": 0.0,
                        }
                    )
                    s["active"] = False
                    continue

                old_plan, old_seats = s["plan"], s["seats"]
                old_mrr = old_seats * PRICE_PER_SEAT[old_plan]
                # expansion: seat adds when utilization is high, plan upgrades when engagement is high
                if util > 0.82 and rng.random() < 0.22:
                    s["seats"] = int(np.ceil(s["seats"] * rng.uniform(1.08, 1.3)))
                idx = PLANS.index(s["plan"])
                if idx < 2 and rng.random() < 0.015 * np.exp(0.8 * e):
                    s["plan"] = PLANS[idx + 1]
                # contraction: seat cuts when utilization is low
                if util < 0.50 and rng.random() < 0.10:
                    s["seats"] = max(1, int(np.floor(s["seats"] * rng.uniform(0.6, 0.9))))
                if idx > 0 and e < -1.2 and rng.random() < 0.05:
                    s["plan"] = PLANS[idx - 1]
                new_mrr = s["seats"] * PRICE_PER_SEAT[s["plan"]]
                if new_mrr != old_mrr:
                    changes.append(
                        {
                            "account_id": acc_id,
                            "month": month,
                            "change_type": "expansion" if new_mrr > old_mrr else "contraction",
                            "old_plan": old_plan,
                            "new_plan": s["plan"],
                            "old_seats": old_seats,
                            "new_seats": s["seats"],
                            "old_mrr": old_mrr,
                            "new_mrr": new_mrr,
                        }
                    )

            subs.append(
                {
                    "account_id": acc_id,
                    "month": month,
                    "plan": s["plan"],
                    "seats": s["seats"],
                    "mrr": float(s["seats"] * PRICE_PER_SEAT[s["plan"]]),
                }
            )
            # engagement drifts
            s["engagement"] = 0.92 * s["engagement"] + float(rng.normal(0, 0.35))
            s["tenure"] += 1

    return {
        "accounts": pd.DataFrame(accounts),
        "subscriptions_monthly": pd.DataFrame(subs),
        "plan_changes": pd.DataFrame(changes),
        "usage_monthly": pd.DataFrame(usage),
    }


def write_duckdb(tables: dict[str, pd.DataFrame], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = duckdb.connect(str(path))
    for name, df in tables.items():
        con.register("tmp_df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM tmp_df")
        con.unregister("tmp_df")
    for t in ("subscriptions_monthly", "plan_changes", "usage_monthly"):
        con.execute(f"ALTER TABLE {t} ALTER month TYPE DATE")
    con.execute("ALTER TABLE accounts ALTER signup_month TYPE DATE")
    con.close()
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/saas.duckdb")
    args = ap.parse_args()
    tables = generate(GeneratorConfig(seed=args.seed))
    out = write_duckdb(tables, args.out)
    for k, v in tables.items():
        print(f"{k:24s} {len(v):>8,d} rows")
    print(f"SYNTHETIC data written to {out}")


if __name__ == "__main__":
    main()
