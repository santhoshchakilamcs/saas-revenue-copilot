"""Power analysis and a simulated A/B test of an upgrade-flow change.

The baseline is measured from the synthetic data: the monthly rate at which Starter and Team
accounts move up a plan tier. The test is sized for a chosen minimum detectable effect (MDE),
then simulated once with the true lift equal to the MDE and validated by Monte Carlo: across many
simulated experiments the share of significant results should be close to the planned power when
the lift is real, and close to alpha when there is no lift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize, proportions_ztest


def baseline_upgrade_rate(plan_changes: pd.DataFrame, subs: pd.DataFrame) -> float:
    eligible = subs[subs["plan"].isin(["Starter", "Team"])]
    order = {"Starter": 0, "Team": 1, "Business": 2, "Enterprise": 3}
    pc = plan_changes.dropna(subset=["old_plan", "new_plan"])
    ups = pc[(pc["old_plan"].isin(["Starter", "Team"])) & (pc["new_plan"].map(order) > pc["old_plan"].map(order))]
    return float(len(ups) / len(eligible))


def sample_size(p0: float, rel_mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    es = proportion_effectsize(p0 * (1 + rel_mde), p0)
    n = NormalIndPower().solve_power(effect_size=es, alpha=alpha, power=power, ratio=1.0, alternative="two-sided")
    return int(np.ceil(n))


def _test(conv_c: int, conv_t: int, n: int) -> tuple[float, float, tuple[float, float]]:
    _, p = proportions_ztest([conv_t, conv_c], [n, n])
    pc, pt = conv_c / n, conv_t / n
    se = np.sqrt(pc * (1 - pc) / n + pt * (1 - pt) / n)
    diff = pt - pc
    return float(p), float(diff), (float(diff - 1.96 * se), float(diff + 1.96 * se))


def run(plan_changes: pd.DataFrame, subs: pd.DataFrame, rel_mde: float = 0.20, alpha: float = 0.05,
        power: float = 0.8, n_sims: int = 2000, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    p0 = baseline_upgrade_rate(plan_changes, subs)
    n = sample_size(p0, rel_mde, alpha, power)
    p1 = p0 * (1 + rel_mde)

    conv_c, conv_t = int(rng.binomial(n, p0)), int(rng.binomial(n, p1))
    p_val, diff, ci = _test(conv_c, conv_t, n)

    def sim(pt: float) -> float:
        c = rng.binomial(n, p0, size=n_sims)
        t = rng.binomial(n, pt, size=n_sims)
        hits = sum(_test(int(a), int(b), n)[0] < alpha for a, b in zip(c, t))
        return hits / n_sims

    return {
        "baseline_upgrade_rate": p0,
        "relative_mde": rel_mde,
        "alpha": alpha,
        "planned_power": power,
        "n_per_arm": n,
        "observed_control_rate": conv_c / n,
        "observed_treatment_rate": conv_t / n,
        "observed_relative_lift": (conv_t - conv_c) / conv_c,
        "p_value": p_val,
        "abs_diff_ci95": ci,
        "empirical_power": sim(p1),
        "empirical_false_positive_rate": sim(p0),
        "n_sims": n_sims,
    }
