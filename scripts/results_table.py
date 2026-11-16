"""Print the README results table from the JSON files the runs wrote to artifacts/.
Every number in the README's results section comes from this script."""
import json
from pathlib import Path

A = Path("artifacts")


def load(name):
    p = A / name
    return json.loads(p.read_text()) if p.exists() else None


r = load("results.json")
s, pc, pe, ex = r["survival"], r["propensity_churn"], r["propensity_expansion"], r["experiment"]
rows = [
    ("Survival (Cox PH)", "Concordance index, 25% holdout", f"{s['cox_concordance_test']:.3f}", f"{s['n_accounts']:,} accounts, {s['n_events']} churn events"),
    ("Survival (Kaplan-Meier)", "12-month survival SMB / Mid-Market / Enterprise", " / ".join(f"{s['km'][k]['survival_12m']:.1%}" for k in ["SMB", "Mid-Market", "Enterprise"]), f"log-rank p = {s['logrank_p']:.1e}"),
    ("Churn propensity (XGBoost)", "AUC, time-based holdout (2025-07..11)", f"{pc['xgb_auc']:.3f}", f"logistic baseline {pc['logreg_auc']:.3f}; top-decile lift {pc['top_decile_lift']:.1f}x"),
    ("Expansion propensity (XGBoost)", "AUC, same holdout", f"{pe['xgb_auc']:.3f}", f"logistic baseline {pe['logreg_auc']:.3f}; top-decile lift {pe['top_decile_lift']:.1f}x"),
    ("Power analysis", "Accounts per arm for +20% relative MDE (alpha 0.05, power 0.8)", f"{ex['n_per_arm']:,}", f"baseline upgrade rate {ex['baseline_upgrade_rate']:.2%}/month"),
    ("A/B simulation (upgrade flow)", "Empirical power / false-positive rate", f"{ex['empirical_power']:.1%} / {ex['empirical_false_positive_rate']:.1%}", f"{ex['n_sims']:,} simulated tests each; one run: lift {ex['observed_relative_lift']:+.1%}, p = {ex['p_value']:.3f}"),
]
for f, label in [("eval_offline_golden.json", "golden 45"), ("eval_offline_holdout.json", "held-out 20")]:
    e = load(f)
    if e:
        m = e["summary"]
        rows.append((f"Agent, offline planner ({label})", "Accuracy vs reference", f"{m['accuracy']:.1%}", f"metric {m['metric_accuracy']:.1%}, refusals {m['refusal_accuracy']:.0%}"))
for f, label in [("eval_ollama_golden_v1_nogrounding.json", "llama3.2, one generic tool, golden"),
                 ("eval_ollama_golden_v2_nogrounding.json", "llama3.2, typed tools, golden"),
                 ("eval_ollama_golden_v2.json", "llama3.2, typed tools + grounding, golden"),
                 ("eval_ollama_holdout_v2.json", "llama3.2, typed tools + grounding, held-out")]:
    e = load(f)
    if e:
        m = e["summary"]
        rows.append((f"Agent, {label}", "Accuracy vs reference", f"{m['accuracy']:.1%}",
                     f"metric {m['metric_accuracy']:.1%}; flagged {m['flagged_rate']:.0%}; unflagged accuracy {m['accuracy_unflagged']:.0%}; wrong and unflagged: {m['wrong_and_unflagged']}"))
        if "judge_agreement_with_exact" in m and "grounding" in label:
            rows.append((f"LLM-as-a-Judge ({label.split(',')[-1].strip()}, {label.split(',')[1].strip()})", "Agreement with exact-match grader",
                         f"{m['judge_agreement_with_exact']:.1%}", f"{m['judge_items']} items incl. corrupted answers; caught {m['judge_caught_corrupted']:.0%} of corrupted"))

print("| Component | Measure | Result | Notes |\n|---|---|---|---|")
for row in rows:
    print("| " + " | ".join(row) + " |")
