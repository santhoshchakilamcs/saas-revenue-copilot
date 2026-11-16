"""Deterministic question -> tool-call planner (the no-API-key, no-LLM backend).

It maps a natural-language metric question to one call of the semantic layer's ``get_metric``
tool. It is also used as an independent second opinion on the LLM's plan: when the two disagree,
the answer is sent to human review.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

# order matters: more specific phrases first
METRIC_PATTERNS: list[tuple[str, list[str]]] = [
    ("nrr", [r"\bnrr\b", r"net revenue retention", r"net dollar retention", r"\bndr\b"]),
    ("grr", [r"\bgrr\b", r"gross revenue retention", r"gross dollar retention"]),
    ("crr", [r"\bcrr\b", r"customer retention", r"logo retention"]),
    ("logo_churn_rate", [r"logo churn rate", r"customer churn rate", r"churn rate"]),
    ("churned_mrr", [r"churned mrr", r"mrr (?:lost|churned)", r"revenue lost to churn", r"(?:lost|lose|loses) to churn"]),
    ("churned_customers", [r"customers? churned", r"accounts? churned", r"churned (?:customers|accounts)", r"cancel"]),
    ("expansion_mrr", [r"expansion", r"upsell"]),
    ("contraction_mrr", [r"contraction", r"downgrade"]),
    ("new_mrr", [r"new mrr", r"mrr from new", r"new business mrr"]),
    ("new_customers", [r"new customers", r"new logos", r"new accounts", r"signed up", r"acquired"]),
    ("consumption", [r"api calls?", r"consumption", r"usage", r"active users", r"feature events", r"support tickets"]),
    ("arr", [r"\barr\b", r"annual recurring revenue", r"annualized revenue", r"run[- ]rate"]),
    ("mrr", [r"\bmrr\b", r"monthly recurring revenue"]),
    ("active_customers", [r"active customers", r"how many customers", r"paying customers", r"customer count", r"number of customers"]),
]
RANGE_METRICS = {"nrr", "grr", "crr"}

REFUSAL_PATTERNS = [
    r"\bwhich (?:customers?|accounts?)\b", r"\blist (?:the |all )?(?:customers?|accounts?)", r"\bnames?\b",
    r"\bemails?\b", r"\bphone\b", r"\baccount[_ ]id\b", r"\bsynth-\d+", r"\braw (?:rows|data|table)",
    r"\bdrop\b|\bdelete\b|\binsert\b|\bupdate\b.*\bset\b", r"\bselect\b.*\bfrom\b",
]


@dataclass
class Plan:
    tool: str  # "get_metric" or "refuse"
    args: dict = field(default_factory=dict)
    confidence: float = 1.0
    notes: list[str] = field(default_factory=list)


def _norm_month(y: int, m: int) -> str:
    return f"{y:04d}-{m:02d}"


def extract_months(q: str) -> list[str]:
    ql = q.lower()
    found: list[tuple[int, str]] = []
    for mt in re.finditer(r"\b(20\d{2})-(0[1-9]|1[0-2])\b", ql):
        found.append((mt.start(), _norm_month(int(mt.group(1)), int(mt.group(2)))))
    for mt in re.finditer(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,? (20\d{2})\b", ql):
        found.append((mt.start(), _norm_month(int(mt.group(2)), MONTHS[mt.group(1)])))
    for mt in re.finditer(r"\bq([1-4]) (20\d{2})\b", ql):
        found.append((mt.start(), _norm_month(int(mt.group(2)), int(mt.group(1)) * 3)))
    for mt in re.finditer(r"\b(?:end of|year[- ]end) (20\d{2})\b", ql):
        found.append((mt.start(), _norm_month(int(mt.group(1)), 12)))
    found.sort()
    out: list[str] = []
    for _, m in found:
        if m not in out:
            out.append(m)
    return out


def extract_segment(q: str) -> str | None:
    ql = q.lower()
    if re.search(r"\bsmb\b|small business", ql):
        return "SMB"
    if re.search(r"mid[- ]?market", ql):
        return "Mid-Market"
    if re.search(r"\benterprise\b", ql):
        return "Enterprise"
    return None


def extract_measure(q: str) -> str:
    ql = q.lower()
    if "active users" in ql:
        return "active_users"
    if "feature events" in ql:
        return "feature_events"
    if "support tickets" in ql:
        return "support_tickets"
    return "api_calls"


def plan(question: str) -> Plan:
    ql = question.lower()
    for pat in REFUSAL_PATTERNS:
        if re.search(pat, ql):
            return Plan("refuse", {"reason": "row-level, identifying or write requests are out of scope; aggregate metrics only"})

    metric = None
    for name, pats in METRIC_PATTERNS:
        if any(re.search(p, ql) for p in pats):
            metric = name
            break
    if metric is None:
        return Plan("refuse", {"reason": "not a metric this layer defines"}, confidence=0.3, notes=["no metric matched"])

    args: dict = {"metric": metric}
    seg = extract_segment(question)
    if seg:
        args["segment"] = seg
    months = extract_months(question)
    conf = 1.0
    notes: list[str] = []

    if metric in RANGE_METRICS:
        yr = re.search(r"(?:over|during|in|for|across) (?:calendar year |cy|fy)?(20\d{2})\b", ql)
        if len(months) >= 2:
            args["start_month"], args["end_month"] = months[0], months[1]
        elif yr and not months:
            y = int(yr.group(1))
            args["start_month"], args["end_month"] = _norm_month(y - 1, 12), _norm_month(y, 12)
        elif len(months) == 1:
            y, m = map(int, months[0].split("-"))
            args["start_month"], args["end_month"] = _norm_month(y - 1, m), months[0]
            conf, notes = 0.6, ["assumed trailing 12 months"]
        else:
            return Plan("get_metric", args, 0.2, ["no period found"])
    else:
        if not months:
            return Plan("get_metric", args, 0.2, ["no month found"])
        args["month"] = months[-1]
        if metric == "consumption":
            args["measure"] = extract_measure(question)
    return Plan("get_metric", args, conf, notes)
