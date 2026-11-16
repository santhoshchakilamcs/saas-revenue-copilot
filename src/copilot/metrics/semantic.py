"""Semantic layer: one place where every revenue metric is defined, in SQL, over DuckDB.

Definitions (all month-end snapshot based; ``month`` is 'YYYY-MM'):

* MRR(m)            sum of subscription MRR for accounts active at month-end m
* ARR(m)            12 x MRR(m)
* active_customers  count of accounts with a subscription snapshot at m
* new_mrr(m)        MRR at m of accounts absent at m-1 and present at m
* expansion_mrr(m)  sum of positive MRR deltas, m-1 -> m, for accounts present in both
* contraction_mrr   sum of |negative MRR deltas|, m-1 -> m, for accounts present in both
* churned_mrr(m)    MRR at m-1 of accounts present at m-1 and absent at m
* churned_customers count of accounts present at m-1 and absent at m
* logo_churn_rate   churned_customers(m) / active_customers(m-1)
* CRR(s, e)         customer retention rate = (E - N) / S, where S = customers at s,
                    E = customers at e, N = customers at e who were not active at s
* NRR(s, e)         MRR at e from accounts active at s / MRR at s (includes expansion, net of
                    contraction and churn; excludes new logos)
* GRR(s, e)         sum over accounts active at s of min(MRR_e, MRR_s) / MRR at s (caps expansion)
* consumption(m)    sum of a usage measure (api_calls, active_users, feature_events) in month m

Every metric takes an optional ``segment`` filter (SMB, Mid-Market, Enterprise).
The agent can only reach data through these functions; it never runs free-form SQL and never
sees row-level account data.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import duckdb

SEGMENTS = ("SMB", "Mid-Market", "Enterprise")
USAGE_MEASURES = ("api_calls", "active_users", "feature_events", "support_tickets")
_MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


class MetricError(ValueError):
    """Raised for an unknown metric, a bad month, or a bad segment."""


def _month(m: str) -> str:
    if not isinstance(m, str) or not _MONTH_RE.match(m):
        raise MetricError(f"month must look like 'YYYY-MM', got {m!r}")
    return f"{m}-01"


def _segment_clause(segment: str | None, alias: str = "a") -> tuple[str, list]:
    if segment is None:
        return "", []
    if segment not in SEGMENTS:
        raise MetricError(f"segment must be one of {SEGMENTS}, got {segment!r}")
    return f" AND {alias}.segment = ?", [segment]


@dataclass(frozen=True)
class MetricDef:
    name: str
    description: str
    params: tuple[str, ...]
    unit: str


class SemanticLayer:
    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con
        self.con.execute(
            """
            CREATE OR REPLACE TEMP VIEW snap AS
            SELECT s.account_id, s.month, s.mrr, s.seats, s.plan, a.segment
            FROM subscriptions_monthly s JOIN accounts a USING (account_id)
            """
        )
        self._registry: dict[str, tuple[MetricDef, Callable]] = {}
        for d, fn in [
            (MetricDef("mrr", "Monthly recurring revenue at month-end", ("month", "segment"), "USD"), self.mrr),
            (MetricDef("arr", "Annual recurring revenue = 12 x MRR at month-end", ("month", "segment"), "USD"), self.arr),
            (MetricDef("active_customers", "Accounts with an active subscription at month-end", ("month", "segment"), "count"), self.active_customers),
            (MetricDef("new_customers", "Accounts active at m but not at m-1", ("month", "segment"), "count"), self.new_customers),
            (MetricDef("churned_customers", "Accounts active at m-1 but not at m", ("month", "segment"), "count"), self.churned_customers),
            (MetricDef("logo_churn_rate", "churned_customers(m) / active_customers(m-1)", ("month", "segment"), "ratio"), self.logo_churn_rate),
            (MetricDef("new_mrr", "MRR from accounts new in month m", ("month", "segment"), "USD"), self.new_mrr),
            (MetricDef("expansion_mrr", "Positive MRR deltas m-1 -> m for retained accounts", ("month", "segment"), "USD"), self.expansion_mrr),
            (MetricDef("contraction_mrr", "Absolute negative MRR deltas m-1 -> m for retained accounts", ("month", "segment"), "USD"), self.contraction_mrr),
            (MetricDef("churned_mrr", "MRR at m-1 of accounts that churned in m", ("month", "segment"), "USD"), self.churned_mrr),
            (MetricDef("crr", "Customer retention rate (E - N) / S between two month-ends", ("start_month", "end_month", "segment"), "ratio"), self.crr),
            (MetricDef("nrr", "Net revenue retention between two month-ends", ("start_month", "end_month", "segment"), "ratio"), self.nrr),
            (MetricDef("grr", "Gross revenue retention between two month-ends", ("start_month", "end_month", "segment"), "ratio"), self.grr),
            (MetricDef("consumption", "Sum of a usage measure in a month (api_calls, active_users, feature_events, support_tickets)", ("month", "measure", "segment"), "count"), self.consumption),
        ]:
            self._registry[d.name] = (d, fn)

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_path(cls, path: str | Path, read_only: bool = True) -> "SemanticLayer":
        return cls(duckdb.connect(str(path), read_only=read_only))

    # ------------------------------------------------------------------ registry
    def list_metrics(self) -> list[MetricDef]:
        return [d for d, _ in self._registry.values()]

    def query(self, metric: str, **kwargs) -> float:
        if metric not in self._registry:
            raise MetricError(f"unknown metric {metric!r}; known: {sorted(self._registry)}")
        d, fn = self._registry[metric]
        unknown = set(kwargs) - set(d.params)
        if unknown:
            raise MetricError(f"{metric} does not take {sorted(unknown)}; params: {d.params}")
        return fn(**kwargs)

    def months(self) -> list[str]:
        rows = self.con.execute("SELECT DISTINCT strftime(month, '%Y-%m') FROM snap ORDER BY 1").fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------ helpers
    def _scalar(self, sql: str, params: list) -> float:
        v = self.con.execute(sql, params).fetchone()[0]
        return float(v) if v is not None else 0.0

    # ------------------------------------------------------------------ point-in-time metrics
    def mrr(self, month: str, segment: str | None = None) -> float:
        seg, p = _segment_clause(segment)
        return self._scalar(f"SELECT SUM(mrr) FROM snap a WHERE month = ?{seg}", [_month(month), *p])

    def arr(self, month: str, segment: str | None = None) -> float:
        return 12.0 * self.mrr(month, segment)

    def active_customers(self, month: str, segment: str | None = None) -> float:
        seg, p = _segment_clause(segment)
        return self._scalar(f"SELECT COUNT(*) FROM snap a WHERE month = ?{seg}", [_month(month), *p])

    # ------------------------------------------------------------------ month-over-month movement
    def _movement(self, month: str, segment: str | None) -> str:
        """SQL for per-account (prev_mrr, cur_mrr) between m-1 and m, full outer joined."""
        _month(month)
        return """
            WITH cur AS (SELECT account_id, mrr, segment FROM snap WHERE month = CAST(? AS DATE)),
                 prev AS (SELECT account_id, mrr, segment FROM snap
                          WHERE month = CAST(? AS DATE) - INTERVAL 1 MONTH)
            SELECT COALESCE(cur.account_id, prev.account_id) AS account_id,
                   COALESCE(cur.segment, prev.segment) AS segment,
                   prev.mrr AS prev_mrr, cur.mrr AS cur_mrr
            FROM cur FULL OUTER JOIN prev USING (account_id)
        """

    def _move_agg(self, expr: str, month: str, segment: str | None) -> float:
        seg, p = _segment_clause(segment, "mv")
        m = _month(month)
        return self._scalar(f"SELECT {expr} FROM ({self._movement(month, segment)}) mv WHERE 1=1{seg}", [m, m, *p])

    def new_customers(self, month: str, segment: str | None = None) -> float:
        return self._move_agg("COUNT(*) FILTER (WHERE prev_mrr IS NULL)", month, segment)

    def churned_customers(self, month: str, segment: str | None = None) -> float:
        return self._move_agg("COUNT(*) FILTER (WHERE cur_mrr IS NULL)", month, segment)

    def new_mrr(self, month: str, segment: str | None = None) -> float:
        return self._move_agg("SUM(cur_mrr) FILTER (WHERE prev_mrr IS NULL)", month, segment)

    def expansion_mrr(self, month: str, segment: str | None = None) -> float:
        return self._move_agg("SUM(cur_mrr - prev_mrr) FILTER (WHERE cur_mrr > prev_mrr)", month, segment)

    def contraction_mrr(self, month: str, segment: str | None = None) -> float:
        return self._move_agg("SUM(prev_mrr - cur_mrr) FILTER (WHERE cur_mrr < prev_mrr)", month, segment)

    def churned_mrr(self, month: str, segment: str | None = None) -> float:
        return self._move_agg("SUM(prev_mrr) FILTER (WHERE cur_mrr IS NULL)", month, segment)

    def logo_churn_rate(self, month: str, segment: str | None = None) -> float:
        prev_active = self._move_agg("COUNT(*) FILTER (WHERE prev_mrr IS NOT NULL)", month, segment)
        if prev_active == 0:
            return 0.0
        return self.churned_customers(month, segment) / prev_active

    # ------------------------------------------------------------------ cohort retention
    def _cohort(self, start_month: str, end_month: str, segment: str | None) -> tuple[str, list]:
        s, e = _month(start_month), _month(end_month)
        if e <= s:
            raise MetricError("end_month must be after start_month")
        seg, p = _segment_clause(segment, "st")
        sql = f"""
            SELECT st.account_id, st.mrr AS start_mrr, en.mrr AS end_mrr
            FROM (SELECT * FROM snap WHERE month = CAST(? AS DATE)) st
            LEFT JOIN (SELECT * FROM snap WHERE month = CAST(? AS DATE)) en USING (account_id)
            WHERE 1=1{seg}
        """
        return sql, [s, e, *p]

    def crr(self, start_month: str, end_month: str, segment: str | None = None) -> float:
        sql, p = self._cohort(start_month, end_month, segment)
        start_n, retained = self.con.execute(
            f"SELECT COUNT(*), COUNT(end_mrr) FROM ({sql})", p
        ).fetchone()
        # (E - N) / S  ==  retained-from-start / S
        return float(retained) / start_n if start_n else 0.0

    def nrr(self, start_month: str, end_month: str, segment: str | None = None) -> float:
        sql, p = self._cohort(start_month, end_month, segment)
        start, end = self.con.execute(
            f"SELECT SUM(start_mrr), SUM(COALESCE(end_mrr, 0)) FROM ({sql})", p
        ).fetchone()
        return float(end) / float(start) if start else 0.0

    def grr(self, start_month: str, end_month: str, segment: str | None = None) -> float:
        sql, p = self._cohort(start_month, end_month, segment)
        start, kept = self.con.execute(
            f"SELECT SUM(start_mrr), SUM(LEAST(COALESCE(end_mrr, 0), start_mrr)) FROM ({sql})", p
        ).fetchone()
        return float(kept) / float(start) if start else 0.0

    # ------------------------------------------------------------------ consumption
    def consumption(self, month: str, measure: str = "api_calls", segment: str | None = None) -> float:
        if measure not in USAGE_MEASURES:
            raise MetricError(f"measure must be one of {USAGE_MEASURES}, got {measure!r}")
        seg, p = _segment_clause(segment)
        return self._scalar(
            f"SELECT SUM(u.{measure}) FROM usage_monthly u JOIN accounts a USING (account_id) "
            f"WHERE u.month = ?{seg}",
            [_month(month), *p],
        )
