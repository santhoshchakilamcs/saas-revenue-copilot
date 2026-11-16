"""Independent pandas reference implementation of the metric definitions.

It shares no code with the SQL semantic layer. It exists for two reasons:
1. tests cross-check every SQL metric against it, so a SQL bug shows up as a mismatch;
2. the agent's golden-set answers are computed here, so the agent is graded against numbers the
   semantic layer did not produce.
"""
from __future__ import annotations

import pandas as pd


class ReferenceMetrics:
    def __init__(self, accounts: pd.DataFrame, subs: pd.DataFrame, usage: pd.DataFrame):
        seg = accounts.set_index("account_id")["segment"]
        self.subs = subs.assign(month=pd.to_datetime(subs["month"]), segment=subs["account_id"].map(seg))
        self.usage = usage.assign(month=pd.to_datetime(usage["month"]), segment=usage["account_id"].map(seg))

    @classmethod
    def from_duckdb(cls, con) -> "ReferenceMetrics":
        return cls(
            con.execute("SELECT * FROM accounts").df(),
            con.execute("SELECT * FROM subscriptions_monthly").df(),
            con.execute("SELECT * FROM usage_monthly").df(),
        )

    def _snap(self, month: str, segment: str | None) -> pd.Series:
        df = self.subs[self.subs["month"] == pd.Timestamp(f"{month}-01")]
        if segment:
            df = df[df["segment"] == segment]
        return df.set_index("account_id")["mrr"]

    @staticmethod
    def _prev(month: str) -> str:
        return (pd.Timestamp(f"{month}-01") - pd.DateOffset(months=1)).strftime("%Y-%m")

    def mrr(self, month, segment=None):
        return float(self._snap(month, segment).sum())

    def arr(self, month, segment=None):
        return 12 * self.mrr(month, segment)

    def active_customers(self, month, segment=None):
        return float(len(self._snap(month, segment)))

    def _pair(self, month, segment):
        cur, prev = self._snap(month, segment), self._snap(self._prev(month), segment)
        return cur, prev

    def new_customers(self, month, segment=None):
        cur, prev = self._pair(month, segment)
        return float(len(cur.index.difference(prev.index)))

    def churned_customers(self, month, segment=None):
        cur, prev = self._pair(month, segment)
        return float(len(prev.index.difference(cur.index)))

    def logo_churn_rate(self, month, segment=None):
        cur, prev = self._pair(month, segment)
        return float(len(prev.index.difference(cur.index)) / len(prev)) if len(prev) else 0.0

    def new_mrr(self, month, segment=None):
        cur, prev = self._pair(month, segment)
        return float(cur[cur.index.difference(prev.index)].sum())

    def _deltas(self, month, segment):
        cur, prev = self._pair(month, segment)
        both = cur.index.intersection(prev.index)
        return cur[both] - prev[both]

    def expansion_mrr(self, month, segment=None):
        d = self._deltas(month, segment)
        return float(d[d > 0].sum())

    def contraction_mrr(self, month, segment=None):
        d = self._deltas(month, segment)
        return float(-d[d < 0].sum())

    def churned_mrr(self, month, segment=None):
        cur, prev = self._pair(month, segment)
        return float(prev[prev.index.difference(cur.index)].sum())

    def crr(self, start_month, end_month, segment=None):
        s, e = self._snap(start_month, segment), self._snap(end_month, segment)
        n_new = len(e.index.difference(s.index))
        return (len(e) - n_new) / len(s) if len(s) else 0.0

    def nrr(self, start_month, end_month, segment=None):
        s, e = self._snap(start_month, segment), self._snap(end_month, segment)
        return float(e.reindex(s.index).fillna(0).sum() / s.sum())

    def grr(self, start_month, end_month, segment=None):
        s, e = self._snap(start_month, segment), self._snap(end_month, segment)
        kept = pd.concat([s, e.reindex(s.index).fillna(0)], axis=1).min(axis=1)
        return float(kept.sum() / s.sum())

    def consumption(self, month, measure="api_calls", segment=None):
        df = self.usage[self.usage["month"] == pd.Timestamp(f"{month}-01")]
        if segment:
            df = df[df["segment"] == segment]
        return float(df[measure].sum())
