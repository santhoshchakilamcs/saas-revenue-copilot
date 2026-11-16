"""LangGraph analyst agent. It answers revenue questions only by calling the semantic layer.

Graph:  plan  ->  (refuse | call_tool)  ->  answer

* ``plan``      the backend chooses a tool call. ``offline`` uses the deterministic planner;
                ``ollama`` asks a local LLM to emit a ``get_metric`` tool call (function calling).
                With an LLM backend the deterministic planner runs too, as a second opinion.
* ``call_tool`` executes the call against the semantic layer (typed, validated params; no SQL
                from the model, aggregate results only).
* ``answer``    writes the answer from the tool result. The number in the answer always comes
                from the tool, never from the model's text, so the model cannot invent a figure.

Confidence: 1.0 when the plan is complete; lower when the planner had to assume a period; 0.5 when
the LLM and the deterministic planner disagree; 0.1 when the tool call failed. The API flags any
answer under ``REVIEW_CONFIDENCE_THRESHOLD`` for human review.
"""
from __future__ import annotations

import json
import os
from typing import Any, Literal, TypedDict

from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

from ..metrics.semantic import MetricError, SemanticLayer
from . import planner

METRIC_NAMES = [
    "mrr", "arr", "active_customers", "new_customers", "churned_customers", "logo_churn_rate", "new_mrr",
    "expansion_mrr", "contraction_mrr", "churned_mrr", "crr", "nrr", "grr", "consumption",
]


class AgentState(TypedDict, total=False):
    question: str
    plan: dict
    alt_plan: dict
    result: dict
    answer: str
    confidence: float
    backend: str
    error: str


def make_tool(sl: SemanticLayer):
    @tool
    def get_metric(
        metric: str,
        month: str | None = None,
        start_month: str | None = None,
        end_month: str | None = None,
        segment: str | None = None,
        measure: str | None = None,
    ) -> str:
        """Return one aggregate SaaS metric from the governed semantic layer.

        metric: one of mrr, arr, active_customers, new_customers, churned_customers, logo_churn_rate,
            new_mrr, expansion_mrr, contraction_mrr, churned_mrr, crr, nrr, grr, consumption.
        month: 'YYYY-MM' month-end, for every metric except crr, nrr and grr.
        start_month, end_month: 'YYYY-MM', only for crr (customer retention rate), nrr (net revenue
            retention) and grr (gross revenue retention). "Over 2025" means 2024-12 to 2025-12.
        segment: optional, one of SMB, Mid-Market, Enterprise. Omit for the whole business.
        measure: only for consumption: api_calls, active_users, feature_events or support_tickets.
        """
        kwargs = {k: v for k, v in dict(month=month, start_month=start_month, end_month=end_month,
                                         segment=segment, measure=measure).items() if v}
        value = sl.query(metric, **kwargs)
        return json.dumps({"metric": metric, "params": kwargs, "value": value})

    return get_metric


PointMetric = Literal[
    "mrr", "arr", "active_customers", "new_customers", "churned_customers", "logo_churn_rate",
    "new_mrr", "expansion_mrr", "contraction_mrr", "churned_mrr",
]
Segment = Literal["SMB", "Mid-Market", "Enterprise"]


def make_typed_tools(sl: SemanticLayer):
    """v2 tool schema: three typed tools with enum parameters, one per metric family."""

    @tool
    def point_metric(metric: PointMetric, month: str, segment: Segment | None = None) -> str:
        """Revenue or customer metric at one month-end ('YYYY-MM').
        arr = annual recurring revenue (12 x MRR). mrr = monthly recurring revenue.
        active_customers = paying customers. new_customers = new logos / new accounts that month.
        churned_customers = how many accounts cancelled that month. logo_churn_rate = churn RATE.
        new_mrr = MRR from new customers. expansion_mrr = upsells and seat adds.
        contraction_mrr = downgrades. churned_mrr = MRR lost to churn.
        segment: SMB, Mid-Market or Enterprise; omit for the whole company."""
        kw = {"month": month, **({"segment": segment} if segment else {})}
        return json.dumps({"metric": metric, "params": kw, "value": sl.query(metric, **kw)})

    @tool
    def retention_metric(metric: Literal["crr", "nrr", "grr"], start_month: str, end_month: str,
                         segment: Segment | None = None) -> str:
        """Retention between two month-ends ('YYYY-MM'). crr = customer / logo retention rate.
        nrr = net revenue (net dollar) retention. grr = gross revenue (gross dollar) retention.
        'Over 2025' or 'during 2025' means start_month 2024-12 and end_month 2025-12.
        segment: SMB, Mid-Market or Enterprise; omit for the whole company."""
        kw = {"start_month": start_month, "end_month": end_month, **({"segment": segment} if segment else {})}
        return json.dumps({"metric": metric, "params": kw, "value": sl.query(metric, **kw)})

    @tool
    def usage_metric(measure: Literal["api_calls", "active_users", "feature_events", "support_tickets"],
                     month: str, segment: Segment | None = None) -> str:
        """Product consumption in one month ('YYYY-MM'): total API calls, active users, feature
        events or support tickets. segment: SMB, Mid-Market or Enterprise; omit for the whole company."""
        kw = {"month": month, "measure": measure, **({"segment": segment} if segment else {})}
        return json.dumps({"metric": "consumption", "params": kw, "value": sl.query("consumption", **kw)})

    return {"point_metric": point_metric, "retention_metric": retention_metric, "usage_metric": usage_metric}


def to_metric_args(tool_name: str, args: dict) -> dict:
    """Normalize a v2 tool call to the flat get_metric argument shape."""
    if tool_name == "usage_metric":
        return {"metric": "consumption", **args}
    return dict(args)


def ground(question: str, plan: dict) -> dict:
    """Grounding guardrail: drop a segment filter the question never mentions. Small models
    tend to add one (often 'Enterprise') to every call."""
    plan = dict(plan)
    if "segment" in plan and planner.extract_segment(question) is None:
        plan.pop("segment")
    return plan


SYSTEM = (
    "You are a revenue analyst. Answer by calling the get_metric tool exactly once with the right "
    "metric and parameters. Months are 'YYYY-MM'. 'End of 2024' is 2024-12; Q2 2025 ends 2025-06; "
    "'over 2025' or 'during 2025' for a retention metric means start_month 2024-12, end_month 2025-12. "
    "Segments are exactly SMB, Mid-Market or Enterprise; omit segment for the whole company. "
    "Retention metrics: crr = customer (logo) retention rate, nrr = net revenue retention, "
    "grr = gross revenue retention. Never answer from memory."
)
SYSTEM_V2 = (
    "You are a revenue analyst. Answer by calling exactly one tool: point_metric for a value at one "
    "month-end, retention_metric for retention between two months, usage_metric for product usage. "
    "Months are 'YYYY-MM'. 'End of 2024' is 2024-12; Q2 2025 ends 2025-06. Pass segment only when the "
    "question names SMB (small business), Mid-Market or Enterprise. Never answer from memory."
)


def _fmt(metric: str, v: float) -> str:
    if metric in {"crr", "nrr", "grr", "logo_churn_rate"}:
        return f"{v * 100:.2f}%"
    if metric in {"active_customers", "new_customers", "churned_customers", "consumption"}:
        return f"{v:,.0f}"
    return f"${v:,.2f}"


def _same(a: dict, b: dict) -> bool:
    keys = ("metric", "month", "start_month", "end_month", "segment", "measure")
    na = {k: a.get(k) for k in keys}
    nb = {k: b.get(k) for k in keys}
    if na["metric"] == "consumption" and nb["metric"] == "consumption":
        na["measure"] = na["measure"] or "api_calls"
        nb["measure"] = nb["measure"] or "api_calls"
    return na == nb


class Copilot:
    def __init__(self, sl: SemanticLayer, backend: str | None = None, model: str | None = None,
                 tool_schema: str = "v2", grounding: bool = True):
        self.sl = sl
        self.grounding = grounding
        self.backend = backend or os.environ.get("AGENT_BACKEND", "offline")
        self.tool_schema = tool_schema
        self.tool = make_tool(sl)
        self.typed_tools = make_typed_tools(sl)
        self.llm = None
        if self.backend == "ollama":
            from langchain_ollama import ChatOllama

            llm = ChatOllama(model=model or os.environ.get("OLLAMA_MODEL", "llama3.2"), temperature=0)
            tools = [self.tool] if tool_schema == "v1" else list(self.typed_tools.values())
            self.llm = llm.bind_tools(tools)
            self.system = SYSTEM if tool_schema == "v1" else SYSTEM_V2
        elif self.backend != "offline":
            raise ValueError(f"unknown backend {self.backend!r}")
        self.graph = self._build()

    # ------------------------------------------------------------------ nodes
    def _plan(self, state: AgentState) -> AgentState:
        det = planner.plan(state["question"])
        det_plan = {"tool": det.tool, **det.args}
        if self.llm is None:
            return {"plan": det_plan, "confidence": det.confidence, "backend": "offline"}
        if det.tool == "refuse" and det.confidence >= 1.0:
            # privacy/write guardrail runs before the model sees the question
            return {"plan": det_plan, "confidence": 1.0, "backend": self.backend}
        msg = self.llm.invoke([("system", self.system), ("human", state["question"])])
        calls = getattr(msg, "tool_calls", None) or []
        if not calls:
            return {"plan": {"tool": "none"}, "alt_plan": det_plan, "confidence": 0.1, "backend": self.backend,
                    "error": "model did not call a tool"}
        args = {k: v for k, v in calls[0]["args"].items() if v not in (None, "", "null", "None")}
        llm_plan = {"tool": "get_metric", **to_metric_args(calls[0]["name"], args)}
        if self.grounding:
            llm_plan = ground(state["question"], llm_plan)
        conf = 0.95 if det.tool == "get_metric" and _same(llm_plan, det.args) else 0.5
        return {"plan": llm_plan, "alt_plan": det_plan, "confidence": conf, "backend": self.backend}

    def _route(self, state: AgentState) -> str:
        return {"get_metric": "call_tool"}.get(state["plan"]["tool"], "refuse")

    def _call_tool(self, state: AgentState) -> AgentState:
        args = {k: v for k, v in state["plan"].items() if k != "tool"}
        try:
            res = json.loads(self.tool.invoke(args))
            return {"result": res}
        except (MetricError, TypeError, ValueError) as e:
            return {"error": str(e), "confidence": 0.1}
        except Exception as e:  # pydantic validation errors from the tool schema
            return {"error": f"{type(e).__name__}: {e}", "confidence": 0.1}

    def _refuse(self, state: AgentState) -> AgentState:
        if state["plan"]["tool"] != "refuse":  # the model failed to call a tool: not a refusal, low confidence
            return {"answer": "I could not map that question to a governed metric.", "result": {}}
        reason = state["plan"].get("reason", "out of scope")
        return {"answer": f"I can't answer that from the semantic layer: {reason}.", "result": {"refused": True}}

    def _answer(self, state: AgentState) -> AgentState:
        if "error" in state and "result" not in state:
            return {"answer": f"The metric call failed: {state['error']}"}
        r = state["result"]
        p = r["params"]
        scope = p.get("segment", "all segments")
        period = f"{p['start_month']} to {p['end_month']}" if "start_month" in p else p.get("month", "")
        extra = f" ({p['measure']})" if "measure" in p else ""
        return {"answer": f"{r['metric']}{extra} for {scope}, {period}: {_fmt(r['metric'], r['value'])}."}

    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("plan", self._plan)
        g.add_node("call_tool", self._call_tool)
        g.add_node("refuse", self._refuse)
        g.add_node("answer", self._answer)
        g.set_entry_point("plan")
        g.add_conditional_edges("plan", self._route, {"call_tool": "call_tool", "refuse": "refuse"})
        g.add_edge("call_tool", "answer")
        g.add_edge("refuse", END)
        g.add_edge("answer", END)
        return g.compile()

    def ask(self, question: str) -> dict[str, Any]:
        out = self.graph.invoke({"question": question})
        return {
            "question": question,
            "answer": out.get("answer"),
            "value": (out.get("result") or {}).get("value"),
            "refused": bool((out.get("result") or {}).get("refused")),
            "plan": out.get("plan"),
            "alt_plan": out.get("alt_plan"),
            "confidence": float(out.get("confidence", 0.0)),
            "backend": out.get("backend", self.backend),
            "error": out.get("error"),
        }
