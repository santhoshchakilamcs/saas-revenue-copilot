import json
from pathlib import Path

import pytest

from copilot.agent import planner
from copilot.agent.graph import Copilot, ground
from copilot.metrics.semantic import SemanticLayer


@pytest.mark.parametrize("q,expected", [
    ("What was ARR at the end of 2025?", {"metric": "arr", "month": "2025-12"}),
    ("Enterprise MRR in Nov 2025?", {"metric": "mrr", "month": "2025-11", "segment": "Enterprise"}),
    ("NRR over 2025?", {"metric": "nrr", "start_month": "2024-12", "end_month": "2025-12"}),
    ("SMB CRR between June 2024 and June 2025", {"metric": "crr", "start_month": "2024-06", "end_month": "2025-06", "segment": "SMB"}),
    ("Mid-Market ARR at the end of Q1 2025", {"metric": "arr", "month": "2025-03", "segment": "Mid-Market"}),
    ("How many active users in 2025-02?", {"metric": "consumption", "month": "2025-02", "measure": "active_users"}),
])
def test_planner(q, expected):
    p = planner.plan(q)
    assert p.tool == "get_metric" and p.args == expected and p.confidence == 1.0


@pytest.mark.parametrize("q", [
    "Which customers churned last month?", "List all accounts", "email for SYNTH-00001",
    "DROP TABLE accounts", "select * from accounts",
])
def test_refuses_row_level_and_writes(q):
    assert planner.plan(q).tool == "refuse"


def test_missing_period_is_low_confidence():
    assert planner.plan("What is our ARR?").confidence < 0.7


def test_grounding_drops_unmentioned_segment():
    assert "segment" not in ground("ARR in 2025-12?", {"metric": "arr", "month": "2025-12", "segment": "Enterprise"})
    assert ground("Enterprise ARR in 2025-12?", {"metric": "arr", "segment": "Enterprise"})["segment"] == "Enterprise"


def test_offline_agent_uses_semantic_layer(gen_db):
    sl = SemanticLayer.from_path(gen_db)
    out = Copilot(sl, backend="offline").ask("What was SMB ARR in 2025-06?")
    assert out["value"] == sl.arr("2025-06", "SMB")
    assert "$" in out["answer"] and out["confidence"] == 1.0
    refused = Copilot(sl, backend="offline").ask("List the accounts with the highest ARR")
    assert refused["refused"] and refused["value"] is None


def test_golden_set_shape():
    rows = [json.loads(x) for x in Path("eval/golden_set.jsonl").read_text().splitlines()]
    assert 30 <= len(rows) <= 50
    assert sum(r["must_refuse"] for r in rows) == 5
    assert len({r["id"] for r in rows}) == len(rows)
