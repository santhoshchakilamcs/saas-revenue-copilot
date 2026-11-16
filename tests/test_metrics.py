"""Semantic-layer metrics: hand-computed values on a tiny dataset, input validation, and an
exhaustive cross-check against the independent pandas reference implementation."""
import duckdb
import pytest

from copilot.metrics.reference import ReferenceMetrics
from copilot.metrics.semantic import MetricError, SemanticLayer

# Jan: A=100 (SMB), B=50 (SMB), C=1000 (Ent).  Feb: A=150 (+50 expansion), B churned (-50),
# C=900 (-100 contraction), D=200 (new SMB).


@pytest.fixture()
def sl(tiny_con):
    return SemanticLayer(tiny_con)


def test_point_metrics(sl):
    assert sl.mrr("2024-01") == 1150
    assert sl.mrr("2024-02") == 1250
    assert sl.arr("2024-02") == 15000
    assert sl.arr("2024-02", segment="SMB") == 12 * 350
    assert sl.active_customers("2024-02") == 3


def test_movements(sl):
    assert sl.new_customers("2024-02") == 1
    assert sl.new_mrr("2024-02") == 200
    assert sl.expansion_mrr("2024-02") == 50
    assert sl.contraction_mrr("2024-02") == 100
    assert sl.churned_customers("2024-02") == 1
    assert sl.churned_mrr("2024-02") == 50
    assert sl.logo_churn_rate("2024-02") == pytest.approx(1 / 3)
    # MRR bridge: start + new + expansion - contraction - churn = end
    assert sl.mrr("2024-01") + 200 + 50 - 100 - 50 == sl.mrr("2024-02")


def test_retention(sl):
    assert sl.nrr("2024-01", "2024-02") == pytest.approx((150 + 0 + 900) / 1150)
    assert sl.grr("2024-01", "2024-02") == pytest.approx((100 + 0 + 900) / 1150)
    assert sl.crr("2024-01", "2024-02") == pytest.approx((3 - 1) / 3)  # (E - N) / S
    assert sl.nrr("2024-01", "2024-02", segment="SMB") == pytest.approx(150 / 150)
    assert sl.crr("2024-01", "2024-02", segment="Enterprise") == 1.0


def test_consumption(sl):
    assert sl.consumption("2024-02") == 1200 + 8000 + 500
    assert sl.consumption("2024-02", measure="active_users", segment="SMB") == 14


@pytest.mark.parametrize("call", [
    lambda s: s.mrr("2024-13"),
    lambda s: s.mrr("Jan 2024"),
    lambda s: s.arr("2024-01", segment="Startup"),
    lambda s: s.consumption("2024-01", measure="rm -rf"),
    lambda s: s.nrr("2024-02", "2024-01"),
    lambda s: s.query("ltv", month="2024-01"),
    lambda s: s.query("arr", month="2024-01", start_month="2024-01"),
])
def test_validation(sl, call):
    with pytest.raises(MetricError):
        call(sl)


def test_segment_is_parameterized_not_interpolated(sl):
    with pytest.raises(MetricError):
        sl.mrr("2024-01", segment="SMB' OR '1'='1")


def test_registry_lists_all(sl):
    names = {m.name for m in sl.list_metrics()}
    assert {"arr", "mrr", "crr", "nrr", "grr", "consumption", "expansion_mrr", "contraction_mrr"} <= names


def test_sql_layer_matches_reference(gen_db):
    con = duckdb.connect(gen_db, read_only=True)
    sl, ref = SemanticLayer(con), ReferenceMetrics.from_duckdb(con)
    months = ["2024-02", "2024-07", "2024-12", "2025-05", "2025-12"]
    point = ["mrr", "arr", "active_customers", "new_customers", "churned_customers", "logo_churn_rate",
             "new_mrr", "expansion_mrr", "contraction_mrr", "churned_mrr"]
    checked = 0
    for seg in [None, "SMB", "Mid-Market", "Enterprise"]:
        for m in months:
            for name in point:
                assert sl.query(name, month=m, segment=seg) == pytest.approx(getattr(ref, name)(m, seg)), (name, m, seg)
                checked += 1
            for measure in ["api_calls", "active_users"]:
                assert sl.consumption(m, measure, seg) == ref.consumption(m, measure, seg)
                checked += 1
        for s, e in [("2024-01", "2024-12"), ("2024-12", "2025-12"), ("2025-03", "2025-09")]:
            for name in ["crr", "nrr", "grr"]:
                assert sl.query(name, start_month=s, end_month=e, segment=seg) == pytest.approx(getattr(ref, name)(s, e, seg))
                checked += 1
    assert checked == 4 * (5 * 12 + 9)
