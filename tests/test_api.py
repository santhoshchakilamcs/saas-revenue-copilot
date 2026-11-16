import pytest
from fastapi.testclient import TestClient

from copilot.api.main import create_app
from copilot.api.security import AuthConfig, UserStore


@pytest.fixture()
def client(gen_db):
    users = UserStore()
    users.add("alice", "viewer-pw", "viewer")
    users.add("root", "admin-pw", "admin")
    app = create_app(db_path=gen_db, users=users, auth=AuthConfig(secret="test-secret-that-is-at-least-32-bytes-long", expire_minutes=5),
                     backend="offline", threshold=0.7)
    return TestClient(app, raise_server_exceptions=False)


def _token(c, u, p):
    r = c.post("/auth/token", data={"username": u, "password": p})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_login_and_me(client):
    h = _token(client, "alice", "viewer-pw")
    assert client.get("/me", headers=h).json() == {"username": "alice", "role": "viewer"}


def test_bad_password_and_missing_or_forged_token(client):
    assert client.post("/auth/token", data={"username": "alice", "password": "nope"}).status_code == 401
    assert client.get("/v1/metrics").status_code == 401
    import jwt
    forged = jwt.encode({"sub": "root", "role": "admin"}, "a-different-secret-also-32-bytes-long!", algorithm="HS256")
    assert client.get("/v1/reviews", headers={"Authorization": f"Bearer {forged}"}).status_code == 401


def test_rbac(client):
    viewer = _token(client, "alice", "viewer-pw")
    admin = _token(client, "root", "admin-pw")
    assert client.get("/v1/reviews", headers=viewer).status_code == 403
    assert client.get("/ops/metrics", headers=viewer).status_code == 403
    assert client.get("/v1/reviews", headers=admin).status_code == 200
    assert client.get("/v1/metrics", headers=admin).status_code == 200  # admin inherits viewer


def test_metric_route_and_errors(client):
    h = _token(client, "alice", "viewer-pw")
    r = client.get("/v1/metrics/arr", params={"month": "2025-12", "segment": "SMB"}, headers=h)
    assert r.status_code == 200 and r.json()["value"] > 0
    bad = client.get("/v1/metrics/arr", params={"month": "December"}, headers=h)
    assert bad.status_code == 422 and bad.json()["error"] == "invalid_metric_request"
    assert client.get("/v1/metrics/ltv", params={"month": "2025-12"}, headers=h).status_code == 422


def test_ask_confident_answer_not_flagged(client):
    h = _token(client, "alice", "viewer-pw")
    r = client.post("/v1/ask", json={"question": "What was NRR over 2025?"}, headers=h).json()
    assert r["flagged"] is False and r["confidence"] == 1.0 and r["value"] > 0


def test_human_in_the_loop_flow(client):
    viewer = _token(client, "alice", "viewer-pw")
    admin = _token(client, "root", "admin-pw")
    # no period -> low confidence -> auto-flagged into the review queue
    low = client.post("/v1/ask", json={"question": "What is our ARR?"}, headers=viewer).json()
    assert low["flagged"] is True and low["review_id"]
    # a viewer flags a confident answer by hand
    ok = client.post("/v1/ask", json={"question": "ARR in 2025-06?"}, headers=viewer).json()
    f = client.post(f"/v1/answers/{ok['answer_id']}/flag", json={"reason": "looks high vs board deck"}, headers=viewer)
    assert f.status_code == 201
    dup = client.post(f"/v1/answers/{ok['answer_id']}/flag", json={"reason": "again"}, headers=viewer).json()
    assert dup["duplicate"] is True
    queue = client.get("/v1/reviews", headers=admin).json()
    assert {q["answer_id"] for q in queue} == {low["answer_id"], ok["answer_id"]}
    rid = f.json()["review_id"]
    assert client.post(f"/v1/reviews/{rid}/resolve", json={"verdict": "corrected"}, headers=admin).status_code == 422
    done = client.post(f"/v1/reviews/{rid}/resolve", json={"verdict": "approved", "note": "matches finance"}, headers=admin)
    assert done.status_code == 200 and done.json()["status"] == "resolved" and done.json()["resolved_by"] == "root"
    assert client.post(f"/v1/reviews/{rid}/resolve", json={"verdict": "approved"}, headers=admin).status_code == 409
    assert client.post("/v1/answers/nope/flag", json={"reason": "xyz"}, headers=viewer).status_code == 404


def test_pii_redacted_in_audit_log(client):
    h = _token(client, "alice", "viewer-pw")
    r = client.post("/v1/ask", json={"question": "ARR in 2025-06? cc jane.doe@example.com"}, headers=h).json()
    stored = client.app.state.store.get_answer(r["answer_id"])
    assert "example.com" not in stored["question"] and "[EMAIL]" in stored["question"]


def test_observability(client):
    h = _token(client, "root", "admin-pw")
    r = client.get("/healthz", headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123"
    client.post("/v1/ask", json={"question": "ARR in 2025-06?"}, headers=h)
    text = client.get("/ops/metrics", headers=h).text
    assert 'route="/v1/ask",method="POST",status="200"' in text
    assert 'agent_answers_total{outcome="answered"}' in text


def test_validation_errors(client):
    h = _token(client, "alice", "viewer-pw")
    assert client.post("/v1/ask", json={"question": ""}, headers=h).status_code == 422
    assert client.post("/v1/ask", json={}, headers=h).status_code == 422
